// MIT License

// Copyright (c) 2023 NeSC-IV

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "sageICP.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "sage_icp/core/Deskew.hpp"
#include "sage_icp/core/Preprocessing.hpp"
#include "sage_icp/core/Registration.hpp"
#include "sage_icp/core/VoxelHashMap.hpp"

namespace sage_icp::pipeline {

sageICP::Vector4dVectorTuple sageICP::RegisterFrame(const std::vector<Eigen::Vector4d> &frame,
                                                    const std::vector<double> &timestamps) {
    const auto &deskew_frame = [&]() -> std::vector<Eigen::Vector4d> {
        if (!config_.deskew) return frame;
        // TODO(Nacho) Add some asserts here to sanitize the timestamps

        //  If not enough poses for the estimation, do not de-skew
        const size_t N = poses().size();
        if (N <= 2) return frame;

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];
        return DeSkewScan(frame, timestamps, start_pose, finish_pose);
    }();
    return RegisterFrame(deskew_frame);
}

sageICP::Vector4dVectorTuple sageICP::RegisterFrame(const std::vector<Eigen::Vector4d> &frame) {
    // Preprocess the input cloud
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    const auto &cropped_frame = Preprocess(frame, config_.max_range, config_.min_range, config_.dynamic_vehicle_filter, config_.dynamic_vehicle_filter_th);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(cropped_frame);  // source为二次降采样的点云，frame_downsample为一次降采样的点云

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    // Compute initial_guess for ICP
    const auto prediction = GetPredictionModel(); // 根据过去两帧之间的相对位姿
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();  // 上一帧的位姿
    const auto initial_guess = last_pose * prediction;  // 预测位姿

    // Run icp
    auto start = std::chrono::high_resolution_clock::now();
    const Sophus::SE3d new_pose = sage_icp::RegisterFrame(source,         //
                                                          sem_map_,     //
                                                          initial_guess,  //
                                                          3.0 * sigma,    // max_correspondence_distance
                                                          sigma / 3.0,
                                                          config_.sem_th);   // kernel
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout<<"ICP time: "<< elapsed.count()<<" s"<<std::endl;
    std::chrono::duration<double> elapsed_all = end - preprocess_start;
    std::cout<<"All time: "<< elapsed_all.count()<<" s"<<std::endl;
    const auto model_deviation = initial_guess.inverse() * new_pose;  // 预测位姿与实际位姿的差值
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    sem_map_.Update(frame_downsample, new_pose);  // 更新局部地图
    poses_.push_back(new_pose);                    // 更新位姿
    return {frame, elapsed.count(), elapsed_all.count()};
}

void sageICP::lasermap_fov_segment(const Sophus::SE3d &new_pose){
    cub_needrm.clear();
    const Eigen::Vector3d &pos_LiD = new_pose.translation(); // 机器人在世界坐标系下的位置
    if (!Globalmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = static_cast<float>(pos_LiD(i)) - config_.MAP_RANGE / 2.0f;
            LocalMap_Points.vertex_max[i] = static_cast<float>(pos_LiD(i)) + config_.MAP_RANGE / 2.0f;
        }
        Globalmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(static_cast<float>(pos_LiD(i)) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(static_cast<float>(pos_LiD(i)) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= config_.MOV_THRESHOLD * static_cast<float>(config_.max_range) || dist_to_map_edge[i][1] <= config_.MOV_THRESHOLD * static_cast<float>(config_.max_range)) need_move = true;
    }
    // std::cout<<dist_to_map_edge<<endl;
    if (!need_move) return;
    std::cout<<"need move!"<<std::endl;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((config_.MAP_RANGE - 2.0f * config_.MOV_THRESHOLD * static_cast<float>(config_.max_range)) * 0.5f * 0.9f, float(static_cast<float>(config_.max_range) * (config_.MOV_THRESHOLD -1)));
    // float mov_dist = max((config_.MAP_RANGE - config_.MOV_THRESHOLD * config_.max_range) * 0.5f * 0.9f, float(config_.max_range * (config_.MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= config_.MOV_THRESHOLD * static_cast<float>(config_.max_range)){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= config_.MOV_THRESHOLD * static_cast<float>(config_.max_range)){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    // points_cache_collect();
    // double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) {
        local_map_.Delete_Point_Boxes(cub_needrm);
        ground_map_.Delete_Point_Boxes(cub_needrm); 
        //kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    }
    // kdtree_delete_time = omp_get_wtime() - delete_begin;
}


sageICP::Vector4dVectorTuple2 sageICP::Voxelize(const std::vector<Eigen::Vector4d> &frame) const {
    // const auto voxel_size = config_.voxel_size_map;
    const auto frame_downsample = sage_icp::VoxelDownsample(frame, config_.voxel_size_road * 0.5, config_.voxel_size_building * 0.5, config_.voxel_size_plant * 0.5,
                                                     config_.voxel_size_object * 0.5, config_.voxel_size_unlabel * 0.5, config_.voxel_size_vehicle * 0.5);
    const auto source = sage_icp::VoxelDownsample(frame_downsample, config_.voxel_size_road * 1.5, config_.voxel_size_building * 1.5, config_.voxel_size_plant * 1.5,
                                                     config_.voxel_size_object * 1.5, config_.voxel_size_unlabel * 1.5, config_.voxel_size_vehicle * 1.5);
    return {source, frame_downsample};
}

double sageICP::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d sageICP::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool sageICP::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

}  // namespace sage_icp::pipeline
