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
    const auto &cropped_frame = Preprocess(frame,
                                        config_.max_range,
                                        config_.min_range,
                                        config_.label_max_range,
                                        config_.dynamic_vehicle_filter,
                                        config_.dynamic_vehicle_filter_th,
                                        config_.voxel_labels[config_.dynamic_vehicle_voxid],
                                        config_.dynamic_remove_lankmark
                                        );

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(cropped_frame);  // source is double downsample pc，frame_downsample is single downsample pc

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    // Compute initial_guess for ICP
    const auto prediction = GetPredictionModel(); // according to the last two poses to predict the current pose
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();  // last pose
    const auto initial_guess = last_pose * prediction;  // initial guess

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

    const auto model_deviation = initial_guess.inverse() * new_pose;  // deviation between initial guess and new pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    sem_map_.Update(frame_downsample, new_pose);  // update map
    poses_.push_back(new_pose);                    // update poses
    return {source, elapsed.count(), elapsed_all.count()};
}

sageICP::Vector4dVectorTuple2 sageICP::Voxelize(const std::vector<Eigen::Vector4d> &frame) const {
    const auto frame_downsample = sage_icp::VoxelDownsample(frame, config_.voxel_labels, config_.voxel_size, 0.5);
    const auto source = sage_icp::VoxelDownsample(frame_downsample, config_.voxel_labels, config_.voxel_size, 1.5);
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
