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
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>
#include <vector>
#include "sage_icp/core/Deskew.hpp"
#include "sage_icp/core/Threshold.hpp"
#include "sage_icp/core/VoxelHashMap.hpp"
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/filters/uniform_sampling.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/registration/transforms.h>
#include <chrono>

namespace sage_icp::pipeline {
struct sageConfig {
    std::vector<std::vector<int>> voxel_labels;
    std::vector<double> voxel_size;
    // map params
    double voxel_size_map = 1.0;
    
    double max_range = 100.0;
    double min_range = 5.0;
    double label_max_range = 50.0;
    double local_map_range = 100.0;
    int basic_points_per_voxel = 20;
    int critical_points_per_voxel = 20;
    std::vector<int> basic_parts_labels;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;
    double sem_th = 0.4;

    // Motion compensation
    bool deskew = false;

    bool dynamic_vehicle_filter = false;
    double dynamic_vehicle_filter_th = 0.5;
    int dynamic_vehicle_voxid = 5;
    std::vector<int> dynamic_remove_lankmark;
};

class sageICP {
public:
    using Vector4dVector = std::vector<Eigen::Vector4d>;
    using Vector4dVectorTuple = std::tuple<Vector4dVector, double, double>;
    using Vector4dVectorTuple2 = std::tuple<Vector4dVector, Vector4dVector>;

    explicit sageICP(const sageConfig &config)
        : config_(config),
        sem_map_(config.voxel_size_map, config_.local_map_range, config_.basic_points_per_voxel, config_.critical_points_per_voxel, config_.basic_parts_labels),
        adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

    sageICP() : sageICP(sageConfig{}) {}

    Vector4dVectorTuple RegisterFrame(const std::vector<Eigen::Vector4d> &frame);
    Vector4dVectorTuple RegisterFrame(const std::vector<Eigen::Vector4d> &frame,
                                      const std::vector<double> &timestamps);
    Vector4dVectorTuple2 Voxelize(const std::vector<Eigen::Vector4d> &frame) const;
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();
    std::vector<Eigen::Vector4d> TransformToLastFrame(const Sophus::SE3d &last_pose,
                                                        const Sophus::SE3d &current_pose,
                                                        const std::vector<Eigen::Vector4d> &points);

    // Extra C++ API to facilitate ROS debugging
    std::vector<Eigen::Vector4d> LocalMap() const { return sem_map_.Pointcloud(); };
    std::vector<Sophus::SE3d> poses() const { return poses_; };
    bool reinitialize() { 
        poses_.clear();
        adaptive_threshold_ = AdaptiveThreshold(config_.initial_threshold, config_.min_motion_th, config_.max_range);
        sem_map_.Clear();
        return true;
    };

private:
    // SAGE-ICP pipeline modules
    std::vector<Sophus::SE3d> poses_;
    sageConfig config_;
    // local map
    VoxelHashMap sem_map_;
    AdaptiveThreshold adaptive_threshold_;

};

}  // namespace sage_icp::pipeline
