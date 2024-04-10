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
#include "sage_icp/core/ikd_Tree.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/transforms.h>
#include <chrono>

namespace sage_icp::pipeline {
using XYZLDTuple = std::tuple<XYZLD, double>;
struct sageConfig {
    // map params
    double voxel_size_map = 1.0;
    double voxel_size_road = 1.0;
    double voxel_size_building = 1.0;
    double voxel_size_plant = 1.0;
    double voxel_size_object = 1.0;
    double voxel_size_unlabel = 1.0;
    double voxel_size_vehicle = 1.0;
    
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;
    double sem_th = 0.4;

    // Motion compensation
    bool deskew = false;

    float MAP_RANGE = 1000.0f; // > DET_RANGE*3
    float MOV_THRESHOLD = 1.5f;

    bool use_global_map = false;
    bool dynamic_vehicle_filter = false;
    double dynamic_vehicle_filter_th = 0.5;
};

class sageICP {
public:
    using Vector4dVector = std::vector<Eigen::Vector4d>;
    using Vector4dVectorTuple = std::tuple<Vector4dVector, double, double>;
    using Vector4dVectorTuple2 = std::tuple<Vector4dVector, Vector4dVector>;


public:
    explicit sageICP(const sageConfig &config)
        : config_(config),
        sem_map_(config.voxel_size_map, config_.max_range, config_.max_points_per_voxel),
        adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

    sageICP() : sageICP(sageConfig{}) {}

public:
    Vector4dVectorTuple RegisterFrame(const std::vector<Eigen::Vector4d> &frame);
    Vector4dVectorTuple RegisterFrame(const std::vector<Eigen::Vector4d> &frame,
                                      const std::vector<double> &timestamps);
    void lasermap_fov_segment(const Sophus::SE3d &new_pose);
    Vector4dVectorTuple2 Voxelize(const std::vector<Eigen::Vector4d> &frame) const;
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();

public:
    // Extra C++ API to facilitate ROS debugging
    XYZLD GlobalMap() {
        XYZLF ().swap(local_map_.PCL_Storage);
        local_map_.flatten(local_map_.Root_Node, local_map_.PCL_Storage, NOT_RECORD);
        XYZLF ().swap(ground_map_.PCL_Storage);
        ground_map_.flatten(ground_map_.Root_Node, ground_map_.PCL_Storage, NOT_RECORD);
        local_map_.PCL_Storage.insert(local_map_.PCL_Storage.end(), ground_map_.PCL_Storage.begin(), ground_map_.PCL_Storage.end());
        XYZLD local_mapd;
        for (auto &ptf : local_map_.PCL_Storage) {
            XYZLD_PointType ptd;
            ptd.x = ptf.x;
            ptd.y = ptf.y;
            ptd.z = ptf.z;
            ptd.label = ptf.label;
            local_mapd.emplace_back(ptd);
        }
        return local_mapd;
    };
    std::vector<Eigen::Vector4d> LocalMap() const { return sem_map_.Pointcloud(); };
    std::vector<Sophus::SE3d> poses() const { return poses_; };

    bool reinitialize() { 
        poses_.clear();
        cub_needrm.clear();
        Globalmap_Initialized = false;
        LocalMap_Points = BoxPointType();
        adaptive_threshold_ = AdaptiveThreshold(config_.initial_threshold, config_.min_motion_th, config_.max_range);
        local_map_.reinit();
        ground_map_.reinit();
        sem_map_.Clear();
        return true;
    };

private:
    // SAGE-ICP pipeline modules
    std::vector<Sophus::SE3d> poses_;
    sageConfig config_;
    KD_TREE<pcl::PointXYZL> local_map_;
    KD_TREE<pcl::PointXYZL> ground_map_;
    vector<BoxPointType> cub_needrm;
    BoxPointType LocalMap_Points;
    bool Globalmap_Initialized = false;
    // local map
    VoxelHashMap sem_map_;
    AdaptiveThreshold adaptive_threshold_;

};

}  // namespace sage_icp::pipeline
