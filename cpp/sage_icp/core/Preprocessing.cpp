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
#include "Preprocessing.hpp"

#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <vector>

namespace {
using Voxel = Eigen::Vector3i;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};
}  // namespace

namespace sage_icp {
std::vector<Eigen::Vector4d> VoxelDownsample(const std::vector<Eigen::Vector4d> &frame, double voxel_size_road, double voxel_size_building, double voxel_size_plant,
                                                     double voxel_size_object, double voxel_size_unlabel, double voxel_size_vehicle) {
    std::vector<double> voxel_size_group = {voxel_size_road, voxel_size_building, voxel_size_plant, voxel_size_object, voxel_size_unlabel, voxel_size_vehicle};
    // 点云分类
    // ******************************************* //
    // 0 : "unlabeled"
    // 1 : "outlier"
    
    // 10: "car"
    // 11: "bicycle"
    // 13: "bus"
    // 15: "motorcycle"
    // 16: "on-rails"
    // 18: "truck"
    // 20: "other-vehicle"
    
    // 30: "person"
    // 31: "bicyclist"
    // 32: "motorcyclist"

    // 40: "road"
    // 44: "parking"
    // 48: "sidewalk"
    // 49: "other-ground"
    
    // 50: "building"
    // 51: "fence"
    // 52: "other-structure"
    
    // 60: "lane-marking"
    // 70: "vegetation"
    // 71: "trunk"
    // 72: "terrain"
    // 80: "pole"
    // 81: "traffic-sign"
    // 99: "other-object"
    
    // 252: "moving-car"
    // 253: "moving-bicyclist"
    // 254: "moving-person"
    // 255: "moving-motorcyclist"
    // 256: "moving-on-rails"
    // 257: "moving-bus"
    // 258: "moving-truck"
    // 259: "moving-other-vehicle"
    std::vector<tsl::robin_map<Voxel, Eigen::Vector4d, VoxelHash>> grid_group(6);
    for (int i = 0; i < 6; i++)
    {
        tsl::robin_map<Voxel, Eigen::Vector4d, VoxelHash> grid;
        grid.reserve(frame.size());
        grid_group.emplace_back(grid);
    }

    for (const auto &point : frame) {
        int label = static_cast<int>(point[3]);
        int group = -1;
        if (label > 39 && label < 50) group = 0; // road
        else if (label > 49 && label < 60) group = 1; // building
        else if (label == 70 || label == 72) group = 2; // vegetation
        else if (label == 71 || label == 60 || (label < 100 && label > 72)) group = 3; // object
        else if (label == 0) group = 4; // unlabel
        else if (label > 9 && label < 21) group = 5; // vehicle
        else continue; 
        const auto voxel = Voxel((point.head<3>() / voxel_size_group[group]).cast<int>()); //获取体素坐标
        if (grid_group[group].contains(voxel)) continue;
        grid_group[group].insert({voxel, point});
    }
    std::vector<Eigen::Vector4d> frame_dowsampled;
    frame_dowsampled.reserve(frame.size());
    for (int i = 0; i < 6; i++)
    {
        for (const auto &[voxel, point] : grid_group[i]) {
            (void)voxel;
            frame_dowsampled.emplace_back(point);
        }
    }
    return frame_dowsampled;
}

std::vector<Eigen::Vector4d> Preprocess(const std::vector<Eigen::Vector4d> &frame,
                                        double max_range,
                                        double min_range,
                                        bool dynamic_vehicle_filter,
                                        double dy_th) {
    std::vector<Eigen::Vector4d> inliers;
    if(dynamic_vehicle_filter){
        pcl::PointCloud<pcl::PointXYZL>::Ptr map_all(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::PointCloud<pcl::PointXYZL>::Ptr map_vehicle(new pcl::PointCloud<pcl::PointXYZL>);
        std::vector<Eigen::Vector4d> vehicle_inliers;
        for (const auto &point : frame) {
            Eigen::Vector4d point_new = point;
            const double norm = point.head<3>().norm(); // Euclidean norm
            if (norm < max_range && norm > min_range){
                if (norm > 50) point_new[3] = 0.0;
                pcl::PointXYZL point_temp;
                point_temp.x = static_cast<float>(point[0]);
                point_temp.y = static_cast<float>(point[1]);
                point_temp.z = static_cast<float>(point[2]);
                point_temp.label = static_cast<uint32_t>(point_new[3]);
                map_all->points.emplace_back(point_temp);
                if(point_temp.label > 9 && point_temp.label < 21){
                    map_vehicle->points.emplace_back(point_temp);
                    vehicle_inliers.emplace_back(point_new);
                }
                else{
                    inliers.emplace_back(point_new);
                }
            }
        }
        /// 创建map树
        pcl::KdTreeFLANN<pcl::PointXYZL> maptree;  // 创建一个KdTreeFLANN对象，输入点云。
	    maptree.setInputCloud(map_all);

        /// 创建kd树
        pcl::search::KdTree<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>);
        tree->setInputCloud(map_vehicle);
        /// 设置分割参数
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
        ec.setClusterTolerance(0.5);	//设置近邻搜索的半径
        ec.setMinClusterSize(5);		//设置最小聚类点数
        ec.setMaxClusterSize(static_cast<int>(map_vehicle->size()));	//设置最大聚类点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(map_vehicle);
        ec.extract(cluster_indices);    //从点云中提取聚类，并将点云索引保存在cluster_indices中

        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++){
            bool is_static_vehicle = false;
            int cluster_size = static_cast<int>(it->indices.size());
            int count_size = 0;
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
                // R半径搜索：在半径r内搜索近邻。
                vector<int> pointIdxRadiusSearch;  // 存储近邻索引
                vector<float> pointRadiusSquaredDistance;   // 存储近邻对应的平均距离
                int b = maptree.radiusSearch(map_vehicle->points[*pit], 0.5f, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                if (b > 0)
                {
                    for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++)
                    {
                        if (map_all->points[pointIdxRadiusSearch[i]].label == 44 || map_all->points[pointIdxRadiusSearch[i]].label == 48)
                        {
                            count_size++;
                            if (count_size > static_cast<int>(dy_th * static_cast<double>(cluster_size)))
                            {
                                is_static_vehicle = true;
                                break;
                            }
                        }
                    }
                }
            }
            if(is_static_vehicle){
                for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++){
                    inliers.emplace_back(vehicle_inliers[*pit]);
                }
            }
        }
    }
    else{
        std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt) {
            Eigen::Vector3d v3point(pt[0], pt[1], pt[2]);
            const double norm = v3point.norm(); // Euclidean norm
            return norm < max_range && norm > min_range;
        });
    }
    return inliers;
}

}  // namespace sage_icp
