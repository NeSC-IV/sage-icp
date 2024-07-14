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
std::vector<Eigen::Vector4d> VoxelDownsample(const std::vector<Eigen::Vector4d> &frame,
                                            const std::vector<std::vector<int>> &voxel_labels,
                                            const std::vector<double> &voxel_size,
                                            double vox_scale) {
    int voxel_size_len = static_cast<int>(voxel_size.size());
    // init grid group
    std::vector<tsl::robin_map<Voxel, Eigen::Vector4d, VoxelHash>> grid_group(voxel_size_len);
    for (int i = 0; i < voxel_size_len; i++)
    {
        tsl::robin_map<Voxel, Eigen::Vector4d, VoxelHash> grid;
        grid.reserve(frame.size());
        grid_group.emplace_back(grid);
    }
    // insert points into grid
    for (const auto &point : frame) {
        int label = static_cast<int>(point[3]);
        int group = -1;
        for (int i = 0; i < voxel_size_len; i++)
        {
            if (std::find(voxel_labels[i].begin(), voxel_labels[i].end(), label) != voxel_labels[i].end())
            {
                group = i;
                break;
            }
        }
        if (group == -1) continue;
        const auto voxel = Voxel((point.head<3>() / (voxel_size[group] * vox_scale)).cast<int>()); //获取体素坐标
        if (grid_group[group].contains(voxel)) continue;
        grid_group[group].insert({voxel, point});
    }
    std::vector<Eigen::Vector4d> frame_dowsampled;
    frame_dowsampled.reserve(frame.size());
    for (int i = 0; i < voxel_size_len; i++)
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
                                        double label_max_range,
                                        bool dynamic_vehicle_filter,
                                        double dy_th,
                                        const std::vector<int> &dynamic_labels,
                                        const std::vector<int> &lankmark) {
    std::vector<Eigen::Vector4d> inliers;
    if(dynamic_vehicle_filter){
        pcl::PointCloud<pcl::PointXYZL>::Ptr map_all(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::PointCloud<pcl::PointXYZL>::Ptr map_vehicle(new pcl::PointCloud<pcl::PointXYZL>);
        std::vector<Eigen::Vector4d> vehicle_inliers;
        for (const auto &point : frame) {
            Eigen::Vector4d point_new = point;
            const double norm = point.head<3>().norm(); // Euclidean norm
            if (norm < max_range && norm > min_range){
                if (norm > label_max_range) point_new[3] = 0.0;
                pcl::PointXYZL point_temp;
                point_temp.x = static_cast<float>(point[0]);
                point_temp.y = static_cast<float>(point[1]);
                point_temp.z = static_cast<float>(point[2]);
                point_temp.label = static_cast<uint32_t>(point_new[3]);
                map_all->points.emplace_back(point_temp);
                if (std::find(dynamic_labels.begin(), dynamic_labels.end(), point_temp.label) != dynamic_labels.end()){
                    map_vehicle->points.emplace_back(point_temp);
                    vehicle_inliers.emplace_back(point_new);
                }
                // if(point_temp.label > 9 && point_temp.label < 21){
                //     map_vehicle->points.emplace_back(point_temp);
                //     vehicle_inliers.emplace_back(point_new);
                // }
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
                std::vector<int> pointIdxRadiusSearch;  // 存储近邻索引
                std::vector<float> pointRadiusSquaredDistance;   // 存储近邻对应的平均距离
                int b = maptree.radiusSearch(map_vehicle->points[*pit], 0.5f, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                if (b > 0)
                {
                    for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++)
                    {
                        if (std::find(lankmark.begin(), lankmark.end(), map_all->points[pointIdxRadiusSearch[i]].label) != lankmark.end())
                        // if (map_all->points[pointIdxRadiusSearch[i]].label == 44 || map_all->points[pointIdxRadiusSearch[i]].label == 48)
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
        for (const auto &point : frame) {
            Eigen::Vector4d point_new = point;
            const double norm = point.head<3>().norm(); // Euclidean norm
            if (norm < max_range && norm > min_range){
                if (norm > label_max_range) point_new[3] = 0.0;
                inliers.emplace_back(point_new);
            }
        }
        // std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt) {
        //     Eigen::Vector3d v3point(pt[0], pt[1], pt[2]);
        //     const double norm = v3point.norm(); // Euclidean norm
        //     return norm < max_range && norm > min_range;
        // });
    }
    return inliers;
}

}  // namespace sage_icp
