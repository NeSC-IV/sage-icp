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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace sage_icp {
struct VoxelHashMap {
    using Vector4dVector = std::vector<Eigen::Vector4d>;
    using Vector4dVectorTuple = std::tuple<Vector4dVector, Vector4dVector>;
    using Voxel = Eigen::Vector3i;
    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector4d> points;
        int num_points_;
        inline void AddPoint(const Eigen::Vector4d &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.emplace_back(point);
            else{
                if(static_cast<int>(point[3]) == 0) ;
                else if ((static_cast<int>(point[3]) > 39 && static_cast<int>(point[3]) < 51)
                || static_cast<int>(point[3]) == 70 || static_cast<int>(point[3]) == 72)
                {
                    for (auto &p : points) {
                        if (static_cast<int>(p[3]) == 0) {
                            p = point;
                            break;
                        }
                    }
                }
                else{
                    if (points.size() < 2 * static_cast<size_t>(num_points_)) points.emplace_back(point);
                    else
                        for (auto &p : points) {
                            if (static_cast<int>(p[3]) == 0) {
                                p = point;
                                break;
                            }
                        }
                }
            }
        }
    };
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector4dVectorTuple GetCorrespondences(const Vector4dVector &points,
                                           double max_correspondance_distance,
                                           double th) const;
    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const std::vector<Eigen::Vector4d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector4d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector4d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector4d> Pointcloud() const;

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};
}  // namespace sage_icp
