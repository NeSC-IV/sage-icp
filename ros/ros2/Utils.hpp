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
#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
#include <vector>
#include <exception>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_field.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <omp.h>
#include <map>

#include "sage_icp/core/ikd_Tree.h"

namespace sage_icp_ros::utils {

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointField = sensor_msgs::msg::PointField;
using Header = std_msgs::msg::Header;
std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

std::map<int, int> label_color = {
                   { 0, 0x000000},
                   { 1, 0xff0000},
                   { 10, 0x6496f5},
                   { 11, 0x64e6f5},
                   { 13, 0x6450fa},
                   { 15, 0x1e3c96},
                   { 16, 0x0000ff},
                   { 18, 0x501eb4},
                   { 20, 0x0000ff},
                   { 30, 0xff1e1e},
                   { 31, 0xff28c8},
                   { 32, 0x961e5a},
                   { 40, 0xff00ff},
                   { 44, 0xff96ff},
                   { 48, 0x4b004b},
                   { 49, 0xaf004b},
                   { 50, 0xffc800},
                   { 51, 0xff7832},
                   { 52, 0xff9600},
                   { 60, 0x96ffaa},
                   { 70, 0x00af00},
                   { 71, 0x873c00},
                   { 72, 0x96f050},
                   { 80, 0xfff096},
                   { 81, 0xff0000},
                   { 99, 0x32ffff},
                    {252, 0x6496f5},
                    {253, 0xff28c8},
                    {254, 0xff1e1e},
                    {255, 0x961e5a},
                    {256, 0x0000ff},
                    {257, 0x6450fa},
                    {258, 0x501eb4},
                    {259, 0x0000ff}
                };


auto GetTimestampField(const PointCloud2 &msg) {
    PointField timestamp_field;
    for (const auto &field : msg.fields) {
        if ((field.name == "t" || field.name == "timestamp" || field.name == "time")) {
            timestamp_field = field;
        }
    }
    if (!timestamp_field.count) {
        throw std::runtime_error("Field 't', 'timestamp', or 'time'  does not exist");
    }
    return timestamp_field;
}

// Normalize timestamps from 0.0 to 1.0
auto NormalizeTimestamps(const std::vector<double> &timestamps) {
    const double max_timestamp = *std::max_element(timestamps.cbegin(), timestamps.cend());
    // check if already normalized
    if (max_timestamp < 1.0) return timestamps;
    std::vector<double> timestamps_normalized(timestamps.size());
    std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                   [&](const auto &timestamp) { return timestamp / max_timestamp; });
    return timestamps_normalized;
}

auto ExtractTimestampsFromMsg(const PointCloud2 &msg, const PointField &field) {
    // Extract timestamps from cloud_msg
    const size_t n_points = msg.height * msg.width;
    std::vector<double> timestamps;
    timestamps.reserve(n_points);

    // Option 1: Timestamps are unsigned integers -> epoch time.
    if (field.name == "t" || field.name == "timestamp") {
        sensor_msgs::PointCloud2ConstIterator<uint32_t> msg_t(msg, field.name);
        for (size_t i = 0; i < n_points; ++i, ++msg_t) {
            timestamps.emplace_back(static_cast<double>(*msg_t));
        }
        // Covert to normalized time, between 0.0 and 1.0
        return NormalizeTimestamps(timestamps);
    }

    // Option 2: Timestamps are floating point values between 0.0 and 1.0
    // field.name == "timestamp"
    sensor_msgs::PointCloud2ConstIterator<double> msg_t(msg, field.name);
    for (size_t i = 0; i < n_points; ++i, ++msg_t) {
        timestamps.emplace_back(*msg_t);
    }
    return timestamps;
}

auto CreatePointCloud2Msg(const size_t n_points, const Header &header, bool timestamp = false) {
    PointCloud2 cloud_msg;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = FixFrameId(cloud_msg.header.frame_id);
    cloud_msg.fields.clear();
    int offset = 0;
    offset = addPointField(cloud_msg, "x", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "y", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "z", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "label", 1, PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "rgb", 1, PointField::UINT32, offset);
    offset += sizeOfPointField(PointField::FLOAT32);
    if (timestamp) {
        // asuming timestamp on a velodyne fashion for now (between 0.0 and 1.0)
        offset = addPointField(cloud_msg, "time", 1, PointField::FLOAT64, offset);
        offset += sizeOfPointField(PointField::FLOAT64);
    }

    // Resize the point cloud accordingly
    cloud_msg.point_step = offset;
    cloud_msg.row_step = cloud_msg.width * cloud_msg.point_step;
    cloud_msg.data.resize(cloud_msg.height * cloud_msg.row_step);
    modifier.resize(n_points);
    return cloud_msg;
}

void FillPointCloud2XYZlRGB(const std::vector<Eigen::Vector4d> &points, PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    sensor_msgs::PointCloud2Iterator<float> msg_label(msg, "label");
    sensor_msgs::PointCloud2Iterator<uint32_t> msg_rgb(msg, "rgb");
    for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z, ++msg_label, ++msg_rgb) {
        const Eigen::Vector4d &point = points[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
        *msg_label = point.w();
        *msg_rgb = label_color[int(point.w())];
    }
}

void FillPointCloud2Timestamp(const std::vector<double> &timestamps, PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<double> msg_t(msg, "time");
    for (size_t i = 0; i < timestamps.size(); i++, ++msg_t) *msg_t = timestamps[i];
}

std::vector<double> GetTimestamps(const PointCloud2 &msg) {
    auto timestamp_field = GetTimestampField(msg);

    // Extract timestamps from cloud_msg
    std::vector<double> timestamps = ExtractTimestampsFromMsg(msg, timestamp_field);

    return timestamps;
}



std::vector<Eigen::Vector4d> PointCloud2ToEigen(const PointCloud2 &msg) {
    std::vector<Eigen::Vector4d> points;
    points.reserve(msg.height * msg.width);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(msg, "z");
    if (msg.fields.size() == 5){
        sensor_msgs::PointCloud2ConstIterator<uint32_t> msg_l(msg, "label");
        for (size_t i = 0; i < msg.height * msg.width; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_l) {
            points.emplace_back(*msg_x, *msg_y, *msg_z, *msg_l);
        }
    }
    else{
        sensor_msgs::PointCloud2ConstIterator<float> msg_l(msg, "label");
        for (size_t i = 0; i < msg.height * msg.width; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_l) {
            points.emplace_back(*msg_x, *msg_y, *msg_z, *msg_l);
        }
    }
    return points;
}

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector4d> &points, const Header &header) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header);
    FillPointCloud2XYZlRGB(points, msg);
    return msg;
}

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector4d> &points,
                               const std::vector<double> &timestamps,
                               const Header &header) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header, true);
    FillPointCloud2XYZlRGB(points, msg);
    FillPointCloud2Timestamp(timestamps, msg);
    return msg;
}


}  // namespace sage_icp_ros::utils
