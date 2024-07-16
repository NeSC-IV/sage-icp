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
#include <sstream>
#include "visualization_msgs/msg/marker.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <omp.h>
#include <map>

namespace sage_icp_ros::utils {

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointField = sensor_msgs::msg::PointField;
using Header = std_msgs::msg::Header;
using Marker = visualization_msgs::msg::Marker;
std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

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
    offset = addPointField(cloud_msg, "label", 1, PointField::UINT8, offset);
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

void FillPointCloud2XYZlRGB(const std::vector<Eigen::Vector4d> &points, const std::map<int, int> &color_list, PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> msg_label(msg, "label");
    sensor_msgs::PointCloud2Iterator<uint32_t> msg_rgb(msg, "rgb");
    for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z, ++msg_label, ++msg_rgb) {
        const Eigen::Vector4d &point = points[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
        *msg_label = static_cast<uint8_t>(point.w());
        *msg_rgb = color_list.at(static_cast<int>(point.w()));
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
        sensor_msgs::PointCloud2ConstIterator<uint8_t> msg_l(msg, "label");
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

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector4d> &points,
                                const Header &header,
                                const std::map<int, int> &color_list) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header);
    FillPointCloud2XYZlRGB(points, color_list, msg);
    return msg;
}

PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector4d> &points,
                                const Header &header,
                                const std::map<int, int> &color_list,
                                const std::vector<double> &timestamps) {
    PointCloud2 msg = CreatePointCloud2Msg(points.size(), header, true);
    FillPointCloud2XYZlRGB(points, color_list, msg);
    FillPointCloud2Timestamp(timestamps, msg);
    return msg;
}

Marker OdomToMarker(const nav_msgs::msg::Odometry &odom_msg,
                    const std::string &key_frame_topic,
                    int &last_marker_id){
    Marker marker;
    marker.header = odom_msg.header;
    marker.ns = key_frame_topic;
    marker.id = last_marker_id++;
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;
    marker.pose = odom_msg.pose.pose;
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    return marker;
}

std::vector<std::vector<int>> EigenToGridMap(const std::vector<Eigen::Vector4d>& points,
                                                const std::vector<std::vector<double>> key_frame_bounds,
                                                const std::vector<int>& key_frame_occ_size) {
    std::vector<std::vector<int>> gridMap(key_frame_occ_size[0], std::vector<int>(key_frame_occ_size[1], 0));
    const double x_resolution = (key_frame_bounds[0][1] - key_frame_bounds[0][0]) / key_frame_occ_size[1]; // Width
    const double y_resolution = (key_frame_bounds[1][1] - key_frame_bounds[1][0]) / key_frame_occ_size[0]; // Height
    for (const auto& point : points) {
        if (point[0] < key_frame_bounds[0][0] || point[0] > key_frame_bounds[0][1] ||
            point[1] < key_frame_bounds[1][0] || point[1] > key_frame_bounds[1][1] ||
            point[2] < key_frame_bounds[2][0] || point[2] > key_frame_bounds[2][1]) {
            continue;
        }
        // move pc to occ frame
        int occ_x = static_cast<int>((point[0] + key_frame_bounds[0][1])/ x_resolution);
        int occ_y = static_cast<int>((point[1] + key_frame_bounds[1][1])/ y_resolution);

        // check if the point is within the bounds of the occupancy grid
        if (occ_x >= 0 && occ_x < key_frame_occ_size[1] && occ_y >= 0 && occ_y < key_frame_occ_size[0]) {
            gridMap[occ_y][occ_x] = 1; // update the occupancy grid
        }
    }
    return gridMap;
}

double compute_occ_overlap(const std::vector<std::vector<int>>& occ_s, const std::vector<std::vector<int>>& occ_t) {
    int overlap = 0;
    int total = 0;
    for (size_t i = 0; i < occ_s.size(); i++) {
        for (size_t j = 0; j < occ_s[0].size(); j++) {
            if (occ_s[i][j] == 1 && occ_t[i][j] == 1) {
                overlap++;
            }
            if (occ_s[i][j] == 1) {
                total++;
            }
        }
    }
    return static_cast<double>(overlap) / total;
}

std::vector<std::vector<int>> unpack_2d_array_int(const std::string& packed_str) {
    std::vector<std::vector<int>> result;
    std::stringstream ss(packed_str);
    std::string row_str;
    
    while (std::getline(ss, row_str, ';')) {
        std::vector<int> row;
        std::stringstream row_ss(row_str);
        std::string value_str;
        
        while (std::getline(row_ss, value_str, ',')) {
            row.push_back(std::stof(value_str));
        }
        
        result.push_back(row);
    }
    
    return result;
}

std::vector<std::vector<double>> unpack_2d_array_double(const std::string& packed_str) {
    std::vector<std::vector<double>> result;
    std::stringstream ss(packed_str);
    std::string row_str;
    
    while (std::getline(ss, row_str, ';')) {
        std::vector<double> row;
        std::stringstream row_ss(row_str);
        std::string value_str;
        
        while (std::getline(row_ss, value_str, ',')) {
            row.push_back(std::stof(value_str));
        }
        
        result.push_back(row);
    }
    
    return result;
}

std::map<int, int> unpack_dict(const std::string& mapAsString) {
    std::map<int, int> result;
    std::stringstream ss(mapAsString);
    std::string item;
    while (std::getline(ss, item, ';')) {
        size_t pos = item.find(",");
        if (pos != std::string::npos) {
            int key = std::stoi(item.substr(0, pos));
            int value = std::stoi(item.substr(pos + 1));
            result[key] = value;
        }
    }
    return result;
}


}  // namespace sage_icp_ros::utils
