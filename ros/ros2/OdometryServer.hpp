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

// SAGE-ICP
#include "sage_icp/pipeline/sageICP.hpp"
#include <fstream>
#include <signal.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <unistd.h>
// ROS2
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "example_interfaces/srv/add_two_ints.hpp"
namespace sage_icp_ros {

class OdometryServer : public rclcpp::Node {
public:
    /// OdometryServer constructor
    OdometryServer();

private:
    /// Register new frame
    void RegisterFrame(const sensor_msgs::msg::PointCloud2::SharedPtr msg_ptr);
    /// pre-matching
    // Sophus::SE3d PreMatching(pcl::PointCloud<pcl::PointXYZ>::Ptr &building_pc);
    /// publish groundtruth
    void RegisterGPS(const geometry_msgs::msg::PoseStamped::SharedPtr msg_ptr);

    void ReinitService(const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
          std::shared_ptr<example_interfaces::srv::AddTwoInts::Response>      response);

private:
    /// Ros node stuff
    size_t queue_size_{1};

    /// Tools for broadcasting TFs.
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    /// Data subscribers.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gps_sub_;

    /// Data publishers.
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_publisher_;

    /// service
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr reinit_service_;
    /// Path publisher
    // nav_msgs::msg::Path path_msg_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_publisher_;
    /// Groundtruth Path publisher
    // nav_msgs::msg::Path gt_path_msg_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr GT_publisher_;

    /// SAGE-ICP
    sage_icp::pipeline::sageICP odometry_;
    sage_icp::pipeline::sageConfig config_;

    /// Global/map coordinate frame.
    std::string odom_frame_{"odom"};
    std::string child_frame_{"base_link"};
    std::string gps_topic_{"gps"};

};

}  // namespace sage_icp_ros
