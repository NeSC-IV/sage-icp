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
#include <Eigen/Core>
#include <vector>

// SAGE-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// SAGE-ICP
#include "sage_icp/pipeline/sageICP.hpp"

// ROS2 headers
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

nav_msgs::msg::Path path_msg_;
nav_msgs::msg::Path gt_path_msg_;
std::vector<Eigen::Vector2d> time_icp;
namespace sage_icp_ros {

OdometryServer::OdometryServer() : rclcpp::Node("odometry_node") {
    // clang-format off
    pc_topic_ = declare_parameter<std::string>("pc_topic", pc_topic_);
    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    odom_frame_ = declare_parameter<std::string>("odom_frame", odom_frame_);
    odom_topic_ = declare_parameter<std::string>("odom_topic", odom_topic_);
    trajectory_topic_ = declare_parameter<std::string>("trajectory_topic", trajectory_topic_);
    publish_frame_ = declare_parameter<bool>("publish_frame", publish_frame_);
    frame_topic_ = declare_parameter<std::string>("frame_topic", frame_topic_);
    local_map_topic_ = declare_parameter<std::string>("local_map_topic", local_map_topic_);
    sub_ground_truth_ = declare_parameter<bool>("sub_ground_truth", sub_ground_truth_);
    gt_topic_ = declare_parameter<std::string>("gt_topic", gt_topic_);
    gt_trajectory_topic_ = declare_parameter<std::string>("gt_trajectory_topic", gt_trajectory_topic_);
    config_.deskew = declare_parameter<bool>("deskew", config_.deskew);
    config_.max_range = declare_parameter<double>("max_range", config_.max_range);
    config_.min_range = declare_parameter<double>("min_range", config_.min_range);
    config_.label_max_range = declare_parameter<double>("label_max_range", config_.label_max_range);
    std::string voxel_labels_str_ = declare_parameter<std::string>("voxel_labels_str");
    config_.voxel_labels = utils::unpack_2d_array_int(voxel_labels_str_);
    config_.voxel_size = declare_parameter<std::vector<double>>("voxel_size", config_.voxel_size);
    config_.dynamic_vehicle_filter = declare_parameter<bool>("dynamic_vehicle_filter", config_.dynamic_vehicle_filter);
    config_.dynamic_vehicle_filter_th = declare_parameter<double>("dynamic_vehicle_filter_th", config_.dynamic_vehicle_filter_th);
    config_.dynamic_vehicle_voxid = declare_parameter<int>("dynamic_vehicle_voxid", config_.dynamic_vehicle_voxid);
    auto temp_vector = declare_parameter<std::vector<long int>>("dynamic_remove_lankmark", std::vector<long int>(config_.dynamic_remove_lankmark.begin(), config_.dynamic_remove_lankmark.end()));
    config_.dynamic_remove_lankmark.clear();
    std::transform(temp_vector.begin(), temp_vector.end(), std::back_inserter(config_.dynamic_remove_lankmark), [](long int val) { return static_cast<int>(val); });
    config_.voxel_size_map = declare_parameter<double>("voxel_size_map", config_.voxel_size_map);
    config_.local_map_range = declare_parameter<double>("local_map_range", config_.local_map_range);
    config_.basic_points_per_voxel = declare_parameter<int>("basic_points_per_voxel", config_.basic_points_per_voxel);
    config_.critical_points_per_voxel = declare_parameter<int>("critical_points_per_voxel", config_.critical_points_per_voxel);
    auto temp_vector2 = declare_parameter<std::vector<long int>>("basic_parts_labels", std::vector<long int>(config_.basic_parts_labels.begin(), config_.basic_parts_labels.end()));
    config_.basic_parts_labels.clear();
    std::transform(temp_vector2.begin(), temp_vector2.end(), std::back_inserter(config_.basic_parts_labels), [](long int val) { return static_cast<int>(val); });
    config_.sem_th = declare_parameter<double>("sem_th", config_.sem_th);
    config_.initial_threshold = declare_parameter<double>("initial_threshold", config_.initial_threshold);
    config_.min_motion_th = declare_parameter<double>("min_motion_th", config_.min_motion_th);
    std::string color_list_str_ = declare_parameter<std::string>("color_list_str");
    color_list_ = utils::unpack_dict(color_list_str_);
    publish_key_frame_ = declare_parameter<bool>("publish_key_frame", publish_key_frame_);
    key_frame_topic_ = declare_parameter<std::string>("key_frame_topic", key_frame_topic_);
    key_marker_topic_ = declare_parameter<std::string>("key_marker_topic", key_marker_topic_);
    key_frame_overlap_ = declare_parameter<double>("key_frame_overlap", key_frame_overlap_);
    std::string key_frame_bounds_str_ = declare_parameter<std::string>("key_frame_bounds_str");
    key_frame_bounds_ = utils::unpack_2d_array_double(key_frame_bounds_str_);
    auto temp_vector3 = declare_parameter<std::vector<long int>>("key_frame_occ_size", std::vector<long int>(key_frame_occ_size_.begin(), key_frame_occ_size_.end()));
    key_frame_occ_size_.clear();
    std::transform(temp_vector3.begin(), temp_vector3.end(), std::back_inserter(key_frame_occ_size_), [](long int val) { return static_cast<int>(val); });

    if (config_.max_range < config_.min_range) {
        RCLCPP_WARN(get_logger(), "[WARNING] max_range is smaller than min_range, settng min_range to 0.0");
        config_.min_range = 0.0;
    }
    // clang-format on
    RCLCPP_INFO(this->get_logger(), "config get ready!");
    // Construct the main SAGE-ICP odometry node
    odometry_ = sage_icp::pipeline::sageICP(config_);

    // Intialize subscribers
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pc_topic_, 10, //rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));
    if (sub_ground_truth_){
        RCLCPP_INFO(this->get_logger(), "Publish groundtruth enable!");
        gt_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            gt_topic_, 10, //rclcpp::SensorDataQoS(),
            std::bind(&OdometryServer::pub_gtpath, this, std::placeholders::_1));
    }

    // Intialize publishers
    rclcpp::QoS qos(rclcpp::KeepLast{queue_size_});
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, qos);
    frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(frame_topic_, qos);
    map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic_, qos);
    key_frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(key_frame_topic_, qos);
    marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(key_marker_topic_, qos);
    path_msg_.header.frame_id = odom_frame_;
    traj_publisher_ = create_publisher<nav_msgs::msg::Path>(trajectory_topic_, qos);
    gt_path_msg_.header.frame_id = odom_frame_;
    GT_publisher_ = create_publisher<nav_msgs::msg::Path>(gt_trajectory_topic_, qos);

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Broadcast a static transformation that links with identity the specified base link to the
    // pointcloud_frame, basically to always be able to visualize the frame in rviz
    // if (base_frame_ != "base_link") {
    //     static auto br = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
    //     geometry_msgs::msg::TransformStamped alias_transform_msg;
    //     alias_transform_msg.header.stamp = this->get_clock()->now();
    //     alias_transform_msg.transform.translation.x = 0.0;
    //     alias_transform_msg.transform.translation.y = 0.0;
    //     alias_transform_msg.transform.translation.z = 0.0;
    //     alias_transform_msg.transform.rotation.x = 0.0;
    //     alias_transform_msg.transform.rotation.y = 0.0;
    //     alias_transform_msg.transform.rotation.z = 0.0;
    //     alias_transform_msg.transform.rotation.w = 1.0;
    //     alias_transform_msg.header.frame_id = base_frame_;
    //     alias_transform_msg.child_frame_id = "base_link";
    //     br->sendTransform(alias_transform_msg);
    // }

    reinit_service_ = create_service<example_interfaces::srv::AddTwoInts>("reinit",
                                                                   std::bind(&OdometryServer::ReinitService, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(), "SAGE-ICP ROS2 odometry node initialized");
}

void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::SharedPtr msg_ptr) {
    // ROS2::Foxy can't handle a callback to const MessageT&, so we hack it here
    // https://github.com/ros2/rclcpp/pull/1598
    const sensor_msgs::msg::PointCloud2 &msg = *msg_ptr;
    const auto points = utils::PointCloud2ToEigen(msg);
    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return utils::GetTimestamps(msg);
    }();

    // Register frame, main entry point to SAGE-ICP pipeline
    const auto &[frame, timeicp, timeall] = odometry_.RegisterFrame(points, timestamps); //frame is downsampling point cloud for registration
    Eigen::Vector2d time_use;
    time_use << timeicp, timeall;
    time_icp.emplace_back(time_use);
    // PublishPose
    const auto pose = odometry_.poses().back(); //Sophus::SE3d
    // Convert from Eigen to ROS types
    const Eigen::Vector3d t_current = pose.translation();
    const Eigen::Quaterniond q_current = pose.unit_quaternion();
    // Broadcast the tf
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = msg.header.stamp;
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = base_frame_;
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    tf_broadcaster_->sendTransform(transform_msg);

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = msg.header.stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;
    odom_msg.pose.pose.orientation.x = q_current.x();
    odom_msg.pose.pose.orientation.y = q_current.y();
    odom_msg.pose.pose.orientation.z = q_current.z();
    odom_msg.pose.pose.orientation.w = q_current.w();
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_publisher_->publish(odom_msg);
    // publish trajectory msg
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.pose = odom_msg.pose.pose;
    pose_msg.header = odom_msg.header;
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_->publish(path_msg_);
    // Publish SAGE-ICP internal data, just for debugging
    if (publish_frame_){
        std_msgs::msg::Header frame_header = msg.header;
        frame_header.frame_id = base_frame_;
        frame_publisher_->publish(utils::EigenToPointCloud2(frame, frame_header, color_list_));
        // Map is referenced to the odometry_frame
        std_msgs::msg::Header local_map_header = msg.header;
        local_map_header.frame_id = odom_frame_;
        const auto &local_map = odometry_.LocalMap();
        map_publisher_->publish(utils::EigenToPointCloud2(local_map, local_map_header, color_list_));
    }

    if (publish_key_frame_){
        std::vector<std::vector<int>> current_key_frame_occ_ = utils::EigenToGridMap(points, key_frame_bounds_, key_frame_occ_size_);
        if (last_key_frame_occ_.size() == 0){
            last_key_frame_occ_ = current_key_frame_occ_;
            std_msgs::msg::Header frame_header = msg.header;
            frame_header.frame_id = base_frame_;
            key_frame_publisher_->publish(utils::EigenToPointCloud2(points, frame_header, color_list_));
            marker_publisher_->publish(utils::OdomToMarker(odom_msg, key_marker_topic_, last_marker_id_));
        }
        else{
            double overlap = utils::compute_occ_overlap(last_key_frame_occ_, current_key_frame_occ_);
            if (overlap < key_frame_overlap_){
                last_key_frame_occ_ = current_key_frame_occ_;
                std_msgs::msg::Header frame_header = msg.header;
                frame_header.frame_id = base_frame_;
                key_frame_publisher_->publish(utils::EigenToPointCloud2(points, frame_header, color_list_));
                marker_publisher_->publish(utils::OdomToMarker(odom_msg, key_marker_topic_, last_marker_id_));
            }
        }
    }

}

void OdometryServer::pub_gtpath(const geometry_msgs::msg::PoseStamped::SharedPtr msg_ptr) {
    const geometry_msgs::msg::PoseStamped &msg = *msg_ptr;
    // publish gt trajectory msg
    geometry_msgs::msg::PoseStamped gt_pose_msg;
    gt_pose_msg.pose = msg.pose;
    gt_pose_msg.header.stamp = msg.header.stamp;
    gt_pose_msg.header.frame_id = odom_frame_;
    gt_path_msg_.poses.push_back(gt_pose_msg);
    GT_publisher_->publish(gt_path_msg_);
}

void OdometryServer::ReinitService(const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
          std::shared_ptr<example_interfaces::srv::AddTwoInts::Response>      response){

    bool map_init = odometry_.reinitialize();
    RCLCPP_INFO(this->get_logger(), "SAGE-ICP ROS2 odometry node reinitialized map: %d", map_init);

    std::string file_dir = std::to_string(request->a);

    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path directory_path = current_file_path.parent_path().parent_path().parent_path();
    std::string homepath = directory_path.string() + "/results/seq" + file_dir;
    std::cout << "saving dir: " << homepath << std::endl;
    if (std::filesystem::exists(homepath) && std::filesystem::is_directory(homepath)) {
        std::cout << "Directory exists!" << std::endl;
    } else {
        std::filesystem::create_directory(homepath);
        std::cout << "Directory does not exist! Creating ..." << std::endl;
    }
    
    std::cout << "Writing time.txt..." << std::endl;
    std::ofstream fout_time;
    fout_time.open(homepath + "/time.txt");
    for(int i=0;i<time_icp.size();i++){
        fout_time << i << " " << time_icp[i][0] << " " << time_icp[i][1] << std::endl;
    }
    fout_time.close();
    
    usleep(5000000); // 5s
    path_msg_.poses.clear();
    gt_path_msg_.poses.clear();
    time_icp.clear();
    std::cout << "Finish clearing memory!"<<std::endl;
    response->sum = map_init;
    std::cout << response->sum <<std::endl;
    reinit_service_->send_response(*request_header, *response);
    std::cout << "Finish sending response!"<<std::endl; 
}


}  // namespace sage_icp_ros

void ctrl_c_handler(int sig) {
    // handel Ctrl+C
    // ros2 topic echo /path > path.txt

    auto now = std::chrono::system_clock::now();

    auto timestamp = std::chrono::system_clock::to_time_t(now);

    auto local_time = std::localtime(&timestamp);

    std::string file_dir = std::to_string(1900 + local_time->tm_year) + std::to_string(1 + local_time->tm_mon) + std::to_string(local_time->tm_mday) + "_" + std::to_string(local_time->tm_hour) + "_" + std::to_string(local_time->tm_min) + "_" + std::to_string(local_time->tm_sec);
    std::filesystem::path current_file_path(__FILE__);
    std::filesystem::path directory_path = current_file_path.parent_path().parent_path().parent_path();
    std::string homepath = directory_path.string() + "/results/" + file_dir;
    if (std::filesystem::exists(homepath) && std::filesystem::is_directory(homepath)) {
        std::cout << "Directory exists!" << std::endl;
    } else {
        std::filesystem::create_directory(homepath);
        std::cout << "Directory does not exist! Creating ..." << std::endl;
    }

    std::cout<<"Writing path.txt..."<<std::endl;
    std::ofstream fout;
    fout.open(homepath + "/path.txt");
    // double timestep = 0.0;
    for(int i=0;i<path_msg_.poses.size();i++){
        fout << path_msg_.poses[i].header.stamp.sec << "." << path_msg_.poses[i].header.stamp.nanosec << " " << path_msg_.poses[i].pose.position.x << " " << path_msg_.poses[i].pose.position.y << " " << path_msg_.poses[i].pose.position.z << " " << path_msg_.poses[i].pose.orientation.x << " " << path_msg_.poses[i].pose.orientation.y << " " << path_msg_.poses[i].pose.orientation.z << " " << path_msg_.poses[i].pose.orientation.w << std::endl;
    }
    fout.close();
    //timestamp (s) tx ty tz qx qy qz qw
    std::cout<<"Writing gt_path.txt..."<<std::endl;
    std::ofstream fout_gt;
    fout_gt.open(homepath + "/gt_path.txt");
    // double timestep_gt = 0.0;
    for(int i=0;i<gt_path_msg_.poses.size();i++){
        fout_gt << gt_path_msg_.poses[i].header.stamp.sec << "." << gt_path_msg_.poses[i].header.stamp.nanosec << " " << gt_path_msg_.poses[i].pose.position.x << " " << gt_path_msg_.poses[i].pose.position.y << " " << gt_path_msg_.poses[i].pose.position.z << " " << gt_path_msg_.poses[i].pose.orientation.x << " " << gt_path_msg_.poses[i].pose.orientation.y << " " << gt_path_msg_.poses[i].pose.orientation.z << " " << gt_path_msg_.poses[i].pose.orientation.w << std::endl;
    }
    fout_gt.close();
    
    std::cout << "Writing time.txt..." << std::endl;
    std::ofstream fout_time;
    fout_time.open(homepath + "/time.txt");
    for(int i=0;i<time_icp.size();i++){
        fout_time << i << " " << time_icp[i][0] << " " << time_icp[i][1] << std::endl;
    }
    fout_time.close();
    
    rclcpp::shutdown();
}


int main(int argc, char **argv) {
    // Ctrl+C handler
    signal(SIGINT, ctrl_c_handler);
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<sage_icp_ros::OdometryServer>());
    return 0;
}
