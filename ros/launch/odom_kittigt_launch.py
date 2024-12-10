
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import sys
current_pkg_path = os.path.join(FindPackageShare("sage_icp").find("sage_icp"), "launch")
sys.path.append(current_pkg_path)
from base_launch import Base_config
class KITTIGT_config(Base_config):
    def __init__(self,
                 color_yaml="semantic-kitti.yaml",
                 publish_frame=True,
                 deskew=False,
                 dynamic_vehicle_filter=False,
                 publish_key_frame=True,
                 sub_correct_pose=True,
                 ):
        super().__init__(color_yaml=color_yaml,
                         publish_frame=publish_frame,
                         deskew=deskew,
                         dynamic_vehicle_filter=dynamic_vehicle_filter,
                         publish_key_frame=publish_key_frame,
                         sub_correct_pose=sub_correct_pose)
        
        # Semantic assisted association
        self.sem_th: float = 0.05

def generate_launch_description():
    
    sage_icp_config = KITTIGT_config(color_yaml="semantic-kitti.yaml",
                                   publish_frame=True,
                                   deskew=False,
                                   dynamic_vehicle_filter=False,
                                   publish_key_frame=True,
                                   sub_correct_pose=True,)
    # SAGE-ICP Node
    sage_icp_node = Node(
                    package="sage_icp",
                    executable="odometry_node",
                    name="odometry_node",
                    output="screen",
                    # remappings=[("pointcloud_topic", LaunchConfiguration("topic"))],
                    parameters=[
                        {
                            "pc_topic": sage_icp_config.pc_topic,
                            "base_frame": sage_icp_config.base_frame,
                            "odom_frame": sage_icp_config.odom_frame,
                            "odom_topic": sage_icp_config.odom_topic,
                            "trajectory_topic": sage_icp_config.trajectory_topic,
                            "publish_frame": sage_icp_config.publish_frame,
                            "frame_topic": sage_icp_config.frame_topic,
                            "local_map_topic": sage_icp_config.local_map_topic,
                            "sub_correct_pose": sage_icp_config.sub_correct_pose,
                            "correct_pose_topic": sage_icp_config.correct_pose_topic,
                            "deskew": sage_icp_config.deskew,
                            "max_range": sage_icp_config.max_range,
                            "min_range": sage_icp_config.min_range,
                            "label_max_range": sage_icp_config.label_max_range,
                            "voxel_labels_str": sage_icp_config.voxel_labels_str,
                            "voxel_size": sage_icp_config.voxel_size,
                            "dynamic_vehicle_filter": sage_icp_config.dynamic_vehicle_filter,
                            "dynamic_vehicle_filter_th": sage_icp_config.dynamic_vehicle_filter_th,
                            "dynamic_vehicle_voxid": sage_icp_config.dynamic_vehicle_voxid,
                            "dynamic_remove_lankmark": sage_icp_config.dynamic_remove_lankmark,
                            "voxel_size_map": sage_icp_config.voxel_size_map,
                            "local_map_range": sage_icp_config.local_map_range,
                            "basic_points_per_voxel": sage_icp_config.basic_points_per_voxel,
                            "critical_points_per_voxel": sage_icp_config.critical_points_per_voxel,
                            "basic_parts_labels": sage_icp_config.basic_parts_labels,
                            "sem_th": sage_icp_config.sem_th,
                            "initial_threshold": sage_icp_config.initial_threshold,
                            "min_motion_th": sage_icp_config.min_motion_th,
                            "color_list_str": sage_icp_config.color_list_str,
                            "publish_key_frame": sage_icp_config.publish_key_frame,
                            "key_frame_topic": sage_icp_config.key_frame_topic,
                            "key_marker_topic": sage_icp_config.key_marker_topic,
                            "key_frame_overlap": sage_icp_config.key_frame_overlap,
                            "key_frame_bounds_str": sage_icp_config.key_frame_bounds_str,
                            "key_frame_occ_size": sage_icp_config.key_frame_occ_size,
                        }
                    ],
                )
    # RVIZ2
    rviz2_node = Node(
                    package="rviz2",
                    executable="rviz2",
                    output={"both": "log"},
                    arguments=["-d", PathJoinSubstitution([FindPackageShare("sage_icp"), "rviz", "semantic_slam.rviz"])],
                    condition=IfCondition("true"),
                    )
    # ROS2 bag play
    # bag_play = ExecuteProcess(
    #                 cmd=["ros2", "bag", "play", sage_icp_config.bagfile],
    #                 output="screen",
    #                 condition=IfCondition(
    #                     PythonExpression(["'", sage_icp_config.bagfile, "' != ''"])
    #                 ),
    #             )
    return LaunchDescription(
        [
            sage_icp_node,
            rviz2_node,
            # bag_play,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()