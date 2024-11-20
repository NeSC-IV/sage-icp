
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python import get_package_share_directory,get_package_prefix
import os
import yaml

def generate_launch_description():
    # SAGE-ICP Node
    sage_icp_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('sage_icp'),
            'launch/odometry_livox_launch.py'))
    )
    # SMENet
    smenet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('smenet_ros2'),
            'smenet_livox_launch.py'))
    )
    # SEM_SAM
    sam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('sem_sam'),
            'launch',
            'optimize_launch.py'))
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
    bag_file = "" # bag file name
    bag_play = ExecuteProcess(
                    cmd=["ros2", "bag", "play", bag_file],
                    output="screen",
                    condition=IfCondition(
                        PythonExpression(["'", bag_file, "' != ''"])
                    ),
                )

    return LaunchDescription(
        [
            sage_icp_node,
            smenet_node,
            sam_node,
            rviz2_node,
            bag_play,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()