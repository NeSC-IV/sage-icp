
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
            'launch',
            'odom_livox_launch.py'))
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
    return LaunchDescription(
        [
            sage_icp_node,
            smenet_node,
            sam_node,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()