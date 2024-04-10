import os
from ament_index_python import get_package_share_directory,get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction,ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    # include another launch file
    sem_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('cylinder3d_ros2'),
            'launch/cylinder3d_ros2.launch.py'))
    )

    odom_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('sage_icp'),
            'launch/odometry.launch.py'))
    )


    return LaunchDescription(
        [
            sem_launch_include,
            odom_launch_include,
        ]
    )
