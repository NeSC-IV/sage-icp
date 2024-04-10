
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


def generate_launch_description():
    current_pkg = FindPackageShare("sage_icp")
    return LaunchDescription(
        [
            # ROS2 parameters
            DeclareLaunchArgument("topic", description="sensor_msg/PointCloud2 topic to process", default_value="/label_points"), # afiter_sem_points
            DeclareLaunchArgument("bagfile", default_value=""),
            DeclareLaunchArgument("visualize", default_value="true"),
            DeclareLaunchArgument("odom_frame", default_value="odom"), # Odometry frame
            DeclareLaunchArgument("child_frame", default_value="base_link"), # Coordinate origin
            DeclareLaunchArgument("ground_truth_topic", default_value="/ground_truth"),
            # SAGE-ICP parameters
            DeclareLaunchArgument("deskew", default_value="false"), # Point cloud deskew
            DeclareLaunchArgument("max_range", default_value="100.0"), # Maximum distance between local map and current point cloud
            DeclareLaunchArgument("min_range", default_value="5.0"), # Minimum distance of current point cloud
            # Voxel grid filter
            DeclareLaunchArgument("voxel_size_map", default_value="0.8"), # Voxel size for map, 0.8, 1.0
            DeclareLaunchArgument("voxel_size_road", default_value="0.6"), # Voxel size for road, 0.8, 1.0
            DeclareLaunchArgument("voxel_size_building", default_value="1.0"), # Voxel size for building, 0.8
            DeclareLaunchArgument("voxel_size_plant", default_value="0.9"), # Voxel size for plant, 1.0
            DeclareLaunchArgument("voxel_size_object", default_value="0.8"), # Voxel size for object, 0.5
            DeclareLaunchArgument("voxel_size_unlabel", default_value="1.0"), # Voxel size for unlabelled, 
            DeclareLaunchArgument("voxel_size_vehicle", default_value="0.6"), # Voxel size for vehicle, 0.5
            # Dynamic vehicle filter
            DeclareLaunchArgument("dynamic_vehicle_filter", default_value="false"), # Dynamic vehicle filter
            DeclareLaunchArgument("dynamic_vehicle_filter_th", default_value="0.1"), # Dynamic vehicle filter threshold

            # global map params
            DeclareLaunchArgument("use_global_map", default_value="false"),
            DeclareLaunchArgument("map_range", default_value="1000.0"), # Maximum distance of local map, 1000  #250
            DeclareLaunchArgument("mov_threshold", default_value="1.5"), # Movement threshold, 1.5  #1.2

            DeclareLaunchArgument("sem_th", default_value="0.05"), # Semantic segmentation threshold, 0.4

            Node(
                package="sage_icp",
                executable="odometry_node",
                name="odometry_node",
                output="screen",
                remappings=[("pointcloud_topic", LaunchConfiguration("topic"))],
                parameters=[
                    {
                        "odom_frame": LaunchConfiguration("odom_frame"),
                        "child_frame": LaunchConfiguration("child_frame"),
                        "max_range": LaunchConfiguration("max_range"),
                        "min_range": LaunchConfiguration("min_range"),
                        "deskew": LaunchConfiguration("deskew"),
                        "voxel_size_map": LaunchConfiguration("voxel_size_map"),
                        "voxel_size_road": LaunchConfiguration("voxel_size_road"),
                        "voxel_size_building": LaunchConfiguration("voxel_size_building"),
                        "voxel_size_plant": LaunchConfiguration("voxel_size_plant"),
                        "voxel_size_object": LaunchConfiguration("voxel_size_object"),
                        "voxel_size_unlabel": LaunchConfiguration("voxel_size_unlabel"),
                        "voxel_size_vehicle": LaunchConfiguration("voxel_size_vehicle"),
                        "dynamic_vehicle_filter": LaunchConfiguration("dynamic_vehicle_filter"),
                        "max_points_per_voxel": 20,
                        "initial_threshold": 2.0,
                        "min_motion_th": 0.1,
                        "gps_topic": LaunchConfiguration("ground_truth_topic"),
                        "use_global_map": LaunchConfiguration("use_global_map"),
                        "map_range": LaunchConfiguration("map_range"),
                        "mov_threshold": LaunchConfiguration("mov_threshold"),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output={"both": "log"},
                arguments=["-d", PathJoinSubstitution([current_pkg, "rviz", "sage_icp_ros2.rviz"])],
                condition=IfCondition(LaunchConfiguration("visualize")),
            ),
            # ExecuteProcess(
            #     cmd=["ros2", "bag", "play", LaunchConfiguration("bagfile")],
            #     output="screen",
            #     condition=IfCondition(
            #         PythonExpression(["'", LaunchConfiguration("bagfile"), "' != ''"])
            #     ),
            # ),
        ]
    )
