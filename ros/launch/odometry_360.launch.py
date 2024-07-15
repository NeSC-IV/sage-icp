
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
import yaml
class Basic_config():
    def __init__(self, color_yaml="semantic-kitti.yaml"):
        # ROS2 parameters
        self.pc_topic: str = "/sem_points" # input pointcloud topic
        self.base_frame: str = "base_link"
        self.odom_frame: str = "odom"
        self.odom_topic: str = "/sage_icp/odometry"
        self.trajectory_topic: str = "/sage_icp/trajectory"

        self.publish_frame: bool = True # publish frame in odom and map for visualization
        self.frame_topic: str = "/sage_icp/frame"
        self.local_map_topic: str = "/sage_icp/local_map"

        self.sub_ground_truth: bool = True
        self.gt_topic: str = "/ground_truth" # input gt_topic
        self.gt_trajectory_topic: str = "/sage_icp/gt_trajectory"
        
        # Pointcloud pre-process
        self.deskew: bool = False # Point cloud deskew
        self.max_range: float = 100.0 # pointcloud max range
        self.min_range: float = 5.0 # pointcloud min range
        self.label_max_range: float = 50.0 # label max range
        
        # Voxel grid filter
        self.voxel_labels = [
            [40, 44, 48, 49],  # road
            [50, 51, 52],  # building
            [70, 72],  # plant
            [60, 71, 80, 81, 99], # object
            [0],  # unlabelled
            [10, 11, 13, 15, 16, 18, 20],  # vehicle
        ]
        self.voxel_labels_str: str = self.pack_2d_array(self.voxel_labels) # pack 2d array to string
        self.voxel_size: list = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]

        # Dynamic cars remove
        self.dynamic_vehicle_filter: bool = True
        self.dynamic_vehicle_filter_th: float = 0.5
        self.dynamic_vehicle_voxid: int = 5 # voxid in voxel_labels
        self.dynamic_remove_lankmark: list = [44, 48] # landmark labels for dynamic remove
        
        # Map
        self.voxel_size_map: float = 1.0
        self.local_map_range: float = 100.0
        self.basic_points_per_voxel: int = 20 # basic part
        self.critical_points_per_voxel: int = 20 # critical part
        self.basic_parts_labels: list = [40, 44, 48, 49, 50, 70, 72] # basic parts labels, others are critical parts
        
        # Semantic assisted association
        self.sem_th: float = 0.8
        
        # KISS-ICP Adaptive threshold
        self.initial_threshold: float = 2.0
        self.min_motion_th: float = 0.1

        # color map
        self.current_pkg = FindPackageShare("sage_icp")
        current_pkg_path_str = self.current_pkg.find("sage_icp")
        label_mapping = os.path.join(current_pkg_path_str, "launch", color_yaml)
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        color_map_bgr = semkittiyaml['color_map']
        self.color_list = []
        for key, value in color_map_bgr.items():
            b, g, r = value
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            self.color_list.append([key,rgb])
        self.color_list_str: str = self.pack_2d_array(self.color_list)

        # Key Frames extract
        self.publish_key_frame: bool = False # publish key frame
        self.key_frame_topic: str = "/sage_icp/key_frame"
        self.key_marker_topic: str = "/sage_icp/key_marker"
        self.key_frame_overlap: float = 0.5 # map sample overlap
        self.key_frame_bounds: list = [[-51.2, 51.2], [-51.2, 51.2], [-4, 2.4]] # Point Cloud Boundaries, used for generate occupancy map
        self.key_frame_bounds_str: str = self.pack_2d_array(self.key_frame_bounds)
        self.key_frame_occ_size: list = [128, 128] # H*W, occ resolution

        # RVIZ2 and ROS2 bag play
        self.visualize: str = "true" # must be string
        self.bagfile: str = ""
    
    def pack_2d_array(self, array_2d):
        return ';'.join([','.join(map(str, row)) for row in array_2d])


def generate_launch_description():
    
    sage_icp_config = Basic_config()
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
                            "sub_ground_truth": sage_icp_config.sub_ground_truth,
                            "gt_topic": sage_icp_config.gt_topic,
                            "gt_trajectory_topic_": sage_icp_config.gt_trajectory_topic,
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
                    arguments=["-d", PathJoinSubstitution([sage_icp_config.current_pkg, "rviz", "sage_icp_ros2.rviz"])],
                    condition=IfCondition(sage_icp_config.visualize),
                    )
    # ROS2 bag play
    bag_play = ExecuteProcess(
                    cmd=["ros2", "bag", "play", sage_icp_config.bagfile],
                    output="screen",
                    condition=IfCondition(
                        PythonExpression(["'", sage_icp_config.bagfile, "' != ''"])
                    ),
                )
    return LaunchDescription(
        [
            sage_icp_node,
            rviz2_node,
            bag_play,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()