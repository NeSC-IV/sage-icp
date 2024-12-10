from launch_ros.substitutions import FindPackageShare
import os
import yaml
class Base_config():
    def __init__(self,
                 color_yaml="semantic-kitti.yaml",
                 publish_frame=True,
                 deskew=False,
                 dynamic_vehicle_filter=False,
                 publish_key_frame=True,
                 sub_correct_pose=True,
                 ):
        # ROS2 base parameters
        self.pc_topic: str = "/label_points" # input pointcloud topic
        self.base_frame: str = "base_link"
        self.odom_frame: str = "odom"
        self.odom_topic: str = "/sage_icp/odometry"
        self.trajectory_topic: str = "/sage_icp/trajectory"

        # Publish frame
        self.publish_frame: bool = publish_frame # publish frame in odom and map for visualization
        self.frame_topic: str = "/sage_icp/frame"
        self.local_map_topic: str = "/sage_icp/local_map"
        
        # Pointcloud pre-process
        self.deskew: bool = deskew # Point cloud deskew
        self.max_range: float = 100.0 # pointcloud max range
        self.min_range: float = 5.0 # pointcloud min range
        self.label_max_range: float = 50.0 # label max range

        # Dynamic cars remove
        self.dynamic_vehicle_filter: bool = dynamic_vehicle_filter
        self.dynamic_vehicle_filter_th: float = 0.5
        self.dynamic_vehicle_voxid: int = 5 # voxid in voxel_labels
        self.dynamic_remove_lankmark: list = [44, 48] # landmark labels for dynamic remove
        
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
        self.voxel_size: list = [0.6, 1.0, 0.9, 0.8, 1.0, 0.6]

        # Map
        self.voxel_size_map: float = 0.8
        self.local_map_range: float = 100.0
        self.basic_points_per_voxel: int = 20 # basic part
        self.critical_points_per_voxel: int = 20 # critical part
        self.basic_parts_labels: list = [40, 44, 48, 49, 50, 70, 72] # basic parts labels, others are critical parts
        
        # Semantic assisted association
        self.sem_th: float = 0.4
        
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
        self.publish_key_frame: bool = publish_key_frame # publish key frame
        self.key_frame_topic: str = "/sage_icp/key_frame"
        self.key_marker_topic: str = "/sage_icp/key_marker"
        self.key_frame_overlap: float = 0.5 # map sample overlap
        self.key_frame_bounds: list = [[-51.2, 51.2], [-51.2, 51.2], [-4, 2.4]] # Point Cloud Boundaries, used for generate occupancy map
        self.key_frame_bounds_str: str = self.pack_2d_array(self.key_frame_bounds)
        self.key_frame_occ_size: list = [128, 128] # H*W, occ resolution

        # Sub SEM-SAM topic
        self.sub_correct_pose: bool = sub_correct_pose
        self.correct_pose_topic: str = "/sem_sam/correct_marker" # correct odometry topic

        # RVIZ2 and ROS2 bag play
        self.visualize: str = "true" # must be string
        self.bagfile: str = ""
    
    def pack_2d_array(self, array_2d):
        return ';'.join([','.join(map(str, row)) for row in array_2d])