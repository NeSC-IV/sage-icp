import os
import sys
import argparse
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix, CameraInfo, PointField
from std_msgs.msg import Header
from ament_index_python import get_package_share_directory,get_package_prefix
import time
import open3d as o3d
from std_srvs.srv import Empty
from rclpy.duration import Duration
from rclpy.serialization import serialize_message
from example_interfaces.msg import Int32
import rclpy
from rclpy.time import Time

import rosbag2_py
import sys
sys.dont_write_bytecode = True
import math
from numpy.linalg import inv
import tf_transformations
import cv2
# from cv_bridge import CvBridge
import progressbar
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pcl2 # point_cloud2.create_cloud() 函数是sensor_msgs.msg.PointCloud2消息的一个帮助函数，它将一系列点的x、y、z坐标和其他属性打包到点云消息中。
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from example_interfaces.srv import AddTwoInts
import pandas as pd

class SemanticKitti_Raw:
    """Load and parse raw data into a usable format"""

    def __init__(self, dataset_path, sequence_number, scanlabel_bool, **kwargs):
        self.data_path = dataset_path + sequence_number
        self.seq = os.listdir(self.data_path)[0]
        self.data_path = os.path.join(self.data_path, self.seq, 'velodyne_points')
        self.frames = kwargs.get('frames', None)

        self.imtype = kwargs.get('imtype', 'png')

        self._get_file_lists(scanlabel_bool)
        #self._load_calib()

        # Add correction for KITTI datasets, can be easilty removed if unwanted
        from kiss_icp.pybind import kiss_icp_pybind
        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
        )
        self._load_timestamps()

    def _get_file_lists(self, scanlabel_bool):

        self.velo_files = sorted(os.listdir(os.path.join(self.data_path, 'data')))
            # os.path.join(self.data_path, 'data', '*.bin')))

        if scanlabel_bool == 1:
            self.label_files = sorted(glob.glob(
                os.path.join(self.data_path, 'labels', '*.label')))

        # if self.frames is not None:

    def _load_timestamps(self):
        timestamp_file = os.path.join(
                self.data_path, 'timestamps.txt')

        self.timestamps = []
        time_fist = False
        first_time = 0.0
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                read_time = line.split(" ")[1]
                timest = read_time.split(":")
                timesec = float(timest[0])*3600 + float(timest[1])*60 + float(timest[2])
                if time_fist == False:
                    first_time = timesec
                    time_fist = True
                number = timesec - first_time
                if number == 0.0:
                    number = 0.0001
                self.timestamps.append(number)

def inv_t(transform):

    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1*R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv

    return transform_inv

def convertdata(labelscan, color_map_bgr):
    labels = []
    rgbs = []

    for counting in range(len(labelscan)):
        sem_id = int(labelscan[counting]) & 0xFFFF
        bgr = color_map_bgr[sem_id]
        rgb = (int(bgr[2]) << 16) | (int(bgr[1]) << 8) | int(bgr[0])
        labels.append(sem_id)
        rgbs.append(rgb)

    return np.array(labels, dtype=np.uint8), np.array(rgbs, dtype=np.uint32)

def get_velo_data_with_label(kitti, velo_frame_id, iter, color_map_bgr):
    velo_data_dir = os.path.join(kitti.data_path, 'velodyne')
    velo_filenames = sorted(os.listdir(velo_data_dir))

    label_data_dir = os.path.join(kitti.data_path, 'labels')
    label_filenames = sorted(os.listdir(label_data_dir))

    datatimes = kitti.timestamps
    dt = datatimes[iter]
    veloname = velo_filenames[iter]
    labelname = label_filenames[iter]

    velo_filename = os.path.join(velo_data_dir, veloname)
    label_filename = os.path.join(label_data_dir, labelname)

    veloscan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
    veloscan3 = veloscan[:, :3].astype(np.float32)
    veloscan3 = kitti.correct_kitti_scan(veloscan3).astype(np.float32)
    
    labelscan = (np.fromfile(label_filename, dtype=np.int32)).reshape(-1,1)
    
    labels, rgbs = convertdata(labelscan, color_map_bgr)
    
    scan = np.zeros((veloscan3.shape[0],), dtype=[
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('label', np.uint8),
                    ('rgb', np.uint32),
            ])
    scan['x'] = veloscan3[:,0]
    scan['y'] = veloscan3[:,1]
    scan['z'] = veloscan3[:,2]
    scan['label'] = labels
    scan['rgb'] = rgbs

    header = Header()
    header.frame_id = velo_frame_id
    time = Time(seconds = float(dt)) # Clock().now()
    header.stamp = time.to_msg()

    fields =[PointField(name='x',  offset=0, datatype=PointField.FLOAT32, count = 1),
            PointField(name='y',  offset=4, datatype=PointField.FLOAT32, count = 1),
            PointField(name='z',  offset=8, datatype=PointField.FLOAT32, count = 1),
            PointField(name='label', offset=12, datatype=PointField.UINT8, count = 1),
            PointField(name='rgb', offset=13, datatype=PointField.UINT32, count = 1),
            ]

    pcl_msg = pcl2.create_cloud(header, fields, scan)
    return pcl_msg

def get_velo_data(kitti, velo_frame_id, iter):
    velo_data_dir = os.path.join(kitti.data_path, 'data')
    # velo_filenames = sorted(os.listdir(velo_data_dir))

    datatimes = kitti.timestamps
    dt = datatimes[iter]
    veloname = kitti.velo_files[iter]
    # if dt is None:
    #     continue

    velo_filename = os.path.join(velo_data_dir, veloname)

    veloscan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
    points = veloscan.astype(np.float64)

    header = Header()
    header.frame_id = velo_frame_id
    time = Time(seconds = float(dt)) # Clock().now()
    header.stamp = time.to_msg()

    fields =[PointField(name='x',  offset=0, datatype=PointField.FLOAT32, count = 1),
            PointField(name='y',  offset=4, datatype=PointField.FLOAT32, count = 1),
            PointField(name='z',  offset=8, datatype=PointField.FLOAT32, count = 1),
            PointField(name='intensity',  offset=12, datatype=PointField.FLOAT32, count = 1)]

    pcl_msg = pcl2.create_cloud(header, fields, points)
    veloname = int(veloname.split(".")[0])

    return pcl_msg, veloname

def read_calib_file(filename):
    """ read calibration file 

        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    """calib1 = np.eye(4,4)
    calib1[0:3, 3] = [0.27, 0.0, -0.08]
    print(calib1)
    calib.append(calib1)

    calib2 = np.eye(4,4)
    calib2[0:3, 3] = [0.27, -0.51, -0.08]
    print(calib2)
    calib.append(calib2)

    calib3 = np.eye(4,4)
    calib3[0:3, 3] = [0.27, 0.06, -0.08]
    print(calib3)
    calib.append(calib3)

    calib4 = np.eye(4,4)
    calib4[0:3, 3] = [0.27, -0.45, -0.08]
    print(calib4)
    calib.append(calib4)"""
    calib_file = open(filename)

    key_num = 0

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4,4))
        
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    
    #print(calib)
    return calib

def read_poses_file(filename):
    pose_file = open(filename)

    poses = {}

    Tr = np.array([[0.99992906, 0.0057743, 0.01041756, 0.77104934],
                 [0.00580536, -0.99997879, -0.00295331, 0.29854144],
                 [0.01040029, 0.00301357, -0.99994137, -0.83628022],
                 [0, 0, 0, 1]])
    Tr_inv = inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[1:5]
        pose[1, 0:4] = values[5:9]
        pose[2, 0:4] = values[9:13]
        pose[3, 3] = 1.0
        poses[int(values[0])] = np.matmul(Tr_inv, np.matmul(pose, Tr)) # lidar_gt
        #return Tr @ poses @ np.linalg.inv(Tr)
        # left = np.einsum("...ij,...jk->...ik", np.linalg.inv(Tr), pose)
        # right = np.einsum("...ij,...jk->...ik", left, Tr)
        # poses2.append(right)

    pose_file.close()
    return poses

def get_pose_msg(kitti, poses, master_frame_id, iter, seq, first_pose_inv):
    # posestamp只存储位姿信息
    datatimes = kitti.timestamps
    dt = datatimes[iter]
    pose = poses[seq]
    pose = np.matmul(first_pose_inv, pose)

    p = PoseStamped()
    p.header.frame_id = master_frame_id
    time = Time(seconds = float(dt))
    p.header.stamp = time.to_msg()

    t = pose[0:3, 3]
    q = tf_transformations.quaternion_from_matrix(pose)

    p.pose.position.x = t[0]
    p.pose.position.y = t[1]
    p.pose.position.z = t[2]

    q_n = q / np.linalg.norm(q) # 四元数归一化

    p.pose.orientation.x = q_n[0]
    p.pose.orientation.y = q_n[1]
    p.pose.orientation.z = q_n[2]
    p.pose.orientation.w = q_n[3]
    return p

class Listener(Node):
    def __init__(self,name,seq):
        super().__init__(name)
        self.iter = 0
        self.get_pose = True
        self.odom_group = []
        self.gt_group = []
        self.sequence_number = str(seq)
        self.scanlabel_bool = 0

        self.kitti = SemanticKitti_Raw("/media/oliver/Elements SE/dataset/kitti_360/data_3d_test_slam/test_", self.sequence_number, self.scanlabel_bool)

        if not os.path.exists(self.kitti.data_path):
            print('Path {} does not exists. Force-quiting....'.format(self.kitti.data_path))
            sys.exit(1)

        if len(self.kitti.timestamps) == 0:
            print('Dataset is empty? Check your semantickitti dataset file')
            sys.exit(1)
        
        self.world_frame_id = 'map'
        self.velo_frame_id = 'velodyne'
        # calibration = read_calib_file(os.path.join(self.kitti.data_path, 'calib.txt'))
        # ground_truth_file_name = "{}.txt".format(self.sequence_number)

        self.ground_truth = read_poses_file(os.path.join("/media/oliver/Elements SE/dataset/kitti_360/data_poses", self.kitti.seq, "poses.txt"))
        self.lenth = self.kitti.timestamps.__len__()
        self.odomx = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.odomy = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        self.gtx = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.gty = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        plt.ion()                  # 开启一个画图的窗口
        self.subscriber = self.create_subscription(Odometry,
                                                  "odometry",
                                                  self.sub_callback,
                                                  10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/velodyne_points',1)
        self.pose_pub = self.create_publisher(PoseStamped, '/ground_truth', 1)
        # 创建service,用于更新seq
        self.cli = self.create_client(AddTwoInts, 'reinit')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

        self.first_pose_inv = inv(self.ground_truth[int(self.kitti.velo_files[0].split(".")[0])])

        timer_period = 0.2 # 定时器周期为0.1s
        self.timer = self.create_timer(timer_period,self.timer_callback) #创建定时器
        self.get_logger().info("Publisher is started, publishing kitti %s to %s" % (seq,"/velodyne_points"))
        self.pbar = tqdm(total=self.lenth)

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        try:
            rclpy.spin_until_future_complete(self, self.future, timeout_sec=10.0)  # 设置超时时间为10秒
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))
            return None
        if self.future is not None and self.future.done():
            if (self.future.result().sum == 1):
                self.get_logger().info('Finish cleaning!')
        return self.future.result()

    def sub_callback(self,msg):
        odom = [msg.header.stamp.sec, msg.header.stamp.nanosec, msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.odom_group.append(odom)
        self.odomx.append(msg.pose.pose.position.x)
        self.odomy.append(msg.pose.pose.position.y)
        plt.clf()              # 清除之前画的图
        plt.plot(self.odomx,self.odomy,color = "g")        # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(self.gtx, self.gty, color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.01)         # 暂停一秒
        plt.ioff()             # 关闭画图的窗口
        self.get_pose = True


    def timer_callback(self):
        if self.get_pose:
            if(self.iter == self.lenth):
                print('Convertion is done, start saving pose ...')
                plt.clf()              # 清除之前画的图
                plt.plot(self.odomx,self.odomy,color = "g")        # 画出当前 ax 列表和 ay 列表中的值的图形
                plt.plot(self.gtx, self.gty, color='r')
                plt.xlabel('x')
                plt.ylabel('y')
                saving_path = "/home/oliver/catkin_ros2/src/sage-icp/results/seq" + str(int(self.sequence_number))
                if not os.path.exists(saving_path): # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(saving_path)
                plt.savefig(saving_path + "/sequence" + self.sequence_number + '.png', bbox_inches='tight')
                with open(saving_path + "/path.txt", 'w') as f:
                    for odom in self.odom_group:
                        f.writelines(str(odom[0]) + '.' + str(odom[1]) + ' ' + str(odom[2]) + ' ' + str(odom[3]) + ' ' + str(odom[4]) + ' ' + str(odom[5]) + ' ' + str(odom[6]) + ' ' + str(odom[7]) + ' ' + str(odom[8]) + '\n')
                with open(saving_path + "/gt_path.txt", 'w') as f:
                    for gt in self.gt_group:
                        f.writelines(str(gt[0]) + '.' + str(gt[1]) + ' ' + str(gt[2]) + ' ' + str(gt[3]) + ' ' + str(gt[4]) + ' ' + str(gt[5]) + ' ' + str(gt[6]) + ' ' + str(gt[7]) + ' ' + str(gt[8]) + '\n')
                response = self.send_request(int(self.sequence_number), 0)
                # plt.ioff()             # 关闭画图的窗口
                # 结束进程
                raise SystemExit           # <--- here is we exit the node

            # if self.scanlabel_bool == 1:
            #     pointcloud = get_velo_data_with_label(self.kitti, self.velo_frame_id, self.iter)

            # elif self.scanlabel_bool == 0:
            pointcloud, pose_seq = get_velo_data(self.kitti, self.velo_frame_id, self.iter)
            pose = get_pose_msg(self.kitti, self.ground_truth, self.world_frame_id, self.iter, pose_seq, self.first_pose_inv) # posestamped: ground_truth to map, odom: ground_truth to map

            self.lidar_pub.publish(pointcloud)
            self.pose_pub.publish(pose)

            gt_pose = [pose.header.stamp.sec, pose.header.stamp.nanosec, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
            self.gt_group.append(gt_pose)
            self.gtx.append(pose.pose.position.x)
            self.gty.append(pose.pose.position.y)
            self.iter +=1
            self.pbar.update(1)
            self.get_pose = False



def main(args=None):
    rclpy.init()
    for seq in range(0, 4): #(0,3)
        print('Start publishing sequence %s' % seq)
        dataset_pub = Listener('dataset_publisher',seq)
        try:
            rclpy.spin(dataset_pub)
        except SystemExit:                 # <--- process the exception 
            rclpy.logging.get_logger("Quitting").info('Done')
        dataset_pub.destroy_node()
        time.sleep(5)
    rclpy.shutdown()  # 停止节点

if __name__ =='__main__':
    # talker()
    main()
    # os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
