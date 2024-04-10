import pykitti
import os
import sys
import argparse
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix, CameraInfo, PointField, Image
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
import utils #import utils.py
from numpy.linalg import inv
import tf_transformations
import cv2
from cv_bridge import CvBridge
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
from scipy.spatial.transform import Rotation
def takeSort(elem):
    return int(elem)

# from kiss_icp.pybind import kiss_icp_pybind
# correct_kitti_scan = lambda frame: np.asarray(
#     kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
# )


# def inv_t(transform):

#     R = transform[0:3, 0:3]
#     t = transform[0:3, 3]
#     t_inv = -1*R.T.dot(t)
#     transform_inv = np.eye(4)
#     transform_inv[0:3, 0:3] = R.T
#     transform_inv[0:3, 3] = t_inv

#     return transform_inv


# def get_velo_data(kitti, velo_frame_id, iter):
#     # velo_data_dir = os.path.join(kitti.data_path, 'data')
#     # velo_filenames = sorted(os.listdir(velo_data_dir))

#     # datatimes = kitti.timestamps
#     # dt = datatimes[iter]
#     # veloname = kitti.velo_files[iter]
#     # if dt is None:
#     #     continue

#     # velo_filename = os.path.join(velo_data_dir, veloname)

#     # veloscan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
#     # points = veloscan.astype(np.float64)

#     points = np.array(kitti.get_velo(iter)).reshape(-1, 4)
#     veloscan3 = points[:, :3].astype(np.float64)
#     velodyne_correct = correct_kitti_scan(veloscan3)
#     points[:, :3] = velodyne_correct

#     header = Header()
#     header.frame_id = velo_frame_id
#     time = Clock().now()
#     header.stamp = time.to_msg()

#     fields =[PointField(name='x',  offset=0, datatype=PointField.FLOAT32, count = 1),
#             PointField(name='y',  offset=4, datatype=PointField.FLOAT32, count = 1),
#             PointField(name='z',  offset=8, datatype=PointField.FLOAT32, count = 1),
#             PointField(name='intensity',  offset=12, datatype=PointField.FLOAT32, count = 1)]

#     pcl_msg = pcl2.create_cloud(header, fields, points)
#     return pcl_msg

# def get_pose_msg(kitti, master_frame_id, iter):
#     # posestamp只存储位姿信息
#     pose = kitti.oxts[iter].T_w_imu
#     # 将imu坐标系下的位姿转换到lidar坐标系下
#     pose = np.matmul(np.linalg.inv(kitti.oxts[0].T_w_imu), pose)
#     p = PoseStamped()
#     p.header.frame_id = master_frame_id
#     time = Clock().now()
#     p.header.stamp = time.to_msg()

#     t = pose[0:3, 3]
#     q = tf_transformations.quaternion_from_matrix(pose)

#     p.pose.position.x = t[0]
#     p.pose.position.y = t[1]
#     p.pose.position.z = t[2]

#     q_n = q / np.linalg.norm(q) # 四元数归一化

#     p.pose.orientation.x = q_n[0]
#     p.pose.orientation.y = q_n[1]
#     p.pose.orientation.z = q_n[2]
#     p.pose.orientation.w = q_n[3]
#     return p

def loadPoses(file_name, toCameraCoord, format):
        '''
            Each line in the file should follow one of the following structures
            time x y z rx ry rz rw
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            if(format == 'tum'):
                # xyz
                P[0, 3] = line_split[1]
                P[1, 3] = line_split[2]
                P[2, 3] = line_split[3]
                # Q to R
                q = np.array([line_split[4],line_split[5],line_split[6],line_split[7]]) #xyzw
                r = Rotation.from_quat(q)
                rot_mat = r.as_matrix()
                P[0:3, 0:3] = rot_mat
                frame_idx = cnt
            elif(format == 'kitti'):
                withIdx = int(len(line_split)==13)
                for row in range(3):
                    for col in range(4):
                        P[row, col] = line_split[row*4 + col + withIdx]
                if withIdx:
                    frame_idx = line_split[0]
                else:
                    frame_idx = cnt
            else :
                print('Unknown pose format!')
                exit()
            # if toCameraCoord:
            #     poses[frame_idx] = self.toCameraCoord(P)
            # else:
            poses[frame_idx] = P
        return poses

class Listener(Node):
    def __init__(self,name,seq,basedir,date,drive):
        super().__init__(name)
        self.iter = 0
        self.get_pose = True
        self.odom_group = []
        self.gt_group = []
        self.sequence_number = str(seq)
        self.scanlabel_bool = 0

        self.data = pykitti.raw(basedir, date, drive)
        
        self.world_frame_id = 'map'
        self.velo_frame_id = 'velodyne'
        # calibration = read_calib_file(os.path.join(self.kitti.data_path, 'calib.txt'))
        # ground_truth_file_name = "{}.txt".format(self.sequence_number)

        # self.ground_truth = read_poses_file(os.path.join("/media/oliver/Elements SE/dataset/kitti_360/data_poses", self.kitti.seq, "poses.txt"))
        self.lenth = self.data.timestamps.__len__()
        self.oursx = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.oursy = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        self.rangex = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.rangey = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        self.kissx = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.kissy = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        self.gtx = []                    # 定义一个 x 轴的空列表用来接收动态的数据
        self.gty = []                    # 定义一个 y 轴的空列表用来接收动态的数据
        plt.ion()                  # 开启一个画图的窗口
        # self.subscriber = self.create_subscription(Odometry,
        #                                           "odometry",
        #                                           self.sub_callback,
        #                                           10)
        # self.lidar_pub = self.create_publisher(PointCloud2, '/velodyne_points',1)
        # self.pose_pub = self.create_publisher(PoseStamped, '/ground_truth', 1)
        self.image_pub = self.create_publisher(Image, '/image_raw', 1)
        self.br = CvBridge()
        # 创建service,用于更新seq
        # self.cli = self.create_client(AddTwoInts, 'reinit')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.req = AddTwoInts.Request()

        # self.first_pose_inv = inv(self.ground_truth[int(self.kitti.velo_files[0].split(".")[0])])

        timer_period = 1 # 定时器周期为0.1s
        self.timer = self.create_timer(timer_period,self.timer_callback) #创建定时器
        self.get_logger().info("Publisher is started, publishing kitti %s to %s" % (seq,"/velodyne_points"))
        self.pbar = tqdm(total=self.lenth)
        # read path
        self.kisspose = loadPoses(os.path.join("/home/oliver/catkin_ros2/src/kiss-icp", 'results', 'kissseq31', 'path.txt'), False, 'tum')
        self.ourspose = loadPoses(os.path.join("/home/oliver/catkin_ros2/src/kiss-icp", 'results', 'oursseq31', 'path.txt'), False, 'tum')
        self.rangepose = loadPoses(os.path.join("/home/oliver/catkin_ros2/src/kiss-icp", 'results', 'rangeseq31', 'path.txt'), False, 'tum')
        self.gtpose = loadPoses(os.path.join("/home/oliver/catkin_ros2/src/kiss-icp", 'results', 'kissseq31', 'gt_path.txt'), False, 'tum')


    # def send_request(self, a, b):
    #     self.req.a = a
    #     self.req.b = b
    #     self.future = self.cli.call_async(self.req)
    #     try:
    #         rclpy.spin_until_future_complete(self, self.future, timeout_sec=10.0)  # 设置超时时间为10秒
    #     except Exception as e:
    #         self.get_logger().error('Service call failed %r' % (e,))
    #         return None
    #     if self.future is not None and self.future.done():
    #         if (self.future.result().sum == 1):
    #             self.get_logger().info('Finish cleaning!')
    #     return self.future.result()

    # def sub_callback(self,msg):
    #     odom = [msg.header.stamp.sec, msg.header.stamp.nanosec, msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    #     self.odom_group.append(odom)
    #     self.odomx.append(msg.pose.pose.position.x)
    #     self.odomy.append(msg.pose.pose.position.y)
    #     plt.clf()              # 清除之前画的图
    #     plt.plot(self.odomx,self.odomy,color = "g")        # 画出当前 ax 列表和 ay 列表中的值的图形
    #     plt.plot(self.gtx, self.gty, color='r')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.pause(0.01)         # 暂停一秒
    #     plt.ioff()             # 关闭画图的窗口
    #     self.get_pose = True


    def timer_callback(self):
        if self.iter < self.lenth:
            # publish image
            num_str = "0"
            if self.iter < 10:
                num_str = "000000000" + str(self.iter) + ".png"
            elif self.iter < 100:
                num_str = "00000000" + str(self.iter) + ".png"
            elif self.iter < 1000:
                num_str = "0000000" + str(self.iter) + ".png"
            image_filename = os.path.join("/media/oliver/Elements SE/dataset/KITTI/2011_09_26/2011_09_26_drive_0027_sync/image_02/data", num_str)
            cv_image = cv2.imread(image_filename)

            encoding = "bgr8"
            image_message = self.br.cv2_to_imgmsg(cv_image, encoding=encoding)
            image_message.header.frame_id = "camera"
            self.image_pub.publish(image_message)

            # draw path
            self.oursx.append(self.ourspose[self.iter][0,3])
            self.oursy.append(self.ourspose[self.iter][1,3])
            self.kissx.append(self.kisspose[self.iter][0,3])
            self.kissy.append(self.kisspose[self.iter][1,3])
            self.rangex.append(self.rangepose[self.iter][0,3])
            self.rangey.append(self.rangepose[self.iter][1,3])
            self.gtx.append(self.gtpose[self.iter][0,3])
            self.gty.append(self.gtpose[self.iter][1,3])
            plt.clf()              # 清除之前画的图
            plt.xlim(-10, 360)
            plt.ylim(-4, 9) 
            plt.plot(self.gtx, self.gty, color='r', label='Ground Truth')
            plt.plot(self.oursx, self.oursy, color='g', label='Ours')
            plt.plot(self.kissx, self.kissy, color='b', label='KISS-ICP')
            plt.plot(self.rangex, self.rangey, color='k', label='Ours-RangeNet')
            start_point = [0, 0]
            plt.plot(start_point[0], start_point[1], 'ko', label='Start Point')
            fontsize_ = 12
            plt.xlabel('x (m)', fontsize=15)
            plt.ylabel('y (m)', fontsize=15)
            # 设置刻度标签的字体大小
            plt.tick_params(axis='both', which='major', labelsize=fontsize_)
            plt.legend(loc='upper left', prop={'size':fontsize_})
            plt.pause(0.01)         # 暂停一秒
            plt.ioff()             # 关闭画图的窗口


            self.iter +=1
            self.pbar.update(1)
        else:
            raise SystemExit           # <--- here is we exit the node


def main(args=None):
    rclpy.init()
    basedir = "/media/oliver/Elements SE/dataset/KITTI"
    if not os.path.exists(basedir):
        print('KITTI dataset not found')
        return
    date_seq = []
    drive_seq = []
    for name in os.listdir(basedir):
        # 获取name前三位字符
        if (name[0:4] != "2011"): continue

        if os.path.isdir(os.path.join(basedir, name)):
            drive_seq_temp = []
            for drive in os.listdir(os.path.join(basedir, name)):
                if os.path.isdir(os.path.join(basedir, name, drive)):
                    drive_name = drive[-9:-5]
                    date_seq.append(name)
                    drive_seq_temp.append(drive_name)
            # 排序drive_seq_temp
            drive_seq_temp.sort(key=takeSort)
            drive_seq = drive_seq + drive_seq_temp
    
    for seq in range(31, 32): #[30,41]
        if seq == 35:
            continue
        print('Start publishing sequence %s' % seq)
        dataset_pub = Listener('dataset_publisher',seq,basedir,date_seq[seq-30], drive_seq[seq-30])
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


