<div align="center">
    <h1>SAGE-ICP</h1>
    <a href=https://ieeexplore.ieee.org/document/10610280>Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://arxiv.org/abs/2310.07237>Arxiv</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://www.youtube.com/watch?v=6bXX2PnEjj4>YouTube</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://www.bilibili.com/video/BV1BH4y1o7N3/?spm_id_from=333.999.0.0>bilibili</a>
    <br />
    <br />
    
[SAGE-ICP](https://ieeexplore.ieee.org/document/10610280) is a semantic information-assisted point-to-point ICP ([ICRA 2024](https://2024.ieee-icra.org/), accepted).  
  <br />
[Jiaming Cui](https://github.com/shashenyiguang), Jiming Chen, [Liang Li](https://github.com/liangli1990)  

  <p align="center">
    <a href="https://www.youtube.com/watch?v=6bXX2PnEjj4"><img alt="SAGE-ICP Demo" src="figure/pipeline.png"></a>
  </p>
</div>
<hr />

## Prerequisites
1. [ROS2](https://www.ros.org/blog/getting-started/)  
2. Dependencies
```sh
sudo apt-get install ros-<ros_distro>-tf-transformations
pip install transforms3d kiss-icp lark-parser
```
3. Semantic segmentation models for LiDAR point clouds, could select one installation.  
[Cylinder3D](https://github.com/NeSC-IV/cylinder3d_ros2), [RangeNet++](https://github.com/NeSC-IV/RangeNetTrt8) 

## Install
```sh
mkdir -p ~/sage-icp/src && cd ~/sage-icp/src
git clone https://github.com/NeSC-IV/sage-icp.git
cd .. && colcon build --symlink-install
source install/setup.bash
```

## Evaluation
First, start a semantic point cloud publishing node:
```sh
# Cylinder3D for VLP-64
ros2 launch cylinder3d_ros2 cylinder3d_ros2.launch.py
# RangeNet++ for VLP-64
ros2 launch rangenet_pp ros2_rangenet.launch.py
# or you can use a script in eval/
python kitti_pub.py
```
To evaluate sage-icp only, please run:
```sh
# for KITTI Odometry using gt semantic label
ros2 launch sage_icp odom_kittigt_launch.py
# for KITTI Odometry using semantic segmentation model
ros2 launch sage_icp odom_kitti_launch.py
# for KITTI raw using semantic segmentation model
ros2 launch sage_icp odom_kittiraw_launch.py
# for KITTI360 using semantic segmentation model
ros2 launch sage_icp odom_kitti360_launch.py
# for LIVOX
ros2 launch sage_icp odom_livox_launch.py
```
To evaluate on Semantic SLAM, please run:
```sh
source install/setup.bash
ros2 launch sage_icp slam_livox_launch.py
```
## topics
### input topic
* topic: `/label_points`
* description: Input LiDAR point cloud with semantic labels
* type: `sensor_msgs/msg/PointCloud2`
* format: (x,y,z,label,rgb)
  * `xyz` (float32)
  * `label` (uint8)
  * `rgb` (uint32)
  
### output topic
* topic: `/sage_icp/key_frame`
* description: Output key frame point cloud in base_link frame
* type: `sensor_msgs/msg/PointCloud2`
* format: (x,y,z,label,rgb)
  * `xyz` (float32)
  * `label` (uint8)
  * `rgb` (uint32)

* topic: `/sage_icp/key_marker`
* description: Output key frame marker
* type: `visualization_msgs/msg/Marker, SPHERE`
* info:
  * `id`: key frame id, start with 0
  * `pose`: key frame transformation from base_link to odom

## Citation

If you use this library for any academic work, please cite our original [Paper](https://ieeexplore.ieee.org/document/10610280).
```sh
@INPROCEEDINGS{10610280,
  author={Cui, Jiaming and Chen, Jiming and Li, Liang},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={SAGE-ICP: Semantic Information-Assisted ICP}, 
  year={2024},
  volume={},
  number={},
  pages={8537-8543},
  keywords={Location awareness;Training;Laser radar;Accuracy;Semantic segmentation;Semantics;Pose estimation},
  doi={10.1109/ICRA57147.2024.10610280}}
```



## Acknowledgements
This repo is built upon [KISS-ICP](https://github.com/PRBonn/kiss-icp), [Cylinder3D](https://github.com/xinge008/Cylinder3D) and [RangeNet++](https://github.com/PRBonn/rangenet_lib).  
This project is free software made available under the MIT License. For details see the LICENSE file.