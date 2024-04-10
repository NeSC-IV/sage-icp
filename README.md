<div align="center">
    <h1>SAGE-ICP</h1>
    <a href=https://arxiv.org/abs/2310.07237>Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://www.youtube.com/watch?v=6bXX2PnEjj4>YouTube</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://www.bilibili.com/video/BV1BH4y1o7N3/?spm_id_from=333.999.0.0>bilibili</a>
    <br />
    <br />
    
[SAGE-ICP](https://arxiv.org/abs/2310.07237) is a semantic information-assisted point-to-point ICP ([ICRA 2024](https://2024.ieee-icra.org/), accepted).  
  <br />
[Jiaming Cui](https://github.com/shashenyiguang), Jiming Chen, [Liang Li](https://github.com/liangli1990)  

  <p align="center">
    <a href="https://www.youtube.com/watch?v=6bXX2PnEjj4"><img alt="SAGE-ICP Demo" src="figure/pipeline.png"></a>
  </p>
</div>
<hr />

## Prerequisites
1. Ubuntu 20.04 (tested) or 22.04  
2. [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)  
> **Note**
>
> Debian packages for ROS 2 Humble Hawksbill are currently available for Ubuntu Jammy (22.04).  
> Source installation is available for Ubuntu Focal (20.04).  

3. [Cylinder3D](https://github.com/NeSC-IV/cylinder3d_ros2), [RangeNet++](https://github.com/NeSC-IV/RangeNetTrt8)  
Semantic segmentation models for LiDAR point clouds, could select one installation.

## Install
```sh
mkdir -p ~/sage-icp/src && cd ~/sage-icp/src
git clone https://github.com/NeSC-IV/sage-icp.git
cd .. && colcon build --symlink-install
```

## Evaluation
To evaluate cylinder3D on KITTI Odometry, please run:
```sh
source install/setup.bash
ros2 launch sage_icp sem_odom.launch.py
```
To evaluate RangeNet++ on KITTI Odometry, please run:
```sh
source install/setup.bash
ros2 launch sage_icp range_odom.launch.py
```

## Citation

If you use this library for any academic work, please cite our original [arxiv paper](https://arxiv.org/abs/2310.07237).

## Acknowledgements
This repo is built upon [KISS-ICP](https://github.com/PRBonn/kiss-icp), [Cylinder3D](https://github.com/xinge008/Cylinder3D) and [RangeNet++](https://github.com/PRBonn/rangenet_lib).  
This project is free software made available under the MIT License. For details see the LICENSE file.