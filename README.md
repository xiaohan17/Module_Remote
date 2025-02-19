# Module_Remote
***
遥感图像处理相关模块整理，方便使用集成相应的模块、骨架代码，能够为后续研究提供相应思路
# 安装
***
1. 克隆该项目  
    ```shell
    git clone https://github.com/xiaohan17/Module_Remote.git
    cd Module_Remote
    ```
2. 创建并激活虚拟环境
    ```shell
    conda create -n conda_MR python==3.8 -y
    conda activate conda_MR
    ```
3. 安装pytorch
    ```shell
    conda install pytorch==2.3.0 cpuonly -c pytorch
    ```
# 目录
***
* Change Detection
  * [Temporal Fusion Attention Modules](Module/Change%20Detection/TFAM/README.md)
* Segmentation
  * [Frequency-aware Feature Fusion](Module/Segmentation/FFD/README.md)
  * [IGM-Att and PC-Att](Module/Segmentation/IGM-PC/README.md)
