## Object Tracking for nuScenes & LiDAR

本项目是一个基于 ROS 的 3D LiDAR 多目标跟踪包，包含：


- **IMM‑UKF‑JPDA 多模型滤波与数据关联**
- **轨迹管理与目标静/动态分类**
- **基于 nuScenes v1.0‑mini 的离线评估与 RViz 可视化**

原始思路参考论文  
*A Multiple-Model PHD/JPDA Filter for Tracking Multiple Extended Targets*（TUDelft），
并在此基础上做了大量工程化修改与 nuScenes 适配。

---

### 环境依赖

- Ubuntu 20.04 / ROS **Noetic**
- C++14 编译器（`catkin_make`）
- PCL（ROS 自带版本即可）
- OpenCV（由 PCL/ROS 依赖提供）
- Python 3 + `nuscenes-devkit`（用于预处理 nuScenes）

安装 nuScenes devkit：

```bash
pip install nuscenes-devkit
```

---

### 数据准备

1. 下载 **nuScenes v1.0‑mini** 数据集，将其放在项目根目录：

```text
object_tracking/
  v1.0-mini/
    v1.0-mini/
    maps/
    ...
```

2. 首次运行脚本会自动在项目下生成预处理结果：

```text
nuscenes_preprocessed/
  scene-0061.txt
  scene-0103.txt
  ...
```

---

### 编译与运行（推荐方式）

项目已经自带一键脚本 `run_nuscenes.sh`，负责：

1. 创建/更新 catkin 工作空间
2. 编译 `object_tracking` 包
3. 预处理 nuScenes v1.0‑mini
4. 运行 IMM‑UKF‑JPDA 跟踪（可选 RViz 可视化）

#### 基本用法

在项目根目录：

```bash
cd /home/wbk/dongbeidaxue/object_tracking
chmod +x run_nuscenes.sh

# 运行全部场景（快速，无可视化）
./run_nuscenes.sh

# 运行指定场景
./run_nuscenes.sh scene-0061

# 带 RViz 可视化（默认 2 Hz 回放）
./run_nuscenes.sh scene-0061 --viz

# 插值到 10 Hz 或 20 Hz
./run_nuscenes.sh scene-0061 --10hz
./run_nuscenes.sh scene-0061 --20hz

# 自定义插值帧率
./run_nuscenes.sh scene-0061 --interp=10
```

脚本会自动在 `~/dongbeidaxue/nuscenes_ws` 下创建/更新 catkin workspace，
并使用 `nuscenes_test.launch` 做可视化。

---

### 可视化说明（RViz）

在 `--viz` 模式下，`nuscenes_test.launch` 会启动 RViz，默认配置在 `rviz/nuscenes.rviz` 中：

- **蓝色半透明框**：nuScenes 提供的检测/标注框（GT）
- **绿色框**：IMM‑UKF‑JPDA 跟踪输出的目标框
- **彩色点**：跟踪目标中心（黄：初始化 / 绿：确认 / 蓝：静态）
- **绿色箭头**：目标速度方向
- **红色箭头**：自车位姿

---

### 代码结构

- `src/tracking/imm_ukf_jpda.cpp`  
  JPDA 数据关联、量测验证、跟踪管理与可视化接入。
- `src/tracking/ukf.cpp` / `include/Ukf.h`  
  多模型 Unscented Kalman Filter（CV / CTRV / RM / CA）及 IMM 模式概率更新。
- `src/tracking/nuscenes_main.cpp`  
  nuScenes 预处理结果 → ROS 消息 / RViz Marker 的桥接代码。
- `scripts/preprocess_nuscenes.py`  
  读取 nuScenes v1.0‑mini，插值到指定帧率、导出文本格式供 C++ 跟踪器使用。
- `visualize_nuscenes_trajectory.py`  
  使用 nuScenes devkit 在 Python 中快速查看某个 agent 的历史/未来轨迹（调试用）。

---

### 注意事项

- 仓库默认 **不包含 nuScenes 原始数据** 与中间结果：
  - `v1.0-mini/`、`nuscenes_preprocessed/`、`backup/` 等目录已在 `.gitignore` 中忽略。
- 若要在 KITTI 或其他数据集上使用，需要自行适配输入格式与坐标系。

---

### 致谢

- 原始工程与思路来源于社区开源项目（如 TUDelft 相关实现）以及  
  GitHub 社区中关于 LiDAR 多目标跟踪的开源代码。
- nuScenes 数据集由 Motional 提供（`https://www.nuscenes.org`）。

