# 在另一个集群运行旋钮仿真（不依赖 T2-z）

当前90°示范使用 **PyBullet + 脚本控制器** 生成。运行这条数据管线不需要GPU、Isaac、T2-z、策略网络或模型检查点。
`src/dexx/tasks/sharpa_contact_adapt/knob_env.py` 另有 Isaac Lab 环境，但它是固定机械臂、22维手动作的实验版本，尚未完成接触/成功转动验证；不能把它当成生成本批29维动作数据的同一个环境。

## 安装

```bash
git clone https://github.com/CathyF9600/sharpa-knob-sim.git
cd sharpa-knob-sim
python3.10 -m venv .venv-knob
source .venv-knob/bin/activate
python -m pip install -r requirements-knob-cpu.txt
```

本仓库是从Dex-X和本地旋钮任务中提取的独立仿真子集，不含策略网络、训练检查点、数据集或T2-z。保留少量共享配置及空包入口，以复用机器人参数而不注册原始策略环境。
CPU管线只需上述依赖，也不需要 `pip install -e .`；入口脚本自行加入仓库 `src` 路径。
在集群分配的CPU作业中执行以下命令，线程和worker数不超过分配的CPU数量。

## 最小完整采集：四档各一条，目标90°

从仓库根目录执行；输出路径可放在新集群的共享存储。目录必须不存在，继续已有批次才加 `--resume`。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python scripts/knob/batch180.py --out logs/knob90_smoke \
  --source assets/tasks/knob --goal-degrees 90 --seed 90100 \
  --per-condition 1 --workers 4 --seconds 50 --max-attempts 12
python scripts/knob/summarize_batch180.py logs/knob90_smoke --per-condition 1 --goal-degrees 90
python scripts/knob/export_rgbd.py logs/knob90_smoke --workers 4
python scripts/knob/preview_rgbd.py logs/knob90_smoke
```

脚本虽然叫 `batch180.py`，实际目标由 `--goal-degrees` 决定。四档轴制动力矩为0/0.005/0.03/0.10 N·m。
每条先模拟5秒准备抓取，随后以实际轴角为起点转90°；筛选满足精度、过冲、保持时间、碰撞等门槛的轨迹。
教师使用仿真真实角度/角速度；预抓取姿态已保存，抓取不依赖视觉网络，也不是强化学习生成。

生成200条时换新输出目录，把 `--per-condition` 改为50、`--max-attempts` 改为200，并给summarize传 `--per-condition 50`。
大量TinyRenderer图像渲染需要CPU时间与磁盘空间；已有200条RGB-D约15.4GB，原始轨迹与临时输出另占空间。
`finish_rgbd_dataset.py` 和 `stream_rgbd_export.py` 是历史4×50交付工具，带200条等固定假设；小样本及自定义数据集用上面的独立命令。

## 文件与接口

- `assets/generated/fr3_with_right_sharpa_wave.urdf`：机械臂与手的组合模型。
- `assets/franka_fr3/`、`assets/sharpa_wave/`：机器人网格和原始许可证，必须保留。
- `assets/tasks/knob/`：旋钮几何、STL/SCAD打印件、刻度、预抓取及闭合关节配置。
- `calib/camera_align/current.npy`、`src/dexx/deploy_config.py`：仿真相机外参、内参和基座/桌面坐标。
- `scripts/knob/cpu_rollouts.py`：`PreviewSim`，`step(target)`接收29维绝对关节目标（弧度），推进1/30秒并返回模拟状态。
- `scripts/knob/batch180.py`、`goal_rollouts.py`：脚本教师与批量采集。
- `scripts/knob/export_rgbd.py`：按保存状态渲染同步RGB-D HDF5，不重新推进物理。
- `src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py`：CPU生成器从源码读取已有碰撞过滤对，需保留；无需导入Isaac。

HDF5每条有N个动作、N+1个观察：`observations/rgb`、`observations/depth_m`、`observations/q_rad`、`observations/dq_rad_s`、`observations/tip_normal_force_world_n`、`actions/joint_position_target_rad`。
`obs[t]`是动作前，`actions[t]`执行后得到`obs[t+1]`。`privileged/*`含阻力/轴角等真值，默认不能当作视觉策略输入。
动作顺序见每条header的`joint_names`，7维FR3+22维Sharpa手；它不是原Dex-X策略的归一化机械臂增量接口。

可将这些HDF5接入其他模型的数据加载器。若要训练在线RL策略，需要自己实现reset/step/奖励及批量运行包装；当前CPU接口不是开箱即用的Gym训练环境。
RGB-D是320×240、30Hz、理想针孔渲染，RGB对齐到Dex-X深度相机光学坐标；没有RealSense噪声与真实彩色镜头标定。

## 可复现边界

完整数据不随Git提交。随机种子控制抓紧幅度与腕部速度；跨CPU/库版本不保证逐位一致，应重新运行质量检查。
物体位置、预抓姿态和场景固定。这批数据不覆盖从任意初始位置自主发现/抓取物体，也不证明实机触觉适应。
不要修改旧姿态后沿用旧数据的几何假设；`prepare_pose.py`求的是早期候选，不应在上述复现流程中覆盖现成的`cpu_pregrasp_180.json`。
Isaac安装与实验环境说明见[KNOB_ADAPTATION.md](KNOB_ADAPTATION.md)；其中历史阶段的120°/180°描述不改变上述90°命令。

## 本次迁移检查（2026-10-20）

- 15项既有任务单元测试通过（测试运行环境额外提供PyTorch；CPU生成器本身不依赖它）。
- 从默认任务资产重新生成四档各一条90°轨迹，全部通过质量门槛，最终角度89.90°–89.95°。
- 将暂存的代码/资产快照解包到新路径，在没有旧日志、T2-z和模型检查点的目录完成29关节物理步进及320×240 RGB-D渲染。
- 四条轨迹已导出3255帧同步RGB-D，导出器的动作/观察对齐、有限深度和重渲染读回检查通过。
- 已核对URDF引用的网格、相机标定与公共配置全部被Git跟踪。
- 以上是当前Linux/Python 3.10 CPU环境验证；不表示已在目标集群或Isaac上验证。
