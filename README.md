# Sharpa Knob Simulation

FR3机械臂 + 右Sharpa Wave灵巧手的被动旋钮仿真、90°示范生成、同步RGB-D导出与可打印道具。
本仓库独立于T2-z，可给其他模型提供数据和仿真组件；不包含模型权重或已有数据集。

- **已验证的CPU管线**：PyBullet，29维绝对关节位置目标，四档轴制动，脚本教师。
- **实验性Isaac Lab环境**：固定机械臂、22维手关节增量控制；尚未验证完整转动成功，和CPU生成器不是同一个控制任务。
- **抓取初始化**：预设物体位置、离线几何/IK预抓姿态、脚本闭合；不是抓取网络或强化学习策略。

## 在新集群运行

```bash
git clone https://github.com/CathyF9600/sharpa-knob-sim.git
cd sharpa-knob-sim
python3.10 -m venv .venv-knob
source .venv-knob/bin/activate
python -m pip install -r requirements-knob-cpu.txt
```

私有仓库需要先配置GitHub访问权限。CPU生成无需GPU、Isaac或PyTorch。
完整四档各一条/各50条采集、质量检查和RGB-D导出命令见 [迁移与运行指南](docs/KNOB_PORTABILITY.md)。
道具尺寸、打印装配及历史任务设计见 [任务说明](docs/KNOB_ADAPTATION.md)。

## 范围与来源

从 [Dex-X / dexx_lab](https://github.com/gcfy63821/dexx_lab) 提取机器人配置与资产，加上本项目的旋钮任务和生成工具；不是上游官方发行。
保留内部`dexx`命名空间是为了兼容几何/配置导入，不依赖安装上游仓库。
精确来源与改动见 [PROVENANCE.md](PROVENANCE.md)。保留 [MIT许可](LICENSE)、[第三方声明](THIRD_PARTY_NOTICES.md) 和机器人资产内的Apache-2.0许可证及NOTICE。

相机是320×240、30Hz的合成对齐RGB-D，使用Dex-X深度相机几何；不含真实RealSense噪声。
这不是已验证的实机部署包，也没有把旋钮转动包装成通用Gym训练接口。
