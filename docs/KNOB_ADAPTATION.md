# Sharpa 触觉 ICL：被动旋钮任务 v0

本次实现选择 **固定轴旋钮**。打印轮体和底座，轴、垫片、锁紧螺母使用标准金属件，避免第一版同时调试打印螺纹、轴向运动和螺纹接触。另附六角轮体，便于比较不同抓握外形；它仍绕固定轴旋转，不是沿螺杆行进的螺母任务。

当前交付是环境代码、可复生成的打印资产、经验记录接口与检查脚本。没有训练模型，也没有接入真机。CPU 逻辑测试通过；Isaac 接触、初始姿态稳定性、闭合动作及完整转动尚待在 Isaac/GPU 机器验证。不要把这些状态写成“任务成功”或“ICL 已验证”。

## 1. 任务定义

| 项目 | v0 实现 |
|---|---|
| 环境 | `sharpa-knob-adapt-v0`，独立 `DirectRLEnv` |
| 机器人 | 仓库现有 FR3 + 右 Sharpa Wave；机械臂保持初始姿态 |
| 控制 | 22 个具名手关节位置目标增量，30 Hz；动作裁剪到 [-1,1]，每步最大 0.02 rad |
| 物体 | 底座固定，轮体只有一个被动转轴；没有目标角度电机 |
| 目标 | 从 0 转到 +180°，误差 ≤5°、角速度 ≤0.1 rad/s，连续保持 15 个控制步 |
| 时长 | 每次尝试最多 8 秒；物理更新 120 Hz |
| 变化条件 | 轮体接触摩擦系数、轴阻力矩、速度阻尼，彼此独立采样 |
| 支持/查询协议 | 同一组条件连续 3 次尝试，重置手与轮体状态，保留调用者的上下文；第 4 次换条件并清空上下文 |

配置入口：`src/dexx/tasks/sharpa_contact_adapt/knob_env_cfg.py`。采用现有手关节增益、armature 和具名关节映射，不依赖原任务的未来示范轨迹。原有 teacher/DAgger 网络与 checkpoint 的输入输出不匹配，不能直接用于这个环境。

### 阻力模型的明确边界

轴上只施加反向阻力：`-sign(ω) * min(τdry + b|ω|, I|ω|/dt)`。后一项限制单个孤立积分步的反向越过零点；CPU 测试检查其耗能性。默认 `τdry=0.002–0.012 N·m`、`b=0.0001–0.0006 N·m·s`，是待标定的工程起始值。

**这不是完整的静摩擦/启动阈值模型**：ω=0 时该阻力为零。真实垫片可能存在明显的启动转矩和 stick-slip，不能由当前模型声称复现。下一阶段用扭矩测量拟合静止和运动两个区域，再决定采用求解器关节摩擦还是显式粘滑模型。轮体摩擦系数范围 0.4–1.0 指材质设置，不等同于指尖接触对的最终有效摩擦系数，后者还受材质组合规则影响。

## 2. 如何表达“我这样动了，但旋钮没转，触觉发生了这些变化”

每个 1/30 秒区间记录 `before → action → after`，而不只记录动作命令：

```text
before:  实际手关节 q、dq，上一目标 q_target，五指接触力/有效标记，旋钮角度和角速度
action:  proposed_action、实际裁剪并限位后的 issued_target_rad、duration_s
after:   实际 q、dq，接触力/有效标记，角度和角速度，终止/超时标记
```

于是模型可以区分以下情况：

- 发出了关节命令，但实际 `Δq≈0`：手指受阻，不能说“手已经按命令动了”。
- `Δq≠0`，角度 `Δθ≈0`，接触力变大：手动了、物体没动，增加了接触载荷。
- `Δq≠0`，`Δθ≈0`，接触力减小/消失：可能脱离接触，需要结合历史确认。
- `Δθ≠0`：物体确实转动，记录达到的位移及随后是否稳定。

这些是观测模式，不能单凭一帧确定“阻力太大”或“发生滑移”的原因。真实滑移还需皮肤剪切、接触位置或视觉相对运动证据。

`recording.py` 将数据写为 JSONL；`experience` 与 `metadata` 分开。物理参数、条件编号和尝试编号只用于分组及评测，不能拼进模型输入。终止步的 `after` 在 Isaac 自动 reset **之前**捕获，避免把重置到零的角度误当成动作效果。

环境当前提供 91 维单步观测：q(22)、dq(22)、issued_target(22)、接触力(15)、有效标记(5)、角度/角速度/目标/目标误差/reset 标记(各1)。精确顺序见 `knob_spec.policy_layout()`。

触觉目前是按指尖过滤的 **指尖—轮体法向接触合力向量**，转到各指尖坐标系；不是高分辨率压力图，不是独立剪切力传感器。无接触的零力与 reset 后无效数据通过有效标记区分。每控制步记录一个样本，尚未输出高频冲击波形。

### 上下文如何接入（下一阶段，尚未实现网络）

把上述区间编码成 token 或数值序列，缓存 `(s_before, issued_action, Δq, Δθ, F_before, F_after, valid, Δt)`。保留数值，语言摘要仅辅助解释。先实现带历史的策略基线，再接视觉/语言骨干；当前有角度真值而无视觉/语言输入，因此这个环境自身不是 VLA。

一次条件块可以安排：第 1 次探测，第 2 次再尝试，第 3 次查询评测。所有尝试采用同一个模型权重 θ；测试阶段不做梯度更新。改变的是上下文，不是模型参数。环境只负责保存物理条件，**上下文缓存由策略调用方维护**。

完整 `env.reset()` 开始新条件块；自动 reset 或 `restart_trial()` 推进同一块中的下一次尝试，到第三次结束后自动换条件。`restart_trial(new_condition=True)` 强制换条件。调用方须比较新旧 condition_id，在块边界清空缓存；ID 只控制生命周期，不能作为策略特征。

评测至少比较无历史、正确历史、打乱动作与结果对应关系、其他条件的历史、去掉触觉五种设置。支持集和查询集不能使用不同的隐藏物理条件；独立保留未见参数区间/组合用于测试。成功率随尝试次数提升只是必要证据，还需排除 reset 状态差异、训练更新和参数泄漏。目前并未实现这些评测或训练。

## 3. 打印与装配

资产在 `assets/tasks/knob/`：

| 文件 | 作用与尺寸 |
|---|---|
| `base.stl` | 底盘直径 90 mm、厚 8 mm；轴座直径 20 mm、高 35 mm |
| `rotor_round.stl` | 圆轮直径 50 mm、高 20 mm，中心孔 8.5 mm |
| `rotor_hex.stl` | 六角轮，对角 50 mm、对边约 43.3 mm、高 20 mm，中心孔 8.5 mm |
| `fixture.scad` | 可编辑装配预览；STL/OpenSCAD 单位 mm |
| `geometry.json` | 仿真与打印共用尺寸源，单位 m |
| `knob.usda` | 单位 m 的圆轮简化仿真资产；没有打印孔和紧固件碰撞 |
| `pregrasp.json` | 离线 IK 求得的候选姿态，不是真机运动指令 |

底座底面有 13.4 mm 对边、5.6 mm 深的六角螺栓头槽。硬件：一根合适长度的 M8 六角头轴/螺栓、平垫片、摩擦垫片、可选碟簧/压缩弹簧、锁紧螺母、桌面夹具。优先让轮体运行于光滑轴段；螺纹杆直接摩擦打印孔可能产生额外卡滞。轴长按实物垫片、弹簧和螺母叠高选择，并检查螺栓头能完全收入底槽。

装配次序：轴从底座下方穿入，轴座顶面放下垫片，再放轮体、上垫片/弹性件、锁紧螺母；夹紧底盘，让底座与轴都不随轮体转动。调整预紧力得到阻力条件；给轮体做一个视觉角度标记。设计中的下垫片间隙为 1 mm，若实物不同，应更新 geometry.json 并重生成。

建议底座大平面、轮体大平面朝下打印。孔径与头槽需要本机打印公差验证，先试打小孔/槽样片，不能把 8.5 mm 设计孔径当成打印后保证尺寸。圆轮和六角轮都无需打印螺纹。当前 USD 只实现圆轮；换六角轮做仿真时还需匹配碰撞形状和惯量，不能只更换实物。

用弹簧秤沿已知半径切向拉动，分别记录刚开始转动的力和稳定转动的力，用 `τ=F×r` 标定阻力；重复测量正反两个方向和不同速度。预紧螺母的拧紧圈数只作装配标签，不当作已知扭矩。

## 4. 复生成与检查命令

所有命令从仓库根目录运行。打印资产生成只需 Python；离线 IK 需 NumPy、SciPy、trimesh，预览还需 viser、matplotlib。USD 检查可选 usd-core。使用独立工具环境，避免修改 Isaac 的依赖。

```bash
python scripts/knob/build_assets.py
# 修改尺寸后用 --spec 保留修改；不传 --spec 会生成默认尺寸：
# python scripts/knob/build_assets.py --spec assets/tasks/knob/geometry.json
python scripts/knob/prepare_pose.py
python scripts/knob/preview.py --export-only
# 或启动只读交互预览：http://127.0.0.1:8080
python scripts/knob/preview.py
PYTHONPATH=src python -m pytest -q tests/test_knob_adaptation.py
```

预览输出 `logs/knob_preview/pregrasp.png`、`scene.viser` 和 `geometry_check.json`。使用真实机器人碰撞网格；采样点穿透检测只是筛查，不包含完整自碰撞、连续碰撞或接触稳定性保证。

进入仓库配置的 Isaac Lab 2.2.1 / Sim 4.5 Python 环境后，按顺序运行：

```bash
python scripts/knob/smoke.py --mode hold --num_envs 1 --steps 180 --headless --result logs/knob_hold.json
python scripts/knob/smoke.py --mode brake --num_envs 1 --steps 180 --headless --result logs/knob_brake.json
python scripts/knob/smoke.py --mode close --num_envs 1 --steps 180 --headless --result logs/knob_close.json --record logs/knob_close.jsonl
```

结果文件禁止覆盖，重跑使用新路径。`hold` 检查零动作不会自动完成任务、角度漂移与机械臂漂移；`brake` 给轴初始速度检查孤立耗能，禁止把它录成机器人经验；`close` 是小幅三指闭合的接触检查，**不是能拧动旋钮的控制器**。脚本还检查有限数值、关节范围及同条件重置。单环境通过后再检查多环境接触过滤与重置独立性。

完成上述检查后，需要补一个真正接触驱动的成功轨迹（脚本/遥操作均可），按 30°、60°、120° 验证可达性，最终确认 +180° 可达且能保持，才能把任务用于策略学习。固定腕部若无法完成，需要先调整抓姿、物体尺寸或增加腕部动作，而非直接扩大训练预算。

## 5. 本机验证状态

- CPU 测试：15 项通过，覆盖阻力耗能、成功保持判据、条件块/RNG、打印网格、尺寸与观测布局、终止经验记录。
- 三个 STL 经 trimesh 检查均封闭，USD 可由 pxr 打开，固定轴与转轴连接关系存在。
- 离线几何预览已生成并查看，最终网格采样未发现手部穿入轮体或桌面；这不是完整碰撞证明，初始位姿仍标记为待接触仿真验证。
- 当前机器没有可用 Isaac Lab/Sim 和 GPU，未执行 Isaac 物理 smoke；未验证打印装配公差或真机性能。

下一步首先解决物理 smoke 和成功轨迹，然后采集条件块数据，最后实现固定参数的上下文策略与对照评测。

## 6. 无 GPU 的数据与动画预览（PyBullet，非 Isaac 验证）

`scripts/knob/cpu_rollouts.py` 提供独立的 CPU 物理诊断。需要 PyBullet 3.2.7、NumPy、SciPy、trimesh、Pillow；本机安装于 `/tmp/dexx-knob-tools`，未修改 Isaac 依赖。

```bash
/tmp/dexx-knob-tools/bin/python scripts/knob/cpu_rollouts.py --out logs/knob_cpu_preview_new --render
# 从已记录的状态重渲染，不重新仿真，不生成接触力：
/tmp/dexx-knob-tools/bin/python scripts/knob/cpu_rollouts.py --out logs/knob_cpu_preview_new --replay
```

每条 6 秒、180 个 30 Hz 动作区间，物理步长 1/960 秒。三条分别为保持、轻探测、更强闭合与拨动；物理参数相同，动作幅度是脚本预设，**没有模型根据上下文学会调整动作**。输出 `01/02/03.jsonl`、各条 GIF、`comparison.gif`、`comparison.png`、`summary.json` 及本次几何/抓姿快照。GIF 以 10 fps 展示记录中的真实关节状态，图表来自同一份记录；角度表盘是数据读数。

该程序发现原预抓取姿态在 Bullet 凸包碰撞下会产生非指尖接触，因此只为 CPU 预览另解了一个碰撞约束抓姿，保存为输出目录的 `cpu_pregrasp.json`，没有覆盖 Isaac 抓姿。Bullet 使用自己的位置伺服，手关节力矩上限为 URDF 上限与 0.25 N·m 的较小者，不含 Isaac armature；机器人自碰撞关闭。两套引擎的动力学与碰撞近似不同，**CPU 轨迹不能标注成 Isaac rollout，也不能直接充当已验证的训练数据**。

CPU JSONL 的关节状态含全部 29 个关节，顺序保存在 header；指尖法向合力以世界坐标保存。它与第 2 节的 Isaac 91 维观测格式不同，读取时必须按 schema 处理。另存全部手部—轮体接触，避免把指节接触误写成指尖触觉。

首次最终预览 `logs/knob_cpu_preview` 的结果：保持时转角为 0°；轻探测最终约 +0.33°；较强拨动最终约 −16.80°。三条都没有完成 +60° 目标。最大单指法向合力分别约 0、4.36、21.50 N；这些数值未做真机标定。最大片段关节越限约 0.00251 rad，保留在 summary 中用于判断数值误差。本次结果展示的是动作与反馈的对应关系，不是任务成功率或 ICL 性能。

### 四档阻力对照

当时目标改为 +120°（现已更新为 +180°），历史数据保留原目标。新增对照位于 `logs/knob_resistance_preview`：

```bash
/tmp/dexx-knob-tools/bin/python scripts/knob/cpu_rollouts.py --out logs/knob_resistance_new --suite resistance --brake-model solver_friction --goal-angle-deg 120 --render
```

四档轴制动转矩上限为 0、0.005、0.03、0.10 N·m。所有轨迹共用初始抓姿、位置命令序列、接触材质和速度阻尼 0.0003 N·m·s；0 档只是没有额外恒定制动，仍有速度阻尼。每档 6 秒、180 个区间，不是四档各一次成功示范。

这里 **CPU 专用** `solver_friction` 使用求解器的零目标速度、有限转矩约束，抵抗静止和运动状态下的轴转动，外加速度阻尼；没有角度伺服，也没有非零目标速度。它与早期显式、每步截断的阻力模型不同，不能把两批数据当成只换了一个参数的同模型对照。Isaac 环境的阻力实现尚未同步迁移。新模型通过孤立转轴衰减检查，四档零动作各检查 1 秒均无转动和接触。

四档最终转角依次为 −3.91°、−2.27°、+30.76°、−1.55°，均未满足 +120° 目标。接触操作中的最终转角不随阻力单调变化，不可用它直接当阻力标签；必须结合整个动作、转动和接触历史分析。中档重新运行得到相同状态轨迹。原始数据保留全部结果，未筛选成单调趋势。

`comparison.gif` 按左上、右上、左下、右下展示四档，曲线统一坐标范围；`01.gif` 至 `04.gif` 是单条，`curves.png` 为四条反馈叠加图。JSONL 保存实际 `before/issued_target/after`，物理条件只在 header 元数据中。`*_validation.json` 分别记录命令一致性、零动作、制动耗能及重复运行的检查结果。最大关节越限约 0.00326 rad，数值和未标定的接触力都保留供诊断，数据不默认用于训练。

### 四档均完成 +120° 的目标对照

用户要求四档都转到同一目标，因此另加 `goal_rollouts.py`。控制器先闭合手指，再通过机械臂让腕部绕轴移动；根据实测角度误差调整腕部转动速度（同一套增益、限速，不读取阻力标签）。轴仍只有被动制动，没有目标角度驱动。此版本 **允许机械臂运动**，不同于原来的固定腕部 22 维动作任务，也不是学习策略或 ICL 结果。

```bash
/tmp/dexx-knob-tools/bin/python scripts/knob/goal_rollouts.py --goal-degrees 120 --out logs/knob_goal120_completed_new
/tmp/dexx-knob-tools/bin/python scripts/knob/render_goal_rollouts.py --goal-degrees 120 --out logs/knob_goal120_completed_new
# 同步 2×2 GIF：重放已有记录，不重新仿真
/tmp/dexx-knob-tools/bin/python scripts/knob/render_goal_gif.py --out logs/knob_goal120_completed_new
```

四档初态及物理设置与阻力对照一致，每次独立运行；统一在首次满足角度误差 ≤0.25°、角速度 ≤0.05 rad/s、连续 1 秒时结束 episode，超时保留失败。命令会随反馈不同而变化，不能再称为“完全相同的动作”。`all_120_degrees.png/pdf` 在同一张图中展示各 episode 的真实终止姿态、实际转角、耗时和完整角度/触觉曲线。

这只验证规定的 1 秒保持窗口，不代表永久锁定。先前固定运行 20 秒的诊断保存于 `logs/knob_goal120_preview`，其中 0.005 N·m 档在达到目标后继续运行会再次漂移；没有删除这份证据。正式完成数据 `logs/knob_goal120_completed` 使用上述统一终止规则重新运行，不能把终止后的静态渲染当作额外物理保持时间。

`all_120_degrees.gif` 以真实时间同步四条轨迹，10 fps；每个小图包含阻力、角度和触觉曲线。较早结束的 episode 停留在最后一个记录状态，并显示 `DONE / replay frozen`。全部结束后额外停帧 2 秒便于查看；这段是展示停帧，不是新模拟的保持时间。具体时间戳见 `gif_replay_metadata.json`。

### 专家示范质量：到达目标不等于轨迹优质

原 `knob_goal120_completed` 四条虽通过末尾成功判据，最大过冲仍为 3.97°、3.87°、6.25°、7.25°；前两档初始抓紧还反转约 16°。`quality_audit.json` 已将它们标为 `diagnostic_not_expert`，不应直接用于专家动作监督。原始记录保留，可作为明确标注的次优尝试或恢复经验；ICL 支持历史可包含这类反馈，查询端的示范动作应另行筛选。

新增 `smooth` 控制器加入角速度反馈、低通滤波、接近目标时减速、腕部速度/加速度限制，并减小初始闭合量。它是工程改进，不是已证明的最优策略；减小过冲、接触力与耗时之间有权衡。不同阻力仍共用控制器参数，不输入阻力标签。

```bash
/tmp/dexx-knob-tools/bin/python scripts/knob/goal_rollouts.py --out logs/knob_smooth_new --controller smooth --seconds 45 --hold-seconds 3
/tmp/dexx-knob-tools/bin/python scripts/knob/audit_goal_data.py logs/knob_goal120_completed logs/knob_smooth_new
```

示范筛查同时检查：过冲 ≤1°、初始反转 ≤1°、累计角行程 ≤125°、末尾误差 ≤0.25°且角速度 ≤0.05 rad/s 连续 3 秒；输出峰值接触力和按统一 1 秒标准计算的首次完成时间供比较。通过者只标为 CPU 候选，仍需碰撞、接触力标定、Isaac 和更广条件分布验证，不能自动叫作 optimal 或直接混进训练集。审计不裁剪、平滑或改写原始角度与力曲线。


### 180°、四档各 50 条的 CPU 数据采集

当前任务目标为 **相对起点 +180°**。批量脚本为 `scripts/knob/batch180.py`；输出目录为 `logs/knob180_pinch_4x50`。实际完成数量与复核结果以该目录的 `manifest.json`、`verification.json` 为准，不把待生成数量当成完成数量。

为避开旧抓姿在约 130° 后的关节行程问题，CPU 批次把夹具从 `[0.48, -0.15, 0.415]` 移到 `[0.33, -0.15, 0.415]`，起抓方向绕竖轴旋转 90°，并重新求解 FR3 冗余关节姿态、避免食指/中指穿插的张开抓姿。闭合时用食指触觉垫和拇指末节壳体形成两侧接触；拇指壳体力只记在接触列表中，不作为拇指触觉垫力。旋钮尺寸、Sharpa Wave 手和四档轴阻力保持原设置。初始几何的 Viser 场景和检查在 `logs/knob180_self_pose/viser`，采样未发现手部穿入桌面或旋钮；这不是完整自碰撞验证。

每次先物理模拟 5 秒抓取准备，再以抓稳后的实际轴角为相对零点。准备过程保存在 `preparation/`，正式任务第一条 `before` 必须等于准备过程最后一条 `after`；不重置轴角、不回写状态、不剪掉任务中的反转和过冲。+180° 是从这个零点算起，原始 `angle_rad` 仍为模拟器绝对轴角。

四档轴阻力为 0、0.005、0.03、0.10 N·m。每条采用确定性 seed，对抓紧尺度施加 ±2% 的变化，腕部最大转速采样自 0.135–0.15 rad/s。各档尝试使用相同 seed 序列，质量筛除后不保证全部一一配对。这是同一夹具、同一初始抓姿附近的小范围示范集，不代表覆盖未见物理条件的泛化数据。

此处是读取轴角的脚本教师；最终夹持方案四档使用相同闭合目标和控制规则，不读取阻力标签。阻力仍保存在元数据中，不能作为待训练模型的观察输入。轴只施加被动制动与黏性阻尼，没有目标角位置驱动。记录 29 个关节的状态、下发位置目标、实际轴角和角速度、5 个指尖的接触法向力以及接触点。数据仍为 PyBullet 结果，未验证 Isaac 接触模型及真实触觉标定。新批次启用自碰撞，只排除机械相邻链节及仓库既有的 7 对机械连接过滤对；食指、中指之间的碰撞不排除。每个控制步记录自碰撞及活动链节与桌面的最大穿透深度，并要求均不超过 2 mm。物理步长统一为 1/960 秒，约束求解迭代数为 150。

质量要求：末尾误差 ≤0.25°、速度 ≤0.05 rad/s 连续 3 秒；全程最大过冲 ≤1°、初始反转 ≤1°、总转角路程 ≤185°；腕部位置 IK 残差 ≤2 mm、实际关节限位超出量 ≤0.01 rad；指尖法向力 ≤50 N 仅作未标定的异常筛查。全部 raw attempts 与被拒原因保留，只将通过者计入每档 50 条。通过这些阈值不等于数学意义的 optimal。

`summarize_batch180.py` 对 200 条逐一重审、核对文件哈希和准备/任务连续性，并检查实测状态轨迹无重复；输出所有轨迹的叠加图。四格 GIF 固定选择每档第一个合格 seed，显示实际相对角度与触觉曲线。白刻度代表起点，绿刻度代表 180° 目标，结束后明确标注停帧。


**无效旧批次**：`logs/knob180_4x50` 与 `logs/knob180_high50` 初期仅通过了转角筛查，随后几何复核发现关闭自碰撞时食指和中指穿插约 10 mm。两批已中止并标为 `diagnostic_not_expert`，合格清单清空；不能因旧转角达标而拿它们作为训练专家示范。原始文件保留，见各目录 `INVALIDATED.json`。

碰撞修正时的 480 Hz / 80 次迭代试验也只作为诊断保存：高阻力能成功，但低阻力末段抖动，未通过 3 秒连续保持和累计角路程要求。最终采集不混用这些不同求解器设置的数据。当前 CPU 接触对离散化和抓姿敏感，不能把这里的成功率解释为真实手上的成功率。


完整采集命令（从仓库根目录运行；`--out` 使用新目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/dexx-knob-tools/bin/python scripts/knob/batch180.py --out logs/knob180_new --per-condition 50 --workers 24 --max-attempts 200 --seconds 65
/tmp/dexx-knob-tools/bin/python scripts/knob/summarize_batch180.py logs/knob180_new
/tmp/dexx-knob-tools/bin/python scripts/knob/render_goal_gif.py --out logs/knob180_new/preview
```

源目录默认使用 `assets/tasks/knob/cpu_pregrasp_180.json` 及既有几何。`accepted/` 只含合格样本，`attempts/` 保留所有候选及失败原因。达到不可逆的质量失败条件（如累计角路程超标）后提前终止该候选，避免继续消耗计算。已启动目录可用 `--resume` 续采。`STOP` 文件用于停止采集，不会删除已生成的数据。
