# MT_DS_LossNorm

从 `../DuCiSC-main/DuCiSC-main` 独立精简。原始目录与其他实验不作修改。

仅保留 VNet 骨干、四尺度深度监督、Mean Teacher 和逐项损失归一化。
无原型对齐、图像融合、融合监督、动态阈值、置信度掩码或边界模块。
网络层、权重命名与深度监督头位置沿用原代码；删除原型用途的特征返回。

训练目标：

```text
L_sup = mean_4_scales((CrossEntropy + foreground_BinaryDiceLoss) / 2)
L_mt  = mean_4_scales(MSE(softmax(student), stopgrad(softmax(teacher))))
L     = L_sup / stopgrad(L_sup) + 0.3 * L_mt / stopgrad(L_mt)
```

分母使用 `detach().clamp_min(1e-8)` 防止零除，非零正常损失保持原始归一化形式。
先汇总四尺度，再分别归一化监督与一致性目标。归一化前的 CE、Dice、监督损失、
一致性损失以及梯度范数均记录到 CSV / TensorBoard。

默认训练设置：seed=42，15000 步，前 2000 步只优化监督损失，之后一致性权重 0.3；
SGD，恒定学习率 0.01，momentum=0.9，weight_decay=1e-4；batch=4（2 标注+2 无标注）。
Teacher 从 Student 复制初值，禁止梯度，通过参数 EMA 更新，decay=0.99，保留初期真实均值修正。
沿用原始 BN 策略：Teacher 保持 train 模式，BN 统计由其无标签前向自行更新；EMA 只更新参数。
Student / Teacher 的独立 Dropout3d(p=0.5) 提供一致性扰动，不新增输入噪声或增强策略。
Student 有标签与无标签图像分两次前向，沿用原来的 BN 批次划分。

| 数据集 | 标注 / 训练例数 | Patch | 在线验证 | 最终测试 |
| --- | --- | --- | --- | --- |
| LA | 8 / 80 | 112×112×80 | test.list，20 例 | test.list，20 例 |
| Pancreas | 6 / 62 | 96×96×96 | test.list，20 例 | test.list，20 例 |
| BraTS19 | 25 / 250 | 96×96×96 | val.list，25 例 | test.list，60 例 |

20% 标注使用 LA 16 例、Pancreas 12 例、BraTS19 50 例，其余训练设置一致。
在启动器中加 `--label_percent 20`，输出分别为 `LA_16l`、`Pancreas_12l`、`BraTS19_50l`，
启动记录为 `launch_manifest_20pct.json`。

默认数据根目录为 `../Dataset`，无需复制或修改数据。
LA / Pancreas 沿用原项目的 test.list 在线选模协议；划分与病例顺序完整记录到
`data_split_manifest.json`，明确标记在线验证与测试是否同一列表。
BraTS19 使用本地单通道、二分类预处理数据，验证和测试列表分开。

第 800 步起，每 200 步对 Student 的最终高分辨率输出作滑窗验证，按 Dice 选最佳 Student。
不启用最大连通域后处理。完成训练后用最佳 Student 自动跑完整 test.list。
Dice / JC 统计全部病例；空预测或空 GT 的表面距离记为 null，均值只统计可定义的病例，
并记录 `surface_metric_cases`，避免将不可定义的 HD95 / ASD 伪记为 0。
表面距离使用体素单位。

使用已有环境启动三个数据集（自动选取三张各有至少 12000 MiB 空闲显存的 GPU）：

```bash
cd /aita_data/home/aita_duanxuelong/MT_DS_LossNorm
../.conda/envs/SSL29/bin/python scripts/launch_three.py
../.conda/envs/SSL29/bin/python scripts/launch_three.py --label_percent 20
```

指定 LA / Pancreas / BraTS19 的 GPU，或单独启动：

```bash
../.conda/envs/SSL29/bin/python scripts/launch_three.py --gpus 4 0 5
../.conda/envs/SSL29/bin/python code/train.py --dataset LA --labelnum 8 --gpu 4 --output_dir Results/seed_42/LA_8l
```

三个子进程脱离启动终端运行；启动记录在 `launch_manifest.json`。
已有输出目录会拒绝覆盖。每个实验目录保存：

- `config.json`、`data_split_manifest.json`、源码快照与来源哈希。
- `run_status.json`、`log.txt`、`training_metrics.csv`、`tensorboard/`。
- `validation_metrics.jsonl`、`latest_validation.json`、`best_metrics.json`。
- `best_model.pth`：最佳 Student；`teacher_at_best_student.pth`：同一步 Teacher。
- `last_checkpoint.pth`：Student、Teacher、优化器、迭代数与随机状态；第 1 步及每 1000 步保存。
- `test_metrics.json`：训练完成后最佳 Student 的独立测试结果。

中断恢复可使用同一配置和 `--resume Results/seed_42/LA_8l/last_checkpoint.pth`。
恢复不重置 EMA 或 SGD momentum，但 DataLoader 从新一轮抽样开始，不保证逐批位级复现。

核心验证：

```bash
../.conda/envs/SSL29/bin/python tests/verify_core.py
```

验证同权重下四尺度预测与原始网络完全一致、四个监督头梯度、Teacher 梯度隔离、
EMA 首步更新，以及损失归一化的梯度和零值行为。
