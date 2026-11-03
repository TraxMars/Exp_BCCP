# Exp_BCCP

边界引导的跨流通道扰动（BCCP），分步在两个基线上进行：`MT_DS`（无 LossNorm）与 `MT_DS_LossNorm`。
每一步只改一个变量；两个基线的 `code/` 只在 LossNorm 相关几行不同（`diff -r MT_DS/code MT_DS_LossNorm/code`）。
`Results/` 是原基线结果，新步骤输出到 `Results_step{N}/`。

## Step 1：只观测，不干预

`code/cbc.py` 在 D3（`x6_up`，1/4）和 D2（`x7_up`，1/2）两个解码器 tap 上计算通道边界贡献
`ψ_c = mean_x ∂L/∂F_c(x) · F_c(x)`（ψ<0：该通道在帮助；组分数 = 组内求和）。

- 有标签：GT 形态学边界带（全分辨率 dilate−erode，半径 2）avg-pool 到 1/4、1/2、1 三个 DS 头，带内 / 带外软目标 CE。
- 无标签：Teacher argmax 的边界带 × Teacher 置信度，带内 / 带外 student-teacher MSE。
- 探针只通过 `autograd.grad` 作用在训练步已有的计算图上，不进入优化目标；
  `tests/verify_cbc.py` 逐位检查参数梯度、BN 缓冲与随机数状态不变。
- 每步 L1 归一化后做 EMA（CHAP 余弦动量 0.5→0.99）；每 50 步写 `channel_diagnostics.csv`，每 500 步写 `cbc_scores.jsonl`。
- 边界专属性在通道空间衡量：`residual_fraction = 1 − cos²(ψ_bd, ψ_rg)`，并看残差排名是否稳定（`lag_rho_res`）。

已验证：LA 上开启观测后前 30 步 loss 与旧基线逐位相同。Pancreas 上基线自身就不能逐位复现
（两次关闭观测的运行在第 30 步左右分叉），开启观测的运行与其中一次重合到 1e-8，偏差来自 GPU 非确定性。
观测开销约 5%（`--cbc_interval 5`），显存峰值不变。

Step 1 关闭在线验证（`--val_interval 0`）：评估在 eval + no_grad 下进行，不影响训练轨迹，这一步只需要诊断。
结论（`summary/step1.md`）：有标签分数稳定，D2 上边界与区域可区分；无标签 MSE 探针基本是噪声
（梯度方向随学生在 Teacher 上 / 下翻转），跨流一致性为随机水平。

## Step 1b：无标签探针改为伪标签 CE，加 oracle（只观测，LA）

- 无标签探针：Teacher 硬伪标签的边界带内 / 外 CE，按 Teacher 置信度加权，与有标签同形。
- oracle（`--cbc_oracle`，仅写日志）：无标签图像对隐藏 GT 的同形探针，得到 ψ^U_GT。
  训练步先清梯度再观测，观测后断言所有参数梯度仍为 None；测试中逐位检查梯度 / BN / 随机数不变。
- 新增记录：伪带与 GT 带 IoU、带内 Teacher 置信度、区域分数稳定性、各流奇偶拆半 Spearman（噪声上限）、
  每次观测的有效图像数（边界带为空或占满的图像被排除）。
- 判据见 `scripts/step1b_report.py`：C1 伪标签分数 500 步稳定性，C2 ρ(ψ^U_伪, ψ^U_GT)，C3 ρ(ψ^L, ψ^U_GT) 相对上限。

## Step 2：无标签 D2 随机通道丢弃（与 1b 并行，LA）

- 无标签 student 的 D2 tap（`x7_up`）逐样本整通道丢弃 p=0.5，按 CHAP 方式重标定（`code/perturbation.py`），
  只重跑 D2 之后的解码器尾部（`VNetDS.decode_from`），对 Teacher 的 D2 / D1 输出做 softmax-MSE。
- 尾部重跑用 batch 统计量但不更新 BN running stats（`frozen_bn_statistics`）。
- 目标：MT_DS 为 `L_sup + w·(L_mt + L_fp)`；LossNorm 为 `N(L_sup) + w·N(L_mt + L_fp)`（并入一致性组再归一化）。
  warmup 内不启用；沿用基线验证协议（800 步起每 200 步）。

```bash
PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
for b in MT_DS MT_DS_LossNorm; do (cd $b && $PY tests/verify_cbc.py && $PY tests/verify_perturbation.py); done
$PY scripts/launch.py --steps step1b step2 --datasets LA --data_root /aita_data/home/aita_duanxuelong/Dataset
$PY scripts/step1b_report.py     # summary/step1b.md，可在训练中途运行
$PY scripts/step2_report.py      # summary/step2.md
```
