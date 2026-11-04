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
结论（`summary/step1.md`）：有标签分数稳定；无标签 MSE 探针基本是噪声
（梯度方向随学生在 Teacher 上 / 下翻转），跨流一致性为随机水平。
有标签 D2 上"边界与区域可区分"经 1b 的 oracle 检验是有标签集的假象（见下）。

## Step 1b：无标签探针改为伪标签 CE，加 oracle（只观测，LA）

- 无标签探针：Teacher 硬伪标签的边界带内 / 外 CE，按 Teacher 置信度加权，与有标签同形。
- oracle（`--cbc_oracle`，仅写日志）：无标签图像对隐藏 GT 的同形探针，得到 ψ^U_GT。
  训练步先清梯度再观测，观测后断言所有参数梯度仍为 None；测试中逐位检查梯度 / BN / 随机数不变。
- 新增记录：伪带与 GT 带 IoU、带内 Teacher 置信度、区域分数稳定性、各流奇偶拆半 Spearman（噪声上限）、
  每次观测的有效图像数（边界带为空或占满的图像被排除）。
- 判据见 `scripts/step1b_report.py`：C1 伪标签分数 500 步稳定性，C2 ρ(ψ^U_伪, ψ^U_GT)，C3 ρ(ψ^L, ψ^U_GT) 相对上限。
  `scripts/step1b_crosscheck.py` 在 EMA 快照上做偏相关与残差迁移检验（`summary/step1b_crosscheck.md`）。

结论（LA，两个基线一致，10k–15k）：
- 伪标签 CE 修好了稳定性（C1 0.96–0.99），但 C2（0.48–0.83）始终低于 C3（0.85–0.89）；
  控制 ψ^L 后 ψ^U_伪 与 ψ^U_GT 的偏相关 ≈0（−0.14~0.12）。伪带与 GT 带 IoU 仅 0.57–0.63，带内置信度却 0.95–0.97。
- 无标签真值上边界≈区域：oracle 残差占比 D3 0.06–0.10、D2 0.02–0.06，ρ(O.bd, O.rg) 0.84–0.95。
  有标签 D2 的"专属性"来自有标签区域损失被记住（≈0.005，oracle 为 0.05–0.07）；有标签残差不迁移（D2 0.00–0.25）。
- 测量口径问题：训练 DS 用最近邻下采样标签，探针在 1/4、1/2 头用 avg-pool 软目标，
  有标签带内 CE 随训练上升（0.66→0.86，贴近 ln2），是三流共有的成分，可能抬高跨流相关。
  已修正：`--cbc_coarse_target nearest`（默认）在粗尺度头用与训练相同的最近邻目标，`soft` 为 LA 运行时的旧口径；
  CSV 新增逐头损失 `{lab,unl,orc}_loss_{bd,rg}_h{1,2,3}`。Pancreas 的 1b 用 nearest 口径。
- 上面两条"边界≈区域"和"L→U 迁移 0.85–0.89"经 1c 证明主要是探针假象（粗尺度头的增益方向 + 软目标），见 1c 结论。

Pancreas 1b（nearest 口径，10k–15k）：哪一流更能预测无标签真值在两个基线上相反——
MT_DS ρ(lab,orc)/ρ(unl,orc) = D3 0.15/0.83、D2 0.57/0.39；LossNorm = D3 0.80/0.26、D2 0.83/0.35。
有标签带外 CE ≈0.000–0.002（被记住），oracle 全分辨率带内 CE 随训练上升（MT_DS 1.38→1.96）。

## Step 1c：去掉增益方向的通道边界专属性（只观测，LA，两个基线，保留 oracle）

动机：1b 中各流 ψ>0 的通道占比 0.69–0.94，oracle 带内 CE 随训练上升——ψ 的排序可能主要是
"通道对 margin 幅度的贡献"（增益），它同时造成 oracle 上边界≈区域和跨流相关偏高。
`code/cbc_gain.py`（`--cbc_tracker 1c`；1b 的 `cbc.py` 路径不变）并列记录：

- hard：三个 DS 头，最近邻目标（与训练相同），在各头分辨率上由该目标重建边界带（半径随步长缩小，至少 1）；
  full：只用全分辨率头（决策探针，无任何重采样）。
- 增益方向：带内 / 带外平均 |margin| 探针的 ψ_gbd、ψ_grg；把 ψ_bd、ψ_rg 都投影到 span{ψ_gbd, ψ_grg} 的正交补
  （保守做法，避免只有一方去掉增益而造成假专属性），再取边界残差 = 去增益 bd 去掉沿去增益 rg 的分量。
- k：D2 头的闭式对比 K_c =（带内前景侧 − 背景侧的 a_c·F_c 均值）/ margin 标准差，a_c = W_1c − W_0c；区域版本用远场。
  只反映辅助头 `out_conv_2` 的读取，作交叉检验。
- 逐头记录：软目标带内 CE 与其熵下限、硬目标带内 / 带外 CE；ψ>0 占比；残差的奇偶拆半与 500 步稳定性。
- 测试 `tests/verify_cbc_gain.py`：重建带几何、投影代数、闭式对比、梯度 / BN / 随机数逐位不变、CSV 与恢复。
  冒烟：MT_DS LA 前 40 步训练轨迹与旧基线一致，峰值显存 5.7 GB。

判据（`scripts/step1c_report.py` → `summary/step1c.md`，预先固定，只看 orc 流第 10k–15k 步，full 为决策探针）：
(i) 去增益后 1−cos² ≥ 0.3、剩余范数占比 ≥ 0.3、orc 残差拆半 ρ ≥ 0.7；
(ii) lab 残差拆半 ρ ≥ 0.7，且 ρ(lab 残差, orc 残差) ≥ 0.5、≥ 0.8 × √(lab 拆半 × orc 拆半)。
两条须在同一 tap 上同时满足（并在两个数据集上）才继续通道级边界路线，任一不满足即停止。
另看去增益后 ρ(lab, orc)：决定"有标签引导 L→U"这一后备路线是否在去掉共享增益后仍成立。

```bash
$PY scripts/launch.py --steps step1c --datasets LA --data_root /aita_data/home/aita_duanxuelong/Dataset
$PY scripts/step1c_report.py
```

结论（LA，两个基线一致，10k–15k）——按预设判据，**停止通道级边界路线**：
- (i) 通过：full 探针上，去增益后 oracle 的边界与区域可分（1−cos² 0.72–0.97，剩余范数 0.45–0.69，残差拆半 0.78–0.87）。
- (ii) 全部不通过：有标签残差自身不可靠（拆半 −0.31~0.56），与 oracle 残差相关 −0.51~0.21；
  伪标签也估计不了（残差拆半 0.18–0.40，与 oracle 相关 0.23–0.48）。有标签全分辨率带内 CE 仅 0.16、带外 ≈0（被记住）。
- 1b 的两条结论是假象：hard/D2 的增益占比 0.97，原始 1−cos² 仅 0.04–0.07（边界≈区域来自粗尺度头的增益）；
  只用全分辨率头时原始 1−cos² 0.45–0.84，ψ>0 占比 0.41–0.52（无增益特征）。
  去增益后 ρ(lab, orc) 从 0.20–0.83 降到 −0.57~0.36：1b 中的 L→U 迁移由共享增益方向携带，"有标签引导 L→U"后备路线也不成立。
- 软目标假象得到确认：有标签 1/4 头软目标带内 CE 1.71–1.85，熵下限 0.33，硬目标仅 0.07。
- k（D2 闭式对比）退化：各流、bd/rg 之间相关均 ≈0.99，1−cos² 0.03，只反映头权重 × 激活幅度。

## Step 2：无标签 D2 随机通道丢弃（与 1b 并行，LA）

- 无标签 student 的 D2 tap（`x7_up`）逐样本整通道丢弃 p=0.5，按 CHAP 方式重标定（`code/perturbation.py`），
  只重跑 D2 之后的解码器尾部（`VNetDS.decode_from`），对 Teacher 的 D2 / D1 输出做 softmax-MSE。
- 尾部重跑用 batch 统计量但不更新 BN running stats（`frozen_bn_statistics`）。
- 目标：MT_DS 为 `L_sup + w·(L_mt + L_fp)`；LossNorm 为 `N(L_sup) + w·N(L_mt + L_fp)`（并入一致性组再归一化）。
  warmup 内不启用；沿用基线验证协议（800 步起每 200 步）。

结论（`summary/step2.md`，单种子）：
- MT_DS：warmup 段与基线逐位相同；之后整条验证曲线都更高（各阶段均值 +2.0 / +1.1 / +0.85，84–95% 的验证点更高），
  HD95 / ASD、最差病例 Dice 也更好。但 L_fp（≈2× L_mt）同时加大了一致性梯度，需 `--fp_drop 0` 对照区分"扰动"与"更多一致性"。
- LossNorm：无增益；后期 3 次单病例失败（最差病例 Dice 24–33）拉低均值，基线后期很稳（std 0.27）。
  warmup 段两次运行在未干预时就相差 2.7pp，LossNorm 的单种子噪声明显更大。

对照 `step2_p0`（`Results_step2/nodrop_d2_p0`，`--fp_drop 0`）：同样的第二次 D2 尾部重跑与一致性项，只是不丢通道，
用于区分"扰动"与"一致性项变多 / 被稀释"。Pancreas 上 1b、step2、step2_p0 × 两个基线共 6 组：

```bash
$PY scripts/launch.py --steps step1b step2 step2_p0 --datasets Pancreas --data_root /aita_data/home/aita_duanxuelong/Dataset
```

Pancreas 结论（单种子；后期 10k–15k 用中位数，MT_DS 基线在 10800 步有一次 44.9 的塌陷会抬高均值差）：
- MT_DS：p=0 比基线 +1.45，p=0.5 仅 +0.20；p=0.5 − p=0 = −0.92（6k–10k −0.95），后期 84% 的验证点 p=0 更好。
  增益来自多出的一致性分支，丢通道本身无益甚至有害。
- LossNorm：p=0.5 / p=0 比基线 +1.97 / +1.76，p=0.5 − p=0 = +0.58，处于噪声内
  （warmup 段未干预时与基线的差就达 +0.7~+3.0）。
- 结论：两个数据集上都没有证据表明随机通道扰动本身优于不丢弃对照。

```bash
PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
for b in MT_DS MT_DS_LossNorm; do (cd $b && $PY tests/verify_cbc.py && $PY tests/verify_perturbation.py); done
$PY scripts/launch.py --steps step1b step2 --datasets LA --data_root /aita_data/home/aita_duanxuelong/Dataset
$PY scripts/step1b_report.py     # summary/step1b.md，可在训练中途运行
$PY scripts/step2_report.py      # summary/step2.md
```
