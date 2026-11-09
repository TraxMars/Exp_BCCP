# Step 1：只观测的 CBC 诊断

数值为各阶段中位数；lab=有标签（GT 边界带），unl=无标签（Teacher 伪边界带）。
lag_* 为 EMA 分数与 500 步前的一致性；raw_* 为相邻两次原始观测的一致性；
residual_fraction = 1 − cos²(ψ_bd, ψ_rg)（通道空间），lag_rho_res 为去区域分量后残差排名的稳定性。

## MT_DS / LA_8l

- 状态：completed，iteration 15000
- 与旧基线训练轨迹：identical through 15000
- d3 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 fail; 10k–15k: G1 fail, G2 fail

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.78 | 0.52 | 0.51 | 0.36 | 0.60 | 0.56 | 0.64 | 0.79 | 0.21 | 0.00 |
| warmup ≤2k | unl | 0.18 | 0.23 | -0.01 | 0.18 | 0.66 | 0.58 | 0.56 | 0.15 | 0.48 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=-0.10, cross_jaccard_bd=0.14, cross_rho_res=0.03 | | | | | | | | | |
| 2k–6k | lab | 0.93 | 0.60 | 0.68 | 0.43 | 0.84 | 0.74 | 0.30 | 0.77 | 0.17 | 0.00 |
| 2k–6k | unl | 0.08 | 0.16 | -0.19 | 0.13 | 0.34 | 0.27 | 0.88 | 0.07 | 0.48 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=-0.03, cross_jaccard_bd=0.14, cross_rho_res=-0.04 | | | | | | | | | |
| 6k–10k | lab | 0.96 | 0.68 | 0.77 | 0.48 | 0.92 | 0.82 | 0.15 | 0.89 | 0.14 | 0.00 |
| 6k–10k | unl | 0.03 | 0.16 | -0.19 | 0.12 | 0.40 | 0.30 | 0.84 | 0.13 | 0.50 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=-0.06, cross_jaccard_bd=0.10, cross_rho_res=-0.09 | | | | | | | | | |
| 10k–15k | lab | 0.99 | 0.88 | 0.82 | 0.51 | 0.94 | 0.88 | 0.11 | 0.98 | 0.11 | 0.00 |
| 10k–15k | unl | 0.33 | 0.23 | -0.20 | 0.12 | 0.23 | 0.16 | 0.94 | 0.37 | 0.50 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=-0.10, cross_jaccard_bd=0.07, cross_rho_res=-0.35 | | | | | | | | | |

- d2 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.59 | 0.45 | 0.19 | 0.22 | -0.28 | 0.18 | 0.91 | 0.48 | 0.20 | 0.00 |
| warmup ≤2k | unl | -0.04 | 0.14 | 0.00 | 0.18 | 0.46 | 0.39 | 0.78 | -0.00 | 0.36 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=0.16, cross_jaccard_bd=0.14, cross_rho_res=0.11 | | | | | | | | | |
| 2k–6k | lab | 0.80 | 0.60 | 0.30 | 0.23 | -0.33 | 0.22 | 0.89 | 0.78 | 0.19 | 0.00 |
| 2k–6k | unl | 0.08 | 0.19 | -0.16 | 0.15 | 0.38 | 0.38 | 0.83 | 0.06 | 0.44 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=0.26, cross_jaccard_bd=0.14, cross_rho_res=0.19 | | | | | | | | | |
| 6k–10k | lab | 0.94 | 0.78 | 0.46 | 0.29 | -0.47 | 0.18 | 0.78 | 0.90 | 0.16 | 0.00 |
| 6k–10k | unl | 0.09 | 0.14 | -0.21 | 0.12 | 0.26 | 0.29 | 0.91 | 0.14 | 0.41 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=0.10, cross_jaccard_bd=0.14, cross_rho_res=0.03 | | | | | | | | | |
| 10k–15k | lab | 0.99 | 1.00 | 0.60 | 0.36 | -0.50 | 0.20 | 0.75 | 0.98 | 0.16 | 0.00 |
| 10k–15k | unl | 0.37 | 0.23 | -0.22 | 0.12 | 0.08 | 0.14 | 0.98 | 0.37 | 0.28 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=0.11, cross_jaccard_bd=0.14, cross_rho_res=0.10 | | | | | | | | | |

## MT_DS / Pancreas_6l

- 状态：completed，iteration 15000
- 与旧基线训练轨迹：identical through 15000
- d3 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.56 | 0.39 | 0.34 | 0.30 | 0.45 | 0.44 | 0.65 | 0.59 | 0.13 | 0.00 |
| warmup ≤2k | unl | 0.30 | 0.23 | 0.20 | 0.27 | 0.56 | 0.51 | 0.65 | 0.08 | 0.40 | 0.10 |
| warmup ≤2k | L↔U | cross_rho_bd=0.02, cross_jaccard_bd=0.14, cross_rho_res=0.11 | | | | | | | | | |
| 2k–6k | lab | 0.83 | 0.52 | 0.57 | 0.37 | 0.82 | 0.75 | 0.33 | 0.69 | 0.19 | 0.00 |
| 2k–6k | unl | 0.03 | 0.19 | 0.10 | 0.23 | 0.58 | 0.43 | 0.64 | 0.06 | 0.45 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=-0.04, cross_jaccard_bd=0.14, cross_rho_res=-0.04 | | | | | | | | | |
| 6k–10k | lab | 0.91 | 0.60 | 0.60 | 0.38 | 0.80 | 0.79 | 0.37 | 0.83 | 0.17 | 0.00 |
| 6k–10k | unl | 0.05 | 0.23 | 0.10 | 0.22 | 0.59 | 0.45 | 0.65 | 0.08 | 0.41 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=-0.09, cross_jaccard_bd=0.14, cross_rho_res=-0.01 | | | | | | | | | |
| 10k–15k | lab | 0.98 | 0.78 | 0.66 | 0.40 | 0.86 | 0.82 | 0.27 | 0.97 | 0.19 | 0.00 |
| 10k–15k | unl | 0.21 | 0.28 | 0.08 | 0.22 | 0.35 | 0.27 | 0.87 | 0.18 | 0.39 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=-0.16, cross_jaccard_bd=0.14, cross_rho_res=-0.01 | | | | | | | | | |

- d2 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.50 | 0.33 | 0.29 | 0.27 | 0.06 | -0.08 | 0.72 | 0.33 | 0.12 | 0.00 |
| warmup ≤2k | unl | 0.17 | 0.23 | 0.23 | 0.29 | 0.55 | 0.26 | 0.50 | 0.10 | 0.23 | 0.10 |
| warmup ≤2k | L↔U | cross_rho_bd=-0.02, cross_jaccard_bd=0.14, cross_rho_res=0.04 | | | | | | | | | |
| 2k–6k | lab | 0.72 | 0.45 | 0.34 | 0.27 | 0.03 | 0.09 | 0.84 | 0.53 | 0.11 | 0.00 |
| 2k–6k | unl | 0.01 | 0.14 | 0.06 | 0.25 | 0.50 | 0.20 | 0.61 | 0.02 | 0.39 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=0.09, cross_jaccard_bd=0.14, cross_rho_res=0.01 | | | | | | | | | |
| 6k–10k | lab | 0.84 | 0.45 | 0.34 | 0.28 | -0.16 | 0.01 | 0.83 | 0.62 | 0.09 | 0.00 |
| 6k–10k | unl | 0.07 | 0.23 | 0.08 | 0.25 | 0.43 | 0.24 | 0.61 | 0.17 | 0.31 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=-0.06, cross_jaccard_bd=0.14, cross_rho_res=-0.07 | | | | | | | | | |
| 10k–15k | lab | 0.96 | 0.78 | 0.46 | 0.31 | -0.14 | 0.12 | 0.97 | 0.85 | 0.12 | 0.00 |
| 10k–15k | unl | 0.21 | 0.23 | 0.05 | 0.24 | 0.26 | -0.03 | 0.76 | 0.27 | 0.31 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=-0.06, cross_jaccard_bd=0.14, cross_rho_res=-0.06 | | | | | | | | | |

## MT_DS_LossNorm / LA_8l

- 状态：completed，iteration 15000
- 与旧基线训练轨迹：first gap >1e-6 at 10; max |Δsup|=0.5757
- d3 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 fail; 10k–15k: G1 fail, G2 fail

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.80 | 0.60 | 0.58 | 0.47 | 0.76 | 0.73 | 0.42 | 0.56 | 0.20 | 0.00 |
| warmup ≤2k | unl | 0.14 | 0.19 | -0.01 | 0.18 | 0.55 | 0.54 | 0.70 | 0.30 | 0.47 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=-0.01, cross_jaccard_bd=0.10, cross_rho_res=0.09 | | | | | | | | | |
| 2k–6k | lab | 0.95 | 0.78 | 0.82 | 0.58 | 0.88 | 0.80 | 0.23 | 0.74 | 0.22 | 0.00 |
| 2k–6k | unl | 0.08 | 0.19 | -0.19 | 0.12 | 0.39 | 0.32 | 0.85 | 0.18 | 0.48 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=-0.03, cross_jaccard_bd=0.10, cross_rho_res=0.10 | | | | | | | | | |
| 6k–10k | lab | 0.97 | 0.88 | 0.86 | 0.64 | 0.94 | 0.85 | 0.12 | 0.88 | 0.23 | 0.00 |
| 6k–10k | unl | 0.15 | 0.19 | -0.18 | 0.12 | 0.32 | 0.30 | 0.90 | 0.19 | 0.48 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=-0.06, cross_jaccard_bd=0.10, cross_rho_res=0.06 | | | | | | | | | |
| 10k–15k | lab | 0.99 | 0.88 | 0.88 | 0.65 | 0.96 | 0.88 | 0.08 | 0.96 | 0.23 | 0.00 |
| 10k–15k | unl | 0.35 | 0.28 | -0.21 | 0.11 | -0.17 | -0.11 | 0.89 | 0.28 | 0.45 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=-0.11, cross_jaccard_bd=0.10, cross_rho_res=-0.03 | | | | | | | | | |

- d2 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.63 | 0.33 | 0.25 | 0.18 | 0.14 | 0.30 | 0.90 | 0.55 | 0.16 | 0.00 |
| warmup ≤2k | unl | 0.09 | 0.23 | 0.01 | 0.20 | 0.59 | 0.45 | 0.65 | 0.15 | 0.34 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=0.21, cross_jaccard_bd=0.14, cross_rho_res=0.29 | | | | | | | | | |
| 2k–6k | lab | 0.71 | 0.33 | 0.34 | 0.22 | -0.37 | 0.11 | 0.86 | 0.57 | 0.06 | 0.00 |
| 2k–6k | unl | 0.11 | 0.14 | -0.16 | 0.14 | 0.10 | 0.12 | 0.82 | 0.08 | 0.44 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=0.08, cross_jaccard_bd=0.14, cross_rho_res=0.00 | | | | | | | | | |
| 6k–10k | lab | 0.90 | 0.60 | 0.55 | 0.30 | -0.45 | 0.10 | 0.80 | 0.82 | 0.06 | 0.00 |
| 6k–10k | unl | 0.24 | 0.23 | -0.20 | 0.15 | -0.08 | 0.03 | 0.85 | 0.12 | 0.44 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=0.07, cross_jaccard_bd=0.14, cross_rho_res=0.02 | | | | | | | | | |
| 10k–15k | lab | 0.98 | 0.78 | 0.67 | 0.37 | -0.35 | 0.14 | 0.88 | 0.96 | 0.06 | 0.00 |
| 10k–15k | unl | 0.25 | 0.23 | -0.19 | 0.16 | -0.33 | -0.21 | 0.71 | 0.19 | 0.38 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=0.27, cross_jaccard_bd=0.23, cross_rho_res=-0.05 | | | | | | | | | |

## MT_DS_LossNorm / Pancreas_6l

- 状态：completed，iteration 15000
- 与旧基线训练轨迹：first gap >1e-6 at 20; max |Δsup|=0.3609
- d3 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.52 | 0.33 | 0.33 | 0.29 | 0.63 | 0.47 | 0.56 | 0.54 | 0.12 | 0.00 |
| warmup ≤2k | unl | 0.29 | 0.33 | 0.25 | 0.28 | 0.68 | 0.47 | 0.53 | 0.14 | 0.32 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=0.09, cross_jaccard_bd=0.19, cross_rho_res=0.19 | | | | | | | | | |
| 2k–6k | lab | 0.79 | 0.52 | 0.52 | 0.35 | 0.81 | 0.62 | 0.34 | 0.68 | 0.14 | 0.00 |
| 2k–6k | unl | 0.01 | 0.16 | -0.08 | 0.17 | 0.61 | 0.49 | 0.63 | 0.07 | 0.44 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=0.02, cross_jaccard_bd=0.14, cross_rho_res=0.08 | | | | | | | | | |
| 6k–10k | lab | 0.91 | 0.68 | 0.64 | 0.39 | 0.83 | 0.70 | 0.31 | 0.89 | 0.16 | 0.00 |
| 6k–10k | unl | 0.10 | 0.19 | -0.09 | 0.16 | 0.43 | 0.35 | 0.77 | 0.15 | 0.45 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=0.00, cross_jaccard_bd=0.14, cross_rho_res=0.12 | | | | | | | | | |
| 10k–15k | lab | 0.97 | 0.88 | 0.63 | 0.39 | 0.83 | 0.74 | 0.32 | 0.97 | 0.14 | 0.00 |
| 10k–15k | unl | 0.32 | 0.28 | -0.09 | 0.15 | 0.10 | -0.00 | 0.93 | 0.25 | 0.38 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=-0.18, cross_jaccard_bd=0.10, cross_rho_res=0.24 | | | | | | | | | |

- d2 关卡：2k–6k: G1 fail, G2 pass; 6k–10k: G1 fail, G2 pass; 10k–15k: G1 fail, G2 pass

| phase | stream | lag_rho | lag_jaccard | raw_rho | raw_jaccard | cos_bd_rg | rho_bd_rg | residual_fraction | lag_rho_res | helpful_fraction | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warmup ≤2k | lab | 0.51 | 0.33 | 0.29 | 0.28 | 0.42 | 0.09 | 0.51 | 0.53 | 0.12 | 0.00 |
| warmup ≤2k | unl | 0.33 | 0.23 | 0.24 | 0.28 | 0.74 | 0.22 | 0.36 | 0.04 | 0.22 | 0.00 |
| warmup ≤2k | L↔U | cross_rho_bd=0.23, cross_jaccard_bd=0.23, cross_rho_res=0.22 | | | | | | | | | |
| 2k–6k | lab | 0.65 | 0.33 | 0.35 | 0.29 | 0.29 | 0.06 | 0.73 | 0.51 | 0.19 | 0.00 |
| 2k–6k | unl | 0.03 | 0.14 | -0.13 | 0.17 | 0.50 | 0.31 | 0.55 | 0.20 | 0.44 | 0.00 |
| 2k–6k | L↔U | cross_rho_bd=0.09, cross_jaccard_bd=0.14, cross_rho_res=0.21 | | | | | | | | | |
| 6k–10k | lab | 0.87 | 0.60 | 0.48 | 0.31 | -0.02 | -0.10 | 0.83 | 0.57 | 0.19 | 0.00 |
| 6k–10k | unl | 0.17 | 0.23 | -0.09 | 0.16 | 0.21 | -0.01 | 0.79 | 0.12 | 0.44 | 0.00 |
| 6k–10k | L↔U | cross_rho_bd=0.03, cross_jaccard_bd=0.14, cross_rho_res=0.03 | | | | | | | | | |
| 10k–15k | lab | 0.97 | 0.78 | 0.51 | 0.33 | -0.08 | -0.23 | 0.95 | 0.82 | 0.22 | 0.00 |
| 10k–15k | unl | 0.39 | 0.33 | -0.09 | 0.17 | -0.05 | -0.11 | 0.86 | 0.35 | 0.38 | 0.00 |
| 10k–15k | L↔U | cross_rho_bd=0.03, cross_jaccard_bd=0.14, cross_rho_res=-0.08 | | | | | | | | | |

