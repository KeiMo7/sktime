# 层级概率预测协调（Hierarchical Probabilistic Reconciliation）验收报告

本目录存放跨层一致的预测区间/分位数协调能力的全部验收证据。所有数字由
`generate_acceptance_report.py` 以固定随机种子离线生成，可重复运行：

```bash
PYTHONPATH=/home/wll/uid69 python3 reports/generate_acceptance_report.py
```

## 1. 交付的能力

新增包 `sktime/forecasting/probabilistic/reconcile/`：

| 模块 | 职责 |
| --- | --- |
| `_structure.py` | 由层级 MultiIndex 构建求和矩阵 S、父子邻接；加和性校验（支持自定义权重） |
| `_validation.py` | 分位数交叉定位（精确到节点/时点/分位对）、帧与 3-D 数组的向量化互转 |
| `_reconciler.py` | 协调引擎：线性投影快速路径 + 约束 QP 回退（FISTA/SLSQP）、边界、保序、协方差加权与降级 |
| `_diagnostics.py` | 名义/实际覆盖率偏差、区间宽度前后对比 |
| `_export.py` | 固定列序、固定层级顺序的区间表/分位数表导出 |
| `_exceptions.py` | 带结构化定位信息的异常 |

`ReconcilerForecaster` 新增**默认关闭**的 `reconcile_pred_int` 开关；关闭时
点预测路径、返回结构、estimator tag 与旧行为逐条一致（见下）。

## 2. 验收结论（对应需求"怎么算做完"）

### 2.1 三层加总一致（含残差）
`coherence_by_strategy.csv`：六种策略的列空间残差与父子加总残差均在 1e-11
量级（浮点误差），无系统性偏差。

### 2.2 分位数不交叉 + 覆盖诊断
协调后交叉数 = 0。`coverage_diagnostics.csv` 同时给出每个层级的名义覆盖率
（0.80）与实际覆盖率及偏差两组数字。

### 2.3 策略差异符合预期
`strategy_comparison.csv`：bu/ols/td_fcst/wls_str 宽度比均接近 1；在聚合层
bu 与 ols、td 的平均差异分别为 0.99、1.43，确认不同策略产生不同但都一致的结果。

### 2.4 边界处理后仍一致且未压成零宽
`acceptance_summary.json -> boundary_handling`：非负+容量（0≤y≤400）约束下
最小值 0、最大值 209.4（未越界），一致性残差 1.4e-13，96.3% 的区间保持正宽度；
仅当边界在数学上迫使上下界相等时才出现零宽（可解释，非协调技巧）。

### 2.5 失败模式（不静默、不留半成品）
- 权重不可加：抛 `HierarchyAdditivityError`，指明层级、节点、实际权重和。
- 输入分位数交叉：抛 `QuantileCrossingError`，报告全部违例位置；显式
  `on_input_crossing="isotonic"` 才修复，修复后交叉数为 0。
- 残差样本不足：MinT 自动降级为 OLS，并在 `warnings_` 标注（见 CSV 末行）。
- QP 不收敛时整体抛错，不返回部分结果。

## 3. 不能让的几条如何保证

1. **不独立相加**：所有端点统一在 S 的列空间内协调；OLS/WLS/MinT 的线性投影
   带负系数，实测会在叶子层诱发交叉，由保序 QP 修正，而不是事后截断。
2. **不压成零宽**：保序约束作用于相邻分位差（非负），仅边界强制时才可能等宽。
3. **边界可解释**：边界通过凸 QP 的 box 约束进入优化，不做后处理 clip（clip
   会破坏一致性）。
4. **入口与返回结构不变**：新参数全部带默认值，`capability:pred_int` 仅在
   显式开启时置 True。
5. **无联网求解**：仅依赖 numpy/scipy/pandas（scipy 为 sktime 核心依赖）。

## 4. 性能（面板批量）

`acceptance_summary.json -> panel_timing`（931 节点 / 900 底层 / 12 步预测 /
5 个分位）：纯线性路径 fit ≈ 0.06s、协调 ≈ 0.09s。
更大规模实测（3661 节点 / 3600 底层）：fit ≈ 1s、协调 ≈ 0.55s。
约束 QP 仅在边界绑定或交叉时按行触发；典型门店级规模（数百节点）端到端秒级。

## 5. 向后兼容

`all_point_paths_identical = true`：全部 8 种点预测策略在开关关闭时，
`predict` 输出与旧实现 `DataFrame.equals` 逐条相同。既有
`sktime/transformations/hierarchical/tests` 测试 48 通过、9 跳过（cvxpy 软依赖
未安装），无回归。

## 6. 文件清单

| 文件 | 内容 |
| --- | --- |
| `acceptance_summary.json` | 全部验收指标的机器可读汇总 |
| `acceptance_run.log` | 生成脚本的人类可读控制台输出 |
| `coherence_by_strategy.csv` | 各策略一致性残差 |
| `coverage_diagnostics.csv` | 各层名义/实际覆盖率 |
| `strategy_comparison.csv` | 策略宽度与求解路径对比 |
| `width_comparison.csv` | 协调前后区间宽度对比 |
| `reconciled_interval_table.csv` | 固定 schema 的区间表 |
| `reconciled_quantile_table.csv` | 固定 schema 的分位数表 |
| `tests/` | pytest 实际运行输出存档 |

## 7. 已知的既有问题（与本特性无关）

`TopdownReconciler` 在 `length=60` 的 `_bottom_hier_datagen` 数据上存在一个
pandas MultiIndex 索引的既有 `KeyError`，在未修改的原始构造器
（`reconcile_pred_int=False`）上同样复现。本特性自身的 td 概率路径
（`_topdown_linear`）不受影响；报告的点预测兼容性改用 length=40 数据规避。
