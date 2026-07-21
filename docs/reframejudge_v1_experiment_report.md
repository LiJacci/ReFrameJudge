# ReFrameJudge-v1 Experiment Report

更新日期：2026-07-21

## 1. 项目目标

ReFrameJudge 的目标是训练一个面向“原图-重新构图图像”pair 的评价器。给定：

```text
source image: 原图
candidate image: 重构图、裁剪图、编辑图或生成图
```

模型需要输出：

- `overall_label`: `win / tie / lose`
- `improvement_score`: 总体改善分，范围 `-2 到 2`
- `composition_gain`: 构图改善分，范围 `1 到 5`
- `content_preservation`: 内容保真分，范围 `1 到 5`
- `visual_naturalness`: 视觉真实自然度，范围 `1 到 5`
- `issue_tags` 和 `reason`: 诊断标签与简短说明

当前阶段的核心问题不是单纯判断图片美不美，而是判断候选图相对原图是否在构图层面更好，同时不能破坏内容语义和视觉真实性。

## 2. ReFrameJudge-v1 数据集

当前使用的数据版本是：

```text
data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_{train,val,test}.jsonl
```

划分规模：

| split | samples |
|---|---:|
| train | 800 |
| val | 100 |
| test | 100 |

test 集标签分布：

| label | count |
|---|---:|
| win | 52 |
| tie | 28 |
| lose | 20 |

数据由三个 subset 组成：

| subset | test samples | 特点 |
|---|---:|---|
| `aesrecon_500` | 20 | AesRecon good/poor 图像对，方向固定为 poor -> good，因此 test 中全部为 `win` |
| `fcdb` | 50 | Flickr Cropping Dataset ranking annotation 转换而来，包含 `win / tie / lose`，是当前最能检验细粒度构图偏好的子集 |
| `reframegen_seedream_strong150` | 30 | 基于 AesRecon source 和 Seedream 生成编辑构建的重构图像对，包含生成图真实性和内容保真问题 |

当前 v1 数据集的主要风险：

- 总规模仍小，训练集只有 800 条。
- `win` 类多于 `tie` 和 `lose`。
- 三个 subset 的标签来源不同，label 语义并不完全一致。
- `tie` 边界天然模糊，是当前模型最主要的错误来源。
- score 标签可能存在离散化和弱标注噪声。

## 3. 评价指标

分类指标：

| 指标 | 含义 |
|---|---|
| Accuracy | 所有样本中预测标签正确的比例 |
| Macro-F1 | 分别计算 `lose/tie/win` 的 F1 后取平均，更关注少数类表现 |
| Weighted-F1 | 按各类别样本数量加权后的 F1，更接近整体样本分布 |
| Confusion Matrix | 展示真实类别和预测类别的对应关系，用于定位类别混淆 |

回归指标：

| 指标 | 含义 |
|---|---|
| MAE | 平均绝对误差，越低越好 |
| Pearson | 线性相关性，越高说明预测分数和真实分数趋势越一致 |
| Spearman | 排序相关性，越高说明预测排序越接近真实排序 |

## 4. R1: CLIP Multi-task MLP

R1 是第一个监督学习 baseline。方法是冻结 CLIP image encoder，分别提取 source 和 candidate 图像特征，然后构造 pairwise feature：

```text
[source, candidate, candidate - source, source * candidate]
```

再训练一个小型 multi-task MLP，同时预测：

- 三分类 `overall_label`
- 四个回归分数

R1 test 结果：

| 模型 | 模式 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| CLIP MLP R1 | source/edit supervised | 0.590 | 0.533 | 0.588 |

R1 混淆矩阵，标签顺序为 `lose / tie / win`：

```text
[[ 8,  8,  4],
 [ 4, 13, 11],
 [ 7,  7, 38]]
```

R1 结论：

- CLIP MLP 是一个稳定的轻量强基线。
- 对 `win` 类表现最好，`tie` 和 `lose` 明显更难。
- CLIP 对四个细粒度分数有一定预测能力，但分数相关性整体有限。

## 5. R2: Qwen3.5-VL Local Judge

R2 使用 Qwen3.5 多模态模型作为评价器。我们放弃了此前 Qwen2.5-VL 路线，统一替换为：

```text
Qwen/Qwen3.5-4B
Qwen/Qwen3.5-9B
```

实验分为两类：

| 设置 | 含义 |
|---|---|
| no-LoRA | 不微调，只用 prompt 让模型直接评价 |
| LoRA | 在 ReFrameJudge-v1 train split 上做 LoRA SFT |

输入模式也尝试了两类：

| 模式 | 输入和输出 |
|---|---|
| source-candidate | 明确告诉模型第一张是 source，第二张是 candidate，输出 `overall_label` 和分数 |
| blind A/B | 不告诉模型哪张是原图，输入 A/B 两张图，输出 `A / B / tie` |

## 6. 总体分类结果

| 模型 | 训练 | 模式 | Accuracy | Macro-F1 | Weighted-F1 | 预测分布 |
|---|---|---|---:|---:|---:|---|
| CLIP MLP R1 | supervised | source/edit | 0.590 | 0.533 | 0.588 | - |
| Qwen3.5-4B | no-LoRA | source-candidate | 0.530 | 0.437 | 0.502 | win 64 / lose 21 / tie 15 |
| Qwen3.5-9B | no-LoRA | source-candidate | 0.500 | 0.456 | 0.514 | win 41 / lose 45 / tie 14 |
| Qwen3.5-4B | LoRA | source-candidate | 0.590 | 0.541 | 0.593 | win 50 / lose 16 / tie 34 |
| Qwen3.5-9B | LoRA | source-candidate | 0.580 | 0.556 | 0.591 | win 41 / lose 26 / tie 33 |
| Qwen3.5-4B | no-LoRA | blind A/B | 0.470 | 0.332 | 0.403 | win 64 / lose 36 / tie 0 |
| Qwen3.5-9B | no-LoRA | blind A/B | 0.470 | 0.359 | 0.428 | win 56 / lose 40 / tie 4 |
| Qwen3.5-4B | LoRA | blind A/B | 0.360 | 0.260 | 0.316 | win 56 / lose 44 / tie 0 |
| Qwen3.5-9B | LoRA | blind A/B | 0.460 | 0.346 | 0.412 | win 45 / lose 55 / tie 0 |

主要结论：

- `source-candidate` 明显优于 `blind A/B`，更适合作为当前主任务形式。
- LoRA 对 `source-candidate` 模式有稳定提升。
- `Qwen3.5-9B LoRA source-candidate` 的 Macro-F1 最高，是当前主结果。
- `Qwen3.5-4B LoRA source-candidate` 的 Accuracy 与 CLIP R1 持平，但 Macro-F1 略高。
- `blind A/B` 几乎不预测 `tie`，不适合作为当前主 baseline。

## 7. 当前主模型分析

当前主模型：

```text
Qwen3.5-9B LoRA source-candidate
```

test 结果：

| Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|
| 0.580 | 0.556 | 0.591 |

预测分布：

| label | target | prediction |
|---|---:|---:|
| win | 52 | 41 |
| tie | 28 | 33 |
| lose | 20 | 26 |

混淆矩阵，标签顺序为 `lose / tie / win`：

```text
[[12,  7,  1],
 [ 6, 14,  8],
 [ 8, 12, 32]]
```

分类别表现：

| class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| lose | 0.462 | 0.600 | 0.522 | 20 |
| tie | 0.424 | 0.500 | 0.459 | 28 |
| win | 0.780 | 0.615 | 0.688 | 52 |

错误分析：

| item | count |
|---|---:|
| total samples | 100 |
| total errors | 42 |
| tie-related errors | 33 |

`tie-related errors` 指真实标签是 `tie` 或预测标签是 `tie` 的错误。42 个错误中有 33 个与 `tie` 有关，说明当前最主要瓶颈不是模型完全不会判断构图，而是 `win / tie / lose` 的边界不稳。

## 8. Subset 分析

`Qwen3.5-9B LoRA source-candidate` 在各 subset 上的表现：

| subset | n | Accuracy | Macro-F1 | target distribution | prediction distribution |
|---|---:|---:|---:|---|---|
| aesrecon_500 | 20 | 1.000 | 0.333 | win 20 | win 20 |
| fcdb | 50 | 0.440 | 0.418 | win 17 / lose 17 / tie 16 | lose 26 / win 13 / tie 11 |
| reframegen_seedream_strong150 | 30 | 0.533 | 0.370 | tie 12 / win 15 / lose 3 | tie 22 / win 8 |

解读：

- `aesrecon_500` 全部为 `win`，模型全部预测为 `win`，Accuracy 为 1.0，但 Macro-F1 只有 0.333，因为该 subset 没有 `lose/tie` 样本。
- `fcdb` 是最难、也最关键的构图偏好子集，当前 Accuracy 只有 0.44。模型在 FCDB 上偏向预测 `lose`，对 crop ranking 的细粒度偏好仍不稳定。
- `reframegen_seedream_strong150` 中模型偏向预测 `tie`，说明生成图场景下，模型可能对内容/真实性风险更保守。

## 9. 回归分数结果

source-candidate 模式下的主要回归结果：

| 模型 | improvement MAE | improvement Pearson | composition MAE | composition Pearson | content MAE | content Pearson | naturalness MAE | naturalness Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP R1 | 0.897 | 0.357 | 0.773 | 0.275 | 0.383 | 0.277 | 0.527 | 0.317 |
| Qwen3.5-4B LoRA source | 0.780 | 0.405 | 0.670 | 0.391 | 0.090 | 0.779 | 0.180 | 0.839 |
| Qwen3.5-9B LoRA source | 0.826 | 0.413 | 0.700 | 0.393 | 0.090 | 0.779 | 0.180 | 0.873 |

解读：

- Qwen3.5 LoRA 在 `composition_gain`、`content_preservation`、`visual_naturalness` 上明显优于 CLIP R1。
- `improvement_score` 上，4B LoRA 的 MAE 更低，9B LoRA 的 Pearson 略高。
- `content_preservation` 和 `visual_naturalness` 的指标非常好，但需要谨慎解释：这两个标签分布可能较窄，模型也可能输出较离散的高分。

## 10. Blind A/B 尝试

blind A/B 的目标是不告诉模型哪张是 source，直接比较 A/B 两张图。这种设置理论上可以减少 “edited image 应该更好” 的先验，但当前结果较差。

主要问题：

- 4B no-LoRA 和 4B LoRA 都完全不预测 `tie`。
- 9B LoRA 也完全不预测 `tie`。
- 9B no-LoRA 只预测了 4 个 `tie`，远低于真实的 28 个。
- 4B LoRA blind A/B 甚至低于 no-LoRA，说明 source-candidate LoRA 训练不能直接迁移到 A/B 输出格式。

因此当前决定：

```text
暂时放弃 blind A/B 作为主线，只保留为诊断实验。
```

如果未来要重新做 blind A/B，需要专门构造 A/B-format 训练数据，并显式训练 `A / B / tie` 输出，而不是复用 source-candidate LoRA。

## 11. Score Calibration 尝试

我们尝试过不用模型直接输出的 `overall_label`，而是用 `improvement_score` 在 val 集上学习阈值：

```text
score <= low_threshold -> lose
low_threshold < score < high_threshold -> tie
score >= high_threshold -> win
```

对 `Qwen3.5-9B LoRA source-candidate`，学习到的阈值为：

```text
low_threshold = -1.2
high_threshold = 0.4
```

结果：

| 方法 | Accuracy | Macro-F1 | Weighted-F1 | 预测分布 |
|---|---:|---:|---:|---|
| 原始 overall_label | 0.580 | 0.556 | 0.591 | win 41 / lose 26 / tie 33 |
| score calibrated | 0.570 | 0.532 | 0.566 | win 52 / lose 26 / tie 22 |

结论：

- calibration 在 val 上略有提升，但没有泛化到 test。
- test Macro-F1 从 0.556 降到 0.532。
- 原因是模型输出的 `improvement_score` 很离散，`0.4` 这个边界分数并不稳定。
- 因此简单 score threshold calibration 暂时不作为主结果，也不继续投入。

## 12. 当前结论

截至目前，ReFrameJudge-v1 上最重要的结论是：

1. **主任务形式应采用 source-candidate，而不是 blind A/B。**

   ReFrameJudge 的实际应用目标就是比较原图和重构图，因此告诉模型 source/candidate 关系是合理的。blind A/B 当前 tie 表现太差。

2. **LoRA 有明确价值。**

   在 source-candidate 模式下：

   ```text
   4B no-LoRA: Macro-F1 0.437
   4B LoRA:    Macro-F1 0.541

   9B no-LoRA: Macro-F1 0.456
   9B LoRA:    Macro-F1 0.556
   ```

3. **当前最佳主结果是 Qwen3.5-9B LoRA source-candidate。**

   虽然 Accuracy 不是最高，但 Macro-F1 最高，说明它在类别均衡性上最好。

4. **CLIP MLP R1 仍是强基线。**

   CLIP R1 的 Accuracy 为 0.590，Macro-F1 为 0.533。Qwen3.5-9B LoRA 的优势还不算巨大，因此后续需要更强的数据和分析支撑。

5. **当前主要瓶颈是 tie 边界。**

   主模型 42 个错误中有 33 个与 tie 有关。下一阶段要重点处理 `win/tie/lose` 边界定义、难例样本和标签一致性。

6. **简单 score calibration 意义有限。**

   当前模型的 score 输出较离散，不能通过简单阈值稳定修正 label。

## 13. 下一步建议

下一阶段建议从“继续换模型”转向“数据质量和难例分析”：

1. **Hard Case Mining**

   使用当前主模型的错误分析 HTML，对错误样本进行人工归因：

   - label 确实错了
   - label 有争议
   - 模型判断错了
   - source/candidate 差异太弱
   - 构图改善但内容损坏
   - 内容保真但构图无明显改善
   - 生成图有 artifact 但模型未识别

2. **构建 ReFrameJudge-v1.1**

   基于 hard case review 清理或增强数据：

   - 修正明显错误标签
   - 移除争议过大的样本
   - 增加 `tie` 边界样本
   - 增加 `lose` 样本
   - 增加“构图变好但真实感/内容变差”的反例
   - 增加真实生成图重构样本

3. **重新训练 R2.2**

   使用 v1.1 重新训练：

   ```text
   Qwen3.5-9B LoRA source-candidate
   ```

   并与当前 v1 结果对比，验证数据质量是否带来提升。

4. **统一实验追踪**

   后续建议维护一个固定结果表，记录：

   - dataset version
   - model
   - mode
   - LoRA config
   - training epochs
   - Accuracy / Macro-F1 / per-class F1
   - regression metrics
   - error review link

## 14. 当前推荐引用结果

如果只汇报一个主结果，建议使用：

```text
Dataset: ReFrameJudge-v1 combined balanced1000
Model: Qwen3.5-9B LoRA
Mode: source-candidate
Test samples: 100
Accuracy: 0.580
Macro-F1: 0.556
Weighted-F1: 0.591
```

同时保留 CLIP R1 作为轻量对照：

```text
Model: CLIP Multi-task MLP
Accuracy: 0.590
Macro-F1: 0.533
Weighted-F1: 0.588
```
