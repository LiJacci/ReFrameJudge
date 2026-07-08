# ReFrameJudge FCDB Baseline Study Report

## 1. 背景与目标

ReFrameJudge 的目标是训练和评测一个面向图像重新构图的评价器。给定一对图像：

```text
I_src: 原始图像
I_edit: 重新构图、裁切、扩图或编辑后的图像
```

模型需要判断：

```text
1. 构图是否变好
2. 主要内容是否保留
3. 图像是否真实自然
```

最终输出可以是：

```text
win / tie / lose
```

也可以是连续偏好分数，例如：

```text
edited 比 source 差很多 -> edited 比 source 好很多
```

本阶段使用 FCDB 作为第一个实验数据源。FCDB 本身不是生成图编辑数据集，而是 crop preference 数据集。它适合用于快速验证“构图偏好判断”这个子问题，但不能覆盖生成图中的内容改变、身份变形、背景幻觉、纹理伪影、透视不自然等问题。

因此，本报告中的 FCDB 实验应被理解为 ReFrameJudge 的 pilot study，而不是最终任务的完整评测。

## 2. FCDB 数据集与任务构造

FCDB 的 ranking annotation 对同一张 Flickr 图片的两个候选裁切框进行比较。每个 pair 有 5 个投票：

```text
vote_for_0
vote_for_1
crop_0
crop_1
```

我们把两个 crop 裁成图像 pair，并构造 ReFrameJudge 标注字段：

```json
{
  "source_image": "...",
  "edited_image": "...",
  "overall_label": "win|tie|lose",
  "source_votes": 2,
  "edited_votes": 3,
  "vote_margin": 1,
  "preference_strength": "weak|strong"
}
```

这里的 `source_image` 和 `edited_image` 只是 pair 的两个方向，并不表示真实的原图和生成编辑图。

### 2.1 基础 5k 二分类

最早的 `fcdb_train/val/test` 使用前 5000 个方向样本，包含正反方向：

```text
train: 4000, win 2000, lose 2000
val:    500, win 250,  lose 250
test:   500, win 250,  lose 250
```

这个版本没有显式区分强弱偏好，3:2 / 2:3 弱偏好也被当成 win/lose。

### 2.2 Strong Binary Split

为了减少弱标签噪声，我们构造了 `fcdb_strong_*`：

```text
strong = 4:1 或 5:0 投票
```

数据规模：

```text
train: 2422, win 1211, lose 1211
val:    314, win 157,  lose 157
test:   286, win 143,  lose 143
```

test 中 vote margin 分布：

```text
abs margin 3: 166
abs margin 5: 120
```

### 2.3 Three-way Split

为了模拟 ReFrameJudge 的 `win/tie/lose` 输出，我们构造了 `fcdb_3way_*`：

```text
5:0 / 4:1 -> win 或 lose
3:2 / 2:3 -> tie
```

数据规模：

```text
train: 4000, lose 1185, tie 1630, win 1185
val:    500, lose 156,  tie 188,  win 156
test:   500, lose 170,  tie 160,  win 170
```

test 中标签与 vote margin 对应：

```text
lose: margin -5, -3
tie:  margin -1,  1
win:  margin  3,  5
```

需要注意：这里的 `tie` 并不是真正的“两个 crop 构图等价”，而是弱投票偏好，即 3:2 或 2:3。这是后续三分类困难的关键原因。

## 3. Baseline 设计

本阶段我们测试了四类 baseline。

### 3.1 CLIP + Logistic Regression

使用冻结的 CLIP ViT-B/32 图像 encoder 提取 source/edit embedding：

```text
f_src
f_edit
```

pair feature：

```text
[f_src, f_edit, f_edit - f_src, f_src * f_edit]
```

再用 Logistic Regression 训练 win/lose 分类器。

### 3.2 DINOv2 + Logistic Regression

流程与 CLIP 相同，只是视觉 backbone 换成 DINOv2。我们也尝试了 CLIP + DINOv2 feature fusion。

DINOv3 因官方权重需要 Hugging Face gated access，当前没有纳入正式结果。

### 3.3 三分类与阈值校准

直接三分类：

```text
CLIP features -> Logistic Regression -> lose/tie/win
```

阈值校准三分类：

```text
先在 strong labels 上训练 pairwise win/lose scorer
再在 val 上搜索 threshold tau

score > tau       -> win
|score| <= tau    -> tie
score < -tau      -> lose
```

### 3.4 Score Regression

我们进一步把 FCDB 投票转成连续目标：

```text
target_score = (edited_votes - source_votes) / total_votes
```

在 FCDB 中总票数为 5，因此目标值是：

```text
-1.0, -0.6, -0.2, 0.2, 0.6, 1.0
```

模型使用 CLIP pair feature + Ridge Regression 拟合连续分数，再在 val 上搜索两个阈值：

```text
score < low_threshold  -> lose
score > high_threshold -> win
otherwise              -> tie
```

### 3.5 Qwen VLM Judge

我们测试了 Qwen 视觉大模型作为 judge：

1. Source/Edit prompt：告诉模型第一张是 source，第二张是 edited/reframed。
2. Blind A/B prompt：只告诉模型 Candidate A 和 Candidate B，不出现 source/edit/edited/reframed。
3. Binary blind A/B：在 strong binary test 上强制 A/B 二选一。

Qwen 通过 OpenAI-compatible API 调用，输出 JSON 标签和 reason。

## 4. 实验结果

### 4.1 基础二分类结果

测试集：`fcdb_test.jsonl`

| model | accuracy | macro F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| CLIP + LogReg | 0.672 | 0.672 | 0.720 |
| DINOv2 + LogReg | 0.616 | 0.616 | 0.634 |
| CLIP + DINOv2 Fusion | 0.624 | 0.624 | 0.673 |

结论：

```text
CLIP 是当前最稳的通用 embedding baseline。
DINOv2 单独不如 CLIP。
简单拼接 CLIP + DINOv2 没有带来提升。
```

### 4.2 Strong Binary 结果

测试集：`fcdb_strong_test.jsonl`

| model | accuracy | macro F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| CLIP + LogReg | 0.636 | 0.636 | 0.672 |
| Qwen Blind A/B Binary | 0.538 | 0.537 | n/a |

CLIP confusion matrix：

| true \ pred | lose | win |
| --- | ---: | ---: |
| lose | 91 | 52 |
| win | 52 | 91 |

Qwen binary confusion matrix：

| true \ pred | lose | win |
| --- | ---: | ---: |
| lose | 84 | 59 |
| win | 73 | 70 |

Qwen raw choice distribution：

```text
choice A: 43
choice B: 243
```

结论：

```text
即使在强二分类设置下，Qwen 单次 blind A/B 仍然不如 CLIP。
主要问题是 Candidate B 位置偏置。
```

### 4.3 Three-way 结果

测试集：`fcdb_3way_test.jsonl`

| model | accuracy | macro F1 |
| --- | ---: | ---: |
| CLIP direct 3-way | 0.392 | 0.393 |
| CLIP threshold 3-way | 0.424 | 0.425 |
| CLIP score regression | 0.424 | 0.426 |
| Qwen source/edit prompt | 0.388 | 0.331 |
| Qwen blind A/B prompt | 0.366 | 0.302 |

CLIP score regression confusion matrix：

| true \ pred | lose | tie | win |
| --- | ---: | ---: | ---: |
| lose | 69 | 71 | 30 |
| tie | 43 | 74 | 43 |
| win | 30 | 71 | 69 |

Qwen blind A/B confusion matrix：

| true \ pred | lose | tie | win |
| --- | ---: | ---: | ---: |
| lose | 94 | 7 | 69 |
| tie | 83 | 3 | 74 |
| win | 79 | 5 | 86 |

结论：

```text
三分类整体困难。
CLIP threshold / score regression 比 direct 3-way 更合理，但提升有限。
Qwen blind A/B 几乎不用 tie，tie recall 只有 3 / 160。
```

### 4.4 Score Regression 结果

CLIP score regression 连续分数结果：

| split | MAE | RMSE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| train | 0.372 | 0.459 | 0.636 | 0.636 |
| val | 0.522 | 0.632 | 0.295 | 0.278 |
| test | 0.576 | 0.709 | 0.239 | 0.278 |

阈值：

```text
low_threshold:  -0.202
high_threshold:  0.202
```

结论：

```text
拟合连续分数的任务定义是合理的。
但 CLIP global embedding 对同图不同 crop 的细粒度构图差异不够敏感。
```

## 5. 关键发现

### 5.1 FCDB 的 tie 标签不是稳定视觉类别

我们最初把 3:2 / 2:3 映射为 tie，但这更准确地说是弱偏好或标注分歧。它并不表示两个 crop 在视觉上完全等价。

因此，直接训练 `lose/tie/win` 三分类会有噪声：

```text
tie 不是一种独立视觉形态
tie 更像偏好分数接近 0 或不确定
```

这也是 threshold / score regression 比 direct 3-way 更合理的原因。

### 5.2 CLIP 是稳健下限，但不够构图专用

CLIP 在二分类和三分类中都是最稳定的 baseline。它优于 DINOv2、简单 fusion 和当前 Qwen prompt。

但是，CLIP 的优势仍然有限。FCDB pair 多是同一张图片的不同 crop，语义内容几乎一样，关键差异在：

```text
主体位置
留白比例
边界裁切
视觉平衡
构图焦点
```

这些不是 CLIP global semantic embedding 最擅长的部分。

### 5.3 DINOv2 和简单融合没有解决问题

DINOv2 更偏局部视觉/自监督表征，但在当前 pairwise logistic regression 设置下不如 CLIP。

简单 feature concat：

```text
CLIP feature + DINOv2 feature
```

也没有提升，说明问题不是单纯“多堆通用特征”。

### 5.4 Qwen 单次 judge 不可靠

Qwen source/edit prompt 中，模型强烈倾向认为 edited image 更好。

Qwen blind A/B prompt 中，模型又强烈倾向 Candidate B：

```text
3-way blind A/B: choice B = 397 / 500
binary blind A/B: choice B = 243 / 286
```

这说明单次 VLM A/B 判断受到明显位置偏置影响，不能直接作为可靠 evaluator。

Qwen 的 reason 经常看似合理，但很多时候是在为位置偏好生成解释，而不是稳定执行构图偏好判断。

### 5.5 Reverse pair 会让评估矩阵天然对称

我们在 FCDB pair 构造中加入了 reverse samples：

```text
A -> B: win
B -> A: lose
```

在 strong test 中：

```text
records: 286
unique unordered comparisons: 143
每个 comparison 都有正反方向
```

CLIP 学到了近似反对称分数：

```text
prob_win(A, B) + prob_win(B, A) ≈ 1
```

因此混淆矩阵严格对称不是 bug，而是数据构造与模型形式共同导致的结果。

但这也意味着 test 的独立样本数应按 143 个 unordered comparisons 理解，而不是 286 个方向样本。

## 6. 对 ReFrameJudge 的启示

FCDB 已经完成了 pilot study 的作用。它帮助我们验证了：

```text
1. 构图偏好可以从 pairwise 数据中学习到一些信号。
2. 通用视觉 embedding 有下限能力，但不足以成为最终方案。
3. 三分类 tie 需要被建模为不确定区间，而不是硬类别。
4. VLM 单次判断有明显 prompt/order bias。
5. 评估协议需要避免 reverse pair 带来的重复统计。
```

但 FCDB 不能代表 ReFrameJudge 的最终任务，因为最终任务涉及生成图编辑中的额外问题：

```text
主体是否变形
身份是否改变
背景是否幻觉
内容是否缺失
纹理是否不自然
透视和光照是否一致
```

这些都不是 FCDB crop ranking 能覆盖的。

## 7. 下一步建议

### 7.1 停止在 FCDB 上无限堆 baseline

当前 FCDB baseline 已经足够暴露问题。继续只在 FCDB 上换模型，边际收益会很低。

### 7.2 构建 ReFrameJudge-v1 真实数据

下一阶段应该开始构建真正的 ReFrameJudge-v1：

```text
原图 -> 重新构图生成图
```

每条样本至少包含：

```text
overall_label: win / tie / lose
composition_gain: 1-5
content_preservation: 1-5
visual_naturalness: 1-5
issue_tags
reason
```

数据来源建议：

```text
1. FCDB / crop 数据：构图偏好基础
2. outpainting / uncrop：扩图和重构
3. subject reposition：主体重定位
4. generated negative cases：内容缺失、主体变形、背景幻觉、伪影
```

### 7.3 改进评估协议

对 FCDB 继续保留作为诊断集，但应该生成 no-reverse eval split：

```text
每个 unordered crop pair 只保留一个方向
控制 win/lose 平衡
避免一个 pair 被正反统计两次
```

### 7.4 VLM 后续只做辅助或双顺序一致性

如果继续探索 Qwen/VLM，不建议使用单次 A/B 输出作为最终标签。更合理的是 dual-order：

```text
order 1: A = image1, B = image2
order 2: A = image2, B = image1
```

映射回 ReFrameJudge label 后：

```text
两次一致 -> 高置信
两次不一致 -> uncertain / tie
```

报告：

```text
coverage
agreement-only accuracy
position-bias rate
```

### 7.5 模型路线

短期内建议保留 CLIP score / threshold 作为弱 baseline，同时把精力转到数据集建设。

当 ReFrameJudge-v1 有 500-1000 条真实生成图 pair 后，再考虑：

```text
CLIP-style pair scorer
VLM rationale distillation
multi-dimensional evaluator
small supervised fine-tuning
```

## 8. 总结

FCDB baseline 的核心结论是：

```text
CLIP 是当前最稳定的弱 baseline。
DINOv2 和简单 fusion 没有提升。
三分类困难主要来自 weak preference 被硬映射为 tie。
score regression 是更合理的任务形式，但 CLIP 特征仍不足。
Qwen 单次 judge 受到严重 prompt/order bias，不适合作为直接 evaluator。
FCDB 适合做构图偏好 pilot，不适合作为 ReFrameJudge 最终任务。
```

因此，下一阶段应从 FCDB baseline 转向 ReFrameJudge-v1 数据集构建，并把评价维度扩展到：

```text
构图提升
内容保留
视觉真实性
```

这才是 ReFrameJudge 最初目标真正需要解决的问题。
