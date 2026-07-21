# ReFrameJudge-v2 Dataset Construction Plan

更新日期：2026-07-21

## 1. 背景与目标

ReFrameJudge-v1 已经完成了 baseline 验证，结论是：

- `source-candidate` 是当前最合理的任务形式。
- Qwen3.5-9B LoRA 在 v1 上是当前最强主模型。
- 当前瓶颈主要来自数据，而不是简单阈值或 blind A/B prompt。
- `tie` 边界、FCDB 子集、生成图真实性和内容保真是主要困难点。

因此下一步应从“继续调 baseline”转向“制作真正属于 ReFrameJudge 的数据集”。

ReFrameJudge-v2 的目标是构建一个高质量、可复现、可扩展的数据集，用于训练和评测“原图-重构图”评价模型。每条样本由一个原图和一个候选重构图组成，模型需要判断候选图是否相对原图更好，并给出多维评分和解释。

核心任务：

```text
Input:
  source image
  candidate image

Output:
  overall_label: win / tie / lose
  improvement_score: -2 to 2
  composition_gain: 1 to 5
  content_preservation: 1 to 5
  visual_naturalness: 1 to 5
  issue_tags
  reason
```

v2 不应只是一个“构图偏好”数据集，而应覆盖真实生成式重构评价中最重要的三类能力：

1. 候选图构图是否更好。
2. 候选图是否保留原图主体、身份、场景语义和重要细节。
3. 候选图是否真实自然，是否有生成 artifact、透视错误、光影错误或不合理细节。

## 2. 数据集定位

ReFrameJudge-v2 应服务两个目的：

| 用途 | 说明 |
|---|---|
| 训练集 | 用于训练 source-candidate 图像评价模型 |
| 评测集 | 用于可靠比较 CLIP/DINO/Qwen/其他 VLM evaluator |

它不应只收集“明显变好”的正样本。真正有价值的是包含大量边界样本和反例：

- 构图更好但内容丢失。
- 构图更好但真实感变差。
- 内容保留很好但构图无明显提升。
- 生成图更美但不像原图。
- 原图已经很好，重构图没有必要修改。
- 候选图局部更好但整体更差。
- 两张图差异很小，应该判为 `tie`。

## 3. 推荐规模

建议采用三阶段建设，不要一次性做过大。

### 3.1 Pilot 阶段

目标：验证流程、标注规范、生成策略和质控机制。

| 项目 | 推荐数量 |
|---|---:|
| 原图数 | 300 |
| 每张原图候选图 | 4 |
| pair 数 | 1,200 |
| 人工精标 pair | 1,200 |
| 预计用途 | 流程验证、标注员校准、第一版 v2 baseline |

Pilot 阶段最重要的不是规模，而是发现问题：

- prompt 是否能稳定生成构图变化。
- 候选图是否经常破坏内容。
- 标注员是否能理解 `win/tie/lose`。
- `tie` 标签是否一致。
- 哪些 issue_tags 最常出现。

### 3.2 Core 阶段

目标：形成第一个真正可训练的数据集。

| 项目 | 推荐数量 |
|---|---:|
| 原图数 | 2,000 |
| 每张原图候选图 | 4 |
| pair 数 | 8,000 |
| 人工精标 pair | 8,000 |
| 推荐划分 | train 6,400 / val 800 / test 800 |

这是我最推荐的第一版正式目标。原因：

- 2,000 张原图已经能覆盖较多场景。
- 8,000 pair 对 LoRA SFT 来说比当前 800 train 明显更有意义。
- 人工标注成本仍可控。
- test 800 足以比当前 test 100 更稳定地比较模型。

### 3.3 Scale 阶段

目标：扩展到更强模型训练和更可靠评测。

| 项目 | 推荐数量 |
|---|---:|
| 原图数 | 10,000 |
| 每张原图候选图 | 4-6 |
| pair 数 | 40,000-60,000 |
| 人工精标 pair | 10,000-20,000 |
| 弱标注/模型辅助 pair | 30,000-40,000 |

Scale 阶段可以采用“人工精标 + 模型预标注 + 人工抽检”的方式，不建议所有样本都纯人工精标。

## 4. 原图收集策略

### 4.1 原图规模建议

第一版正式数据集建议：

```text
Core v2:
  2,000 source images
  4 candidate images per source
  8,000 source-candidate pairs
```

不要少于 1,000 原图。少于 1,000 时，模型容易记住场景和生成风格，评测也不稳定。

也不建议第一步直接做 10,000 原图。因为标注规范和生成策略还没有完全稳定，过早扩大规模会把错误放大。

### 4.2 原图来源

推荐采用多源收集，但 v2 主体应尽量使用授权明确、可再分发或至少可记录来源的数据。

候选来源：

| 来源 | 作用 | 风险 |
|---|---|---|
| 自有拍摄/自建图片 | 最干净，适合发布 | 成本高，场景覆盖有限 |
| Unsplash/Pexels/Wikimedia 等开放授权图片 | 适合扩展真实照片场景 | 需要记录 license 和 attribution |
| COCO/OpenImages 等研究数据 | 场景丰富，主体多样 | 需要遵守原数据 license |
| AesRecon/已有研究数据 | 可连接 v1 经验 | 可能不适合作为最终可发布主体 |
| 用户生成或团队收集图片 | 贴近真实应用 | 需处理隐私和授权 |

建议 v2 Core 采用：

```text
60% 开放授权真实照片
20% 现有研究数据中 license 清晰的照片
20% 自有/团队拍摄或自建场景
```

### 4.3 场景覆盖

原图应按场景分层采样，避免全是人像或风景。

推荐场景比例：

| 场景 | 比例 | 说明 |
|---|---:|---|
| 单人/多人照片 | 20% | 人像、街拍、活动照 |
| 物体/商品/静物 | 15% | 适合测试主体居中、留白、比例 |
| 室内场景 | 15% | 家居、餐厅、办公室、展馆 |
| 城市/街景/建筑 | 15% | 适合测试透视、裁切、主体位置 |
| 自然风景 | 15% | 适合测试地平线、前中后景、视觉引导 |
| 动物/植物/食物 | 10% | 适合测试细节和真实感 |
| 复杂场景 | 10% | 多主体、遮挡、杂乱背景、低光、运动模糊 |

### 4.4 原图质量要求

原图应满足：

- 最短边建议大于 768 px。
- 无明显水印、边框、截图 UI、拼图。
- 图像主体或视觉中心可识别。
- 不能含有隐私敏感信息，如清晰证件、车牌、聊天记录、医疗信息。
- 成人、暴力、仇恨、政治敏感等内容应先排除，除非后续明确要做安全评测。

### 4.5 原图元数据

每张 source image 应保存 source manifest：

```json
{
  "source_id": "rfjv2_src_000001",
  "image_path": "data/reframejudge_v2/source/rfjv2_src_000001.jpg",
  "source_url": "",
  "license": "CC0 / custom / research-only / internal",
  "photographer": "",
  "scene_type": "portrait",
  "main_subject": "person",
  "aspect_ratio": "3:2",
  "width": 1536,
  "height": 1024,
  "quality_tags": ["clear_subject", "natural_light"],
  "privacy_checked": true
}
```

## 5. 候选重构图生成策略

每张原图建议生成 4 张候选图。不要只生成“尽量变好”的图，而要有可控的好坏层次。

推荐每张原图生成：

| candidate 类型 | 每源数量 | 目标 |
|---|---:|---|
| strong positive recomposition | 1 | 构图明显更好，内容和真实性基本保持 |
| mild positive / subtle edit | 1 | 小幅改善，容易形成 `tie/win` 边界 |
| risky edit | 1 | 构图可能更好，但内容或真实感可能受损 |
| negative / unnecessary edit | 1 | 候选图无改善或变差，用于 `lose` 样本 |

这样 2,000 原图可以得到：

```text
2,000 source images * 4 candidates = 8,000 pairs
```

### 5.1 生成方式

建议混合使用三种生成方式：

#### A. 几何构图变换

包括：

- crop
- zoom in
- shift
- rotate small angle
- aspect-ratio conversion
- rule-of-thirds crop
- subject-centered crop

优点：

- 可控、便宜、容易生成大量样本。
- 内容保真高。
- 可以生成明确的 `win/tie/lose` 边界样本。

缺点：

- 不能覆盖真正生成图的 artifact。
- 候选图可能只是裁剪，不够代表生成式重构。

建议占比：

```text
30%
```

#### B. 生成式图像编辑

包括：

- outpainting
- background extension
- subject reposition
- viewpoint-like reframing
- composition-aware regeneration
- remove clutter / add negative space
- adjust horizon / framing

优点：

- 最贴近 ReFrameJudge 的最终目标。
- 能产生内容保真和真实自然度问题。

缺点：

- 成本高。
- 质量不稳定。
- 需要严格过滤水印、伪影、语义漂移。

建议占比：

```text
50%
```

#### C. 检索或已有编辑对

包括：

- 公开 cropping preference 数据。
- aesthetic reconstruction 数据。
- 真实修图前后对。
- 人工编辑前后对。

优点：

- 可以提供真实人类偏好的补充。

缺点：

- 任务定义可能与生成式重构不完全一致。
- 标签语义需要统一。

建议占比：

```text
20%
```

### 5.2 构图生成目标

生成 prompt 不应只写“make it better composition”。应指定明确构图操作。

推荐构图操作 taxonomy：

| edit_type | 说明 |
|---|---|
| `crop` | 裁剪掉干扰区域，改善主体占比 |
| `zoom_in` | 拉近主体，增强视觉焦点 |
| `zoom_out` | 扩展画面，增加环境信息或呼吸感 |
| `shift_subject` | 调整主体位置，例如从居中移到三分线 |
| `extend_canvas` | 向某一方向扩图，改善留白或方向感 |
| `aspect_ratio_change` | 横图转竖图、竖图转横图、适配海报/封面比例 |
| `horizon_balance` | 调整地平线或建筑线条 |
| `declutter` | 减少背景干扰 |
| `foreground_background_balance` | 改善前景、中景、背景层次 |
| `negative_space` | 增加适当留白 |
| `failed_edit` | 故意或自然产生的失败重构 |

### 5.3 Prompt 设计

每张原图先由 VLM 生成 source analysis：

```json
{
  "main_subject": "a person standing near a window",
  "scene_type": "indoor portrait",
  "current_composition_issues": [
    "subject too centered",
    "too much empty space on the left",
    "background clutter"
  ],
  "safe_edit_directions": [
    "crop tighter around the upper body",
    "extend space toward the looking direction",
    "reduce background clutter without changing identity"
  ],
  "do_not_change": [
    "person identity",
    "clothing",
    "pose",
    "window lighting"
  ]
}
```

再根据 source analysis 生成不同候选图 prompt。

示例 prompt 模板：

```text
Recompose the image to improve photographic composition.
Preserve the main subject, identity, pose, clothing, scene semantics, lighting, and important objects.
Change only the framing and surrounding canvas when possible.

Composition goal:
  Move the main subject closer to the right third line and add natural negative space on the left.

Do not:
  change the subject identity
  add new important objects
  remove important scene elements
  alter facial features
  create text, watermark, logo, or frame

Output should look like a natural photograph.
```

### 5.4 负样本生成

负样本非常重要。否则模型会学到“候选图总是更好”。

负样本来源：

- 过度裁剪主体。
- 裁掉关键物体。
- 留白方向错误。
- 主体过小或过大。
- 破坏地平线或透视。
- 背景扩展出现 artifact。
- 人脸、手、文字、物体变形。
- 构图变化很小，但生成质量变差。
- 原图已经很好，候选图强行修改。

推荐每张原图至少有 1 个 risky/negative candidate。

## 6. 最终 pair 数与标签分布

### 6.1 Core v2 推荐规模

```text
source images: 2,000
candidates per source: 4
total pairs: 8,000
```

推荐 split：

| split | source images | pairs |
|---|---:|---:|
| train | 1,600 | 6,400 |
| val | 200 | 800 |
| test | 200 | 800 |

划分原则：

- 按 source image 划分，不能同一原图的不同 candidate 同时出现在 train 和 test。
- 同一外部来源、同一拍摄者、同一 URL 的近重复图片尽量放在同一 split。
- test 集保留更高质量人工标注，不允许模型辅助标签直接进入最终 test。

### 6.2 推荐标签分布

理想目标：

| label | 比例 | Core v2 pair 数 |
|---|---:|---:|
| win | 35% | 2,800 |
| tie | 30% | 2,400 |
| lose | 35% | 2,800 |

不要追求完全三等分。真实任务中 `tie` 边界本来就难，30% 已经足够让模型学习“无明显改善”。

每个 split 应尽量保持相似分布：

```text
train: win/tie/lose roughly 35/30/35
val:   win/tie/lose roughly 35/30/35
test:  win/tie/lose roughly 35/30/35
```

### 6.3 每源候选图设计

每个 source 的 4 个 candidate，目标标签大致为：

| candidate | 预期标签 | 说明 |
|---|---|---|
| candidate_1 | win | 明显构图改善 |
| candidate_2 | tie/win | 轻微改善或边界样本 |
| candidate_3 | tie/lose | risky edit，构图可能好但内容/真实感受损 |
| candidate_4 | lose | 明显无改善或更差 |

注意：这是生成目标，不是最终标签。最终标签必须由标注决定。

## 7. 标签体系

### 7.1 主标签

`overall_label`：

| label | 定义 |
|---|---|
| `win` | candidate 相比 source 整体更好，构图改善明确，且内容保真和真实感没有明显抵消收益 |
| `tie` | 没有清晰赢家；或构图略好但内容/真实感损失抵消；或两图差异太弱 |
| `lose` | candidate 相比 source 整体更差；构图无改善，或内容/真实感损坏明显 |

关键原则：

```text
overall_label 不是单独的 composition label。
它是 composition_gain、content_preservation、visual_naturalness 的综合判断。
```

### 7.2 分数标签

`improvement_score`：

| 分数 | 含义 |
|---:|---|
| -2 | 明显更差 |
| -1 | 略差 |
| 0 | 基本持平 |
| 1 | 略好 |
| 2 | 明显更好 |

`composition_gain`：

| 分数 | 含义 |
|---:|---|
| 1 | 构图明显变差 |
| 2 | 构图略差或无效修改 |
| 3 | 构图基本持平 |
| 4 | 构图有改善 |
| 5 | 构图明显改善 |

`content_preservation`：

| 分数 | 含义 |
|---:|---|
| 1 | 主体或语义严重改变 |
| 2 | 重要内容丢失或错误 |
| 3 | 有可见内容变化，但主体大体保留 |
| 4 | 内容基本保留，仅轻微变化 |
| 5 | 内容高度保留 |

`visual_naturalness`：

| 分数 | 含义 |
|---:|---|
| 1 | 严重 artifact，不像真实图像 |
| 2 | 明显不自然或结构错误 |
| 3 | 有轻微 artifact，但整体可接受 |
| 4 | 基本自然 |
| 5 | 非常自然，无明显问题 |

### 7.3 推荐 overall_label 决策规则

人工标注时可以使用以下规则辅助判断：

```text
win:
  composition_gain >= 4
  content_preservation >= 4
  visual_naturalness >= 4
  improvement_score > 0

tie:
  composition_gain around 3
  or improvement_score around 0
  or composition_gain >= 4 but content/naturalness <= 3

lose:
  improvement_score < 0
  or composition_gain <= 2
  or content_preservation <= 2
  or visual_naturalness <= 2
```

这不是硬规则。最终以人工综合判断为准，但标注员需要在 reason 中说明例外原因。

### 7.4 issue_tags

建议保留多标签诊断字段。

正向构图标签：

```text
better_framing
better_subject_placement
better_subject_scale
better_balance
better_negative_space
better_visual_focus
better_leading_lines
better_horizon
better_foreground_background
less_clutter
better_aspect_ratio
```

负向构图标签：

```text
overcropped
subject_too_small
subject_too_large
bad_subject_placement
bad_balance
bad_negative_space
important_region_cut
worse_horizon
worse_visual_focus
more_clutter
awkward_aspect_ratio
```

内容问题：

```text
identity_changed
subject_changed
object_missing
object_added
scene_semantics_changed
text_changed
face_changed
pose_changed
clothing_changed
background_changed_too_much
```

真实感问题：

```text
artifact
distorted_face
distorted_hand
distorted_object
lighting_inconsistent
perspective_error
texture_error
blur_or_smear
unnatural_boundary
watermark_or_text_artifact
```

边界/标注诊断：

```text
minor_difference
ambiguous_preference
composition_content_tradeoff
composition_naturalness_tradeoff
source_already_good
candidate_style_shift
low_confidence_label
```

## 8. 人工标注方案

### 8.1 标注界面

标注界面应固定为 source-candidate 模式：

```text
Left: source image
Right: candidate image
```

每条样本显示：

- source image
- candidate image
- edit_type
- 可选：生成 prompt，不建议默认展示，避免标注员被 prompt 影响
- 放大查看功能

标注员需要填写：

```json
{
  "overall_label": "win/tie/lose",
  "improvement_score": -2,
  "composition_gain": 1,
  "content_preservation": 1,
  "visual_naturalness": 1,
  "issue_tags": [],
  "reason": "",
  "label_confidence": "high/medium/low"
}
```

### 8.2 标注顺序

建议标注员按以下顺序判断：

1. 先看 source，理解主体、场景和原始构图。
2. 再看 candidate，判断候选图做了什么变化。
3. 只评价 candidate 相对 source 的变化，不评价单图绝对美学。
4. 先给 `composition_gain`。
5. 再给 `content_preservation`。
6. 再给 `visual_naturalness`。
7. 最后综合给 `overall_label` 和 `improvement_score`。
8. 选择 issue_tags。
9. 写一句简短 reason。

### 8.3 标注员人数

推荐：

| 数据部分 | 标注人数 |
|---|---:|
| train 普通样本 | 1 人标注 + 抽检 |
| train 难例样本 | 2 人标注 |
| val | 2 人标注，不一致仲裁 |
| test | 3 人标注，多数投票或专家仲裁 |

Pilot 阶段建议所有样本至少 2 人标注，用于校准规范。

Core 阶段建议：

```text
train: 1 annotator + 20% audit
val:   2 annotators + adjudication
test:  3 annotators + adjudication
```

### 8.4 标注一致性

需要统计：

- `overall_label` agreement
- Cohen's kappa 或 Fleiss' kappa
- 各分数的 MAE / correlation
- 哪些 issue_tags 最容易不一致

建议阈值：

| 指标 | 目标 |
|---|---:|
| overall_label raw agreement | >= 70% |
| win/lose agreement | >= 80% |
| tie-related agreement | >= 60% |
| composition_gain 平均差异 | <= 1 |
| content_preservation 平均差异 | <= 1 |
| visual_naturalness 平均差异 | <= 1 |

如果 pilot 阶段达不到这些目标，应先修标注规范，而不是扩大数据。

### 8.5 仲裁规则

如果标注员不一致：

| 情况 | 处理 |
|---|---|
| `win` vs `lose` | 必须专家仲裁 |
| `win` vs `tie` | 检查是否为轻微改善或有内容/真实感损失 |
| `lose` vs `tie` | 检查是否损失是否严重到整体变差 |
| 分数差 >= 2 | 必须复核 |
| `label_confidence=low` | 进入 hard case pool |

仲裁输出应保存：

```json
{
  "annotator_labels": [...],
  "adjudicated_label": "tie",
  "adjudication_reason": "composition is slightly improved but the face is distorted, so no clear overall win"
}
```

## 9. 标注质量控制

### 9.1 Gold Samples

制作 100-200 条 gold samples：

- 明显 win
- 明显 lose
- 明显 tie
- 构图好但内容坏
- 构图好但真实感坏
- 微小变化

每个标注员开始正式标注前，需要通过 gold sample 校准。

建议门槛：

```text
overall_label accuracy >= 75%
win/lose severe mistake <= 5%
```

### 9.2 重复样本

随机插入 5%-10% 重复样本，检测标注员自身一致性。

### 9.3 时间过滤

记录每条样本标注时间：

- 过快样本进入抽检。
- 大量过快且错误率高的标注员需要重训。

### 9.4 模型辅助预检

可以使用当前 Qwen3.5-9B LoRA 模型预标注，但只能作为辅助：

- 不允许直接进入 test 标签。
- 可以帮助排序 hard cases。
- 可以帮助发现明显生成失败样本。
- 可以帮助给 issue_tags 初稿。

### 9.5 数据过滤

以下样本应剔除或单独标记：

- source 或 candidate 图像加载失败。
- candidate 带水印、文字、边框。
- candidate 完全不对应 source。
- candidate 改变主体身份。
- 图像涉及隐私或敏感内容。
- 两图分辨率极低，无法判断。
- 标注员一致性极差且专家也难以判断。

## 10. 数据生成与标注流水线

推荐目录结构：

```text
data/reframejudge_v2/
  source/
  candidates/
  manifests/
    source_manifest.jsonl
    candidate_generation_manifest.jsonl
  annotations/
    raw_annotations.jsonl
    adjudicated_annotations.jsonl
    reframejudge_v2_core.jsonl
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  reports/
    source_quality_report.md
    generation_quality_report.md
    annotation_agreement_report.md
```

### 10.1 Step 1: 收集 source images

输入：

```text
开放授权图片 / 自有图片 / 研究数据图片
```

输出：

```text
source_manifest.jsonl
```

处理：

- 去重和近重复检测。
- 分辨率检查。
- 水印和截图过滤。
- 隐私过滤。
- 场景类型分类。
- source_id 分配。

### 10.2 Step 2: source analysis

使用 VLM 或人工为每张 source 生成构图分析：

```json
{
  "source_id": "rfjv2_src_000001",
  "main_subject": "",
  "scene_type": "",
  "composition_issues": [],
  "safe_edit_directions": [],
  "do_not_change": []
}
```

### 10.3 Step 3: 生成 candidate prompts

每张 source 生成 4 个 candidate prompt：

```text
positive_strong
positive_mild
risky
negative
```

### 10.4 Step 4: 生成 candidate images

记录所有生成元数据：

```json
{
  "candidate_id": "rfjv2_cand_000001_01",
  "source_id": "rfjv2_src_000001",
  "candidate_path": "data/reframejudge_v2/candidates/rfjv2_cand_000001_01.png",
  "generation_model": "",
  "generation_mode": "outpainting",
  "prompt": "",
  "negative_prompt": "",
  "seed": 123,
  "edit_type": "extend_canvas",
  "intended_label": "win",
  "generation_status": "success"
}
```

### 10.5 Step 5: 自动质检

自动检查：

- 图像存在性。
- 分辨率。
- 宽高比。
- 与 source 的 CLIP/DINO 相似度。
- OCR 水印/文字。
- 图像 hash 去重。
- NSFW/敏感内容过滤。

自动质检不能决定最终标签，只决定是否进入人工标注。

### 10.6 Step 6: 人工标注

按照第 8 节标注规范进行。

### 10.7 Step 7: 仲裁与清洗

输出 adjudicated annotations：

```json
{
  "id": "rfjv2_pair_000001",
  "source_image": "...",
  "edited_image": "...",
  "overall_label": "win",
  "improvement_score": 1,
  "composition_gain": 4,
  "content_preservation": 5,
  "visual_naturalness": 4,
  "issue_tags": ["better_subject_placement", "better_negative_space"],
  "reason": "Candidate improves subject placement and preserves the scene.",
  "label_confidence": "high",
  "annotator_count": 3,
  "agreement": 0.67,
  "adjudicated": true
}
```

### 10.8 Step 8: split 构建

按 source_id 划分：

```text
train: 80%
val: 10%
test: 10%
```

保证：

- source 不泄漏。
- 场景比例接近。
- label 分布接近。
- generation_model 分布接近。
- test 中包含足够 hard cases。

## 11. 人工标注成本估算

假设每条 pair 平均 45 秒：

| pair 数 | 单人小时 | 说明 |
|---:|---:|---|
| 1,200 | 15 小时 | Pilot，一人标注 |
| 1,200 * 2 | 30 小时 | Pilot，双人标注 |
| 8,000 | 100 小时 | Core train 单人标注 |
| 8,000 双人全标 | 200 小时 | 成本较高，但质量更好 |

推荐 Core 成本控制方案：

```text
train 6,400: 单人标注 + 20% 抽检
val 800: 双人标注 + 仲裁
test 800: 三人标注 + 仲裁
```

粗略总量：

```text
train: 6,400 + 1,280 audit = 7,680 annotation units
val: 800 * 2 = 1,600 annotation units
test: 800 * 3 = 2,400 annotation units
total: 11,680 annotation units
```

如果每条 45 秒，总计约：

```text
146 annotation hours
```

## 12. 推荐实施路线

### Phase 0: 规范冻结

时间：1-2 天

产出：

- v2 schema
- source manifest schema
- candidate generation manifest schema
- annotation guideline
- issue_tags 列表
- gold sample 初版

### Phase 1: Pilot 300 source / 1,200 pair

时间：1-2 周

目标：

- 跑通 source 收集、生成、自动质检、人工标注、仲裁、split。
- 验证标签一致性。
- 找出 prompt 和生成失败类型。
- 训练一次 Qwen3.5-9B LoRA，看是否优于 v1。

验收标准：

```text
overall_label agreement >= 70%
test Macro-F1 比 v1 不低
tie-related error ratio 下降
```

### Phase 2: Core 2,000 source / 8,000 pair

时间：3-6 周

目标：

- 建立第一个正式 v2 数据集。
- 获得稳定 train/val/test。
- 完成 R2.2 模型训练。
- 完成 v1 vs v2 的消融对比。

验收标准：

```text
Qwen3.5-9B LoRA source-candidate Macro-F1 明显超过 v1
FCDB-like crop/reframe 子集不再明显崩坏
tie F1 提升
```

### Phase 3: Scale 10,000 source / 40,000+ pair

时间：后续扩展

目标：

- 支持更大模型训练。
- 支持公开 benchmark。
- 引入更多生成模型和真实应用样本。

## 13. 我建议现在立刻做什么

下一步不要直接生成 8,000 pair。应该先做 Pilot。

具体行动：

1. 建立 v2 目录结构。
2. 写 source manifest schema 和 candidate manifest schema。
3. 从现有可用图片中筛 300 张 source。
4. 为每张 source 生成 source analysis。
5. 每张 source 生成 4 个 candidate prompt。
6. 先生成 50 source * 4 candidate = 200 pair 的 mini-pilot。
7. 人工标注 200 pair，检查一致性和标签难点。
8. 修正 prompt 和 guideline。
9. 再扩到 300 source / 1,200 pair。

推荐第一批 mini-pilot 场景分布：

| 场景 | source 数 |
|---|---:|
| portrait/person | 10 |
| object/product | 8 |
| indoor | 8 |
| city/building | 8 |
| landscape | 8 |
| food/animal/plant | 4 |
| complex scene | 4 |

## 14. 关键原则

1. **按 source 划分 split，防止泄漏。**

2. **一定要保留失败生成图。**

   失败图不是垃圾，它们是训练 evaluator 判断真实性和内容保真的关键。

3. **不要只做美图偏好。**

   ReFrameJudge 要判断的是“相对原图的重构质量”，不是单图美学分数。

4. **tie 是核心，不是边角料。**

   v1 最大问题就是 tie 边界。v2 必须主动设计 tie 样本。

5. **人工标注优先保证 test。**

   如果预算有限，test 应该最高质量，train 可以接受一定弱标注和抽检。

6. **记录生成过程。**

   每张 candidate 必须知道由什么 prompt、什么模型、什么 edit_type 生成，否则后续无法分析失败来源。

7. **先小规模闭环，再扩大。**

   mini-pilot -> pilot -> core，比直接做大数据更稳。

## 15. 推荐 v2 Core 最终规格

如果只保留一个明确版本目标，我建议：

```text
Dataset name: ReFrameJudge-v2-Core
Source images: 2,000
Candidate images per source: 4
Total pairs: 8,000
Split: 6,400 train / 800 val / 800 test
Label distribution: win 35% / tie 30% / lose 35%
Annotation:
  train: 1 annotator + 20% audit
  val: 2 annotators + adjudication
  test: 3 annotators + adjudication
Primary model target:
  Qwen3.5-9B LoRA source-candidate
Primary metrics:
  Accuracy, Macro-F1, per-class F1, regression MAE/Pearson/Spearman
Primary improvement target:
  tie F1 and FCDB-like composition subset performance
```

这是一个现实、可控、并且能显著推进 ReFrameJudge 的数据建设目标。
