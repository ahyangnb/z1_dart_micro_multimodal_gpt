# 显微镜级多模态 GPT

这个目录是一个独立的、最原始级别的多模态 GPT 学习样例。

它不读取真实照片，不依赖 PyTorch、TensorFlow、NumPy 或任何图片库。代码把“图片”压缩成 4x4 黑白像素，再把图片切成 2x2 patch token，和文字 token 放进同一个 causal self-attention 序列里，最后像 GPT 一样逐字生成答案。

运行：

```bash
dart run z1_dart_micro_multimodal_gpt/micro_multimodal_gpt.dart
```

指定图片：

```bash
dart run z1_dart_micro_multimodal_gpt/micro_multimodal_gpt.dart tree
dart run z1_dart_micro_multimodal_gpt/micro_multimodal_gpt.dart house
dart run z1_dart_micro_multimodal_gpt/micro_multimodal_gpt.dart cross
```

快速验证：

```bash
Z1_DART_MM_GPT_STEPS=20 dart run z1_dart_micro_multimodal_gpt/micro_multimodal_gpt.dart sun
```

默认训练 200 步，通常几秒内可以跑完。步数太少时模型可能答错，这正好可以观察“还没学会”的状态。

## 它实现了什么

这个小引擎训练一个极小的自回归模型，让它学会：

```text
输入：一张 4x4 玩具图片 + 问题“图里有什么”
输出：太阳 / 树 / 房子 / 十字
```

程序会依次打印：

1. 原始 4x4 像素图片。
2. 图片切成的 2x2 patch token。
3. 问题文本切成的字符 token。
4. 图片 token 和文字 token 拼成的 causal sequence。
5. `<bos>` 位置对前文的 attention 权重。
6. `lm_head + softmax` 给出的下一个字概率。
7. 自回归生成出来的答案。

## 最小算法骨架

| 代码里的东西 | 多模态 GPT / VLM 里的概念 | 意思 |
| --- | --- | --- |
| `ImageGrid.rows` | 原始图像 | 这里用 0/1 像素代替真实图片 |
| `patches()` | image patching | 把图片切成小块，类似 ViT 的 patch |
| `vision_proj` | 视觉投影层 | 把每个 patch 的像素变成 embedding |
| `TinyTokenizer` | text tokenizer | 把问题和答案切成字符 token |
| `wme` | modality embedding | 告诉模型这个 token 来自图片还是文字 |
| `wpe` | position embedding | 告诉模型 token 在序列里的位置 |
| `forward()` | GPT 前向传播 | self-attention + MLP + lm head |
| `answerTargets()` | 训练目标 | 用上一个 token 预测下一个答案 token |
| `generate()` | 自回归解码 | 每次预测一个字，直到 `<eos>` |

## 和纯文字 GPT 的关系

根目录里的 GPT 学的是：

```text
文字 token -> embedding -> attention -> 预测下一个文字 token
```

这个目录只多加了一步：

```text
图片 -> patch token -> embedding
```

之后图片 token 和文字 token 被拼进同一个序列：

```text
img0 img1 img2 img3 图 里 有 什 么 <sep> <bos>
```

模型仍然做同一件事：

```text
看前面的 token，预测下一个 token。
```

所以多模态 GPT 的核心直觉可以先这样理解：

```text
不是让文字模型“神秘地看懂图片”，
而是先把图片也变成 token，再让 attention 学会在图片 token 和文字 token 之间建立关系。
```

## 它和真实多模态大模型的差距

这个样例故意保留最小骨架，没有实现：

- 真实图片读取、缩放、归一化
- 大型视觉 encoder
- 大规模图文对齐数据
- 预训练、指令微调、RLHF
- 高效张量计算、GPU、batch 训练
- 多层 transformer、KV cache、量化部署

这些不是漏做，而是刻意不做。第一步先看清楚“图片如何进入 GPT”。

## 建议怎么玩

- 把 `demoImages` 里的 4x4 图案改掉，观察模型是否还能学会。
- 把 `defaultSteps` 调小，看模型从乱猜到会答的过程。
- 在 `buildTrainingExamples()` 里加新的问题模板。
- 把 `patchSize` 改成 1，观察图片 token 变多以后训练速度和 attention 的变化。
- 给 `ImageGrid` 加一个新类别，再看需要多少训练步才能稳定生成。
