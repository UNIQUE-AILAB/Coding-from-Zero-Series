# Coding from Zero Series

**从论文到代码，复现深度学习中的经典架构与算法。**

由[华中科技大学联创团队 AI 组（UNIQUE AI LAB）](https://github.com/UNIQUE-AILAB)维护。

## 关于这个仓库

读懂一个模型，可以从亲手实现它开始。

本仓库收录深度学习经典架构与算法的学习型复现，使用 PyTorch 张量运算和基础网络层搭建核心模块，逐步理解模型结构、数学原理与训练过程之间的对应关系。

我们希望通过这些实现，回答一些具体的问题：注意力是怎样计算的？图像如何变成 token？VAE 为什么需要重参数化？扩散模型如何从噪声生成图像？

这里的“从零”指亲手实现关键结构与算法，使用 PyTorch 提供的自动求导、基础算子和优化器。各实现的覆盖范围见下表；代码以理解核心机制为目标，实验规模与设置可能与原论文不同。

## 已有内容

| 架构 / 算法 | 代码入口 | 学习重点 | 当前覆盖 |
| --- | --- | --- | --- |
| Transformer | [Transformer.ipynb](./Transformer.ipynb) | 位置编码、多头注意力、编码器与解码器、注意力掩码 | 结构实现与前向验证 |
| Vision Transformer（ViT） | [ViT.ipynb](./ViT.ipynb) | Patch Embedding、Transformer Encoder、分类头 | 结构实现与前向验证 |
| U-Net | [UNet.ipynb](./UNet.ipynb) | 编码器与解码器、上下采样、跳跃连接 | 结构实现与 MNIST 分类示例 |
| Variational Autoencoder（VAE） | [VAE/vae.py](./VAE/vae.py) | 编码器、重参数化、解码器、重建损失与 KL 散度 | 核心模块与损失函数 |
| DDPM | [ddpm_diffusion_model.ipynb](./ddpm_diffusion_model.ipynb) | 前向加噪、时间步嵌入、噪声预测、反向采样 | MNIST 训练与采样示例 |
| Diffusion Transformer（DiT） | [DiT/DiT_diffusion_model.ipynb](./DiT/DiT_diffusion_model.ipynb) | Patch 表示、时间条件调制、Transformer 去噪网络 | MNIST 像素空间中的简化训练与采样示例 |

U-Net 示例通过全局平均池化将输出用于分类，便于观察训练过程。DiT 示例直接处理图像像素，用于理解 Transformer 去噪网络与扩散流程的衔接。

关于 DiT 与 DDPM 的联系，可以结合阅读：[DiT 和 DDPM 的关系说明](./DiT/DiT和DDPM关系说明.md)。

## 建议怎样阅读？

可以根据兴趣选择一条路线：

- **注意力与视觉模型**：Transformer → ViT → DiT。
- **生成模型**：VAE → U-Net → DDPM → DiT。

阅读每个实现时，建议先确认模型的输入、输出和任务，再逐个理解模块，跟踪张量形状，最后结合损失函数与训练、采样流程，把整个算法串起来。

运行实验时，可以先使用小批量数据和较小的模型检查流程，再调整训练规模。尝试改变一个设计，观察结果怎样变化，也是理解模型的好方法。

## 开始使用

克隆仓库：

```bash
git clone https://github.com/UNIQUE-AILAB/Coding-from-Zero-Series.git
cd Coding-from-Zero-Series
```

在 Python 环境中，按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/)安装适合操作系统与计算设备的 `torch` 和 `torchvision`，再安装 Notebook 与可视化工具：

```bash
python -m pip install jupyterlab matplotlib
jupyter lab
```

打开感兴趣的 `.ipynb` 文件，阅读说明并按顺序运行相应单元格。包含 MNIST 训练的示例会下载数据；运行训练前可按设备情况调整批量大小、模型规模和训练轮数。

`VAE/vae.py` 提供可导入的模型与损失函数，训练循环和数据加载需要自行编写。

## 一起完善

欢迎通过 Issue 或 Pull Request 交流和贡献：

- 补充经典架构、算法与关键模块的实现。
- 修正公式、代码或解释中的问题。
- 添加张量形状说明、结构图和阅读笔记。
- 补充训练、评估、可视化与对比实验。

新增实现时，建议说明参考论文与代码来源、依赖环境、运行方法，以及相对原论文做出的简化。若包含实验结果，请同时记录配置与评估方式，方便他人复现和讨论。

## 参考论文

- Transformer — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- ViT — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- U-Net — [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- VAE — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- DDPM — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- DiT — [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

更多学习资料与技术笔记，见 [AI 入门指北](https://guidebook.hustunique.com/docs/AI%E5%85%A5%E9%97%A8%E6%8C%87%E5%8C%97)和 [AI 组主页](https://unique-ailab.github.io/)。
