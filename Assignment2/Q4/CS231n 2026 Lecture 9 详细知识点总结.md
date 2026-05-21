# 一、知识点总结）

## 1. Transformer 回顾
### 1.1 Transformer 结构
- **Encoder（编码器）**：多层堆叠，包含：
  - **多头自注意力（Multi-head Self-Attention）**
  - **前馈网络（MLP）**
  - **层归一化（Layer Norm）**
  - **残差连接（Residual）**
- **Decoder（解码器）**：
  - **掩码多头自注意力（Masked Self-Attention）**：只能看前面的词，防止偷看未来信息。
  - **多头注意力（Encoder-Decoder Attention）**：从编码器输出中“读取信息”。
- **位置编码（Positional Encoding）**：给序列加位置信息，因为 Transformer 本身不感知顺序。

### 1.2 三种序列建模方式对比
1. **RNN**：
   - 优点：天然处理序列，记忆长依赖。
   - 缺点：串行计算，**不能并行**，慢。
2. **CNN（卷积）**：
   - 优点：可并行、快。
   - 缺点：感受野有限，**长序列需要堆很多层**。
3. **Self-Attention（Transformer）**：
   - 优点：每个位置**直接看全局**，并行度高。
   - 缺点：复杂度 $O(n^2)$，序列长时计算量大。

---

## 2. ViT（Vision Transformer）：把图片当句子
### 2.1 核心思想
把一张图片切成很多小方块（patch），当成“单词”，用 Transformer 做图像分类。

### 2.2 步骤（详细）
1. **切分图像**：
   - 输入：$224 \times 224 \times 3$
   - 切成 $16 \times 16$ 的小块 → 共 $14 \times 14 = 196$ 个 patch。
2. **拉直+线性投影**：
   - 每个 patch：$16 \times 16 \times 3 = 768$ 维
   - 用线性层映射到 $D$ 维向量：
     $$z_i = W \cdot \text{patch}_i + b$$
3. **加位置编码**：
   - 给每个 patch 向量加位置信息：
     $$z_i' = z_i + E_{\text{pos}}$$
4. **加分类token（class token）**：
   - 在序列最前面加一个可学习向量：
     $$z_0 = \text{class\_token}$$
5. **送入 Transformer Encoder**：
   - 所有 patch + class token 一起做自注意力。
6. **输出分类**：
   - 用 class token 输出类别：
     $$y = \text{Linear}(z_0)$$

### 2.3 ViT 小结
- **不用卷积，全靠注意力**。
- 适合大模型、大数据，小数据不如 CNN。

---

## 3. Transformer 改进（Tweaks）
### 3.1 Pre-Norm（常用）
- 把 Layer Norm 放在**残差内部**，训练更稳：
  $$x \to \text{LN}(x) \to \text{Attn} \to +x$$

### 3.2 RMSNorm
- 简化版归一化，不用均值，只用方差：
  $$y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma_i$$

### 3.3 SwiGLU（MLP 改进）
- 替换传统激活，提升表达力：
  $$y = (\sigma(xW_1) \odot xW_2) W_3$$

### 3.4 MoE（混合专家）
- 每个层放多个 MLP（专家），每次只激活少数：
  - 参数量暴涨，但**计算量可控**。
  - 超大模型（如 GPT-4）常用。

---

## 4. 四大视觉任务对比
### 4.1 图像分类（Classification）
- 输出：整张图一个标签（如 cat）。
- 无空间信息。

### 4.2 语义分割（Semantic Segmentation）
- 输出：**每个像素一个类别**（天空、草地、猫）。
- 不区分个体：两只狗都标 dog。

### 4.3 目标检测（Object Detection）
- 输出：**物体框 + 类别**。
- 区分个体：两只狗各一个框。

### 4.4 实例分割（Instance Segmentation）
- 输出：**每个物体的像素轮廓 + 类别**。
- 最精细：能抠出每只动物的形状。

---

## 5. 语义分割（Semantic Segmentation）
### 5.1 早期方法：滑动窗口（Sliding Window）
- 对每个像素，取周围小图块 → CNN 分类。
- **致命问题**：窗口重叠太多，重复计算，**极慢**。

### 5.2 全卷积网络 FCN（Fully Convolutional Network）
核心：**全是卷积，不用全连接层**，支持任意尺寸输入。
- 流程：
  1. 卷积+池化 → 不断下采样（变小、变深）。
  2. 上采样 → 恢复原图大小。
- 下采样：
  - 池化、步长卷积：
    $$H_{\text{out}} = \frac{H_{\text{in}} - f + 2p}{s} + 1$$
- 上采样三种方式：
  1. **最近邻插值**：简单、快，但边缘模糊。
  2. **反池化（Unpooling）**：记录池化最大值位置，还原。
  3. **转置卷积（Transposed Convolution，可学习上采样）**：
     - 输入 $2\times2$ → 输出 $4\times4$。
     - 核心：**每个输入像素 × 卷积核 → 错位相加**。
     $$O = \text{ConvTranspose2d}(I, W, s=2, p=1)$$

### 5.3 U-Net（医学分割标杆）
- **U 型对称结构**：
  1. **编码器（左，下采样）**：
     - 多次卷积 + 池化 → 特征变小、变深。
     - 作用：抓全局语义，但**丢细节**。
  2. **解码器（右，上采样）**：
     - 转置卷积上采样 → 特征变大。
  3. **跳跃连接（Skip Connection）**：
     - 把编码器对应层的**高分辨率特征**直接拼接到解码器。
     - 公式：
       $$\text{concat}(x_{\text{enc}}, x_{\text{dec}})$$
- 优点：**细节+语义都保住**，小数据也好用。

---

## 6. 目标检测（Object Detection）
### 6.1 单目标检测：分类 + 回归
- 输入图片 → CNN → 特征向量。
- 两个输出头：
  1. 分类：Softmax 输出类别概率。
     $$s_c = \text{Softmax}(W_c \cdot f + b_c)$$
  2. 回归：输出框坐标 $(x,y,w,h)$。
     $$b = W_b \cdot f + b_b$$
- 损失：
  $$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}}$$

### 6.2 多目标检测：输出数量不固定
- 一张图可能 0/1/2/… 个物体，输出长度可变。
- 早期笨办法：**滑动窗口 + CNN 分类**。
  - 几千个窗口，每个都跑 CNN → **超级慢**。

### 6.3 R-CNN（开山之作）
步骤：
1. **选择性搜索（Selective Search）**：找约 2000 个候选框。
2. **裁剪 + 缩放到 224×224**。
3. **每个框单独跑 CNN 提特征**。
4. **SVM 分类 + 框回归修正**。
- 问题：2000 次 CNN 前向 → **极慢**。

### 6.4 Fast R-CNN：只跑一次 CNN
核心改进：**整张图先跑 CNN，再在特征图上裁剪**。
1. 整张图 → CNN → 一张大特征图。
2. 候选框 → 在特征图上对应位置裁剪。
3. **ROI Pooling**：把不同大小特征 → 固定尺寸。
4. 一个网络同时输出：类别 + 框偏移。
- 速度提升几十倍。

### 6.5 Faster R-CNN：用 CNN 找候选框（RPN）
核心：**Region Proposal Network（RPN）替代 Selective Search**。
1. 整张图 → CNN → 特征图。
2. **Anchor（锚框）**：特征图每个格子预放 9 个不同大小/比例框。
3. RPN 做两件事：
   - 分类：框里有没有物体（二分类）。
   - 回归：微调锚框位置。
4. 输出几百个高质量候选框 → 送 Fast R-CNN 分类。
- 整个模型**端到端、最快、工业界主流**。

---

## 7. 模型可视化（Visualization）
### 7.1 卷积核可视化
- AlexNet/VGG/ResNet 的第一层卷积核：
  - AlexNet：边缘、颜色块、简单纹理。
  - VGG：更细的线条、纹理。
  - ResNet：更复杂、抽象的纹理。
- 卷积核是**训练出来的，不是固定的**。

### 7.2 显著性图（Saliency Map）
- 看：**哪些像素对分类最重要**。
- 计算：求类别分数对输入像素的梯度：
  $$S = \frac{\partial s_c}{\partial x}$$
- 亮区：模型最关注的地方（如狗的脸）。

### 7.3 CAM（类激活图）
- 只用最后一层卷积 + 分类权重：
  $$M_c = \sum_k w_{c,k} A_k$$
- 局限：只能用最后一层，且依赖全局平均池化。

### 7.4 Grad-CAM（最常用）
- 用**梯度**计算权重，可用于任意层：
  1. 算梯度：$\frac{\partial s_c}{\partial A_k}$
  2. 求均值权重：$\alpha_k = \frac{1}{HW}\sum_{h,w}\frac{\partial s_c}{\partial A_{h,w,k}}$
  3. 加权求和：$M_c = \text{ReLU}\left(\sum_k \alpha_k A_k\right)$
- 优点：**任意层、不用改模型、效果好**。

---

# 二、学习总结报告

---

# 课程：CS231n Lecture 9
# 主题：Detection、Segmentation、Visualization、Transformer
# 学习总结

## 一、课程概述
本节课系统讲解了**Transformer、ViT、语义分割、目标检测、模型可视化**五大核心主题，从基础模型结构到工业界主流算法，再到模型解释性方法，构建了现代计算机视觉任务的完整知识体系。课程重点围绕“**从CNN到Transformer**”“**从分类到分割/检测**”两条主线展开，是深度学习视觉方向的核心必修课。

---

## 二、核心知识点总结

### 1. Transformer 基础
#### （1）结构组成
- Encoder：多头自注意力 + MLP + 层归一化 + 残差连接
- Decoder：掩码自注意力 + 多头注意力 + MLP
- 位置编码：补充序列顺序信息，弥补注意力无顺序感知的缺陷。

#### （2）三种序列建模对比
- RNN：串行、不可并行、长依赖强，复杂度 $O(n)$
- CNN：可并行、局部感受野，复杂度 $O(n)$
- Self-Attention：全局建模、可并行，复杂度 $O(n^2)$

### 2. ViT（Vision Transformer）
#### （1）核心思想
把图像切成 $16\times16$ 的 patch，当成“单词”，用 Transformer 做图像分类，**完全抛弃卷积**。

#### （2）完整流程
1. 图像切分：$224\times224 \to 14\times14=196$ 个 patch
2. 线性投影：每个 patch（768维）→ $D$ 维向量
   $$z_i = W \cdot \text{patch}_i + b$$
3. 位置编码：$z_i' = z_i + E_{\text{pos}}$
4. 加分类token：序列首位添加可学习向量
5. Transformer Encoder 全局建模
6. 分类输出：用 class token 预测类别

### 3. Transformer 实用改进
- Pre-Norm：归一化放残差内部，训练更稳定
- RMSNorm：简化归一化，公式：
  $$y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma_i$$
- SwiGLU：改进 MLP 激活，提升表达力
  $$y = (\sigma(xW_1) \odot xW_2) W_3$$
- MoE（混合专家）：多专家 MLP，每次激活少数，参数量暴涨、计算量可控

### 4. 语义分割（Semantic Segmentation）
#### （1）任务定义
给每个像素打类别标签，输出和原图尺寸相同的分割图。

#### （2）FCN：全卷积网络
- 核心：无全连接层，支持任意尺寸输入
- 下采样：池化/步长卷积，公式：
  $$H_{\text{out}} = \frac{H_{\text{in}} - f + 2p}{s} + 1$$
- 上采样三方法：最近邻插值、反池化、**转置卷积（可学习上采样）**
  $$O = \text{ConvTranspose2d}(I, W, s=2, p=1)$$

#### （3）U-Net：医学分割标杆
- U 型对称结构：编码器（下采样）+ 解码器（上采样）
- 跳跃连接：拼接编码器高分辨率特征，公式：
  $$\text{concat}(x_{\text{enc}}, x_{\text{dec}})$$
- 优势：兼顾全局语义与细节，小数据场景效果极佳

### 5. 目标检测（Object Detection）
#### （1）任务定义
输出物体边界框 + 类别，区分不同物体实例。

#### （2）演进路线
1. 滑动窗口：暴力穷举，**极慢**
2. R-CNN：选择性搜索 + 逐个框 CNN，慢
3. Fast R-CNN：整张图一次 CNN + ROI Pooling，快
4. Faster R-CNN：RPN 替代选择性搜索，端到端、工业界主流

#### （3）RPN 核心
- 锚框（Anchor）：特征图每个格子预放 9 个框
- 双任务：
  - 分类：框内是否有物体
  - 回归：微调框位置，输出 $(dx, dy, dw, dh)$

#### （4）单目标检测损失
$$\mathcal{L} = \mathcal{L}_{\text{cls}}(\text{Softmax}) + \mathcal{L}_{\text{box}}(\text{L2})$$

### 6. 模型可视化（Visualization）
#### （1）卷积核可视化
- 不同网络卷积核差异：AlexNet（简单纹理）→ VGG（细腻纹理）→ ResNet（抽象特征）
- 卷积核**训练得到，非固定**

#### （2）显著性图（Saliency Map）
- 计算像素梯度：
  $$S = \frac{\partial s_c}{\partial x}$$
- 亮区为模型关注的关键区域

#### （3）Grad-CAM（最常用）
1. 梯度求均值：$\alpha_k = \frac{1}{HW}\sum_{h,w}\frac{\partial s_c}{\partial A_{h,w,k}}$
2. 加权求和：$M_c = \text{ReLU}\left(\sum_k \alpha_k A_k\right)$
- 可用于任意层，无需修改模型，效果最优

---

## 三、学习收获
1. 掌握了 Transformer 的核心结构与改进方法，理解了 ViT 如何将 NLP 模型迁移到视觉任务。
2. 理清了语义分割从 FCN 到 U-Net 的演进逻辑，掌握了上采样、跳跃连接的核心作用。
3. 理解了目标检测从滑动窗口到 Faster R-CNN 的技术迭代，明确了 RPN、Anchor、ROI Pooling 的核心原理。
4. 学会了模型可视化方法（卷积核、显著性图、Grad-CAM），能解释模型的决策依据，提升模型可解释性。

---

## 四、难点与理解
1. **转置卷积**：反向卷积、错位相加、实现可学习上采样，是分割任务的核心。
2. **RPN 锚框机制**：预定义多尺度框、双任务输出，解决多目标检测的输出不固定问题。
3. **Grad-CAM 梯度计算**：通过梯度加权特征图，实现任意层的可视化，公式推导需重点理解。
