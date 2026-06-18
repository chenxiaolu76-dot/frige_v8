# 智能冰箱视觉管理系统

> **GitHub**: [https://github.com/chenxiaolu76-dot/frige_v8](https://github.com/chenxiaolu76-dot/frige_v8)

基于数字图像处理与 YOLOv8 的课程期末项目，面向“冰箱食材管理”这一明确生活场景，交付一套可运行的图像处理系统 App。系统以冰箱内部照片为输入，完成图像预处理、食材检测、数量统计、面积估算、空间定位、热力图分析，并进一步输出补货提醒与空间整理建议。

## 项目定位

本项目不是一个通用目标检测 Demo，而是一套面向实际应用场景的智能冰箱视觉管理系统。它对应课程要求中的三层目标：

1. 基础要求：实现一套完整的数字图像处理系统或 App。
2. 难度系数 1.3：在任务 1 基础上实现目标检测与定位。
3. 难度系数 1.5：增加实际应用，场景明确，并提供至少 2 个应用功能。

本项目选择的实际场景是“家庭智能冰箱食材管理”，当前已经落地的应用功能包括：

1. 食材识别与库存统计。
2. 食材位置分析与冰箱区域热力图。
3. 补货提醒。
4. 空间整理建议。

## 当前实现能力

### 1. 基础图像处理

系统支持以下基础图像处理功能：

1. 图像读取与显示。
2. 灰度化。
3. 对比度增强。
4. 高斯/中值去噪。
5. Canny 边缘检测。
6. 形态学处理。

### 2. 目标检测与定位

系统使用 YOLOv8 食材检测模型，当前已接入 12 类新训练权重（`model/fridge_12class_best.pt`）：

> **数据集来源说明**: 12 类模型的训练数据来自以下公开数据集的组合，覆盖了冰箱常见食材场景。

1. apple
2. banana
3. orange
4. tomato
5. cucumber
6. potato
7. onion
8. carrot
9. green_pepper
10. egg
11. milk
12. bread

#### 📦 数据集来源

12 类模型的训练数据来自以下公开数据集的组合：

| 阶段 | 数据集 | 来源 | 覆盖类别 | 数据规模 |
|------|--------|------|----------|----------|
| **Stage 1** | **LVIS Fruits & Vegetables**（果蔬检测子集） | [GitHub](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) | apple, banana, orange, tomato, cucumber, potato, onion, carrot, green_pepper | 训练 4,869 / 验证 1,055 / 测试 91 |
| **Stage 2** | **LVIS Fruits & Vegetables** + **Refrigerator Contents**（冰箱内容物数据集） | LVIS: 同上 · Refrigerator: [Kaggle](https://www.kaggle.com/datasets/surendraallam/refrigerator-contents) | 全部 12 类（新增 egg, milk, bread） | 训练 5,670 / 验证 1,156 / 测试 192 |

项目包含两个模型权重，对应不同的训练阶段：

- **`model/fresh_stage1_best.pt`**（9 类第一版模型）：仅使用 Stage 1 的 LVIS Fruits & Vegetables 数据集训练，覆盖 9 类果蔬（apple ~ green_pepper），是第一阶段的初始模型。
- **`model/fridge_12class_best.pt`**（12 类当前模型）：采用**两阶段训练策略**——先加载 Stage 1 的预训练权重 `best_stage1.pt`，再合并 Refrigerator Contents 数据集（含 egg、milk、bread 等冰箱场景图片）进行微调，最终产出 12 类模型。

两个数据集的详细信息：

- **LVIS Fruits & Vegetables**：从 LVIS 大规模实例分割数据集中过滤出果蔬子集，转为 YOLO 格式。共 8,221 张图片，涵盖 9 类常见果蔬。该仓库同时提供了 YOLOv8 微调基线模型。
- **Refrigerator Contents**（[Kaggle](https://www.kaggle.com/datasets/surendraallam/refrigerator-contents)）：7 个冰箱常见食材类别（Banana、Bread、Eggs、Milk、Potato、Spinach、Tomato），每类约 150 张图片，~1,050 张总计。原始标注为 Caffe 格式，经转换为 Pascal VOC 后再转为 YOLO 格式。合并时排除了 Spinach 类别。

> 合并过程记录在 `smart-fridge-final-train-bundle/data/merged_12class_yolo/merge_report.yaml` 中。训练脚本见 `smart-fridge-final-train-bundle/scripts/train_stage2_yolo.py`。

系统输出每个目标的类别、置信度、边界框和中心点，可直接用于定位与后续分析。

### 3. 面积估算

系统会根据检测框估算食材投影面积。当前部署的 12 类模型默认使用固定像素系数完成面积估算，后续可扩展接入标定参考物以提升换算精度。

### 4. 实际应用功能

围绕“智能冰箱管理”场景，系统当前提供以下应用层功能：

1. 库存清单生成：统计每类食材数量与面积。
2. 区域位置分析：判断食材位于上层/中层/下层、左侧/中间/右侧。
3. 热力图可视化：展示食材在冰箱中的集中区域。
4. 补货提醒：依据默认库存阈值提示需要优先补充的食材。
5. 空间整理建议：依据区域拥挤程度和面积集中情况给出整理建议。

## 系统界面

当前系统以 `Streamlit` 形式交付，属于可运行的轻量级 App，适合课程验收、答辩演示和功能联调。页面分为以下几个部分：

1. 图像总览。
2. 统计分析。
3. 空间布局。
4. 管理建议。
5. 结果导出。

这套形态已经满足“做一个 App”的课程要求，因为它是一个可交互、可运行、可展示结果、可导出数据的软件系统，而不只是脚本。

## 目录结构

```text
smart-fridge-vision/
├─ README.md
├─ requirements.txt
├─ app/
│  ├─ main.py
│  ├─ core/
│  ├─ ui/
│  ├─ utils/
│  └─ models/
├─ config/
├─ data/
├─ docs/
├─ scripts/
├─ tests/
├─ train_bundle/
└─ 智能冰箱/
```

核心目录说明：

1. `app/core/`：预处理、检测、面积估算、位置分析、热力图与库存管理核心逻辑。
2. `app/ui/`：Streamlit App 界面。
3. `config/`：系统参数、类别映射、冰箱区域划分。
4. `tests/`：关键模块测试。
5. `智能冰箱/`：新训练模型与训练记录。

## 运行方法

建议使用已经配置好的 Conda 环境：

```powershell
conda activate D:\conda-envs\smart-fridge-vision-py311
cd D:\learn\数字图像处理\smart-fridge-vision
streamlit run app\ui\streamlit_app.py
```

## 测试状态

当前核心模块测试已通过：

1. 面积估算测试。
2. 位置分析测试。
3. 图像预处理测试。

在 `D:\conda-envs\smart-fridge-vision-py311` 环境中运行结果为 `17 passed`。

## 答辩时建议如何介绍

可以用下面这条主线来讲：

1. 本项目首先完成了数字图像处理的基础功能。
2. 在此基础上引入 YOLOv8，实现了食材目标检测与定位，满足任务 1 的增强要求。
3. 然后把检测结果放进真实应用场景“智能冰箱管理”中，继续做库存统计、面积估算、空间定位和热力图分析。
4. 最后进一步输出补货提醒和整理建议，使系统从“会识别”升级为“能辅助管理”。

## 后续可扩展方向

如果答辩老师继续追问扩展性，可以说明后续可接入：

1. 过期时间管理。
2. 菜谱推荐。
3. 实例分割替代检测框面积估算。
4. 深度估计辅助体积测量。
5. 摄像头实时采集或嵌入式硬件联动。

## 一句话总结

本项目交付的是一套面向智能冰箱场景的图像处理 App，不仅完成了课程要求的基础图像处理与目标检测，还通过补货提醒和空间整理建议，将系统提升为一个具有明确实际应用价值的视觉管理系统。
