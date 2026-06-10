# 智能冰箱食材识别与空间管理系统 AI 提示词

## 使用说明

这份提示词用于让 AI 协助完成项目设计、代码实现、界面开发、算法模块拆分、文档整理和实验报告撰写。使用时可以直接整体复制，也可以按模块拆开使用。

建议在使用时明确要求：

1. 代码尽量模块化
2. 优先使用 Python 技术栈
3. 优先保证可运行，再考虑高级扩展
4. 默认项目当前不实现新鲜度判断，作为后续扩展功能保留

---

## 一、项目背景提示词

你现在要协助完成一个大学《数字图像处理》课程期末项目。项目名称为：

`基于图像处理与 YOLOv8 的智能冰箱食材识别与空间管理系统`

项目面向日常饮食场景，核心目标是让用户拍摄冰箱内部图片后，系统自动完成以下任务：

1. 食材识别与定位
2. 食材数量统计
3. 食材面积/体积估算
4. 食材摆放位置分析
5. 冰箱空间热力图生成

项目主体是纯软件系统，重点体现数字图像处理、目标检测、图像测量和空间分析能力。机器人或机械臂只是后续扩展方向，本期不作为核心实现内容。

当前版本不把“食品新鲜度判断”作为核心功能，这部分只保留为后续扩展。

---

## 二、项目目标提示词

请围绕以下目标帮助我完成项目：

1. 用 Python 构建一个可运行的智能冰箱食材管理系统
2. 使用 OpenCV 实现基础图像处理功能
3. 使用 YOLOv8 实现食材检测与定位
4. 基于检测结果实现食材数量统计
5. 基于参考物体法或图像区域分析实现食材面积估算
6. 基于冰箱空间分区规则实现食材位置分析
7. 生成冰箱热力图，展示各区域食材分布密度
8. 最终做成一个可视化交互系统，优先使用 Streamlit

---

## 三、技术栈约束提示词

请严格按照以下技术栈给出设计和代码：

- Python 3.10 或 3.11
- OpenCV
- NumPy
- scikit-image
- matplotlib
- Pandas
- PyYAML
- Ultralytics YOLOv8
- Streamlit

可扩展但当前不作为必须项：

- MiDaS 或 ZoeDepth
- python-docx
- pyserial

不要默认引入复杂前端框架，不要优先使用 Flask、Django、Vue、React 这类技术，除非我明确要求。

---

## 四、项目结构提示词

请围绕以下项目结构生成代码、设计模块或补全文档：

```text
smart-fridge-vision/
├─ README.md
├─ requirements.txt
├─ app/
│  ├─ main.py
│  ├─ ui/
│  │  ├─ streamlit_app.py
│  │  └─ components.py
│  ├─ core/
│  │  ├─ image_preprocess.py
│  │  ├─ detector.py
│  │  ├─ segmenter.py
│  │  ├─ area_estimator.py
│  │  ├─ position_analyzer.py
│  │  ├─ heatmap_generator.py
│  │  └─ inventory_manager.py
│  ├─ utils/
│  │  ├─ image_io.py
│  │  ├─ config_loader.py
│  │  ├─ draw.py
│  │  └─ logger.py
│  └─ models/
│     └─ yolo/
├─ config/
│  ├─ classes.yaml
│  ├─ fridge_regions.yaml
│  └─ system.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ samples/
│  └─ outputs/
├─ notebooks/
│  ├─ data_exploration.ipynb
│  └─ algorithm_test.ipynb
├─ scripts/
│  ├─ train_yolo.py
│  ├─ eval_yolo.py
│  └─ export_results.py
├─ docs/
│  ├─ project_plan.md
│  ├─ api_notes.md
│  └─ report_assets/
└─ tests/
   ├─ test_preprocess.py
   ├─ test_area_estimator.py
   └─ test_position_analyzer.py
```

生成内容时请遵守这个目录结构，不要随意改动主目录分层。

---

## 五、功能约束提示词

当前项目必须完成的核心功能：

1. 图像上传与显示
2. 基础图像处理
3. YOLOv8 食材识别与定位
4. 食材数量统计
5. 食材面积估算
6. 食材摆放位置分析
7. 冰箱热力图生成
8. 可视化结果展示

当前项目暂不作为核心实现的功能：

1. 食品新鲜度判断
2. 保质期识别
3. 菜谱推荐
4. 烹饪动作识别
5. 机械臂控制
6. 深度估计法体积测量

这些功能可以在方案说明里作为扩展项保留，但不要默认写成当前必须实现的内容。

---

## 六、代码生成提示词

当你为这个项目生成代码时，请遵守以下要求：

1. 优先生成可运行的最小版本
2. 每个模块职责清晰，避免把所有逻辑写进一个文件
3. 函数命名清晰，尽量使用英文
4. 适当添加注释，但不要过多
5. 如果是 Streamlit 页面，保持结构简洁，便于演示
6. 如果是算法模块，优先保证输入输出明确
7. 不要为了“高级感”引入不必要的复杂依赖

如果你要生成代码，请优先从以下模块开始：

1. `app/ui/streamlit_app.py`
2. `app/core/image_preprocess.py`
3. `app/core/detector.py`
4. `app/core/position_analyzer.py`
5. `app/core/heatmap_generator.py`
6. `app/core/inventory_manager.py`
7. `app/core/area_estimator.py`

---

## 七、面积估算功能提示词

请特别注意，食材面积估算是本项目的重要亮点之一。

目标：

1. 不只是判断食材类别和个数
2. 还要估算食材的投影面积
3. 用于区分完整食材和部分食材，例如半个洋葱和一个洋葱

建议优先采用：

`参考物体法`

即在图像中放入一个已知尺寸的参考物体，例如硬币或标准卡片，根据参考物体像素尺寸与真实尺寸换算食材的实际面积。

如果要生成实现方案，请优先给出：

1. 参考物体检测思路
2. 像素与真实尺寸换算方法
3. 面积估算函数设计
4. 如何与 YOLO 检测结果结合

如需扩展，再提及 MiDaS 或 ZoeDepth，但不要把深度估计作为默认主方案。

---

## 八、位置分析功能提示词

食材摆放位置分析是本项目另一个重要亮点。

请按以下规则理解：

1. 将冰箱区域划分为上层、中层、下层
2. 再划分为左侧、中间、右侧
3. 根据检测框中心点或分割区域中心点，将食材映射到对应空间区域

目标输出示例：

- 苹果：中层右侧
- 牛奶：上层左侧
- 洋葱：下层中间

在此基础上进一步生成：

1. 食材位置列表
2. 冰箱区域占用统计
3. 冰箱热力图

如果要生成代码，请优先拆成：

1. 区域划分配置读取
2. 坐标映射函数
3. 区域统计函数
4. 热力图绘制函数

---

## 九、文档写作提示词

如果你需要帮我写文档，请按课程项目风格输出，适合用于：

1. 开题报告
2. 项目设计说明书
3. 课程答辩 PPT 文案
4. 项目总结

文档内容要突出：

1. 项目背景与现实意义
2. 数字图像处理课程知识的应用
3. YOLOv8 在食材识别中的作用
4. 面积估算和空间管理这两个创新点
5. 项目的可扩展性，例如深度估计和机械臂联动

不要把文档写成泛泛的 AI 套话，要紧扣冰箱食材管理这个具体场景。

---

## 十、完整通用提示词

下面是一段可以直接发给 AI 的完整提示词：

你现在是一名负责大学《数字图像处理》课程项目的开发助手，请帮助我完成一个名为“基于图像处理与 YOLOv8 的智能冰箱食材识别与空间管理系统”的项目。该项目面向日常饮食场景，核心任务是让用户上传冰箱内部图像后，系统自动完成食材识别、定位、数量统计、面积估算、摆放位置分析和热力图生成。项目主体是纯软件实现，重点体现数字图像处理、目标检测、图像测量和空间分析能力。当前不把食品新鲜度判断作为核心功能，它只作为后续扩展项保留。

请你基于以下技术栈进行设计和实现：Python 3.10/3.11、OpenCV、NumPy、scikit-image、matplotlib、Pandas、PyYAML、Ultralytics YOLOv8、Streamlit。不要默认引入复杂 Web 框架。请按模块化方式设计代码，优先保证最小可运行版本，再逐步扩展。

请围绕以下功能展开：
1. 图像上传与显示
2. 基础图像处理
3. YOLOv8 食材检测与定位
4. 食材数量统计
5. 基于参考物体法的食材面积估算
6. 冰箱空间分区与食材位置分析
7. 冰箱区域热力图生成
8. 可视化结果展示

请严格按照以下目录结构组织代码：

```text
smart-fridge-vision/
├─ README.md
├─ requirements.txt
├─ app/
│  ├─ main.py
│  ├─ ui/
│  │  ├─ streamlit_app.py
│  │  └─ components.py
│  ├─ core/
│  │  ├─ image_preprocess.py
│  │  ├─ detector.py
│  │  ├─ segmenter.py
│  │  ├─ area_estimator.py
│  │  ├─ position_analyzer.py
│  │  ├─ heatmap_generator.py
│  │  └─ inventory_manager.py
│  ├─ utils/
│  │  ├─ image_io.py
│  │  ├─ config_loader.py
│  │  ├─ draw.py
│  │  └─ logger.py
│  └─ models/
│     └─ yolo/
├─ config/
│  ├─ classes.yaml
│  ├─ fridge_regions.yaml
│  └─ system.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ samples/
│  └─ outputs/
├─ notebooks/
│  ├─ data_exploration.ipynb
│  └─ algorithm_test.ipynb
├─ scripts/
│  ├─ train_yolo.py
│  ├─ eval_yolo.py
│  └─ export_results.py
├─ docs/
│  ├─ project_plan.md
│  ├─ api_notes.md
│  └─ report_assets/
└─ tests/
   ├─ test_preprocess.py
   ├─ test_area_estimator.py
   └─ test_position_analyzer.py
```

其中，面积估算优先使用参考物体法，不要默认使用深度估计；位置分析需要输出上层/中层/下层、左侧/中间/右侧等相对位置；热力图需要反映冰箱各区域堆放密度。请先给出总体设计，再逐步生成代码。

---

## 十一、推荐使用方式

你后面可以直接把这份文件中的内容用在以下场景：

1. 让 AI 生成项目代码骨架
2. 让 AI 写开题报告或项目说明
3. 让 AI 设计实验方案
4. 让 AI 帮你拆队友分工
5. 让 AI 优化某个具体模块

如果需要更高质量输出，建议每次只让 AI 聚焦一个子模块，不要一次要求把所有代码全部写完。
