# Partner 运行说明

## 1. 项目用途

这是一个基于 `OpenCV + YOLOv8 + Streamlit` 的智能冰箱视觉管理项目运行包。

当前运行包已经包含：

1. Streamlit 前端页面
2. 食材检测逻辑
3. 两份模型权重
4. 默认运行配置

## 2. 推荐环境

建议使用：

1. Windows 10 / 11
2. Python 3.11
3. CPU 可运行，GPU 可选

## 3. 安装依赖

在运行包根目录打开终端后执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install torch torchvision
```

如果已经有可用的 `torch` 环境，可以跳过第二行。

## 4. 启动项目

方式一：直接双击

```text
run_app.bat
```

方式二：命令行启动

```powershell
python -m streamlit run app\ui\streamlit_app.py --server.fileWatcherType none
```

启动后浏览器访问：

```text
http://localhost:8501
```

## 5. 可切换模型

页面左侧支持切换两份模型：

1. 当前 12 类新模型
2. 之前的生鲜阶段模型

模型文件位于：

```text
model/
├─ fridge_12class_best.pt
└─ fresh_stage1_best.pt
```

## 6. 默认识别类别

当前主模型识别 12 类：

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

## 7. 常见问题

### 7.1 `No module named streamlit`

说明当前 Python 环境没有安装 Streamlit，执行：

```powershell
python -m pip install -r requirements.txt
```

### 7.2 `No module named torch`

说明当前 Python 环境没有安装 PyTorch，执行：

```powershell
python -m pip install torch torchvision
```

### 7.3 页面打开但识别结果很差

建议：

1. 左侧开启“检测时直接使用原图”
2. 适当降低置信度阈值，例如 `0.10`
3. 切换另一份模型做对比

## 8. 目录说明

运行包中核心目录如下：

```text
app/          主程序
config/       配置文件
model/        模型权重
.streamlit/   Streamlit 配置
data/         日志和样例目录
```

