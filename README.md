# <div align="center">xTrainer</div>

<div align="center">Version: 1.0.0-dev</div>
<div align="center">Author: Li Linfeng</div>

**<div align="center">Language: [English](README_EN.md)</div>**

---

## 简介
这是一个基于`PyTorch`的分类，分割，多任务训练和推理框架，支持`PyTorch`原生模型和自定义模型，设计多种优化方案，模块化的设计了整个训练推理框架

---

## 特点
- 支持`Linux`，`Windows`，`Macos`
- 支持`分类`任务，`分割`任务，`多任务`
- 支持`训练`和`推理`
- 支持训练数据`预加载`
- 支持`自定义模型`
- 支持`MLflow`参数跟踪
- 支持模块化添加Loss
- 分类任务中数据`平衡采样`
- 可视化`混淆矩阵`
- 自动平衡多任务训练中数据量

---

## 安装
```bash
cd <your_workspace>
git clone https://github.com/akira4O4/xTrainer.git
cd xTrainer
pip install -r requirements.txt
```
---

## 配置文件
### 默认配置文件路径
`xTrainer/configs/default.yaml`
### 配置参数解析

| 参数名字                    | 默认值         | 数据类型        | 描述                                                                                             |
|-------------------------|-------------|-------------|------------------------------------------------------------------------------------------------|
| `mode`                  | `train`     | `str`       | 运行模式<br/>训练：`train`<br/>测试：`test`                                                              |
| `task`                  |             | `str`       | 任务类型<br/>分类：classification<br/>分割：segmentation<br/>多任务：multitask                               |
| `project`               |             | `str`       | 项目路径                                                                                           |
| `experiment`            | `exp`       | `str`       | 每次实验名称                                                                                         |
| `seed`                  | `0`         | `int`       | 随机种子                                                                                           |
| `topk`                  | `[1,2]`     | `List[int]` | 分类topk范围                                                                                       |
| `device`                | `0`         | `int`       | 模型运行设备                                                                                         |
| `epochs `               | `100`       | `int`       | 最大轮训次数                                                                                         |
| `workers`               | ` 0`        | `int`       | dataloader多进程数                                                                                 |
| `not_val`               | `False`     | `bool`      | 是否进行验证，True：只训练不验证                                                                             |
| `model`                 |             | `str`       | 模型名称                                                                                           |
| `pretrained`            | ` True`     | `bool`      | 是否加载预训练模型，模型来自Pytorch Hub                                                                      |
| `weight`                |             | `str`       | 预训练模型路径，来自本地模型                                                                                 |
| `wh`                    | `[256,256]` | `List[int]` | 输入图像宽高                                                                                         |
| `amp`                   | `True`      | `bool`      | 是否使用自动混合精度进行训练                                                                                 |
| `cache`                 | `False`     | `bool`      | 是否使用数据预加载<br/>开启后程序会提前**全部**加载所有数据                                                             |
| `deterministic`         | `True`      | `bool`      | 用于启用确定性模式                                                                                      |
| `save_period`           | `5`         | `int`       | 每训练x次就进行一次模型保存                                                                                 |
| `classification.batch`  |             | `int`       | 分类任务的batch数                                                                                    |
| `classifiction.classes` |             | `int`       | 分类任务的类别数                                                                                       |
| `classification.train ` |             | `str`       | 分类任务的训练数据路径                                                                                    |
| `classification.val`    |             | `str`       | 分类任务的验证数据路径                                                                                    |
| `segmentation.batch`    |             | `int`       | 分割任务的batch数                                                                                    |
| `segmentation.classes`  |             | `int`       | 分割任务的类别数（**分割类别数需要包含背景**）                                                                      |
| `segmentation.train `   |             | `str`       | 分割任务的训练数据路径                                                                                    |
| `segmentation.val  `    |             | `str`       | 分割任务的验证数据路径                                                                                    |
| `optimizer`             | `auto`      | `str`       | 优化器名称<br/>auto="AdamW"<br/>支持优化器=["Adam", "Adamax", "AdamW", "NAdam", "RAdam"，"RMSProp"，"SGD"] |
| `cos_lr `               | `False`     | `bool`      | 是否使用余弦退火学习率                                                                                    |
| `lr0`                   | `0.001`     | `float`     | 初始学习率                                                                                          |
| `lrf`                   | `0.01`      | `float`     | 最低学习率下降比例，最低学习率=`lr0*lrf  `                                                                    |
| `momentum`              | `0.937`     | `float`     | 优化器冲量                                                                                          |
| `alpha`                 | `auto`      | `List[int]` | Focal Loss参数                                                                                   |
| `gamma`                 | `2.0`       | `float`     | Focal Loss参数                                                                                   |
| `smooth`                | `1.0`       | `float`     | 分割loss中的稳定参数<br/>极小目标：1e-6<br/>正常目标：1.0                                                        |
| `loss_sum_weights`      | `[1,1]`     | `List[int]` | 多任务中，分类loss于分割loss加权比例                                                                         |
| `seg_loss_sum_weights`  | `[1,1,1]`   | `List[int]` | 多个分割loss中的加权比例                                                                                 |
| `source`                |             | `str`       | 测试数据路径                                                                                         |
| `test_weight`           |             | `str`       | 测试权重路径                                                                                         |
| `cls_thr`               |             | `List[int]` | 分类任务阈值                                                                                         |
| `seg_thr`               |             | `List[int]` | 分割任务阈值（**不需要包含背景**）                                                                            |
| `mlflow_url`            | `localhost` | `str`       | mlflow URI                                                                                     |
| `mlflow_port`           | `5000  `    | `int`       | mlflow端口                                                                                       |

---

## 如何使用自定义模型

---

## 训练数据格式

### 分类任务

### 分割任务

### 多任务

---

## 训练



---

## 预测

---

## 导出ONNX、TorchScript
