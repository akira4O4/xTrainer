from typing import List, Tuple
import torch
import numpy as np

__all__ = [
    'topk_accuracy',
    'safe_mean',
    'mean_iou_v1',
    'mean_iou_v2',
    'compute_confusion_matrix_classification',
    'compute_confusion_matrix_segmentation',
    ''
]


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> List[float]:
    """Computes the top-k accuracy for the specified values of k

    Args:
        output (torch.Tensor): The model output with shape (batch_size, num_classes).
        target (torch.Tensor): The ground truth labels with shape (batch_size).
        topk (Tuple[int, ...]): A tuple specifying the top-k accuracies to compute.

    Returns:
        List[float]: A list of top-k accuracy values.
    """
    maxk = max(topk)
    batch_size = target.size(0)  # (N,C)

    # 获取预测分数最高的 k 个索引
    values, indices = output.topk(maxk, 1, True, True)
    indices = indices.t()  # 转置以适应 target 的形状 (N,C)->(C,N)
    expand_target = target.view(1, -1).expand_as(indices)  # (1,N)->(C,N)
    correct = indices.eq(expand_target)

    res: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # top-k result
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def iou_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> List[float]:
    """Computes IoU for each class.

    Args:
        pred (torch.Tensor): The predicted segmentation map with shape (H, W).
        target (torch.Tensor): The ground truth segmentation map with shape (H, W).
        num_classes (int): The number of classes including the background.

    Returns:
        List[float]: A list containing IoU for each class.
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            ious.append(float('nan'))  # If no ground truth exists for this class, ignore it in IoU calculation
        else:
            ious.append((intersection / union).item())
    return ious


def safe_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the mean of a tensor, ignoring NaN values."""
    # data=[1,Nan,2,Nan]
    # mask=[True,False,True,False]
    # mean=[1,2].mean()
    mask = ~torch.isnan(tensor)
    return tensor[mask].mean()


def mean_iou_v1(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> float:
    """Computes the mean IoU across all classes.

    Args:
        pred (torch.Tensor): The predicted segmentation map with shape (H, W).
        target (torch.Tensor): The ground truth segmentation map with shape (H, W).
        num_classes (int): The number of classes including the background.

    Returns:
        float: The mean IoU.
    """
    ious = iou_per_class(pred, target, num_classes)
    return safe_mean(torch.tensor(ious)).item()


def compute_confusion_matrix_classification(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """Computes the confusion matrix for classification tasks.

    Args:
        pred (torch.Tensor): The predicted labels with shape (N,) where N is the number of samples.
        target (torch.Tensor): The ground truth labels with shape (N,) where N is the number of samples.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The confusion matrix with shape (num_classes, num_classes).
    """
    if pred.size(0) != target.size(0):
        raise ValueError("The size of pred and target must be the same.")

    pred = torch.argmax(pred, dim=1)

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(target, pred):
        confusion_matrix[t, p] += 1

    return confusion_matrix


def compute_confusion_matrix_segmentation(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """Computes the confusion matrix for segmentation tasks.

    Args:
        pred (torch.Tensor): The predicted labels with shape (H, W) or (N, H, W).
        target (torch.Tensor): The ground truth labels with shape (H, W) or (N, H, W).
        num_classes (int): The number of classes.

    Returns:
        np.ndarray: The confusion matrix with shape (num_classes, num_classes).
    """
    if pred.dim() == 3:
        pred = pred.view(-1)
        target = target.view(-1)

    elif pred.dim() == 2:
        pred = pred.flatten()
        target = target.flatten()

    else:
        raise ValueError("Input tensors should be 2D or 3D tensors.")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
        confusion_matrix[t, p] += 1

    return confusion_matrix


def compute_iou_from_confusion(confusion_matrix: np.ndarray) -> List[float]:
    """Computes IoU for each class from the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): The confusion matrix with shape (num_classes, num_classes).

    Returns:
        List[float]: A list containing IoU for each class.
    """
    ious = []
    nc: int = confusion_matrix.shape[0]  # num of classes
    for cls in range(nc):
        intersection = confusion_matrix[cls, cls]  # 对角线数值
        union = confusion_matrix[cls, :].sum() + confusion_matrix[:, cls].sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # If no ground truth exists for this class, ignore it in IoU calculation
        else:
            ious.append(intersection / union)
    return ious


def mean_iou_v2(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> float:
    """Computes the mean IoU across all classes.

    Args:
        pred (torch.Tensor): The predicted segmentation map with shape (H, W).
        target (torch.Tensor): The ground truth segmentation map with shape (H, W).
        num_classes (int): The number of classes including the background.

    Returns:
        float: The mean IoU.
    """
    confusion_matrix = compute_confusion_matrix_segmentation(pred, target, num_classes)
    ious = compute_iou_from_confusion(confusion_matrix)
    return float(np.nanmean(ious))


import itertools
import matplotlib.pyplot as plt
import numpy as np


# 绘制混淆矩阵

def draw_confusion_matrix(
    cm,
    classes: List[str],
    save: str,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap=plt.cm.Blues
) -> None:
    """
    绘制并保存混淆矩阵的图像。

    参数:
    - cm : ndarray of shape (n_classes, n_classes)
        计算出的混淆矩阵的值
    - classes : list of str
        混淆矩阵中每一行每一列对应的列名
    - normalize : bool, optional, default: False
        True:显示百分比, False:显示个数
    - title : str, optional, default: 'Confusion Matrix'
        混淆矩阵图像的标题
    - cmap : Colormap, optional, default: plt.cm.Blues
        混淆矩阵图像的颜色映射
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 反转 y 轴
    plt.gca().invert_yaxis()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 保存图像
    plt.savefig(save)


if __name__ == '__main__':
    # num_classes = 3  # 假设有3个类别
    # pred = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
    # target = torch.tensor([0, 1, 1, 0, 2], dtype=torch.int64)
    #
    # conf_matrix = compute_confusion_matrix_classification(pred, target, num_classes)
    # print("Confusion Matrix for Classification:\n", conf_matrix)

    batch_size = 5
    num_classes = 10
    topk = (1, 3, 5)

    # 随机生成模型输出和真实标签
    torch.manual_seed(0)
    output = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    print(output.shape)
    print(target.shape)

    # 计算 top-k 精度
    accuracies = topk_accuracy(output, target, topk)
    for k, acc in zip(topk, accuracies):
        print(f"Top-{k} Accuracy: {acc:.2f}%")

    num_classes = 5  # 假设有5个类别，包括背景
    pred = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ], dtype=torch.int64)
    # pred = torch.zeros((num_classes, num_classes))

    target = torch.tensor([
        [0, 1, 1, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4],
        [0, 0, 2, 3, 4]
    ], dtype=torch.int64)

    miou1 = mean_iou_v1(pred, target, num_classes)
    miou2 = mean_iou_v2(pred, target, num_classes)
    print(f"Mean IoU 1: {miou1:.4f}")
    print(f"Mean IoU 2: {miou2:.4f}")
