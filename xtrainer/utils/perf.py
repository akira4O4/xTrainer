from typing import List, Tuple
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

__all__ = [
    'safe_mean',
    'compute_iou',
    'topk_accuracy',
    'compute_confusion_matrix_classification',
    'compute_confusion_matrix_segmentation',
    'draw_confusion_matrix'
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


def compute_iou_with_hw(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> List[float]:
    """
    pred (torch.Tensor): The predicted segmentation map with shape (H, W).
    target (torch.Tensor): The ground truth segmentation map with shape (H, W).
    num_classes (int): The number of classes including the background.
    """
    assert pred.shape[0] == target.shape[0], "Predicted and target batch sizes must be the same"
    assert pred.shape[1] == target.shape[1], "Predicted and target spatial dimensions must be the same"
    assert len(pred.shape) == 2, "Predicted tensor must be 2D (H, W)"
    assert len(target.shape) == 2, "Target tensor must be 2D (H, W)"

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


def compute_iou_with_nhw(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> List[List[float]]:
    """
    pred (torch.Tensor): The predicted segmentation map with shape (N,  H, W).
    target (torch.Tensor): The ground truth segmentation map with shape (N, H, W).
    num_classes (int): The number of classes including the background.
    """
    assert pred.shape[0] == target.shape[0], "Predicted and target batch sizes must be the same"
    assert pred.shape[2:] == target.shape[2:], "Predicted and target spatial dimensions must be the same"
    assert len(pred.shape) == 3, "Predicted tensor must be 3D (N, H, W)"
    assert len(target.shape) == 3, "Target tensor must be 3D (N, H, W)"

    bs = pred.shape[0]
    batch_ious = []

    for b in range(bs):
        ious = compute_iou_with_hw(pred[b], target[b], num_classes)
        batch_ious.append(ious)

    return batch_ious


def compute_iou_with_nchw(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> List[List[float]]:
    """
    pred (torch.Tensor): The predicted segmentation map with shape (N, C, H, W).
    target (torch.Tensor): The ground truth segmentation map with shape (N, 1, H, W).
    num_classes (int): The number of classes including the background.
    """
    assert pred.shape[0] == target.shape[0], "Predicted and target batch sizes must be the same"
    assert pred.shape[2:] == target.shape[2:], "Predicted and target spatial dimensions must be the same"
    assert len(pred.shape) == 4, "Predicted tensor must be 4D (N, C, H, W)"
    assert len(target.shape) == 4 and target.shape[1] == 1, "Target tensor must be 4D (N, 1, H, W)"

    bs = pred.shape[0]
    batch_ious = []

    # Convert prediction to class labels
    pred_labels: torch.Tensor = pred.argmax(dim=1)  # Shape (N, H, W)
    target_labels: torch.Tensor = target.squeeze(1)  # Shape (N, H, W)

    for b in range(bs):
        ious = compute_iou_with_hw(pred_labels[b], target_labels[b], num_classes)

        batch_ious.append(ious)

    return batch_ious


def safe_mean(tensor: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(tensor)
    return tensor[mask].mean()


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> float:
    # [H,W]
    if pred.dim() == 2:
        iou = compute_iou_with_hw(pred, target, num_classes)
        return safe_mean(torch.tensor(iou)).item()

    # [N,H,W]
    elif pred.dim() == 3:
        bs_iou = compute_iou_with_nhw(pred, target, num_classes)
        all_iou = [safe_mean(torch.tensor(iou)).item() for iou in bs_iou]
        return safe_mean(torch.tensor(all_iou)).item()

    # [N,C,H,W]
    elif pred.dim() == 4:
        bs_iou = compute_iou_with_nchw(pred, target, num_classes)
        all_iou = [safe_mean(torch.tensor(iou)).item() for iou in bs_iou]
        return safe_mean(torch.tensor(all_iou)).item()


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

    if pred.dim() == 2:  # (n,nc)->(n,)
        pred = torch.argmax(pred, dim=-1)

    assert target.dim() == 1, f'target.dim != 1'

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(target, pred):
        confusion_matrix[t, p] += 1

    return confusion_matrix


def compute_confusion_matrix_segmentation(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """
    pred (torch.Tensor): The predicted labels with shape (H, W) or (N, H, W).
    target (torch.Tensor): The ground truth labels with shape (H, W) or (N, H, W).
    num_classes (int): The number of classes.
    """
    if pred.dim() == 4:  # shape=(N, C, H, W)
        pred = pred.argmax(dim=1)  # Convert to shape=(N, H, W)

    if target.dim() == 4 and target.shape[1] == 1:  # shape=(N, 1, H, W)
        target = target.squeeze(1)  # Convert to shape=(N, H, W)

    if pred.dim() == 3:  # shape=(N, H, W)
        N = pred.shape[0]
        pred = pred.view(N, -1)
        target = target.view(N, -1)

    elif pred.dim() == 2:  # shape=(H, W)
        pred = pred.flatten()
        target = target.flatten()
    else:
        raise ValueError("Input tensors should be 2D, 3D or 4D tensors.")

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
    plt.close()
