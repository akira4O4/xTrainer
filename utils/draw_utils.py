import numpy as np
import matplotlib.pyplot as plt

__all__ = ['draw_confusion_matrix', 'draw_loss']


def draw_confusion_matrix(
        numclasses: int,
        labels: list,
        matrix: np.ndarray,
        save_path: str
) -> None:
    if len(labels) != numclasses:
        return

    # 绘制混淆矩阵
    plt_title = 'Confusion Matrix'
    plt.title(plt_title)

    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(numclasses):
        for y in range(numclasses):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(numclasses), labels)

    labels = list(map(lambda x: 'pre_' + x, labels))
    plt.xticks(range(numclasses), labels)  # X轴字体倾斜45°
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def draw_loss(
        x: list, y: list,
        xlabel: str = 'Batch', ylabel: str = 'Loss',
        save_path: str = 'temp/loss.png'
) -> None:
    plt.plot(x, y, '.-')
    plt_title = 'Loss'
    plt.title(plt_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
