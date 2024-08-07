from xtrainer.utils.perf import compute_iou
import torch


def test_compute_iou_2d():
    pred = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
    target = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    num_classes = 2  # 假设有两个类别：0 和 1

    iou = compute_iou(pred, target, num_classes)
    print("2D IoU:", iou)  # 应打印IoU值
    assert isinstance(iou, float), "Expected output to be a float."


def test_compute_iou_3d():
    pred = torch.tensor([[[0, 1, 1], [1, 0, 1], [0, 1, 0]],
                         [[1, 1, 0], [0, 1, 1], [1, 0, 0]]])
    target = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 0, 0]],
                           [[1, 0, 0], [0, 1, 1], [1, 0, 0]]])
    num_classes = 2  # 假设有两个类别：0 和 1

    iou = compute_iou(pred, target, num_classes)
    print("3D IoU:", iou)  # 应打印每个批次的IoU值
    assert isinstance(iou, float), "Expected output to be a float."


def test_compute_iou_4d():
    pred = torch.tensor(
        [[[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]],

         [[[1, 1, 0], [0, 1, 1], [0, 0, 1]],
          [[0, 0, 1], [1, 1, 0], [1, 0, 0]]]]
    )

    target = torch.tensor(
        [[[[0, 1, 0], [1, 1, 0], [0, 0, 1]]]
            , [[[0, 1, 0], [1, 1, 0], [0, 0, 1]]]]
    )
    num_classes = 2  # 假设有两个类别：0 和 1

    iou = compute_iou(pred, target, num_classes)
    print("4D IoU:", iou)  # 应打印每个批次的IoU值
    assert isinstance(iou, float), "Expected output to be a float."


def test_compute_iou_no_overlap():
    pred = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    target = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    num_classes = 2  # 假设有两个类别：0 和 1

    iou = compute_iou(pred, target, num_classes)
    print("IoU with no overlap:", iou)  # 应处理无重叠情况，确保返回nan或0
    assert isinstance(iou, float), "Expected output to be a float."


def test_compute_iou_perfect_overlap():
    pred = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    target = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    num_classes = 2  # 假设有两个类别：0 和 1

    iou = compute_iou(pred, target, num_classes)
    print("IoU with perfect overlap:", iou)  # 应处理完全重叠情况，IoU应为1
    assert isinstance(iou, float), "Expected output to be a float."


test_compute_iou_2d()
test_compute_iou_3d()
test_compute_iou_4d()
test_compute_iou_no_overlap()
test_compute_iou_perfect_overlap()
