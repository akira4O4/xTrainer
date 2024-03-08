from typing import Optional
from .task import Task
__all__ = ['AverageMeter', 'ProgressMeter', 'Logger']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=f':4e'):  # 4e
        self.name = name

        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

        self.fmt = fmt

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # Name: 0.001
        # 4f
        return f' [{self.name}]: {format(float(self.avg), f".6f")}'


class ProgressMeter:
    def __init__(self, num_batches: int, meters, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    @staticmethod
    def _get_batch_fmtstr(num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'  # {:nd}

        return '[' + fmt + '/' + fmt.format(num_batches) + ']'  # [{:nd}/m]

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))


class Logger:
    def __init__(
            self,
            task: Task,
            total_step: Optional[int] = 0,
            topn: Optional[int] = 2,
            prefix: Optional[str] = '',
    ) -> None:
        self.progress = None
        self.prefix = prefix
        self.task = task

        self._total_loss = AverageMeter('Total Loss', f':.4e')  # 4e
        self._cls_meters = []
        self._seg_meters = []
        self._total_step = total_step
        self.cls_loss = None
        self.top1 = None
        self.topn = None
        self.MIoU = None
        self.loss_seg = None

        if self.task in [Task.CLS, Task.MultiTask]:
            self.cls_loss = AverageMeter('Classification Loss', f':.{8}e')  # 8e
            self.top1 = AverageMeter('Top1', f':{6}.{2}f')  # 6.2
            self.topn = AverageMeter(f"Top{topn}", f':{6}.{2}f')  # 6.2
            self._cls_meters += [self.cls_loss, self.top1, self.topn]

        if self.task in [Task.SEG, Task.MultiTask]:
            self.MIoU = AverageMeter(f"MIoU", f':{6}.{2}f')  # 6.2
            self.loss_seg = AverageMeter('Segmentation Loss', f':.{8}e')  # 8e
            self._seg_meters += [self.loss_seg, self.MIoU]

    def set_total_step(self, step: int) -> None:
        self._total_step = step

    def display(self, curr_step: int, epoch: int) -> None:
        meters = None
        if self.task == Task.CLS:
            meters = self._cls_meters
        elif self.task == Task.SEG:
            meters = self._seg_meters
        elif self.task == Task.MultiTask:
            meters = self._cls_meters + self._seg_meters

        self.progress = ProgressMeter(
            self._total_step,
            meters,
            prefix=self.prefix + f" Epoch: [{epoch}]"
        )
        self.progress.display(curr_step)

    def clear(self) -> None:
        self._total_loss.reset()
        if self.cls_loss:
            self.cls_loss.reset()

        if self.loss_seg:
            self.loss_seg.reset()

        if self.top1:
            self.top1.reset()

        if self.topn:
            self.topn.reset()
