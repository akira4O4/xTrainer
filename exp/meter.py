from typing import Optional
from utils.util import Task
from helper.precision import data_precision
__all__ = ['AverageMeter', 'ProgressMeter', 'Logger']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=f':{data_precision.Medium}e'):  # 4e
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
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
        return f' [{self.name}]: {format(float(self.avg), f".{data_precision.High}f")}'


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'

        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
        self._total_loss = AverageMeter('Total Loss', f':.{data_precision.Medium}e')  # 4e
        self._cls_meters = []
        self._seg_meters = []
        self._total_step = total_step
        self.loss_cls=None
        self.top1=None
        self.topn=None
        self.MIoU=None
        self.loss_seg=None

        if task in [Task.CLS, Task.MultiTask]:
            self.loss_cls = AverageMeter(
                'Classification Loss', f':.{data_precision.Hightest}e')  # 8e
            self.top1 = AverageMeter(
                'Top1', f':{data_precision.High}.{data_precision.Low}f')  # 6.2
            self.topn = AverageMeter(
                f"Top{topn}", f':{data_precision.High}.{data_precision.Low}f')  # 6.2
            self._cls_meters += [self.loss_cls, self.top1, self.topn]

        if task in [Task.SEG, Task.MultiTask]:
            self.MIoU = AverageMeter(
                f"MIoU", f':{data_precision.High}.{data_precision.Low}f')  # 6.2
            self.loss_seg = AverageMeter(
                'Segmentation Loss', f':.{data_precision.Hightest}e')  # 8e
            self._seg_meters += [self.loss_seg, self.MIoU]

    def set_total_step(self, step: int) -> None:
        self._total_step = step

    def display(self, task: Task, curr_step: int, epoch: int) -> None:
        meters = None
        if task == Task.CLS:
            meters = self._cls_meters
        elif task == Task.SEG:
            meters = self._seg_meters
        elif task == Task.MultiTask:
            meters = self._cls_meters + self._seg_meters

        self.progress = ProgressMeter(
            self._total_step,
            meters,
            prefix=self.prefix + f" Epoch: [{epoch}]"
        )
        self.progress.display(curr_step)

    def clear(self) -> None:
        self._total_loss.reset()
        if self.loss_cls:
            self.loss_cls.reset()
        
        if self.loss_seg:
            self.loss_seg.reset()
        
        if self.top1:
            self.top1.reset()
        
        if self.topn:
            self.topn.reset()
