class Task:
    def __init__(self, task: str) -> None:
        self.task: str = task
        self.CLS: bool = False
        self.SEG: bool = False
        self.MT: bool = False

        if task.lower() == 'multitask':
            self.MT = True
        elif task.lower() == 'classification':
            self.CLS = True
        elif task.lower() == 'segmentation':
            self.SEG = True

    def __str__(self) -> str:
        return self.task
