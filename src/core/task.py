class Task:
    CLS: bool = False
    SEG: bool = False
    MT: bool = False
    TASK: str = ''

    def __init__(self, task: str):  # noqa
        self.TASK = task

        if task.lower() == 'multitask':
            self.MT = True
            self.CLS = True
            self.SEG = True

        elif task.lower() == 'classification':
            self.CLS = True

        elif task.lower() == 'segmentation':
            self.SEG = True

    def __str__(self) -> str:
        return self.TASK


if __name__ == '__main__':
    task = Task("classification")
    print(task.CLS)
    print(task.SEG)
    print(task.MT)
