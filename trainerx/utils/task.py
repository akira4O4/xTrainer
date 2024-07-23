class Task:
    CLS: bool = False
    SEG: bool = False
    MT: bool = False

    def __init__(self, task: str):  # noqa
        self.TASK = task

        if task.lower() == 'multitask':
            self.MT = True
            # self.CLS = False
            # self.SEG = False

        elif task.lower() == 'classification':
            # self.MT = False
            self.CLS = True
            # self.SEG = False

        elif task.lower() == 'segmentation':
            # self.MT = False
            # self.CLS = False
            self.SEG = True

    def __str__(self) -> str:
        return self.TASK


if __name__ == '__main__':
    task = Task("classification")
    print(task.CLS)
    print(task.SEG)
    print(task.MT)
