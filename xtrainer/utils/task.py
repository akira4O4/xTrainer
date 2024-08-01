class Task:
    def __init__(self, task: str) -> None:
        self.TASK: str = task
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
        return self.TASK



if __name__ == '__main__':
    task = Task("classification")
    print(task.CLS)
    print(task.SEG)
    print(task.MT)
