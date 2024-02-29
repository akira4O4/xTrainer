import numpy as np


class BalancedBatchSampler:
    def __init__(self, labels, n_classes: int, n_samples: int):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # 这里指的是对应lebel已经使用的数据个数
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, size=self.n_classes, replace=True)  #
            indices = []

            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                                            class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                # 当数据的使用数量大于这个lebel具有的数据长度的时候，重置已使用数据的数量
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices

            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size  # return num of batch
