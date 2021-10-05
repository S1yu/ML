import numpy as np
from torch.utils.data.dataset import Dataset


class MyDataSet(Dataset):
    def __init__(self, root, lable):
        self.data = root
        self.lable = lable

    def __getitem__(self, item):
        data = self.data[item]
        lables = self.lable[item]
        return data, lables

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    source = np.loadtxt("D:\project\IrisDateset\iris.csv", delimiter=",", usecols=(1, 2, 3, 4), skiprows=1)
    print(source.shape)
