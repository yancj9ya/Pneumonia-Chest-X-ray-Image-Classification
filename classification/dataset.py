# dataset
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



# 1.读取目录结构
# 2.读取图片路径和标签
# 3.根据索引返回图片和标签


class Pdataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 读取目录结构
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # 读取图片路径和标签
        self.data = []
        self._get_img_info()

    def __getitem__(self, index):
        # 根据索引返回图片和标签
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("L")  # 灰度图
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if not self.data:
            raise ValueError(
                f"{self.root_dir} is empty. Please check the root directory."
            )
        return len(self.data)

    def _get_img_info(self):
        # 读取图片路径和标签
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
                    img_path = os.path.join(cls_dir, img_name)
                    self.data.append((img_path, self.class_to_idx[cls_name]))


if __name__ == "__main__":
    # 测试数据集
    root_dir = "chest_xray/train/"
    normMean = [0.5]
    normStd = [0.5]
    normTransform = transforms.Normalize(normMean, normStd)
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            normTransform,
        ]
    )
    dataset = Pdataset(root_dir, train_transform)
    # print(dataset.data[::-1])
    # print(dataset.class_to_idx)
    # 测试dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        imgs, labels = batch
        print(imgs.shape)
        print(labels)
        break