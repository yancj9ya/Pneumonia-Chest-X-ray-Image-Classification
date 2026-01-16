import torch
import torch.nn as nn
import torchvision.transforms as transforms
import dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from mytool import Trainer, make_logger

# 训练集transform
# 归一化
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
# 验证集transform
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normTransform,
    ]
)

# 加载数据集
# 训练集
train_set = dataset.Pdataset(
    root_dir="I:/PythonCode/pytorch/chest_xray/train/", transform=train_transform
)
# 验证集
val_set = dataset.Pdataset(
    root_dir="I:/PythonCode/pytorch/chest_xray/test/", transform=val_transform
)

# dataloader
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# 使用预训练模型ResNet50


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# 替换
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, cooldown=3, mode="max"
)

# 训练模型次数
num_epochs = 30

if __name__ == "__main__":
    logger = make_logger("train.log")
    trainer = Trainer()
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = trainer.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 验证
        val_loss, val_acc = trainer.validate_one_epoch(
            model, val_loader, criterion, device
        )
        # 记录日志
        logger.info(
            f"epoch:{epoch} train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} lr:{optimizer.param_groups[0]['lr']:.6f}"
        )
        # 更新学习率
        scheduler.step(val_acc)

        # 保存模型
        if val_acc > best_acc or epoch == num_epochs - 1:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")
