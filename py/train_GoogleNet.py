import torch
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
# 从同一项目文件夹下加载另一个py文件
from GoogLeNet import GoogLeNet_V1

# 训练集使用随机增强，验证集使用确定性预处理，避免验证结果被随机扰动影响。
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 定义训练集路径、测试集路径、批大小、训练次数、学习率
project_root = Path(__file__).resolve().parents[1]
data_root = project_root / 'data' / 'dogncat'
output_root = project_root / 'outputs' / 'dogncat'
train_path = data_root / 'train'
val_path = data_root / 'val'
batch_size = 128
EPOCH = 100
learning_rate = 0.004
lr_reduce_factor = 0.5
lr_patience = 3
min_learning_rate = 1e-5
# 定义数据加载器
train_dataset = datasets.ImageFolder(root=train_path, transform = train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(root=val_path, transform = val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4)


def print_gpu_status(use_gpu):
    print('CUDA available:', torch.cuda.is_available())
    print('GPU enabled for training:', use_gpu)

    if not torch.cuda.is_available():
        print('No CUDA GPU detected; training will run on CPU.')
        return

    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    current_device = torch.cuda.current_device()

    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
        marker = '*' if use_gpu and device_id == current_device else ' '
        print(
            '%s GPU %d: %s | total: %.2f GB | allocated: %.2f GB | reserved: %.2f GB'
            % (marker, device_id, props.name, total_memory, allocated_memory, reserved_memory)
        )


if __name__ == '__main__':
    output_root.mkdir(parents=True, exist_ok=True)
    # 实例化网络
    net = GoogLeNet_V1()
    # 是否使用GPU
    use_gpu=torch.cuda.is_available()
    print_gpu_status(use_gpu)
    if(use_gpu):
        net = net.cuda()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=lr_reduce_factor,
        patience=lr_patience,
        min_lr=min_learning_rate
    )
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(EPOCH):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            # 获取数据并放置于GPU或CPU中
            inputs, labels = data
            if(use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 前向计算、损失计算、梯度计算、误差反向传播、梯度更新
            outputs = net.forward(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印状态信息
            running_loss += loss.item()
            if step % 100 == 99:    # 每100批次打印一次
                print('[epoch:%d, step:%5d] loss: %.3f' %(epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
        # 每个epoch结束后保存权重
        # torch.save(net.state_dict(), output_root / ('weights_%d.pth' % epoch))
        ########验证集精度#######
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # 不计算梯度，节省时间
            for (images, labels) in val_loader:
                if(use_gpu):
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = net(images)
                numbers, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total if total else 0.0
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        print('Accuracy of the network on the val images: %.2f %% | lr: %.6f' % (accuracy, current_lr))
    torch.save(net.state_dict(), output_root / 'final_weights.pth')
    print('Finished Training')

# Final: 95.72%
