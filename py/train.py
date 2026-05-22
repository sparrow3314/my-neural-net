import torch
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
from net import VGGnet
# 从同一项目文件夹下加载另一个py文件
from GoogLeNet import GoogLeNet_V1

# 数据预处理操作：尺度缩放、中心裁切、转换为张量并同时进行归一化到(0,1)、标准化为(-1,1)
image_transform = transforms.Compose([
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
batch_size = 32
EPOCH = 50
learning_rate = 0.001
# 定义数据加载器
train_dataset = datasets.ImageFolder(root=train_path, transform = image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder(root=val_path, transform = image_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':
    output_root.mkdir(parents=True, exist_ok=True)
    # 实例化网络
    net = GoogLeNet_V1()
    # 是否使用GPU
    use_gpu=torch.cuda.is_available()
    if(use_gpu):
        net = net.cuda()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(EPOCH):
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
        correct = 0
        total = 0
        with torch.no_grad():
            # 不计算梯度，节省时间
            for (images, labels) in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = net(images)
                numbers, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the val images: %d %%' % (100 * correct / total))
    torch.save(net.state_dict(), output_root / 'final_weights.pth')
    print('Finished Training')


