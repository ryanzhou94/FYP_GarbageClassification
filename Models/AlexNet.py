import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

print("Loading data...")
# Load image data (227*227*3)
training_data = np.load("../TrainingData/training_garbage_data_AlexNet.npy", allow_pickle=True)
print("Data is loaded.")
# training_data: 2527张图
# 每张图有2个维度：图片和标签
# 图像：227*227*3

# 参考借鉴：
# https://blog.csdn.net/Gilgame/article/details/85056344?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.conv1 = torch.nn.Sequential(
            # input_size = 227*227*3
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
            # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(
            # input_size = 27*27*96
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
            # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(
            # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
            # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(
            # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
            # output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(
            #input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
            #output_size = 6*6*256
        )

        # Fully-connected layer
        self.dense = torch.nn.Sequential(
            # input_size = 6*6*256
            torch.nn.Linear(9216, 4096),    # 将9216个特征点flatten -> 4096
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(4096, 6)
            # is there a softmax?
        )

    # 正向传播过程
    def forward(self, x):
        conv1_out = self.conv1(x)   # input x must be 227*227*3
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        # Flatten the result
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = AlexNet().to(device)




# Create an optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
# Set the loss function
loss_function = nn.MSELoss().to(device)

# Seperate X and y. PS: this step is very slow
# X: data
# y: labels
print("Start to extract data and labels...")
X = torch.Tensor([i[0] for i in training_data]).to(device)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data]).to(device)
print("Data and labels are extracted.")

TRAIN_PCT = 0.9
train_size = int(len(X) * TRAIN_PCT)

train_X = X[:train_size]
train_y = y[:train_size]

test_X = X[train_size:]
test_y = y[train_size:]

BATCH_SIZE = 512
EPOCHS = 150

print("Start to train...")
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_x = train_X[i:i + BATCH_SIZE].view(-1, 3, 227, 227)
        batch_y = train_y[i:i + BATCH_SIZE]
        net.zero_grad()
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()


correct = 0
total = 0

print("Start to predict...")
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 3, 227, 227))[0]
        predicted_class = torch.argmax(net_out)
        if real_class == predicted_class:
            correct += 1
        total += 1

print("AlexNet Accuracy: ", round(correct/total, 3))