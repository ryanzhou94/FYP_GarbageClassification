import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

print("Loading data...")
# Load image data (32*32)
training_data = np.load("../TrainingData/training_garbage_data_LeNet.npy", allow_pickle=True)
print("Data is loaded.")

# 参考借鉴
# https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 因为LeNet太过于精简，这里不采用Sequential的方式来实现LeNet
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(1, 6, 5)     # 6个5*5的卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)    # 16个5*5的卷积核

        # input feature vector: 400 = 16*5*5; output: 120
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(self.sigmoid((self.conv1(x))))
        x = self.pool(self.sigmoid((self.conv2(x))))
        # Flatten the feature vector
        x = x.view(-1, self.num_flat_feature(x))
        x = self.sigmoid(self.fc1(x)) # 400 -> 120
        x = self.sigmoid(self.fc2(x)) # 120 -> 84
        x = self.fc3(x)               # 84 -> 6
        return x

    # Flatten feature vectors from convolution layers to FC layers
    def num_flat_feature(self, x):
        #x.size(): (BATCH_SIZE, 16, 5, 5); size = (16, 5, 5)
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)

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
EPOCHS = 200

print("Start to train...")
# 注意，下面这串代码在执行的时候有一些显示上的不清楚的地方
# 因为batch size的原因，所以下面的进度条里显示单个单位 比如/225里面的 1代表一个batch，而非一张图片
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_x = train_X[i:i + BATCH_SIZE].view(-1, 1, 32, 32)
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
        net_out = net(test_X[i].view(-1, 1, 32, 32))[0]
        predicted_class = torch.argmax(net_out)
        if real_class == predicted_class:
            correct += 1
        total += 1

print("LeNet Accuracy: ", round(correct/total, 3))
