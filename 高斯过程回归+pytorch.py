import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self,x_train,y_train,num_one_train):
        super(dataset,self).__init__()
        self.n_train = x_train.size(0)//(num_one_train//2)-1
        self.stride = num_one_train//2
        self.num_one_train = num_one_train
        rand = torch.randperm(x_train.size(0))
        self.shuffled_x = x_train[rand]
        self.shuffled_y = y_train[rand]

    def __getitem__(self, index):
        return self.shuffled_x[index*self.stride:index*self.stride+self.num_one_train],\
               self.shuffled_y[index * self.stride:index * self.stride+self.num_one_train]
    def __len__(self):
        return self.n_train

class MLP(nn.Module):
    def __init__(self, ninp, nhid):
        super(MLP, self).__init__()
        self.inlayer = nn.Linear(ninp,nhid)
        self.hidlayer1 = nn.Linear(nhid, nhid)
        # self.bn = nn.LayerNorm(nhid)
        self.hidlayer2 = nn.ReLU()
        self.outlayer = nn.Linear(nhid,nhid)
    def forward(self,x):
        out = self.inlayer(x)
        out = self.hidlayer1(out)
        out = self.hidlayer2(out)
        out = self.outlayer(out)
        out = F.normalize(out, dim=-1)
        return out
class Gauss_re(nn.Module):
    def __init__(self, ninp, nhid,alpha=0.00001):
        super(Gauss_re, self).__init__()
        self.embed = MLP(ninp,nhid)
        self.alpha = alpha

    def forward(self,x,x_,y):
        out = torch.zeros((x_.size(0),x_.size(1),y.size(-1)))
        x = self.embed(x)
        x_ = self.embed(x_)
        for i in range(x.size(0)):
            K_ = torch.cosine_similarity(x_[i].unsqueeze(1),x[i].unsqueeze(0),dim=2)
            K_ = torch.sqrt(K_**2)
            K = torch.cosine_similarity(x[i].unsqueeze(1),x[i].unsqueeze(0),dim=2)
            K = torch.sqrt(K**2)
            out[i] = K_@torch.inverse(K+self.alpha*torch.eye(K.size(0)))@y[i]
        return out
    # def forward(self):
x = torch.linspace(0,4*np.pi,1000).reshape((-1,1))
y = torch.sin(x)
train_x = x[torch.randperm(x.size(0))[:39]][None,:,:]
train_y = torch.sin(train_x)
test_x = torch.linspace(0,4*np.pi,256).reshape((1,-1,1))
test_y = torch.sin(test_x)

BATCH_SIZE = 4
dataset = dataset(x_train=x,y_train=y,num_one_train=40)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

model = Gauss_re(ninp=1,nhid=10)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_func = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.99)
for epoch in range(1000):
    model.train()
    avg_loss = 0
    num_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        # 这里是只取训练数据的意思吗，X_batch和y_batch是怎么分开的?
        # 答：X_batch和y_batch是一一对应的，只不过顺序打乱了，参考torch.utils.data.ipynb
        output = model.forward(X_batch[:,:-1,:],X_batch[:,-1:,:],y_batch[:,:-1,:])
        loss = loss_func(output, y_batch[:,-1:,0])
        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_loss+=loss.data.item()
        num_loss+=X_batch.size(0)
    model.eval()
    with torch.no_grad():
        pred_y = model.forward(train_x,test_x,train_y)
        loss = loss_func(pred_y, test_y)
    print("损失值:{}%：".format(100*loss.data.item()/test_x.size(0)))
fig, ax = plt.subplots()
# 绘制线条
ax.plot(test_x[0], test_y[0], label='true')
ax.plot(test_x[0], pred_y[0], label='pred')

# 添加标题和标签
ax.set_title('true and pred')
ax.set_xlabel('x')
ax.set_ylabel('y')

# 添加图例
ax.legend()

# 显示图形
plt.show()
print(1)