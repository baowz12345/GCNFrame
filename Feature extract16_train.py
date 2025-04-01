# 这个文件是用训练数据做模型的
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d, Linear, Sigmoid
import torch.nn as nn
# 假设你已经有了节点的特征
# df = pd.read_csv('1.csv')  # 替换为你的文件路径
# features = torch.tensor(df.values, dtype=torch.float)
path= "./data_tyy/feature/"#
filename='allngtdm'

all_dataset=f"{filename}.txt"
all_dataset= np.genfromtxt("{}{}".format(path, all_dataset),
                               dtype=np.dtype(str))
# 新增
all_dataset=all_dataset.tolist()
label_1 = [item for item in all_dataset if item[-1] == '1']
label_0 = [item for item in all_dataset if item[-1] == '0']
split_point1 = len(label_1)// 2
split_point0 = len(label_0) // 2
sub_arrays1_qian = label_1[:split_point1]
sub_arrays1_hou = label_1[-split_point1:]
sub_arrays0_qian = label_0[:split_point0]
sub_arrays0_hou = label_0[-split_point0:]
# 新增
train_dataset=sub_arrays1_qian+sub_arrays0_qian
test_dataset=sub_arrays1_hou+sub_arrays0_hou
train1=np.array(train_dataset)
train1 = pd.DataFrame(train1)
train1.to_csv(f'./data_tyy/feature_separate/{filename}_train.csv', header=False,index=False)
test2=np.array(test_dataset)
test2 = pd.DataFrame(test2)
test2.to_csv(f'./data_tyy/feature_separate/{filename}_test.csv', header=False,index=False)
half_of_train=len(train_dataset)
half_of_test=len(test_dataset)

data_for_train = np.array(train_dataset)#！！！3
train_str=data_for_train[0:half_of_train, 0:-1]

train_data=[]
for i in range(len(train_str)):
    test1=[]
    for num in train_str[i]:
        num1=float(num)
        test1.append(num1)
    train_data.append(test1)
features = torch.tensor(train_data)



# 获取节点数
num_nodes = features.shape[0]

# 生成所有可能的边缘对（这里是无向图的示例）
all_edges = [[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)]

# 随机打乱边缘对
np.random.shuffle(all_edges)

# 选择大约50%的边缘
num_edges_to_select = len(all_edges) // 2
selected_edges = all_edges[:num_edges_to_select]


# 将选中的边缘转换为PyTorch张量
edge_index = torch.tensor(selected_edges, dtype=torch.long).t().contiguous()#一个奇怪的转置

# 创建Data对象
data = Data(x=features, edge_index=edge_index)


# 生成标签数据：前3807个为1，后3806个为0
# labels = torch.cat((torch.ones(468, dtype=torch.long), torch.zeros(467, dtype=torch.long)))
train_labels_str=data_for_train[0:half_of_train, -1]
labels = torch.tensor([int(x) for x in train_labels_str])

data.y = labels


print(data)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # 第一层GCN，输出维度16
        self.bn2 = BatchNorm1d(16)
        self.conv2 = GCNConv(16, num_classes)
        self.conv3 = GCNConv(2, 16)
        #self.fc = nn.Linear(2, 64)# 第二层GCN，输出维度为类别数
        #self.sigmoid = Sigmoid()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层GCN后接ReLU激活函数
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = F.relu(x)

        # Dropout层，防止过拟合
        #x = F.dropout(x, training=self.training)
        # 第二层GCN
        x = self.conv2(x, edge_index)#训练F.log_softmax(x, dim=1)
        # print(x.shape)
        # x = self.conv3(x, edge_index)#提取特征x!!!1
        # print(x.shape)
        #x = self.fc(x)
        #x = self.sigmoid(x)
        #F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)#!!!2
    # 假设你有一个数据对象data，它有节点特征和类别标签
def compute_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)



num_node_features = data.num_node_features
print('节点数量：',num_node_features)
num_classes = len(data.y.unique())  # 假设data.y包含节点的类别标签
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCEWithLogitsLoss()

model = GCN(num_node_features=num_node_features, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)
best_accuracy = 0.0
best_epoch = 0
model.train()

def train():
    for epoch in range(300):
        optimizer.zero_grad()#梯度归零
        out = model(data)#调用创建的GCN模型
        #out = out.squeeze()
        loss = F.nll_loss(out, data.y)
        #loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()#更新参数
        acc = compute_accuracy(out, data.y)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {acc}')
        if acc >= 0.9:
            print(f'Reached desired accuracy of 0.9 or higher. Stopping training.')
            # 保存模型
            torch.save(model.state_dict(), f'./models/feature16/tyy_{filename}.pt')
            best_accuracy = acc
            best_epoch = epoch + 1
            break

def feature_extract():
    model = GCN(num_node_features, 2)

    # 加载已保存的模型参数
    model.load_state_dict(torch.load(f'./models/feature16/tyy_{filename}.pt'))

    # 将模型设置为评估模式
    model.eval()

    # 在数据上进行特征提取
    with torch.no_grad():
        features = model(data)
    features=features.numpy()
    shuchu = np.concatenate((features, train_labels_str.reshape(-1, 1)), axis=1)

    df = pd.DataFrame(shuchu)  # 如果 features 是一个张量，则使用 features.numpy() 将其转换为NumPy数组
    # 将DataFrame保存为CSV文件
    df.to_csv(f'./output/feature16_lab/{filename}.csv', index=False)  #将索引列（行号）排除在CSV文件之外


#  第一步：模型预训练，先在forward更改模型组件
train()

#  第二步：使用预训练的模型提取进一步特征
# feature_extract()
