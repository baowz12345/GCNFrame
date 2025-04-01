import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch import optim

# 定义卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.01)
        self.pool = nn.MaxPool1d(pool_size)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.batch_norm(x)
        return x

# 定义多头注意力块
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_output_weights

# 定义模型
class CBLANE(nn.Module):
    def __init__(self, input_shape, heads):
        super(CBLANE, self).__init__()
        self.zero_pad = nn.ZeroPad2d((3, 3, 0, 0))
        self.conv_block_0 = ConvBlock(4, 256, 8, 0, 1)
        self.conv_block_1 = ConvBlock(256, 128, 4, 2, 1)  # Conv1D padding=2 == ZeroPadding1D(1)
        self.conv_block_2 = ConvBlock(128, 64, 2, 1, 2)
        self.conv_block_3 = ConvBlock(64, 64, 2, 1, 2)

        self.query_conv = nn.Conv1d(64, 64, 8, padding=4)
        self.multihead_attn = MultiHeadAttentionBlock(64, heads)

        self.bidirectional_lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64, 21907)
        self.dense2 = nn.Linear(21907, 21907)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整输入形状为 (batch, channels, seq_len)
        x = self.zero_pad(x)
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        query = self.query_conv(x)

        query = query.permute(2, 0, 1)  # 调整形状为 (seq_len, batch, embed_dim)
        key_value = x.permute(2, 0, 1)

        # 打印张量形状以调试


        attn_output, attn_output_weights = self.multihead_attn(query, key_value, key_value)
        attn_output = attn_output.permute(1, 2, 0)  # 调整回 (batch, embed_dim, seq_len)

        # 确保张量形状匹配
        if attn_output.shape != x.shape:
            min_len = min(attn_output.shape[2], x.shape[2])
            attn_output = attn_output[:, :, :min_len]
            x = x[:, :, :min_len]

        attn_output = attn_output * x  # Multiply

        attn_output = attn_output.permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.bidirectional_lstm(attn_output)
        lstm_out, _ = self.lstm(lstm_out[:, :, :64])  # 只传递前64个通道

        lstm_out = self.flatten(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        output = self.dense(lstm_out)
        output = self.dense2(output)
        output = self.sigmoid(output)
        return output

# 使用示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CBLANE(input_shape=(4096, 4), heads=8).to(device)



train_data = np.load('chr1_seq_section0.npy')
train_labels = np.load('chr1_label_section0.npy')

# 转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(device)

# 创建数据集和数据加载器
dataset = TensorDataset(train_data_tensor, train_labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.BCELoss()  # 二分类任务使用的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # 可以根据需要调整

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        labels = labels  # 确保标签形状与输出匹配

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')