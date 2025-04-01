for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    #out = out.squeeze()
    loss = F.nll_loss(out, data.y)
    #loss = criterion(out, data.y.float())
    loss.backward()
    optimizer.step()
    acc = compute_accuracy(out, data.y)
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {acc}')
    if acc >= 0.9:
        print(f'Reached desired accuracy of 0.9 or higher. Stopping training.')
        # 保存模型
        torch.save(model.state_dict(), 'model.pt')
        best_accuracy = acc
        best_epoch = epoch + 1
        break








model = GCN(num_node_features, 2)

# 加载已保存的模型参数
model.load_state_dict(torch.load('model.pt'))

# 将模型设置为评估模式
model.eval()

# 在数据上进行特征提取
with torch.no_grad():
    features = model(data)


df = pd.DataFrame(features.numpy())  # 如果 features 是一个张量，则使用 features.numpy() 将其转换为NumPy数组

# 将DataFrame保存为CSV文件
df.to_csv('features.csv', index=False)  # 将索引列（行号）排除在CSV文件之外