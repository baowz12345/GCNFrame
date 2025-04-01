from GCNFrame import Biodata, GCNmodel
import torch
from torch_geometric.data import DataLoader
import numpy as np


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Biodata(fasta_file="example_data/output1.fasta",
                   label_file=None)
    dataset = data.encode(thread=20)
    model = torch.load("GCN_model.pt", map_location=device)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, follow_batch=['x_src', 'x_dst'])
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            # pred = pred.argmax(dim=1)

            pred = pred.cpu().numpy()

    output_file_path = "example_data/noncoding2.npy"

# 使用 numpy.save() 保存数组为二进制文件
    np.save(output_file_path, pred)

    # model =  GCNmodel.model(label_num=2, other_feature_dim=206).to(device)
    # GCNmodel.train(dataset, model, weighted_sampling=True)
    # GCNmodel.test(model_name="GCN_model.pt", fasta_file="example_data/nature_2017.fasta")
