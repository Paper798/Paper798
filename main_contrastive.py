import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import torch.nn as nn
from sklearn.metrics import roc_auc_score,classification_report
from torch_geometric.data import Data, DataLoader
from util_loss import SupConLoss
from torch.nn.functional import normalize

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
def create_graph(adj,head):
    struc_map = {}
    for i in range(len(adj)):
        struc_map[head[i]] = []
        for j in range(len(adj[i, :])):
            if i != j and adj[i, j]:
                struc_map[head[i]].append(head[j])

    return struc_map


def build_loc_net(struc, feature_map=[]):
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes


pickle_path = 'dataset_contrastive'

with open(pickle_path,"rb") as f:
    data = pickle.load(f)
f.close()

features,block = data

with open('final_test.pickle','rb') as f:
    feature_test = pickle.load(f)
f.close()

with open('final_label.pickle','rb') as f:
    label_test = pickle.load(f)
f.close()


features = torch.nan_to_num(features, nan=0.0)
feature_test = torch.nan_to_num(feature_test,nan=0.0)


with open('adj.pkl',"rb") as f:
    adj_mat = pickle.load(f)

adj,head = adj_mat
struc = create_graph(adj,head)
graph = build_loc_net(struc, head)


# define the GraphAutoencoder class
class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphAutoencoder, self).__init__()
        self.enc1 = GATConv(in_channels, in_channels)
        self.enc2 = GATConv(in_channels, hidden_channels)
        self.dec1 = GATConv(hidden_channels, in_channels)
        self.dec2 = GATConv(in_channels, in_channels)
        self.m = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = self.enc1(x, edge_index)
        x = F.relu(x)
        h = self.enc2(x, edge_index)
        x = F.relu(h)
        x = self.dec1(x,edge_index)
        x = F.relu(x)
        x = self.dec2(x,edge_index)
        return x,h

# define a custom dataset for multiple graphs with continuous node features
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
#
# create a list of PyG Data objects representing multiple graphs with continuous node features
data_list = []
for i in range(features.shape[0]):
    x_1 = features[i,:,0,:]
    x_2 = features[i,:,1,:]
    edge_index = torch.tensor(graph)
    data_1 = Data(x=x_1, edge_index=edge_index)
    data_2 = Data(x=x_2, edge_index=edge_index)
    data_list.append([data_1,data_2,block[i]])

test_list = []
for i in range(feature_test.shape[0]):
    x = feature_test[i, :, :]  # 10 nodes with 2-dimensional continuous features
    edge_index = torch.tensor(graph)
    data = Data(x=x, edge_index=edge_index)
    test_list.append(data)

# create a custom dataset using the list of PyG Data objects
dataset = MyDataset(data_list)

testdataset = MyDataset(test_list)
#
# create a data loader to iterate over the dataset during training
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size,drop_last =True, shuffle=True)
testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
#
# create a GraphAutoencoder model
model = GraphAutoencoder(in_channels=22, hidden_channels=6).to(device)
projection_head = nn.Sequential(nn.Linear(13*6, 13*22),nn.ReLU(),
                            nn.Linear(13*22, 128),
        ).to(device)
        
#
# define the optimizer and loss function
optimizer = torch.optim.Adam(list(model.parameters())+list(projection_head.parameters()), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
criterion = torch.nn.MSELoss().to(device)
max_norm = 1.0
num_epochs = 1
lst = []

contrastive_loss = SupConLoss()


for epoch in range(num_epochs):
    model.train()
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        recon_1,h_1 = model(data[0].x.to(device), data[0].edge_index.to(device))
        #print(h_1.shape)
        recon_2, h_2 = model(data[1].x.to(device), data[1].edge_index.to(device))

        h_1 = h_1.view(batch_size,-1)
        h_2 = h_2.view(batch_size,-1)
        z_1 = normalize(projection_head(h_1),dim = 1)
        z_2 = normalize(projection_head(h_2),dim = 1)

        z_1 = z_1.unsqueeze(1)
        z_2 = z_2.unsqueeze(1)

        z = torch.cat((z_1,z_2),dim = 1)



        loss = 0.5*(criterion(recon_1, data[0].x.to(device)) + criterion(recon_2, data[1].x.to(device))) + 0.1*contrastive_loss(z,labels = data[2])


        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

Threshold = loss.item()
print(Threshold)
# check the reconstruction of the input graphs
model.eval()
anomaly_score = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        recon,_ = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(recon, data.x.to(device)).cpu()
        anomaly_score.append(loss.item())

binary_anomaly = [1 if x >= Threshold else 0 for x in anomaly_score]
