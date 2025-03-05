from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
import rdkit
from rdkit import Chem
tqdm.pandas()

class GCNet(torch.nn.Module):
    def __init__(self, v_in, conv_dim, net_dim, n_extra_inputs, heads):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(v_in, conv_dim)
        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.gat = GATConv(conv_dim, conv_dim, heads=heads)
        self.conv2 = GCNConv(conv_dim * heads, conv_dim)       
        self.mlp1 = nn.Linear(conv_dim + n_extra_inputs, net_dim + n_extra_inputs)    
        self.mlp2 = nn.Linear(net_dim + n_extra_inputs, net_dim)       
        self.mlp3 = nn.Linear(net_dim, 1)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)      
        x = F.relu(x)      
        x = self.dropout(x)
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)   
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp3(x)
        return x
    
class AccumulationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, device, dataloader, optimizer, loss_fn):
    model.train()
    loss_accum = AccumulationMeter()
    for batch in dataloader:
        X, y = batch, batch.y
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y.reshape(-1, 1))
        loss_accum.update(loss.item(), y.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_accum.avg

def test(model, device, dataloader, loss_fn):
    model.eval()
    loss_accum = AccumulationMeter()
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch, batch.y
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y.reshape(-1, 1))
            loss_accum.update(loss.item(), y.size(0))

    return loss_accum.avg

df = pd.read_csv('Data/clean_data/tox_iqr_data.csv')
train_df = df[df['Split'] == 'Training'].copy()
test_df = df[df['Split'] == 'Testing'].copy()
y_train = train_df['logEC50'].values
y_test = test_df['logEC50'].values
df['mol_SMILES'] = df['IL_SMILES'].apply(Chem.MolFromSmiles)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

possible_atom_list = ['C', 'O', 'N', 'Cl','Br', 'P','S','F','B', 'I' ,'Si']
possible_hybridization = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D2
    ]
possible_num_bonds = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge = [0, 1, -1]
possible_num_Hs  = [0, 1, 2, 3]

def atom_features(atom):
    Symbol = atom.GetSymbol()
    Type_atom = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom = [atom.IsInRing()]
    Aromaticity = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + Bonds_atom + Formal_charge + num_Hs
    return np.array(results).astype(np.float32)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def bond_features(bond):
    bt = bond.GetBondType()
    # Features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]
    return np.array(bond_feats).astype(np.float32)
 
def comp_to_graph(df, mol_column, target):
    def process_component(comp):
        edge_attr = []
        edge_index = get_bond_pair(comp)
        for bond in comp.GetBonds():
            edge_attr.extend([bond_features(bond)] * 2)
        return edge_attr, edge_index 

    def create_graph_object(nodes, edges_indx, edges_info, y):
        graph = Data(x=nodes, edge_index=edges_indx, edge_attr=edges_info)
        graph.y = torch.tensor(y, dtype=torch.float)
        return graph

    def create_graph(comp, y_val):
        atoms = comp.GetAtoms()
        node_features = [atom_features(atom) for atom in atoms]
        edge_attr, edge_index = process_component(comp)
        nodes_info = torch.tensor(np.array(node_features), dtype=torch.float32)
        edges_indx = torch.tensor(np.array(edge_index), dtype=torch.long)
        edges_info = torch.tensor(np.array(edge_attr), dtype=torch.float32)
        graph = create_graph_object(nodes_info, edges_indx, edges_info, y_val)
        return graph

    graphs = []
    comps = df[mol_column].tolist()
    ys = df[target].tolist()
    for y, comp in tqdm(zip(ys, comps), total=len(ys)):
        graph = create_graph(comp, y)
        graphs.append(graph)
    return graphs

df['g_ILs'] = comp_to_graph(df, 'mol_SMILES', 'logEC50')
train_df = df[df['Split'] == 'Training']
test_df = df[df['Split'] == 'Testing']
train_dataloader = DataLoader(train_df['g_ILs'].tolist(), batch_size=32, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_df['g_ILs'].tolist(), batch_size=32, shuffle=False, drop_last=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GCNet(31, 100, 100, 0,4).to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 800
train_loss_save, test_loss_save = [], []
for epoch in tqdm(range(epochs)):
    print("epoch=",epoch)
    train_loss = train(model, device, train_dataloader, optimizer, loss_fn)
    print("train_loss=",train_loss)
    train_loss_save.append(train_loss)
    test_loss = test(model,device,test_dataloader,loss_fn)
    print("test_loss=",test_loss)
    test_loss_save.append(test_loss)

torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict(), 
    },'Tox_SG_GCN_GAT.pth') 
print("The relevant parameters have been saved to the path : Tox_SG_GCN_GAT.pth!")
train_mse = np.sqrt(np.array(train_loss_save))
test_mse = np.sqrt(np.array(test_loss_save))
model = GCNet(31, 100, 100, 0,4).to(device)
checkpoint = torch.load('Tox_SG_GCN_GAT.pth')
model.load_state_dict(checkpoint['model_state_dict'])
y_hat_training, y_hat_testing = [] ,[]
model.eval() 
with torch.no_grad():
  for y_storage, dataloader in zip([y_hat_training, y_hat_testing], [train_dataloader, test_dataloader]):
    for batch in dataloader:
      X, y = batch, batch.y
      X, y = X.to(device), y.to(device)
      y_hat = model(X).cpu().numpy().tolist() 
      y_storage.extend(y_hat)

metrics = [r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error]
metric_names = ['R2', 'MAPE', 'MAE', 'RMSE']
metrics_dict = {}
for split, (y_true, y_pred) in zip(['train', 'test'], [(y_train, y_hat_training), (y_test, y_hat_testing)]):
    metrics_dict[split] = {}
    for name, metric in zip(metric_names, metrics):
      metrics_dict[split][name] = metric(y_true, y_pred)
for split, metrics in metrics_dict.items():
    print(f'Split: {split}')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')
    print()

