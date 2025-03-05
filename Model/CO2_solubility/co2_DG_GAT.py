from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.nn.pool import global_add_pool
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from mordred.Polarizability import APol, BPol
from mordred.TopoPSA import TopoPSA
tqdm.pandas()

class GATNet(torch.nn.Module):
    def __init__(self, v_in, conv_dim, net_dim, n_extra_inputs):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(v_in, conv_dim, heads=4, concat=True)
        self.gat2 = GATConv(conv_dim * 4, conv_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(conv_dim * 4)
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.mlp1 = nn.Linear(conv_dim + n_extra_inputs, net_dim + n_extra_inputs)
        self.mlp2 = nn.Linear(net_dim + n_extra_inputs, net_dim)
        self.mlp3 = nn.Linear(net_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        T = torch.reshape(data.Temp, (-1, 1))
        P = torch.reshape(data.P, (-1, 1))
        HB = torch.reshape(data.HB, (-1, 1))
        ap_Cation = torch.reshape(data.ap_Cation, (-1, 1))
        ap_Anion = torch.reshape(data.ap_Anion, (-1, 1))
        bp_Cation = torch.reshape(data.bp_Cation, (-1, 1))
        bp_Anion = torch.reshape(data.bp_Anion, (-1, 1))
        topopsa_Cation = torch.reshape(data.topopsa_Cation, (-1, 1))
        topopsa_Anion = torch.reshape(data.topopsa_Anion, (-1, 1))
        x = torch.cat((x, T, P, HB, ap_Cation, ap_Anion, bp_Cation, bp_Anion, topopsa_Cation, topopsa_Anion), 1)
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.dropout(x)
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

df = pd.read_csv('Data/clean_data/co2_iqr_data.csv')
train_df = df[df['Split'] == 'Training'].copy()
test_df = df[df['Split'] == 'Testing'].copy()
features = ['Temperature(K)', 'Pressure(bar)']
X_train = train_df[features].values
y_train = train_df['Xco2(mole fraction)'].values
X_test = test_df[features].values
y_test = test_df['Xco2(mole fraction)'].values
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
norm_feature_names = ['T_norm', 'P_norm']
train_df[norm_feature_names] = X_train_norm
test_df[norm_feature_names] = X_test_norm

df = pd.concat([train_df, test_df])
df['mol_Cation'] = df['Cation_SMILES'].apply(Chem.MolFromSmiles)
df['mol_Anion'] = df['Anion_SMILES'].apply(Chem.MolFromSmiles)

ap_fun = APol()
df['ap_Cation'] = df['mol_Cation'].apply(ap_fun)
df['ap_Anion'] = df['mol_Anion'].apply(ap_fun)
bp_fun = BPol()
df['bp_Cation'] = df['mol_Cation'].apply(bp_fun)
df['bp_Anion'] = df['mol_Anion'].apply(bp_fun)
topopsa_fun = TopoPSA()
df['topopsa_Cation'] = df['mol_Cation'].apply(topopsa_fun)
df['topopsa_Anion'] = df['mol_Anion'].apply(topopsa_fun)

def get_hb_sites(mol):
    return min(rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol))

df['hb_Cation'] = df['mol_Cation'].apply(get_hb_sites)
df['hb_Anion'] = df['mol_Anion'].apply(get_hb_sites)

def get_inter_hb_sites(row):
    mol1 = row['mol_Cation']
    mol2 = row['mol_Anion']
    return min(rdMolDescriptors.CalcNumHBA(mol1), rdMolDescriptors.CalcNumHBD(mol2)) + \
           min(rdMolDescriptors.CalcNumHBA(mol2), rdMolDescriptors.CalcNumHBD(mol1))
df['inter_hb_sites'] = df.apply(get_inter_hb_sites, axis=1)

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
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    return np.array(bond_feats).astype(np.float32)

def sys2graph(df, mol_column_1, mol_column_2, target, single_system=False):
    def process_component(comp):
        edge_attr = []
        edge_index = get_bond_pair(comp)
        for bond in comp.GetBonds():
            edge_attr.extend([bond_features(bond)] * 2)
        return edge_attr, edge_index

    def combine_graph_info(node_feat_c1, edge_indx_c1, edge_attr_c1, node_feat_c2, edge_indx_c2, edge_attr_c2):
        num_features_c1 = len(node_feat_c1)
        nodes_info = torch.tensor(np.array(node_feat_c1 + node_feat_c2), dtype=torch.float32)
        edges_indx = torch.cat((torch.tensor(np.array(edge_indx_c1), dtype=torch.long),
                                torch.tensor(np.array(edge_indx_c2), dtype=torch.long) + num_features_c1), 1)
        edges_info = torch.tensor(np.array(edge_attr_c1 + edge_attr_c2), dtype=torch.float32)
        return nodes_info, edges_indx, edges_info

    def create_graph_object(nodes, edges_indx, edges_info, y):
        graph = Data(x=nodes, edge_index=edges_indx, edge_attr=edges_info)
        graph.y = torch.tensor(y, dtype=torch.float)
        return graph

    def create_graph(c1, c2, y_val):
        atoms_c1 = c1.GetAtoms()
        atoms_c2 = c2.GetAtoms()
        node_features_c1 = [atom_features(atom) for atom in atoms_c1]
        node_features_c2 = [atom_features(atom) for atom in atoms_c2]
        edge_attr_c1, edge_index_c1 = process_component(c1)
        edge_attr_c2, edge_index_c2 = process_component(c2)
        nodes_info, edges_indx, edges_info = combine_graph_info(node_features_c1, edge_index_c1, edge_attr_c1,
                                                               node_features_c2, edge_index_c2, edge_attr_c2)
        graph = create_graph_object(nodes_info, edges_indx, edges_info, y_val)
        return graph

    graphs = []
    c1 = [df[mol_column_1]] if single_system else df[mol_column_1].tolist()
    c2 = [df[mol_column_2]] if single_system else df[mol_column_2].tolist()
    ys = [df[target]] if single_system else df[target].tolist()
    Ts = [df['T_norm']] if single_system else df['T_norm'].tolist()
    Ps = [df['P_norm']] if single_system else df['P_norm'].tolist()
    HBs = [df['inter_hb_sites']] if single_system else df['inter_hb_sites'].tolist()
    ap_Cations = [df['ap_Cation']] if single_system else df['ap_Cation'].tolist()
    ap_Anions = [df['ap_Anion']] if single_system else df['ap_Anion'].tolist()
    bp_Cations = [df['bp_Cation']] if single_system else df['bp_Cation'].tolist()
    bp_Anions = [df['bp_Anion']] if single_system else df['bp_Anion'].tolist()
    topopsa_Cations = [df['topopsa_Cation']] if single_system else df['topopsa_Cation'].tolist()
    topopsa_Anions = [df['topopsa_Anion']] if single_system else df['topopsa_Anion'].tolist()

    for y, comp1, comp2, T, P, HB, ap_Cation, ap_Anion, bp_Cation, bp_Anion, topopsa_Cation, topopsa_Anion in tqdm(zip(ys, c1, c2, Ts, Ps, HBs, ap_Cations, ap_Anions, bp_Cations, bp_Anions, topopsa_Cations, topopsa_Anions), total=len(ys)):
        graph = create_graph(comp1, comp2, y)
        graph.Temp = T
        graph.P = P
        graph.HB = HB
        graph.ap_Cation = ap_Cation
        graph.ap_Anion = ap_Anion
        graph.bp_Cation = bp_Cation
        graph.bp_Anion = bp_Anion
        graph.topopsa_Cation = topopsa_Cation
        graph.topopsa_Anion = topopsa_Anion
        graphs.append(graph)
    return graphs

df['g_ILs'] = sys2graph(df, 'mol_Cation', 'mol_Anion', 'Xco2(mole fraction)')
train_df = df[df['Split'] == 'Training']
test_df = df[df['Split'] == 'Testing']
train_dataloader = DataLoader(train_df['g_ILs'].tolist(), batch_size=32, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_df['g_ILs'].tolist(), batch_size=32, shuffle=False, drop_last=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GATNet(31,9, 9, 9).to(device) 
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 800

train_loss_save, test_loss_save = [], []
for epoch in tqdm(range(epochs)):
    print("epoch=", epoch)
    train_loss = train(model, device, train_dataloader, optimizer, loss_fn)
    print("train_loss=", train_loss)
    train_loss_save.append(train_loss)
    test_loss = test(model, device, test_dataloader, loss_fn)
    print("test_loss=", test_loss)
    test_loss_save.append(test_loss)

torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'CO2_DG_GAT.pth')
print("The relevant parameters have been saved to the path: CO2_DG_GAT.pth!")
train_mse = np.sqrt(np.array(train_loss_save))
test_mse = np.sqrt(np.array(test_loss_save))

model = GATNet(31,9, 9, 9).to(device)
checkpoint = torch.load('CO2_DG_GAT.pth')
model.load_state_dict(checkpoint['model_state_dict'])
y_hat_training, y_hat_testing = [], []
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
