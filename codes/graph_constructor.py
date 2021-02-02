from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import numpy as np
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot
import pickle
import os
from dgl.data.chem import BaseBondFeaturizer
from functools import partial
import warnings
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing
from itertools import repeat
warnings.filterwarnings('ignore')


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def graphs_from_mol(m1, m2, add_self_loop=False, add_3D=False):
    """
    :param m1: ligand molecule
    :param m2: pocket molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return: 
    complex: graphs contain m1, m2 and complex
    """
    # the distance threshold to determine the interaction between ligand atoms and protein atoms
    dis_threshold = 5
    # small molecule
    new_order1 = rdmolfiles.CanonicalRankAtoms(m1)
    mol1 = rdmolops.RenumberAtoms(m1, new_order1)

    # pocket
    new_order2 = rdmolfiles.CanonicalRankAtoms(m2)
    mol2 = rdmolops.RenumberAtoms(m2, new_order2)

    # construct graphs
    g1 = dgl.DGLGraph()  # small molecule
    g2 = dgl.DGLGraph()  # pocket

    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g1.add_nodes(num_atoms_m1)
    g2.add_nodes(num_atoms_m2)

    if add_self_loop:
        nodes1 = g1.nodes()
        g1.add_edges(nodes1, nodes1)
        nodes2 = g2.nodes()
        g2.add_edges(nodes2, nodes2)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g1.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u)
        dst2.append(v)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g2.add_edges(src_ls2, dst_ls2)

    # add interaction edges, only consider the euclidean distance within dis_threshold
    g3 = dgl.DGLGraph()
    g3.add_nodes(num_atoms)
    dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    node_idx = np.where(dis_matrix < dis_threshold)
    src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
    dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
    g3.add_edges(src_ls3, dst_ls3)

    # assign atom features
    # 'h', features of atoms
    g1.ndata['h'] = AtomFeaturizer(mol1)['h']
    g2.ndata['h'] = AtomFeaturizer(mol2)['h']

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    g1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    g2_d = torch.tensor(dis_matrix_P[src_ls2, dst_ls2], dtype=torch.float).view(-1, 1)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
    g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # efeats1
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g1.edata['e'] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g2.edata['e'] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g1.edata['e'] = torch.cat([g1.edata['e'], g1_d * 0.1], dim=-1)
    g2.edata['e'] = torch.cat([g2.edata['e'], g2_d * 0.1], dim=-1)
    g3.edata['e'] = g3_d * 0.1

    if add_3D:
        g1.ndata['pos'] = mol1.GetConformers()[0].GetPositions()
        g2.ndata['pos'] = mol2.GetConformers()[0].GetPositions()

        # calculate the 3D info for g1
        src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g1.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th], dim=-1)

        # calculate the 3D info for g2
        src_nodes, dst_nodes = g2.find_edges(range(g2.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g2.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g2), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g2.edata['e'] = torch.cat([g2.edata['e'], D3_info_th], dim=-1)
        g1.ndata.pop('pos')
        g2.ndata.pop('pos')

    return g1, g2, g3


def graphs_from_mol_vs(path, key):
    """
    :param path:
    :param key:
    :return:
    """
    add_self_loop = False
    try:
        with open(path, 'rb') as f:
            mol1, mol2 = pickle.load(f)
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = 5
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')
        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
            print(key)
        else:
            status = True
    except:
        g = None
        g3 = None
        status = False
    if status:
        # save_graphs('/data2/dejunjiang/wspy/dpi/lit_pcba_jdj/PKM2ChimeraPrepared5A/Graphs/'+key, [g])
        # save_graphs(r'C:\Users\15766\Desktop\Graphs' + '\\' + key, [g])
        # save_graphs('/data2/dejunjiang/wspy/dpi/lit_pcba_jdj/PKM2ChimeraPrepared5A/Graphs3/'+key, [g3])
        # save_graphs(r'C:\Users\15766\Desktop\Graphs3' + '\\' + key, [g3])
        # return {key: (num_atoms_m1, num_bonds1 * 2, num_atoms_m2, num_bonds2 * 2)}
        pass


def graphs_from_mol_v2(path):
    add_self_loop = False
    try:
        with open(path, 'rb') as f:
            mol1, mol2 = pickle.load(f)
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = 8
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')
        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
        else:
            status = True
    except:
        g = None
        g3 = None
        status = False
    return status, g, g3


def graphs_from_mol_mul(dir, key, label, graph_dic_path, dis_threshold=8.0, path_marker='\\'):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """

    add_self_loop = False
    try:
        with open(dir, 'rb') as f:
            mol1, mol2 = pickle.load(f)
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')
        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
            print(key)
        else:
            status = True
    except:
        g = None
        g3 = None
        status = False
    if status:
        # linux
        # with open(graph_dic_path+key, 'wb') as f:
        #     pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)
        # window
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)


def collate_fn(data_batch):
    graphs1, graphs2, graphs3, Ys, dirs = map(list, zip(*data_batch))
    bg1 = dgl.batch(graphs1)
    bg2 = dgl.batch(graphs2)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg1, bg2, bg3, Ys, dirs


def collate_fn_mul(data_batch):
    graphs1, graphs2, graphs3, Ys, masks, dirs = map(list, zip(*data_batch))
    bg1 = dgl.batch(graphs1)
    bg2 = dgl.batch(graphs2)
    bg3 = dgl.batch(graphs3)
    Ys = torch.stack(Ys, dim=0)
    masks = torch.stack(masks, dim=0)
    return bg1, bg2, bg3, Ys, masks, dirs


def collate_fn_cat(data_batch):
    graphs1, graphs2, graphs3, Ys, global_feats, dirs = map(list, zip(*data_batch))
    bg1 = dgl.batch(graphs1)
    bg2 = dgl.batch(graphs2)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    global_feats = torch.tensor(global_feats, dtype=torch.float)
    return bg1, bg2, bg3, Ys, global_feats, dirs


def collate_fn_vs(data_batch):
    graphs, graphs3, Keys, Ys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.tensor(Ys, dtype=torch.float), dim=-1)
    return bg, bg3, Keys, Ys


def collate_fn_vs_v2(data_batch):
    graphs, graphs3, Keys, Nodes, Ys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.tensor(Ys, dtype=torch.float), dim=-1)
    return bg, bg3, Keys, Nodes, Ys


def collate_fn_v2(data_batch):
    graphs, graphs3, Ys, = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys


def collate_fn_v2_MulPro(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    graphs, graphs3, Ys, keys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys, keys


def collate_fn_v2_2d(data_batch):
    graphs, graphs3, Ys, = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    bg.edata['e'] = bg.edata['e'][:, :11]  # only consider the 2d info for the ligand and protein graphs
    # mask the distance info in graphs3, this is compromised method
    bg3.edata['e'] = torch.zeros(bg3.number_of_edges(), 1)
    return bg, bg3, Ys


def collate_fn_v2_mask_protein(data_batch):
    graphs, graphs3, Ys, Nodes = map(list, zip(*data_batch))
    # mask matrix for protein info
    mask = []
    for i, node_info in enumerate(Nodes):
        ligand_atoms, ligand_edges, protein_atoms, protein_edges = list(node_info.values())[0]
        mask.append(torch.ones(ligand_atoms, 1))  # ligand atoms
        mask.append(torch.zeros(protein_atoms, 1))  # protein atoms
    mask = torch.cat(mask)
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    # mask the interaction info in graphs3
    bg3.edata['e'] = torch.zeros(bg3.number_of_edges(), 1)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys, mask


def collate_fn_v2_mask_ligand(data_batch):
    graphs, graphs3, Ys, Nodes = map(list, zip(*data_batch))
    # mask matrix for ligand info
    mask = []
    for i, node_info in enumerate(Nodes):
        ligand_atoms, ligand_edges, protein_atoms, protein_edges = list(node_info.values())[0]
        mask.append(torch.zeros(ligand_atoms, 1))  # ligand atoms
        mask.append(torch.ones(protein_atoms, 1))  # protein atoms
    mask = torch.cat(mask)
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    # mask the interaction info in graphs3
    bg3.edata['e'] = torch.zeros(bg3.number_of_edges(), 1)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys, mask


def collate_fn_v2_(data_batch):
    """
       used for trained on the original complex dataset and tested on the ligand-only and protein-only datasets
       :param data_batch:
       :return:
       """
    graphs, graphs3, Ys, Nodes = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys


class GraphDataset(object):
    def __init__(self, labels, data_dirs, cache_file_path, add_3D):
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.data_dirs = data_dirs
        self.cache_file_path = cache_file_path
        self.add_3D = add_3D
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_file_path):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            for i, dir in enumerate(self.data_dirs):
                print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                with open(dir, 'rb') as f:
                    lig, pro = pickle.load(f)
                g1, g2, g3 = graphs_from_mol(lig, pro, add_3D=self.add_3D)
                self.graphs.append({'lig': g1, 'pro': g2, 'com': g3})
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

    def __getitem__(self, indx):
        graphs = self.graphs[indx]
        return graphs['lig'], graphs['pro'], graphs['com'], self.labels[indx], self.data_dirs[indx]

    def __len__(self):
        return len(self.data_dirs)


class GraphDatasetMul(object):
    def __init__(self, labels, data_dirs, cache_file_path, add_3D):
        self.labels = F.zerocopy_from_numpy(np.nan_to_num(labels).astype(np.float32))
        self.masks = F.zerocopy_from_numpy((~np.isnan(labels)).astype(np.float32))
        self.data_dirs = data_dirs
        self.cache_file_path = cache_file_path
        self.add_3D = add_3D
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_file_path):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            for i, dir in enumerate(self.data_dirs):
                print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                with open(dir, 'rb') as f:
                    lig, pro = pickle.load(f)
                g1, g2, g3 = graphs_from_mol(lig, pro, add_3D=self.add_3D)
                self.graphs.append({'lig': g1, 'pro': g2, 'com': g3})
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

    def __getitem__(self, indx):
        graphs = self.graphs[indx]
        return graphs['lig'], graphs['pro'], graphs['com'], self.labels[indx], self.masks[indx], self.data_dirs[indx]

    def __len__(self):
        return len(self.data_dirs)


class GraphDatasetCat(object):
    def __init__(self, labels, global_feats, data_dirs, cache_file_path, add_3D):
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.data_dirs = data_dirs
        self.cache_file_path = cache_file_path
        self.add_3D = add_3D
        self.global_feats = global_feats
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_file_path):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            for i, dir in enumerate(self.data_dirs):
                print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                with open(dir, 'rb') as f:
                    lig, pro = pickle.load(f)
                g1, g2, g3 = graphs_from_mol(lig, pro, add_3D=self.add_3D)
                self.graphs.append({'lig': g1, 'pro': g2, 'com': g3})
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

    def __getitem__(self, indx):
        graphs = self.graphs[indx]
        return graphs['lig'], graphs['pro'], graphs['com'], self.labels[indx], self.global_feats[indx], self.data_dirs[
            indx]

    def __len__(self):
        return len(self.data_dirs)


class GraphDatasetVS(object):
    def __init__(self, keys, cache_file_path1, cache_file_path2):
        self.keys = keys
        self.cache_file_path1 = cache_file_path1
        self.cache_file_path2 = cache_file_path2

    def __getitem__(self, indx):
        g = load_graphs(self.cache_file_path1, [indx])
        g3 = load_graphs(self.cache_file_path2, [indx])
        key = self.keys[indx]
        Y = 1 if 'active' in key else 0
        return g[0][0], g3[0][0], key, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetVS1(object):
    def __init__(self, keys, cache_file_path1, cache_file_path2):
        self.keys = keys
        self.graphs, _ = load_graphs(cache_file_path1)
        self.graphs3, _ = load_graphs(cache_file_path2)

    def __getitem__(self, indx):
        g = self.graphs[indx]
        g3 = self.graphs3[indx]
        key = self.keys[indx]
        Y = 1 if 'active' in key else 0
        return g, g3, key, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetVS2(object):
    """
    This class was used for masking part of the inputs
    """

    def __init__(self, keys, nodes, cache_file_path1, cache_file_path2):
        self.keys = keys
        self.nodes = nodes
        self.graphs, _ = load_graphs(cache_file_path1)
        self.graphs3, _ = load_graphs(cache_file_path2)

    def __getitem__(self, indx):
        g = self.graphs[indx]
        g3 = self.graphs3[indx]
        key = self.keys[indx]
        node = self.keys[indx]
        Y = 1 if 'active' in key else 0
        return g, g3, key, node, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetVS1MP(object):
    """
    This class was used for distributed training, each worker only load its assigned dataset,
    not copy the whole dataset.
    """

    def __init__(self, keys, cache_file_path1, cache_file_path2, idx_list):
        """
        :param keys:
        :param cache_file_path1:
        :param cache_file_path2:
        :param idx_list: list of index of graph to be loaded. If not specified, will load all graphs from file
        and corresponding keys
        """
        keys_arr = np.array(keys)
        self.keys = keys_arr[idx_list].tolist() if idx_list else keys
        del keys_arr
        self.graphs, _ = load_graphs(cache_file_path1, idx_list)
        self.graphs3, _ = load_graphs(cache_file_path2, idx_list)

    # def __getitem__(self, indx):
    #     g = self.graphs[indx]
    #     g3 = self.graphs3[indx]
    #     key = self.keys[indx]
    #     Y = 1 if 'active' in key else 0
    #     return g, g3, key, Y
    def __getitem__(self, indx):
        g = self.graphs[indx]
        g3 = self.graphs3[indx]
        key = self.keys[indx]
        Y = 0 if 'inactive' in key else 1
        return g, g3, key, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetVS1MP_Mask(object):
    """
    This class was used for distributed training, each worker only load its assigned dataset,
    not copy the whole dataset, and also support mask part of the inputs
    """

    def __init__(self, keys, nodes, cache_file_path1, cache_file_path2, idx_list):
        """
        :param keys:
        :param cache_file_path1:
        :param cache_file_path2:
        :param idx_list: list of index of graph to be loaded. If not specified, will load all graphs from file
        and corresponding keys
        """
        keys_arr = np.array(keys)
        nodes_arr = np.array(nodes)
        self.keys = keys_arr[idx_list].tolist() if idx_list else keys
        self.nodes = nodes_arr[idx_list].tolist() if idx_list else nodes
        del keys_arr
        del nodes
        self.graphs, _ = load_graphs(cache_file_path1, idx_list)
        self.graphs3, _ = load_graphs(cache_file_path2, idx_list)

    def __getitem__(self, indx):
        g = self.graphs[indx]
        g3 = self.graphs3[indx]
        key = self.keys[indx]
        node = self.nodes[indx]
        Y = 1 if 'active' in key else 0
        return g, g3, key, node, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetVS1MP_V2(object):
    """
    This class was used for distributed training, each worker only load its assigned dataset,
    not copy the whole dataset.
    """

    def __init__(self, keys, graphs, graphs3):
        self.keys = keys
        self.graphs = graphs
        self.graphs3 = graphs3

    def __getitem__(self, indx):
        g = self.graphs[indx]
        g3 = self.graphs3[indx]
        key = self.keys[indx]
        Y = 1 if 'active' in key else 0
        return g, g3, key, Y

    def __len__(self):
        return len(self.keys)


class GraphDatasetV2(object):
    def __init__(self, labels, data_dirs, cache_file_path1, cache_file_path2, add_3D):
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.data_dirs = data_dirs
        self.cache_file_path1 = cache_file_path1
        self.cache_file_path2 = cache_file_path2
        self.add_3D = add_3D
        self.unresolved_idx = []
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_file_path1) and os.path.exists(self.cache_file_path2):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path1, 'rb') as f:
                self.graphs = pickle.load(f)
            with open(self.cache_file_path2, 'rb') as f:
                self.graphs3 = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            self.graphs3 = []
            for i, dir in enumerate(self.data_dirs):
                print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                status, g, g3 = graphs_from_mol_v2(dir)
                if status:
                    self.graphs.append(g)
                    self.graphs3.append(g3)
                else:
                    self.unresolved_idx.append(i)
            with open(self.cache_file_path1, 'wb') as f:
                pickle.dump(self.graphs, f)
            with open(self.cache_file_path2, 'wb') as f:
                pickle.dump(self.graphs3, f)

    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], self.labels[indx]

    def __len__(self):
        return len(self.graphs)


class GraphDatasetV2MulPro(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, keys, labels, data_dirs, graph_ls_path, graph_dic_path, num_process=6, dis_threshold=8.0,
                 add_3D=True, path_marker='\\'):
        """
        :param keys: the keys for the complexs, list
        :param labels: the corresponding labels for the complexs, list
        :param data_dirs: the corresponding data_dirs for the complexs, list
        :param graph_ls_path: the cache path for the total graphs objects (graphs.bin, graphs3.bin), labels, keys
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param num_process: the numer of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the atom-pair interactions
        :param add_3D: add the 3D geometric features to the edges of graphs
        :param path_marker: '\\' for windows and '/' for linux
        """
        self.origin_keys = keys
        self.origin_labels = labels
        self.origin_data_dirs = data_dirs
        self.graph_ls_path = graph_ls_path
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.add_3D = add_3D
        self.dis_threshold = dis_threshold
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.graph_ls_path+self.path_marker+'g.bin'):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_path + self.path_marker + 'g.bin', 'rb') as f:
                self.graphs = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'rb') as f:
                self.graphs3 = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'rb') as f:
                self.keys = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'rb') as f:
                self.labels = pickle.load(f)
        else:
            graph_dic_paths = repeat(self.graph_dic_path, len(self.origin_data_dirs))
            dis_thresholds = repeat(self.dis_threshold, len(self.origin_data_dirs))
            path_markers = repeat(self.path_marker, len(self.origin_data_dirs))

            print('Generate complex graph...')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(graphs_from_mol_mul,
                         zip(self.origin_data_dirs, self.origin_keys, self.origin_labels, graph_dic_paths,
                             dis_thresholds, path_markers))
            pool.close()
            pool.join()

            # collect the generated graph for each complex
            self.graphs = []
            self.graphs3 = []
            self.labels = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.graphs.append(graph_dic['g'])
                    self.graphs3.append(graph_dic['g3'])
                    self.labels.append(graph_dic['label'])
            # store to the disk
            with open(self.graph_ls_path + self.path_marker + 'g.bin', 'wb') as f:
                pickle.dump(self.graphs, f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'wb') as f:
                pickle.dump(self.graphs3, f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'wb') as f:
                pickle.dump(self.keys, f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'wb') as f:
                pickle.dump(self.labels, f)

        # delete the temporary files
        cmdline = 'rm -rf %s' % (self.graph_dic_path + self.path_marker + '*')  # graph_dic_path
        os.system(cmdline)

    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)


class GraphDatasetV2Mask(object):
    def __init__(self, labels, data_dirs, nodes, cache_file_path1, cache_file_path2, add_3D):
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.data_dirs = data_dirs
        self.nodes = nodes
        self.cache_file_path1 = cache_file_path1
        self.cache_file_path2 = cache_file_path2
        self.add_3D = add_3D
        self.unresolved_idx = []
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_file_path1) and os.path.exists(self.cache_file_path2):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path1, 'rb') as f:
                self.graphs = pickle.load(f)
            with open(self.cache_file_path2, 'rb') as f:
                self.graphs3 = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            self.graphs3 = []
            for i, dir in enumerate(self.data_dirs):
                print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                status, g, g3 = graphs_from_mol_v2(dir)
                if status:
                    self.graphs.append(g)
                    self.graphs3.append(g3)
                else:
                    self.unresolved_idx.append(i)
            with open(self.cache_file_path1, 'wb') as f:
                pickle.dump(self.graphs, f)
            with open(self.cache_file_path2, 'wb') as f:
                pickle.dump(self.graphs3, f)

    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], self.labels[indx], self.nodes[indx]

    def __len__(self):
        return len(self.graphs)
