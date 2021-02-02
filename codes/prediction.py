import rdkit
from graph_constructor import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import torch
import pandas as pd
import os
from dgl.data.utils import split_dataset
from sklearn.metrics import mean_squared_error
import argparse

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    keys = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, Ys, Keys = batch
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            outputs = model(bg, bg3)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            keys.append(Keys)
    return true, pred, keys


lr = 10 ** -3.5
epochs = 5000
batch_size = 128
num_workers = 0
tolerance = 0.0
patience = 70
l2 = 10 ** -6
repetitions = 3
# paras for model
node_feat_size = 40
edge_feat_size_2d = 12
edge_feat_size_3d = 21
graph_feat_size = 128
num_layers = 2
outdim_g3 = 128
d_FC_layer, n_FC_layer = 128, 2
dropout = 0.15
n_tasks = 1
mark = '3d'
path_marker = '/'



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_ls_path', type=str, default='./examples/graph_ls_path',
                           help="absolute path for storing graph list objects")
    argparser.add_argument('--graph_dic_path', type=str, default='./examples/graph_dic_path',
                           help="absolute path for storing graph dictionary objects (temporary files)")
    argparser.add_argument('--model_path', type=str, default='./model/pdb2016_10A_20201230_3d_2_ciap1.pth',
                           help="absolute path for storing pretrained model")
    argparser.add_argument('--cpu', type=bool, default=True,
                           help="using cpu for the prediction (default:True)")
    argparser.add_argument('--gpuid', type=int, default=0,
                           help="the gpu id for the prediction")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating graph objects")
    argparser.add_argument('--input_path', type=str, default='./examples/ign_input',
                           help="the absoute path for storing ign input files")
    args = argparser.parse_args()
    graph_ls_path, graph_dic_path, model_path, cpu, gpuid, num_process, input_path = args.graph_ls_path, \
                                                                                     args.graph_dic_path, \
                                                                                     args.model_path, \
                                                                                     args.cpu, \
                                                                                     args.gpuid, \
                                                                                     args.num_process, \
                                                                                     args.input_path


    keys = os.listdir(input_path)
    labels = []
    data_dirs = []
    for key in keys:
        data_dirs.append(input_path + path_marker + key)
        labels.append(0)
    limit = None
    dis_threshold = 12

    # generating the graph objective using multi process
    test_dataset = GraphDatasetV2MulPro(keys=keys[:limit], labels=labels[:limit], data_dirs=data_dirs[:limit],
                                        graph_ls_path=graph_ls_path,
                                        graph_dic_path=graph_dic_path,
                                        num_process=num_process, dis_threshold=dis_threshold, path_marker=path_marker)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_v2_MulPro)

    DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                 graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                 d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks)
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%s" % gpuid)
    DTIModel.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    DTIModel.to(device)

    test_true, test_pred, key = run_a_eval_epoch(DTIModel, test_dataloader, device)
    test_true = np.concatenate(np.array(test_true), 0).flatten()
    test_pred = np.concatenate(np.array(test_pred), 0).flatten()
    key = np.concatenate(np.array(key), 0).flatten()

    res = pd.DataFrame({'key': key, 'true': test_true, 'pred': test_pred})
    res.to_csv('./stats/prediction_results.csv', index=False)
