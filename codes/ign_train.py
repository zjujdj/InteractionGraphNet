# File Name: ign_train.py
# E-mail: jiang_dj@zju.edu.cn
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
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import argparse
path_marker = '/'
limit = None
num_process = 48


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, bg3, Ys, _ = batch
        bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
        outputs = model(bg, bg3)
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, Ys, keys = batch
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            outputs = model(bg, bg3)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=200, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=70, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=10 ** -6, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=3, help="the number of independent runs")
    argparser.add_argument('--node_feat_size', type=int, default=40)
    argparser.add_argument('--edge_feat_size_2d', type=int, default=12)
    argparser.add_argument('--edge_feat_size_3d', type=int, default=21)
    argparser.add_argument('--graph_feat_size', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3, help='the number of intra-molecular layers')
    argparser.add_argument('--outdim_g3', type=int, default=200, help='the output dim of inter-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--model_save_dir', type=str, default='./model_save', help='path for saving model')
    argparser.add_argument('--mark', type=str, default='3d')
    args = argparser.parse_args()
    gpuid, lr, epochs, batch_size, num_workers, model_save_dir = args.gpuid, args.lr, args.epochs, args.batch_size, args.num_workers, args.model_save_dir
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions
    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks, mark = args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks, args.mark

    HOME_PATH = os.getcwd()
    all_data = pd.read_csv('./examples/PDB2016ALL.csv')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    # data
    train_dir = './examples/binding_affinity/training/complex'
    valid_dir = './examples/binding_affinity/validation/complex'
    test_dir = './examples/binding_affinity/test/complex'

    # training data
    train_keys = os.listdir(train_dir)
    train_labels = []
    train_data_dirs = []
    for key in train_keys:
        train_labels.append(all_data[all_data['PDB'] == key]['label'].values[0])
        train_data_dirs.append(train_dir + path_marker + key)


    # validtion data
    valid_keys = os.listdir(valid_dir)
    valid_labels = []
    valid_data_dirs = []
    for key in valid_keys:
        valid_labels.append(all_data[all_data['PDB'] == key]['label'].values[0])
        valid_data_dirs.append(valid_dir + path_marker + key)

    # testing data
    test_keys = os.listdir(test_dir)
    test_labels = []
    test_data_dirs = []
    for key in test_keys:
        test_labels.append(all_data[all_data['PDB'] == key]['label'].values[0])
        test_data_dirs.append(test_dir + path_marker + key)


    # generating the graph objective using multi process
    train_dataset = GraphDatasetV2MulPro(keys=train_keys[:limit], labels=train_labels[:limit], data_dirs=train_data_dirs[:limit],
                                        graph_ls_path='./examples/binding_affinity/training/graph_ls_path',
                                        graph_dic_path='./examples/binding_affinity/training/graph_dic_path',
                                        num_process=num_process, path_marker=path_marker)


    valid_dataset = GraphDatasetV2MulPro(keys=valid_keys[:limit], labels=valid_labels[:limit], data_dirs=valid_data_dirs[:limit],
                                         graph_ls_path='./examples/binding_affinity/validation/graph_ls_path',
                                         graph_dic_path='./examples/binding_affinity/validation/graph_dic_path',
                                         num_process=num_process, path_marker=path_marker)
    test_dataset = GraphDatasetV2MulPro(keys=test_keys[:limit], labels=test_labels[:limit],
                                         data_dirs=test_data_dirs[:limit],
                                         graph_ls_path='./examples/binding_affinity/test/graph_ls_path',
                                         graph_dic_path='./examples/binding_affinity/test/graph_dic_path',
                                         num_process=num_process, path_marker=path_marker)

    stat_res = []
    for repetition_th in range(repetitions):
        set_random_seed(repetition_th)
        print('the number of train data:', len(train_dataset))
        print('the number of valid data:', len(valid_dataset))
        print('the number of test data:', len(test_dataset))
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)

        # model
        DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                     graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                     d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks)
        print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(DTIModel)
        device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
        DTIModel.to(device)
        optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)
        dt = datetime.datetime.now()
        filename = './model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond)
        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance,
                                filename=filename)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device)

            # validation
            train_true, train_pred, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
            valid_true, valid_pred, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
            #
            early_stop = stopper.step(valid_rmse, DTIModel)
            end = time.time()
            if early_stop:
                break
            print(
                "epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(DTIModel)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_v2_MulPro)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn_v2_MulPro)
        train_true, train_pred, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
        valid_true, valid_pred, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)
        test_true, test_pred, _ = run_a_eval_epoch(DTIModel, test_dataloader, device)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()

        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()

        pd_tr = pd.DataFrame({'key': train_keys, 'train_true': train_true, 'train_pred': train_pred})
        pd_va = pd.DataFrame({'key': valid_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
        pd_te = pd.DataFrame({'key': test_keys, 'test_true': test_true, 'test_pred': test_pred})

        pd_tr.to_csv('./stats/{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond), index=False)
        pd_va.to_csv('./stats/{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond), index=False)
        pd_te.to_csv('./stats/{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond), index=False)
        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
                                                    r2_score(train_true, train_pred), \
                                                    mean_absolute_error(train_true, train_pred), \
                                                    pearsonr(train_true, train_pred)
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(mean_squared_error(valid_true, valid_pred)), \
                                                    r2_score(valid_true, valid_pred), \
                                                    mean_absolute_error(valid_true, valid_pred), \
                                                    pearsonr(valid_true, valid_pred)
        test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
                                                r2_score(test_true, test_pred), \
                                                mean_absolute_error(test_true, test_pred), \
                                                pearsonr(test_true, test_pred)

        print('***best %s model***' % repetition_th)
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
            test_rmse, test_r2, test_mae, test_rp[0]))
        stat_res.append([repetition_th, 'train', train_rmse, train_r2, train_mae, train_rp[0]])
        stat_res.append([repetition_th, 'valid', valid_rmse, valid_r2, valid_mae, valid_rp[0]])
        stat_res.append([repetition_th, 'test', test_rmse, test_r2, test_mae, test_rp[0]])

    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'rmse', 'r2', 'mae', 'rp'])
    stat_res_pd.to_csv(
        './stats/{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond),
        index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'train'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'valid'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'valid'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'test'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'test'].std().values[-4:])
