import os
import os.path as osp
import torch
from torch import Tensor
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import random
import h5py


#############################################
#           General IO functions
#############################################
def read_patch_data(path: str, dtype:str='torch', key='features'):
    r"""Read patch data from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
        key (string): Key of return data, default 'features'.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            pdata = hf[key][:]
    elif ext == '.pt':
        pdata = torch.load(path, map_location=torch.device('cpu'))
    elif ext == '.npy':
        pdata = np.load(path)
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(pdata, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(pdata)
    elif isinstance(pdata, Tensor) and dtype == 'numpy':
        return pdata.numpy()
    else:
        return pdata

def read_patch_coord(path: str, dtype:str='torch'):
    r"""Read patch coordinates from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        coords = hf['coords'][:]

    if isinstance(coords, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(coords)
    else:
        return coords 

def read_datasplit_npz(path: str):
    data_npz = np.load(path)
    
    pids_train = [str(s) for s in data_npz['train_patients']]
    if 'val_patients' in data_npz:
        pids_val = [str(s) for s in data_npz['val_patients']]
    else:
        pids_val = None
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

def read_maxt_from_table(path: str, at_column='t'):
    df = pd.read_csv(path)
    return df[at_column].max()

#############################################
#  IO functions for classification models
#############################################
def retrieve_from_table_clf(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id'):
    """Get info from table, oriented to classification tasks"""
    assert level in ['slide', 'patient']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 'label']
    for c in assert_columns:
        assert c in df.columns

    if patient_ids is None or len(patient_ids) == 0:
        patient_ids = list(df[pid_column].unique()) # get rid of possible duplicates
        print(f"[info] got an empty patient list, so all the patients in {table_path} are used by default.")

    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    for i in df.index:
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[warning] Patient ID {} not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _label = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = _label

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] = _label

    if shuffle:
        if level == 'patient':
            pid = random.shuffle(pid)
        else:
            sid = random.shuffle(sid)

    res = []
    for r in ret:
        res.append(eval(r))
    return res

def save_prediction_clf(uids, y_true, y_pred, save_path, binary=True, edl_output=False):
    r"""Save classification prediction.

    Args:
        y_true (Tensor or ndarray): true labels, typically with shape (N, ).
        y_pred (Tensor or ndarray): predicted values, typically with shape (N, num_cls).
        save_path (string): path to save.
        binary (boolean): if it is a binary prediction.
        edl_output (boolean): if it is the output of EDL networks.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.squeeze().numpy()
    if isinstance(y_pred, Tensor):
        if len(y_pred.shape) == 1: # pred is direct prob
            y_pred = y_pred.numpy()
        else: # pred is logit
            if edl_output: # save the y_pred (i.e., the alpha parameters of Dirichlet or Beta)
                binary = False
                y_pred = y_pred.numpy()
            else:
                y_pred = softmax(y_pred, dim=1) # apply softmax to logit outputs (N, num_cls)
                y_pred = y_pred.squeeze().numpy()

    assert len(uids) == len(y_true)
    assert len(uids) == len(y_pred)

    save_data = {'uids': uids, 'y': y_true}
    cols = ['uids', 'y']
    if binary:
        save_data['y_hat'] = y_pred[:, 1]
        cols.append('y_hat')
    else:
        prefix = 'y_hat_' if not edl_output else 'alpha_'
        ncol = y_pred.shape[-1]
        for i in range(ncol):
            _col = prefix + str(i)
            save_data[_col] = y_pred[:, i]
            cols.append(_col)

    df = pd.DataFrame(save_data, columns=cols)
    df.to_csv(save_path, index=False)

def save_prediction_clf_ins(uids, y_true, y_pred, save_path, binary=True, edl_output=False):
    save_data = {'uids': uids, 'y': y_true, 'y_hat': y_pred}
    df = pd.DataFrame(save_data, columns=['uids', 'y', 'y_hat'])
    df.to_csv(save_path, index=False)
