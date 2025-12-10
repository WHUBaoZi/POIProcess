
import json
import pickle as pkl
import numpy as np
# import h5py
from scipy.io import loadmat
import os
import os.path as osp

def load_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    return data

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data

def load_npy(path):
    data = np.load(path, allow_pickle=True, encoding="latin1")
    return data

def save_pkl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pkl.dump(data, f)

    
def save_json(out_path, data):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(data, f)

def save_obj(v, f=None, file_name='output.obj'):
    os.makedirs(osp.dirname(file_name), exist_ok=True)
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def load_mat(path):
    f = loadmat(path)
    
    return f

