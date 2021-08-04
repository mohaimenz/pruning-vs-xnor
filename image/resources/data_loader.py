import os;
import numpy as np;
import torch;
import resources.cifar_subsets as subsets;

def load_train_data(opt, epoch=1):
    path = os.path.join(opt.dataset_path, opt.dataset, 'train/train{}.npz'.format(epoch));
    # print(path);
    return load_data(opt, path);

def load_test_data(opt):
    path = os.path.join(opt.dataset_path, opt.dataset, 'test/test.npz');
    # print(path);
    return load_data(opt, path);

def load_data(opt, path):
    data = np.load(path, allow_pickle=True);
    X = None;
    Y = None;
    #icl ranges from 10-90. If not icl then it is original cifar-10 or cifar-100
    if opt.isICL:
        y = data['y'];
        subset_classes = [];
        subset_idx = [];
        Y = [];
        opt.subset = eval('subsets.icl{}'.format(opt.classes));
        for idx, value in enumerate(y):
            if value in opt.subset:
                # print(idx, '--', value, '--', opt.subset.index(value));
                subset_classes.append(value);
                target = opt.subset.index(value);
                Y.append(target);
                subset_idx.append(idx);
        X = data['x'][subset_idx];
    else:
        X = data['x'];
        Y = data['y'];
    X = torch.tensor(X).to(opt.device);
    Y = torch.tensor(Y).to(opt.device);
    return X, Y;
