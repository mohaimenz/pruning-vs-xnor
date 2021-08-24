import os;
import numpy as np;
import torch;

def load_train_data(opt, epoch=1):
    if opt.dataset == 'audioevent':
        path = os.path.join(opt.data, opt.dataset, 'aug-train-{}khz/train{}.npz'.format(opt.sr//1000, epoch));
    else:
        path = os.path.join(opt.data, opt.dataset, 'aug-data-{}khz/sp-{}/{}.npz'.format(opt.sr//1000, opt.split, 'train/train{}'.format(epoch)));

    return load_data(opt, path);

def load_test_data(opt):
    test_data_path = 'test_data_{}khz/fold{}_test4000.npz';
    # print(opt.dataset);
    if opt.dataset == 'us8k':
        test_data_path = 'test_data_{}khz/fold{}_test.npz';
    elif  opt.dataset == 'audioevent':
        test_data_path = 'aug-test-{}khz/test.npz';
    path = os.path.join(opt.data, opt.dataset, test_data_path.format(opt.sr//1000, opt.split));
    return load_data(opt, path);

def load_data(opt, path):
    data = np.load(path, allow_pickle=True);
    X = None;
    Y = None;
    if (opt.dataset=='esc50' and opt.nClasses < 50) or (opt.dataset=='audioevent' and opt.nClasses < 28):
        y = np.argmax(data['y'], axis=1);
        subset_classes = [];
        subset_idx = [];
        Y = [];
        for idx, value in enumerate(y):
            if value in opt.subset:
                subset_classes.append(value);
                target = opt.subset.index(value);
                label = np.zeros((1, len(opt.subset)));
                label[:, target] = 1
                Y.extend(label);
                subset_idx.append(idx);
        X = data['x'][subset_idx];
    else:
        X = data['x'];
        Y = data['y'];

    X = torch.tensor(np.moveaxis(X, 3, 1)).to(opt.device);
    Y = torch.tensor(Y, dtype=torch.float32).to(opt.device);

    return X, Y;
