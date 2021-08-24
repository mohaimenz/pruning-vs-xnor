import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='ACDNet Sound Classification');

    # General settings
    parser.add_argument('--netType', default='ACDNet',  required=False);
    parser.add_argument('--dataset', required=False, default='esc50', choices=['esc10', 'esc50', 'us8k', 'audioevent']);
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning');
    parser.add_argument('--strongAugment', default=True,  action='store_true', help='Add scale and gain augmentation');
    parser.add_argument('--data', default='/path/to/audioi/datasets/', required=False, help='Path to dataset');

    #Basic Net Settings
    parser.add_argument('--nClasses', type=int, default=50);
    parser.add_argument('--nCrops', type=int, default=10);
    parser.add_argument('--nFolds', type=int, default=5);
    parser.add_argument('--split', type=int, default=0);
    parser.add_argument('--sr', type=int, default=20000);
    parser.add_argument('--inputLength', type=int, default=30225);

    #Leqarning settings
    parser.add_argument('--batchSize', type=int, default=64);
    parser.add_argument('--weightDecay', type=float, default=5e-4);
    parser.add_argument('--momentum', type=float, default=0.9);
    parser.add_argument('--nEpochs', type=int, default=2000);
    parser.add_argument('--LR', type=float, default=0.1);
    parser.add_argument('--warmup', type=int, default=10);

    #Handling unknown arguments
    p, unknown = parser.parse_known_args();
    # print(unknown);
    for i in unknown:
        if i.startswith('--'):
            parser.add_argument(i, default=unknown[unknown.index(i)+1]);

    opt = parser.parse_args();
    opt.schedule = [0.3, 0.6, 0.9];
    opt.splits = [i for i in range(1, opt.nFolds + 1)];
    opt.seed = 42;
    opt.xnor = True if hasattr(opt, 'xnor') and int(opt.xnor) == 1 else False;

    return opt


def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.netType));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.LR));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| Splits: {}'.format(opt.splits));
    print('| Classes: {}'.format(opt.nClasses));
    print('+------------------------------+');
