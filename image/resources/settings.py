import os;
import argparse;
import torch;

def parse():
    parser = argparse.ArgumentParser(description='CIFAR DataSet Preparation');
    parser.add_argument('--net_type', default='RESNET-18',  required=False);
    parser.add_argument('--dataset', required=False, default='cifar10', choices=['cifar10', 'cifar100']);
    parser.add_argument('--icl', type=int, default=0, help='Incremental Training');
    parser.add_argument('--classes', type=int, default=10);

    args = parser.parse_args();
    args.dataset_path = '';
    args.batch_size = 64;
    args.epochs = 400;
    args.lr = 0.1;
    args.momentum = 0.9;
    args.weight_decay = 5e-4;
    args.warmup = 10;
    args.schedule = [0.12, 0.30, 0.75];
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    args.isCuda = True if torch.cuda.is_available() else False;
    args.seed = 42;
    args.isICL = True if args.icl == 1 else False
    return args;


def display_info(args):
    print('+------------------------------+');
    print('| {} image classification using {}'.format(args.dataset.upper(), args.net_type.upper()));
    print('+------------------------------+');
    print('| Epochs  : {}'.format(args.epochs));
    print('| LR start   : {}'.format(args.lr));
    print('| LR schedule : {}'.format(args.schedule));
    print('| warmup   : {}'.format(args.warmup));
    print('| BatchSize: {}'.format(args.batch_size));
    print('| Classes: {}'.format(args.classes));
    print('| ICL Training: {}'.format(args.isICL));
    print('+------------------------------+');
