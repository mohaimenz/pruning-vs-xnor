import os;
import numpy as np;
import argparse;
import time;
import torch;
import torchvision;
import torchvision.transforms as transforms;
from torch.utils.data import DataLoader;

class DatasetGenerator:
    def __init__(self, opt):
        self.opt = opt;
        self.cifar10_mean = (0.4914, 0.4822, 0.4465);
        self.cifar10_std = (0.2470, 0.2435, 0.2616);
        self.cifar100_mean = (0.5071, 0.4865, 0.4409);
        self.cifar100_std = (0.2673, 0.2564, 0.2762);

        self.dataX = None;
        self.dataY = None;

    def generate_train_data(self):
        cifarObj = torchvision.datasets.CIFAR10 if self.opt.dataset == 'cifar10' else torchvision.datasets.CIFAR100;
        mean = self.cifar10_mean if self.opt.dataset == 'cifar10' else self.cifar100_mean;
        std = self.cifar10_std if self.opt.dataset == 'cifar10' else self.cifar100_std;
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]);
        trainingObj = cifarObj(root=self.opt.dataset_path+self.opt.dataset+'/train', train=True, download=True, transform=transform);
        training_loader = DataLoader(trainingObj, shuffle=True, batch_size=100, drop_last=True);

        for e in range(1, self.opt.epochs+1):
            start_time = time.perf_counter();
            x = [];
            y = [];
            for idx, (images, labels) in enumerate(training_loader):
                x.extend(images.numpy());
                y.extend(labels.numpy());
            x = np.asarray(x);
            y = np.asarray(y);
            np.savez_compressed('{}/{}/train/train{}'.format(self.opt.dataset_path, self.opt.dataset, e), x=x, y=y);
            print('Train{} took {} sec'.format(e, time.perf_counter()-start_time));
            if e==10:
                break;


    def generate_test_data(self):
        cifarObj = torchvision.datasets.CIFAR10 if self.opt.dataset == 'cifar10' else torchvision.datasets.CIFAR100;
        mean = self.cifar10_mean if self.opt.dataset == 'cifar10' else self.cifar100_mean;
        std = self.cifar10_std if self.opt.dataset == 'cifar10' else self.cifar100_std;
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]);
        trainingObj = cifarObj(root=self.opt.dataset_path+self.opt.dataset+'/test', train=False, download=True, transform=transform);
        training_loader = DataLoader(trainingObj, shuffle=True, batch_size=100, drop_last=True);

        x = [];
        y = [];
        for idx, (images, labels) in enumerate(training_loader):
            x.extend(images.numpy());
            y.extend(labels.numpy());
        x = np.asarray(x);
        y = np.asarray(y);
        np.savez_compressed('{}/{}/test/test'.format(self.opt.dataset_path, self.opt.dataset), x=x, y=y);
        print('Finished test data generation');


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR DataSet Preparation');
    args = parser.parse_args();
    args.dataset = 'cifar100';
    args.dataset_path = '';
    args.classes = 100;
    args.batch_size = 64;
    args.train = True;
    args.epochs = 200;
    data_gen = DatasetGenerator(args);
    data_gen.generate_train_data();
    data_gen.generate_test_data();
