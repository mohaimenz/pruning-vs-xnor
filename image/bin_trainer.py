import sys;
import os;
import glob;
import math;
import argparse;
import numpy as np;
import random;
import time;
import torch;
import torch.optim as optim;

import nets.models as models;
import resources.data_loader as data_loader;
import resources.settings as settings;
sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.binop_lib as binop_lib;
import calculator as calc;

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.trainX = None;
        self.trainY = None;
        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        # define the binarization operator
        self.bin_op = None;

    def Train(self):
        train_start_time = time.time();
        net = models.BinResnet18(self.opt.classes);
        self.bin_op = binop_lib.BinOp(net);
        print(net);
        if self.opt.isCuda:
            net.cuda();
        calc.summary(net, (3,32,32), xnor_net=True);
        exit();
        param_dict = dict(net.named_parameters());
        params = [];
        for key, value in param_dict.items():
            params += [{'params':[value], 'lr': self.opt.lr,
                'weight_decay': self.opt.weight_decay,
                'key':key}];

        optimizer = optim.Adam(params, lr=self.opt.lr, weight_decay=self.opt.weight_decay);
        # optimizer = optim.SGD(net.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=5e-4);
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt.epochs);
        lossFunc = torch.nn.CrossEntropyLoss()

        self.testX, self.testY = data_loader.load_test_data(self.opt);

        for epochIdx in range(self.opt.epochs):
            epoch_start_time = time.time();
            # optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            self.trainX, self.trainY = data_loader.load_train_data(self.opt, epochIdx+1);
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = len(self.trainX)//self.opt.batch_size;
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.__get_batch(batchIdx);

                # zero the parameter gradients
                optimizer.zero_grad();

                # process the weights including binarization
                self.bin_op.binarization();

                outputs = net(x);

                pred = outputs.data.max(1, keepdim=True)[1]
                running_acc += pred.eq(y.data.view_as(pred)).cpu().sum();
                # print(running_acc);
                loss = lossFunc(outputs, y);
                loss.backward();

                # restore weights
                self.bin_op.restore();
                self.bin_op.updateBinaryGradWeight();

                optimizer.step();
                running_loss += loss.item();

            scheduler.step();
            tr_acc = (running_acc / (n_batches*self.opt.batch_size))*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.__validate(net, lossFunc);
            #Save best model
            self.__save_model(val_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(self.to_hms(total_time_taken)));

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.epochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.lr * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batch_size : (index+1)*self.opt.batch_size];
        y = self.trainY[index*self.opt.batch_size : (index+1)*self.opt.batch_size];
        return x.to(self.opt.device), y.to(self.opt.device);

    def __validate(self, net, lossFunc):
        net.eval();
        self.bin_op.binarization()
        loss = 0
        correct = 0
        y_pred = None;
        for idx in range(math.ceil(len(self.testX)/self.opt.batch_size)):
            x = self.testX[idx*self.opt.batch_size : (idx+1)*self.opt.batch_size];
            y = self.testY[idx*self.opt.batch_size : (idx+1)*self.opt.batch_size];
            scores = net(x);
            loss += lossFunc(scores, y).data.item()
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum();

        acc = 100. * float(correct) / len(self.testX);
        self.bin_op.restore()
        net.train();
        return acc, loss;

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {:.4f}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            epochIdx+1, self.opt.epochs, self.to_hms(epoch_time), self.to_hms(train_time), self.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            icl = 'icl_' if self.opt.isICL else '';
            ds = 'cifar{}'.format(self.opt.classes) if self.opt.isICL else self.opt.dataset;
            fname = "resnet/models/ts/bin_{}{}_a{:.2f}_e{}.pt";
            old_model = fname.format(icl, ds, self.bestAcc, self.bestAccEpoch);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            torch.save(net.state_dict(), fname.format(icl, ds, self.bestAcc, self.bestAccEpoch));

    def to_hms(self, time):
        h = int(time // 3600)
        m = int((time - h * 3600) // 60)
        s = int(time - h * 3600 - m * 60)
        if h > 0:
            line = '{}h{:02d}m'.format(h, m)
        else:
            line = '{}m{:02d}s'.format(m, s)

        return line

if __name__ == '__main__':
    opt = settings.parse();
    opt.lr = 0.001;
    opt.weight_decay = 1e-4;
    settings.display_info(opt);
    torch.manual_seed(opt.seed);
    if opt.isCuda:
        torch.cuda.manual_seed(opt.seed);

    trainer = Trainer(opt);
    print('---Training Staterted---');
    trainer.Train();
