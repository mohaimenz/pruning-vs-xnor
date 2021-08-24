import sys;
import os;
import glob;
import math;
import numpy as np;
import pandas as pd;
import random;
import time;
import torch;
import torch.optim as optim;

import nets.models as models;
import resources.utils as U;
import resources.settings as opts;
import resources.esc_subsets as subsets;
import resources.data_loader as data_loader;

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
        self.opt.subset = eval('subsets.esc{}'.format(self.opt.nClasses));
        # define the binarization operator
        self.bin_op = None;

    def Train(self):
        train_start_time = time.time();
        print('Starting DeepSound model training for SPLIT-{}'.format(self.opt.split));
        ch_config = None;
        if hasattr(self.opt, 'model_path') and self.opt.model_path != "":
            ch_config = torch.load(self.opt.model_path, map_location=self.opt.device)['config'];
        if self.opt.xnor:
            net = models.GetBINACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, ch_config).to(self.opt.device);
            self.bin_op = binop_lib.BinOp(net);
        else:
            net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, ch_config).to(self.opt.device);

        # print(net);
        calc.summary(net, (1,1,self.opt.inputLength), xnor_net=self.opt.xnor);

        optimizer = None;
        scheduler = None;
        if self.opt.xnor:
            param_dict = dict(net.named_parameters());
            params = [];
            for key, value in param_dict.items():
                params += [{'params':[value], 'lr': self.opt.LR,
                    'weight_decay': self.opt.weightDecay,
                    'key':key}];
            optimizer = optim.Adam(params, lr=self.opt.LR, weight_decay=self.opt.weightDecay);
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt.nEpochs);
        else:
            optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');

        self.testX, self.testY = data_loader.load_test_data(self.opt);

        for epochIdx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            if not self.opt.xnor:
                optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            self.trainX, self.trainY = data_loader.load_train_data(self.opt, epochIdx+1);
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = len(self.trainX)//self.opt.batchSize;
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.__get_batch(batchIdx);

                # zero the parameter gradients
                optimizer.zero_grad();

                if self.opt.xnor:
                    # process the weights including binarization
                    self.bin_op.binarization();

                net.requires_grad = True;
                # forward + backward + optimize
                outputs = net(x);
                # print(outputs);
                # exit();
                running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                loss = lossFunc(outputs.log(), y);
                running_loss += loss.item();
                loss.backward();
                # self.plot_grad_flow(net.named_parameters())
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)

                if self.opt.xnor:
                    # restore weights
                    self.bin_op.restore();
                    self.bin_op.updateBinaryGradWeight();

                optimizer.step();

            if self.opt.xnor:
                scheduler.step();

            # print('Running ACC: {:.3f}'.format(running_acc));
            tr_acc = (running_acc/n_batches)*100;
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
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        y = self.trainY[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        return x.to(self.opt.device), y.to(self.opt.device);

    def __validate(self, net, lossFunc):
        net.eval();
        if self.opt.xnor:
            self.bin_op.binarization()
        # with torch.no_grad():
        y_pred = None;
        batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
        for idx in range(math.ceil(len(self.testX)/batch_size)):
            x = self.testX[idx*batch_size : (idx+1)*batch_size];
            scores = net(x);
            y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

        acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc);
        if self.opt.xnor:
            self.bin_op.restore()
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        # with torch.no_grad():
        #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
        y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
        if self.opt.dataset=='us8k':
            y_target = y_target.argmax(dim=1);
        else:
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);

        acc = (((y_pred==y_target)*1).float().mean()*100).item();
        # valLossFunc = torch.nn.KLDivLoss();
        loss = lossFunc(y_pred.float().log(), y_target.float()).item();

        return acc, loss;

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'SP{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {:.4f}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            self.opt.split, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            fname = "audio/trained_models/f{}_{}_a{:.2f}_e{}.pt";
            old_model = fname.format(self.opt.split, self.opt.model_name, self.bestAcc, self.bestAccEpoch);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            torch.save(net.state_dict(), fname.format(self.opt.split, self.opt.model_name, self.bestAcc, self.bestAccEpoch));

def TrainModel(opt):
    opts.display_info(opt);
    trainer = Trainer(opt);
    trainer.Train();

if __name__ == '__main__':
    opt = opts.parse();
    opt.cuda = torch.cuda.is_available();
    opt.device = torch.device("cuda:0" if opt.cuda else "cpu");
    # opt.xnor=True;
    # print(opt.xnor);
    # exit();

    ###This options are provided through slurm job script.
    ###To run this script directly, uncomment the following lines
    # opt.dataset = 'esc50';
    # opt.data = '/path/to/datasets/';
    # opt.model_path = 'audio/models/mini_acdnet.pt';
    # opt.xnor=True;
    # opt.nClasses = 50;
    # opt.model_name = 'esc50_mini_acdnet';

    if opt.xnor:
        opt.LR = 0.001;
        opt.weightDecay = 1e-4;
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed);
    splits = None;
    if opt.dataset == 'audioevent':
        opt.sr = 16000;
        opt.inputLength = 51215;
        opt.nEpochs = 1500;
    elif opt.dataset == 'us8k':
        splits = eval(opt.folds_to_train);
        opt.nEpochs = 1200;
    else:
        splits = opt.splits;

    if opt.dataset == 'audioevent':
        TrainModel(opt);
    else:
        for split in splits:
            print('+-- Split {} --+'.format(split));
            opt.split = split;
            TrainModel(opt);
