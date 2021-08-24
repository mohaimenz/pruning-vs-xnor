import sys;
import os;
import glob;
import math;
import random;
import torch;

import resources.settings as opts;
import resources.utils as U;
import resources.esc_subsets as subsets;
import resources.data_loader as data_loader;
import nets.models as models;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import calculator as calc;

class Trainer:
    def __init__(self, opt=None, split=0):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.trainX = None;
        self.trainY = None;
        self.opt.subset = eval('subsets.esc{}'.format(self.opt.nClasses));

    def load_train_data(self):
        print('Preparing calibration dataset..');
        self.trainX, self.trainY = data_loader.load_train_data(self.opt, 1);
        print('Calibration dataset is ready');
        self.opt.batchSize = 64;

    def load_test_data(self):
        if(self.testX is None):
            self.testX, self.testY = data_loader.load_test_data(self.opt);

    def __validate(self, net, testX, testY):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                #print(x.shape);
                # exit();
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc = self.__compute_accuracy(y_pred, self.testY);
        return acc;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target):
        # with torch.no_grad():
        #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
        y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
        print(y_pred.shape);
        if self.opt.dataset=='us8k':
            y_target = y_target.argmax(dim=1);
        else:
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);

        acc = (((y_pred==y_target)*1).float().mean()*100).item();
        return acc;

    def __load_model(self, quant=False):
        state = {};
        state['config'] = [7, 20, 10, 14, 22, 31, 35, 41, 51, 67, 69, 48];
        # state['weight'] = torch.load(self.opt.model_path, map_location=self.opt.device)['weight'];
        state['weight'] = torch.load(self.opt.model_path, map_location=self.opt.device);
        if quant:
            net = models.GetACDNetModel(input_len=self.opt.inputLength, nclass=self.opt.nClasses, sr=self.opt.sr, channel_config=state['config'], quantize=True).to(self.opt.device);
        else:
            net = models.GetACDNetModel(input_len=self.opt.inputLength, nclass=self.opt.nClasses, sr=self.opt.sr, channel_config=state['config']).to(self.opt.device);
        net.load_state_dict(state['weight']);

        return net;

    def __calibrate(self, net):
        self.load_train_data();
        net.eval();
        with torch.no_grad():
            for i in range(1,2):
                x_pred = None;
                for idx in range(math.ceil(len(self.trainX)/self.opt.batchSize)):
                    x = self.trainX[idx*self.opt.batchSize : (idx+1)*self.opt.batchSize];
                    #print(x.shape);
                    # exit();
                    scores = net(x);
                    x_pred = scores.data if x_pred is None else torch.cat((x_pred, scores.data));

                x_pred = x_pred.argmax(dim=1);
                x_target = self.trainY.argmax(dim=1);

                acc = (((x_pred==x_target)*1).float().mean()*100).item();
                print('calibrate accuracy is: {:.2f}'.format(acc));
        return acc;

    def QuantizeModel(self):
        net = self.__load_model(True);
        config = net.ch_config;
        net.eval();
        # print(net.sfeb);
        # print(net.tfeb);
        # exit();
        #Fuse modules to
        torch.quantization.fuse_modules(net.sfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.sfeb, ['3','4','5'], inplace=True);

        torch.quantization.fuse_modules(net.tfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['4','5','6'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['7','8','9'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['11','12','13'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['14','15','16'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['18','19','20'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['21','22','23'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['25','26','27'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['28','29','30'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['33','34','35'], inplace=True);

        # Specify quantization configuration
        net.qconfig = torch.quantization.get_default_qconfig('qnnpack');
        torch.backends.quantized.engine = 'qnnpack';
        print(net.qconfig);

        torch.quantization.prepare(net, inplace=True);

        # Calibrate with the training data
        self.__calibrate(net);

        # Convert to quantized model
        torch.quantization.convert(net, inplace=True);
        print('Post Training Quantization: Convert done');

        print("Size of model after quantization");
        torch.save(net.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p"))
        os.remove('temp.p')

        self.load_test_data();
        val_acc = self.__validate(net, self.testX, self.testY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));
        self.opt.quant_model_path = '{}/audio/trained_models/quantized_models/{}_{:.2f}.pt'.format(os.getcwd(), self.opt.model_name, val_acc);
        torch.jit.save(torch.jit.script(net), self.opt.quant_model_path);

    def TestModel(self, quant=False):
        if quant:
            net = torch.jit.load(self.opt.quant_model_path);
        else:
            net = self.__load_model();
            calc.summary(net, (1,1,self.opt.inputLength));
        self.load_test_data();
        net.eval();
        val_acc = self.__validate(net, self.testX, self.testY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));

    def GetModelSize(self):
        orig_net_path = self.opt.model_path;
        print('Full precision model size (KB):', os.path.getsize(orig_net_path)/(1024));
        quant_net_path = os.getcwd()+'/20khz_l0_tay_full_80_86.5_1403_quant.onnx';
        print('Quantized model size (KB):', os.path.getsize(quant_net_path)/(1024))


if __name__ == '__main__':
    opt = opts.parse();
    opt.cuda = False; #This quantization schema does not support GPU/CUDA
    opt.device = torch.device("cuda:0" if opt.cuda else "cpu");
    opt.dataset='us8k';
    if opt.dataset == 'us8k':
        opt.data = '/path/to/dataset/';
    opt.nClasses = 50;
    opt.model_path = 'path/to/model/to/quantize';
    opt.model_name = 'file_name_for_quantized_model';
    opt.split = 9;
    if opt.dataset == 'audioevent':
        opt.sr = 16000;
        opt.inputLength = 51215;

    trainer = Trainer(opt);

    print('Testing performance of the provided model.....');
    trainer.TestModel();

    print('Quantization process is started.....');
    trainer.QuantizeModel();
    print('Quantization done');

    print('Testing quantized model.');
    trainer.TestModel(True);
    print('Finished');
