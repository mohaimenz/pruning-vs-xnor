import os;
import numpy as np;
import random;
import resources.utils as U;
import torch;

class Generator():
    #Generates data for Keras
    def __init__(self, samples, labels, options, train=True):
        random.seed(42);
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.train = train;
        self.batch_size = options.batchSize;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.preprocess_funcs = self.preprocess_setup();

    def __reset__(self, opt):
        self.batch_size = opt.batchSize;
        self.opt = opt;

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size));
        #return len(self.samples);

    def __getitem__(self, batchIndex):
        #Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        return batchX, batchY

    def generate_batch(self, batchIndex):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        for i in range(self.batch_size):
            if self.train:
                while True:
                    sound1, label1 = self.data[random.randint(0, len(self.data) - 1)];
                    sound2, label2 = self.data[random.randint(0, len(self.data) - 1)];
                    if label1 != label2:
                        break;

                sound1 = self.preprocess(sound1);
                sound2 = self.preprocess(sound2);

                # Mix two examples
                r = np.array(random.random());
                sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32);
                eye = np.eye(self.opt.nClasses);
                # print(eye)
                label = (eye[int(label1)] * r + eye[int(label2)] * (1 - r)).astype(np.float32);
                # print(label)
                #For stronger augmentation
                sound = U.random_gain(6)(sound).astype(np.float32);

            else:
                sound, target = self.data[i];
                sound = self.preprocess(sound).astype(np.float32);
                label = np.zeros((self.opt.nCrops, self.opt.nClasses));
                label[:,target] = 1;

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);
        if not self.train:
            sounds = sounds.reshape(sounds.shape[0]*sounds.shape[1], sounds.shape[2]);
            labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2]);
        return sounds, labels;

    def preprocess_setup(self):
        if self.train:
            funcs = []
            # if self.opt.strongAugment:
            funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound;

def setup(opt):
    trainGen = None;
    valGen = None;
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True);
    train_sounds = dataset['train'].item()['sounds'];
    train_labels = dataset['train'].item()['labels'];
    test_sounds = dataset['test'].item()['sounds'];
    test_labels = dataset['test'].item()['labels'];

    trainGen = Generator(train_sounds, train_labels, opt, True);
    valGen = Generator(test_sounds, test_labels, opt, False);

    return trainGen, valGen;
