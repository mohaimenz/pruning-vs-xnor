#Only for Audio Event (freesound.org): https://data.vision.ee.ethz.ch/cvl/ae_dataset/
import sys
import os
import subprocess

import glob
import numpy as np
import wavio


def main():
    mainDir = os.path.abspath('..');
    ae_path = os.path.join(mainDir, 'datasets/audioevent');

    if not os.path.exists(ae_path):
        os.mkdir(ae_path)

    sr = 16000;

    dst_path = os.path.join(ae_path, 'wav{}.npz'.format(sr // 1000));
    create_dataset(ae_path, dst_path);

def convert_sr(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path);

    for root, dirs, files in sorted(os.walk(os.path.join(src_path, 'train'))):
        if len(dirs) > 0:
            train_path = os.path.join(dst_path, 'train');
            if not os.path.exists(train_path):
                os.mkdir(train_path);
            for d in dirs:
                dir_path = os.path.join(train_path, d);
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path);
            continue;

        for f in files:
            src_file_path = os.path.join(root, f);
            dst_file_path = os.path.join(dst_path, 'train', os.path.split(root)[-1], f);
            subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(src_file_path, sr, dst_file_path), shell=True);

    for root, dirs, files in sorted(os.walk(os.path.join(src_path, 'test'))):
        test_path = os.path.join(dst_path, 'test');
        if not os.path.exists(test_path):
            os.mkdir(test_path);
        for f in files:
            src_file_path = os.path.join(root, f);
            dst_file_path = os.path.join(test_path, f);
            subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(src_file_path, sr, dst_file_path), shell=True);

def create_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path));
    ae_dataset = {};
    ae_dataset['train'] = {};
    ae_dataset['test'] = {};
    classes = None;
    sounds = [];
    labels = [];
    for root, dirs, files in sorted(os.walk(os.path.join(src_path, 'train'))):
        if len(dirs) > 0:
            classes = sorted(dirs);
            continue;

        for f in files:
            wav_file = os.path.join(root, f);
            sound = get_trimed_sound(wav_file);
            l = os.path.split(root)[-1];
            label = classes.index(l);
            sounds.append(sound);
            labels.append(label);

    ae_dataset['train']['sounds'] = sounds;
    ae_dataset['train']['labels'] = labels;

    sounds = [];
    labels = [];
    for root, dirs, files in sorted(os.walk(os.path.join(src_path, 'test'))):
        for f in files:
            wav_file = os.path.join(root, f);
            sound = get_trimed_sound(wav_file);
            lbl = '_'.join(f.split('_')[:-1]);
            label = classes.index(lbl);
            sounds.append(sound);
            labels.append(label);

    ae_dataset['test']['sounds'] = sounds;
    ae_dataset['test']['labels'] = labels;

    np.savez(dst_path, **ae_dataset);

def get_trimed_sound(wav_file):
    sound = wavio.read(wav_file).data.T[0];
    start = sound.nonzero()[0].min();
    end = sound.nonzero()[0].max();
    sound = sound[start: end + 1]; # Remove silent sections
    return sound;

if __name__ == '__main__':
    main()
