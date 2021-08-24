# pruning-xnor
Official repository for the research article -- 

##### Note: Please run all the python scripts from a TERMINAL

## Audio Classification 
### Dataset preparation
1. Please follow official repository for [ACDNet](https://github.com/mohaimenz/acdnet) to prepare the audio datasets for these experiments
2. For AudioEvent dataset, the scripts are in this repository audio/resources/ae_

### Training ACDNet, Micro-ACDNet and Mini-ACDNet
Example: python audio/trainer.py  --model_path 'path/to/mini/or/micro/acdnet' --dataset 'esc10 or esc50 or us8k' --xnor 0 --nClasses 50 --model_name 'file_name_tosave_the_trained_model'
1. ACDNet on ESC-10: `python audio/trainer.py  --model_path '' --dataset 'esc10' --data '/path/to/dataset/' --xnor 0 --nClasses 10 --model_name 'model_name'`
2. ACDNet on us8k: `python audio/trainer.py  --model_path '' --dataset 'us8k' --data '/path/to/dataset/' --xnor 0 --nClasses 10 --model_name 'model_name'`
3. ACDNet on AudioEvent (20 class): `python audio/trainer.py  --model_path '' --dataset 'audioevent' --data '/path/to/dataset/' --xnor 0 --nClasses 20 --model_name 'model_name'`
4. Mini-ACDNet on ESC-50: `python audio/trainer.py  --model_path 'audio/models/mini_acdnet.pt' --dataset 'esc50' --data '/path/to/dataset/' --xnor 0 --nClasses 50 --model_name 'model_name'`
5. Micro-ACDNet on ESC-10: `python audio/trainer.py  --model_path 'audio/models/micro_acdnet.pt' --dataset 'esc10' --data '/path/to/dataset/' --xnor 0 --nClasses 10 --model_name 'model_name'`

### Training XNOR-Net for ACDNet, Micro-ACDNet and Mini-ACDNet
1. XACDNet on ESC-50: `python audio/trainer.py  --model_path '' --dataset 'esc50' --data '/path/to/dataset/' --xnor 1 --nClasses 50 --model_name 'xacdnet_esc50'`
2. XMicroACDnet on ESC-40: `python audio/trainer.py  --model_path 'audio/models/micro_acdnet.pt' --dataset 'esc40' --data '/path/to/dataset/' --xnor 1 --nClasses 40 --model_name 'xmicro_acdnet_esc40'`
3. XMiniAcdnet on US8k: `python audio/trainer.py  --model_path 'audio/models/mini_acdnet.pt' --dataset 'us8k' --data '/path/to/dataset/' --xnor 1 --nClasses 10 --folds_to_train '[1,2]' --model_name 'xmini_acdnet_us8k'`

##### Note: You can use the pretrained micro and mini acdnets files or use your own mini and micro acdnet versions using [ACDNet](https://github.com/mohaimenz/acdnet) repository

### Quantization of MicroACDNet
1. Update settings `opt.dataset, opt.data, opt.model_path, opt.model_name, opt.split` in `audio/quantization.py` 
3. Run: `python audio/quantization.py`

## Image Classification
### Dataset preparation
1. Create a folder to store the datasets and copy the path
2. Update `args.datasetpath` with the copied path in `image/resources/prepare_dataset.py`
3. Update `args.datasetpath` with the copied path in `image/resources/prepare_dataset_cifar100.py`
4. Update `args.dataset_path` in `image/resources/settings.py`
5. To prepare CIFAR-10, run in terminal: `python image/resources/prepare_dataset.py`
6. To prepare CIFAR-100, run in terminal: `python image/resources/prepare_dataset_cifar100.py`

### Train RESNET-18 (Use Terminal)
1. CIFAR-10: `python resnet/trainer.py  --dataset 'cifar10' --icl 0 --classes 10`
2. CIFAR-100: `python resnet/trainer.py  --dataset 'cifar100' --icl 0 --classes 100`
3. For incremental training (e.g. CIFAR-30): `python resnet/trainer.py  --dataset 'cifar100' --icl 1 --classes 30`

### Train XNOR-Net version of RESNET-18
1. CIFAR-10: `python resnet/bin_trainer.py  --dataset 'cifar10' --icl 0 --classes 10`
2. CIFAR-100: `python resnet/bin_trainer.py  --dataset 'cifar100' --icl 0 --classes 100`
3. For incremental training (e.g. CIFAR-30): `python resnet/bin_trainer.py  --dataset 'cifar100' --icl 1 --classes 30`

