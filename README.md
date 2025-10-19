# QAA-LNS
Source Code for the paper "Bitwidth-Specific Logarithmic Arithmetic Towards Future Hardware Accelerated Training"

## Environment Setup

To set up your Python environment, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/AnonymousCoder8/qaa-lns.git
cd qaa-lns
```
2. Setup the conda environment
```bash
conda create -n lns -c conda-forge python=3.10.9 pillow numpy=1.24.2 tqdm cupy cudatoolkit cudatoolkit-dev
```
2. Activate the conda environment
  ```bash
  conda activate lns
  ```
## Download Datasets

Download the CIFAR-100 dataset from this link ```https://www.cs.toronto.edu/~kriz/cifar.html```, then unzip the dataset in the ```data``` folder

To download the TinyImageNet-200 dataset, run the following command then unzip the data in the ```data``` folder
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

## Reproducing Results
You can reproduce the results from our paper as follows:
1. Define all of your training parameters and experiment config in the folder `configs/` (or use an existing config file)
2. Run the experiment using the following command:
```bash
bash run.sh --config <config_file_name> --cc <gpu_compute_capability> [--save_logs]
``` 
3. The results will be saved in the folder `logs/` if the `--save_logs` option is passed

(Our code currently only runs on GPUs and you need to specify the compute capability of your GPU, e.g., 75 for RTX 2080Ti, in order to compile the CUDA code correctly. You can find the compute capability of your GPU [here](https://developer.nvidia.com/cuda-gpus))

## Citing Our Work