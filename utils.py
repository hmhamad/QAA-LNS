import os.path as osp
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
import matplotlib.patheffects as pe
import os 
import json

def parse_config(config_file):
    config = ConfigParser()
    with open(config_file) as file:
        config.read_string('[default]\n' + file.read())

    arithmetic = config.get('default', 'arithmetic')
    network = config.get('default', 'network')
    total_bits = config.get('default', 'total_bits')
    lr = config.get('default', 'lr')
    epochs = config.get('default', 'epochs')
    optim = config.get('default', 'optim')
    optim = optim.split("#")[0]
    d_float_thresh = config.get('default', 'd_float_thresh')
    return arithmetic, network, total_bits, optim, lr, epochs, d_float_thresh

def plot_dvals(log_dirs):    
    # Iterate over all the input directories
    for i, log_dir in enumerate(log_dirs):
        # Define a figure and set size
        # fig, ax = plt.subplots(1, 3)
        # fig.set_size_inches(6, 2)
        # fig.tight_layout()

        arithmetic, network, total_bits, optim, lr, epochs = parse_config(f"{log_dir}/config.conf")
        print(f"arithmetic:{arithmetic}, network:{network}, total bits:{total_bits}, optim:{optim}")
        # fig.suptitle(f"arithmetic:{arithmetic}, network:{network}, total bits:{total_bits}, optim:{optim}")
        # Read stats
        training_data = np.loadtxt(f"{log_dir}/stats.txt", skiprows=1, usecols=(1,2,3,4))
        dstat_files = glob(f"{log_dir}/*.npy")
        # Array to store all the dvals
        combined_dvals = np.array([])
        for j, file in enumerate(dstat_files):
            arr = np.load(file).reshape(-1)
            combined_dvals = np.append(combined_dvals, arr)
            plt.hist(arr, bins=np.linspace(0, 25, 100), density=True)
            plt.savefig(f"plots/{log_dir.split('/')[-1]}_{j}")
        print(len(np.where(combined_dvals > 8)))

def plot_val_metrics(log_dirs):
    print("\n Generating Validation Metrics Plot \n")
    for i, log_dir in enumerate(log_dirs):


        arithmetic, network, total_bits, optim, lr, epochs, _ = parse_config(f"{log_dir}/config.conf")
        # print(f"arithmetic:{arithmetic}, network:{network}, total bits:{total_bits}, optim:{optim}")

        metric_data = np.loadtxt(f"{log_dir}/stats.txt", skiprows=1, usecols=(1,2,3,4))
        val_acc = metric_data[:, -2]
        val_loss = metric_data[:, -1]

        acc_path = f"ACC_{arithmetic}_{network}_{optim}_{lr}_{epochs}.png"
        loss_path =  f"LOSS_{arithmetic}_{network}_{optim}_{lr}_{epochs}.png"

        plt.figure()
        plt.plot(val_acc, label="Validation Accuracy", c='g', path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
        plt.plot([np.argmax(val_acc)], [val_acc[np.argmax(val_acc)]], marker='*', markersize=20, markeredgecolor="red", label="Best Accuracy")
        plt.ylabel("Validation Accuracy")
        plt.xlabel("Epochs")
        plt.title(f" Arithmetic = {arithmetic} Network = {network} Optimizer = {optim} ")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(f"{log_dir}/{acc_path}")
        plt.figure()
        plt.plot(val_loss, label="Validation Loss", c='r', path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
        plt.plot([np.argmax(val_acc)], [val_loss[np.argmax(val_acc)]], marker='*', markersize=20, markeredgecolor="red", label="Best Accuracy")
        plt.ylabel("Validation Loss")
        plt.xlabel("Epochs")
        plt.title(f" Arithmetic = {arithmetic} Network = {network} Optimizer = {optim} ")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(f"{log_dir}/{loss_path}")

    print("\n Generated Validation Metrics Plot \n")

def plot_combined_val_metrics(experiments):
    colors = ['r','g','b','y']
    print("\n Generating Validation Metrics Plot \n")

    for experiment in experiments:
        labels = experiment['labels']
        savepath = experiment['savepath']
        log_dirs = sorted([name for name in os.listdir(savepath) if os.path.isdir(os.path.join(savepath,name))])
        for dir_name in log_dirs:
            print(dir_name)
        log_dirs = [os.path.join(savepath,dir_name) for dir_name in log_dirs]

        plt.figure()
        plt.ylabel("Validation Accuracy")
        plt.xlabel("Epochs")
        for i, log_dir in enumerate(log_dirs):

            arithmetic, network, total_bits, optim, lr, epochs, _ = parse_config(f"{log_dir}/config.conf")

            metric_data = np.loadtxt(f"{log_dir}/stats.txt", skiprows=1, usecols=(1,2,3,4))
            val_acc = metric_data[:, -2]

            plt.plot(val_acc, label=labels[i], color=colors[i], path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
            plt.plot([np.argmax(val_acc)], [val_acc[np.argmax(val_acc)]], marker='*', color=colors[i], markersize=10, markeredgecolor="black")

        plt.title(f"LNS vs Float, Network {network}, Optim {optim}, lr {lr}, epochs {epochs}")
        plt.legend(loc='best')
        plt.grid()
        savename = os.path.join(savepath,'val_acc_plot.png')
        if os.path.isfile(savename):
            os.remove(savename)
        plt.savefig(savename)
    print("\n Generated Validation Metrics Plot \n")

def custom_plot(dirs,labels,savepath):
    colors = ['r','g','b','y']
    plt.figure()
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epochs")
    for i, log_dir in enumerate(dirs):

        metric_data = np.loadtxt(f"{log_dir}/stats.txt", skiprows=1, usecols=(1,2,3,4))
        val_acc = metric_data[:, -2]

        plt.plot(val_acc, label=labels[i], color=colors[i], path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        plt.plot([np.argmax(val_acc)], [val_acc[np.argmax(val_acc)]], marker='*', color=colors[i], markersize=10, markeredgecolor="black")

    plt.title(f"Two Layer MLP on FashionMNIST")
    plt.legend(loc='best')
    plt.grid()
    savename = os.path.join(savepath,'val_acc_plot.png')
    if os.path.isfile(savename):
        os.remove(savename)
    plt.savefig(savename)


def generate_log_summary(log_dirs):

    formatted_row = "{: <20} {: <12} {: <15} {: <10} {: <15} {: <6} {: <6} {: <10} {: <7}"

    print(formatted_row.format(*["Date", "Arithmetic", "Optim", "Network", "Best Accuracy", "lr", "Epochs", "Bit Width", "d_thresh"]))

    for i, log_dir in enumerate(log_dirs):
        folder = osp.basename(log_dir)
        date = "-".join(folder.split("-")[:6])
        arithmetic, network, total_bits, optim, lr, epochs, d_float_thresh = parse_config(f"{log_dir}/config.conf")
        # Read stats
        training_data = np.loadtxt(f"{log_dir}/stats.txt", skiprows=1, usecols=(1,2,3,4))
        if len(training_data.shape) == 2:
            print(formatted_row.format(*[date, arithmetic, optim, 
                                        network, max(training_data[:, 2]), lr, 
                                        len(training_data), total_bits, d_float_thresh]))
        else:
            print(formatted_row.format(*[date, arithmetic, optim, 
                                        network, "NaN", lr, 
                                        "NaN"], total_bits, d_float_thresh))

if __name__ == "__main__":
    # assert len(sys.argv) == 2, "Log directory missing!"
    log_dirs = glob("logs/*")
    #plot_dvals(log_dirs)
    # generate_log_summary(glob("logs/*"))

    abs_filepath = os.path.abspath(__file__)
    main_dir = os.path.dirname(abs_filepath)
    exp_logs_dir = os.path.join(main_dir,'exp_logs')
    datasets = ['fmnist','cifar10']
    networks = ['net5','net7']
    optimizers = ['sgd','adam']
    experiments = []
    for dataset in datasets:
        dataset_dir = os.path.join(exp_logs_dir,dataset)
        for network in networks:
            network_dir = os.path.join(dataset_dir,network)
            for optimizer in optimizers:
                optimizer_dir = os.path.join(network_dir,optimizer)
                config_file_path = os.path.join(network_dir,optimizer_dir,'config.json')
                if os.path.exists(config_file_path):
                    config = json.load(open(config_file_path))
                    config['savepath'] = os.path.join(dataset_dir,optimizer_dir)
                    experiments.append(config)
    # plot_combined_val_metrics(experiments)

    dirs = ['/home/hhamad/lns-cnn/logs/2023-4-24-10-10-2-lns-fmnist-net1-sgd',
            '/home/hhamad/lns-cnn/logs/2023-4-24-10-10-29-lns-fmnist-net1-sgd',
            '/home/hhamad/lns-cnn/logs/2023-4-24-10-21-58-lns-fmnist-net1-sgd',
            '/home/hhamad/lns-cnn/logs/2023-4-24-10-22-50-lns-fmnist-net1-sgd',
    ]
    labels = ['LUT BLOCK', 'LIN TREE', 'LUT TREE', 'LIN BLOCK']
    savepath = '/home/hhamad/lns-cnn/exp_logs'
    custom_plot(dirs,labels,savepath)