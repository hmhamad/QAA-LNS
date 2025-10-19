import argparse
import cupy as cp
import numpy as np
from datetime import datetime
import os
import shutil
from tqdm import tqdm
import pickle

# load args


def parse_config(filename):
    args = argparse.Namespace()
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                try:
                    if value == "float":
                        pass
                    else:
                        value = eval(value)
                except:
                    pass
                setattr(args, key, value)
    return args


def init_optimizer(args, model, n_batches):
    if args.optim == "sgd":
        optimizer = SGD(
            args.lr,
            args.mom,
            model,
            args.epochs,
            n_batches,
            args.weight_decay,
            args.arithmetic,
        )
    elif args.optim == "adam":
        raise NotImplementedError(
            f"Optimizer {args.optim} needs Loss Scaling and Weight Decay to be implemented"
        )
        optimizer = Adam(
            args.lr,
            0.9,
            0.999,
            1e-8,
            model,
            args.epochs,
            n_batches,
            args.weight_decay,
            args.arithmetic,
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optim} not implemented")
    return optimizer


def init_scheduler(args, optimizer):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min
    )
    return scheduler


def max_grad(model):
    max_grad = -1
    from src.python.layers import AbstractLayer, ResidualBlock

    for layer in model.layers.values():
        if isinstance(layer, AbstractLayer):
            if hasattr(layer, "W"):
                max_grad = max(max_grad, abs(layer.dL_dW.F).max())
            if hasattr(layer, "b"):
                max_grad = max(max_grad, abs(layer.dL_db.F).max())
        elif isinstance(layer, ResidualBlock):
            for sub_layer in layer.layers.values():
                if isinstance(sub_layer, AbstractLayer):
                    if hasattr(sub_layer, "W"):
                        max_grad = max(max_grad, abs(sub_layer.dL_dW.F).max())
                    if hasattr(sub_layer, "b"):
                        max_grad = max(max_grad, abs(sub_layer.dL_db.F).max())
            if layer.shortcut:
                for sub_layer in layer.shortcut_layers.values():
                    if isinstance(sub_layer, AbstractLayer):
                        if hasattr(sub_layer, "W"):
                            max_grad = max(max_grad, abs(sub_layer.dL_dW.F).max())
                        if hasattr(sub_layer, "b"):
                            max_grad = max(max_grad, abs(sub_layer.dL_db.F).max())
    return max_grad.x


def main_loop(
    model,
    train_dataloader,
    val_dataloader,
    epochs: int,
    optimizer,
    scheduler,
    device: str,
    train_params: dict,
    save_logs: bool = False,
    logdir: str = None,
):
    model.to(device)
    optimizer.to(device)
    save_model = True

    start_epoch = optimizer.current_epoch

    train_acc = np.zeros(epochs, dtype=np.float32)
    train_loss = np.zeros(epochs, dtype=np.float32)
    val_acc = np.zeros(epochs, dtype=np.float32)
    val_loss = np.zeros(epochs, dtype=np.float32)

    for epoch in range(start_epoch, epochs):
        print("*" * 50)
        print(f"Epoch {epoch+1}/{epochs}")
        train_epoch_loss, train_epoch_acc, maxgrad = train_epoch(
            model, train_dataloader, epoch, optimizer, scheduler, device, train_params
        )
        val_epoch_loss, val_epoch_acc = eval_epoch(
            model, val_dataloader, device, train_params
        )

        train_acc[epoch] = train_epoch_acc
        train_loss[epoch] = train_epoch_loss
        val_acc[epoch] = val_epoch_acc
        val_loss[epoch] = val_epoch_loss

        best_epoch = np.argmax(val_acc)
        # Save Current Model and Optimizer States. Also Save Stats
        if save_logs:
            if maxgrad < 1e3 and save_model:
                model.save_state(logdir)
                optimizer.save_state(logdir)
                scheduler.save_state(logdir)
                np_rng_state = np.random.get_state()
                cp_rng_state = cp.random.get_random_state()
                # save rng state
                with open(logdir + "/rng_state", "wb") as f:
                    pickle.dump(
                        {"np_rng_state": np_rng_state, "cp_rng_state": cp_rng_state}, f
                    )
            else:
                save_model = False
                print(
                    f"Epoch {epoch+1}/{epochs}: Skipping model save due to maxgrad={maxgrad:.3f}"
                )
            # save stats
            medin_last_10 = (
                np.median(val_acc[epoch - 9 : epoch + 1])
                if epoch >= 9
                else np.median(val_acc[: epoch + 1])
            )
            with open(logdir + "/stats.txt", "a") as f_stats:
                f_stats.write(
                    f"\n{epoch+1}/"
                    + str(epochs)
                    + f"\t  {train_acc[epoch]:.4f}  "
                    + f"\t  {train_loss[epoch]:.4f}  \t  {val_acc[epoch]:.4f}  "
                    + f"  \t{val_loss[epoch]:.4f}  \t{medin_last_10:.4f}"
                )

    return val_loss, val_acc


def train_epoch(
    model, dataloader, epoch: int, optimizer, scheduler, device: str, train_params: dict
):
    model.train()

    maxgrad = -1
    running_acc = 0
    running_loss = 0
    total = 0
    progress_bar = tqdm(
        dataloader, ascii=True, smoothing=1
    )  # smoothing=1 for instantaneous speed
    for nb, (x, y) in enumerate(progress_bar):
        # Move data to device
        x = x.to(device)
        y = y.to(device)
        # Forward Propagation
        out = model.forward(x, train_params)
        # Backward Propagation
        model.backward(out, y, train_params)
        # Step through Optimizer and Update Weights
        optimizer.step(model, train_params)
        # Update Learning Rate
        scheduler.step(epoch + nb / len(dataloader))
        # Calculate and Update Loss/Acc statistics
        batch_loss, batch_correct = calc_stats(
            out,
            y,
            dataloader.dataset.nClasses,
            model.layers,
            train_params,
            model.arithmetic,
        )
        running_acc += batch_correct
        running_loss += batch_loss
        total += y.shape[1]
        epoch_maxgrad = max_grad(model)  # to detect overflow
        maxgrad = max(maxgrad, epoch_maxgrad)
        if epoch_maxgrad > 1e3:
            print("nb", nb)
            print("epoch", epoch)
            print("epoch_maxgrad", epoch_maxgrad)
            # quit program
            exit()
        progress_bar.set_postfix(
            {"train_loss": running_loss / total, "train_acc": running_acc / total},
            maxgrad=maxgrad,
        )
    return running_loss / total, running_acc / total, maxgrad


def eval_epoch(model, dataloader, device: str, train_params: dict):
    model.eval()

    running_acc = 0
    running_loss = 0
    total = 0
    progress_bar = tqdm(dataloader, ascii=True)
    for nb, (x, y) in enumerate(progress_bar):
        # Move data to device
        x = x.to(device)
        y = y.to(device)
        # Forward Propagation
        yhat = model.forward(x, train_params)
        # Calculate and Update Loss/Acc statistics
        batch_loss, batch_correct = calc_stats(
            yhat,
            y,
            dataloader.dataset.nClasses,
            model.layers,
            train_params,
            model.arithmetic,
        )
        running_acc += batch_correct
        running_loss += batch_loss
        total += y.shape[1]
        progress_bar.set_postfix(
            {"val_loss": running_loss / total, "val_acc": running_acc / total}
        )
    return running_loss / total, running_acc / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_name", type=str, help="Simulation Name", default="no-title"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Config File Name, Python uses it to save it in logging directory",
        default="N",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Hardware Device for Training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fmnist",
        help="Choose Dataset from {fmnist,cifar10}",
    )
    parser.add_argument(
        "--arithmetic",
        type=str,
        default="lns",
        help="Arithmetic Datatype for processing, from {float,lns}",
    )
    parser.add_argument(
        "--lns_bits",
        type=int,
        default=32,
        help="Total Number of Bits for LNS Datatype, including sign bit and zero flag bit",
    )
    parser.add_argument(
        "--frac_bits",
        type=int,
        default=20,
        help="Number of Fractional Bits for LNS Datatype",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=5,
        help="Network Architecture Number. All architecture configurations are stored in file net_arch.py",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of Training Epochs"
    )
    parser.add_argument("--batchsize", type=int, default=128, help="Batch Size")
    parser.add_argument(
        "--optim",
        type=str,
        default="sgd",
        help="Choose Optimizer from {sgd,adam,percentdelta}",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning Rate for Optimizer"
    )
    parser.add_argument(
        "--mom", type=float, default=0.9, help="Momentum Value for Optimizer"
    )
    parser.add_argument(
        "--T_0",
        type=int,
        default=100,
        help="T_0 for CosineAnnealingWarmRestarts Scheduler",
    )
    parser.add_argument(
        "--T_mult",
        type=int,
        default=1,
        help="T_mult for CosineAnnealingWarmRestarts Scheduler",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-3,
        help="eta_min for CosineAnnealingWarmRestarts Scheduler",
    )
    parser.add_argument(
        "--sum_approx",
        type=int,
        default=1,
        help="LNS Summation Approximation, 0=Uniform-LUT and 1=Piecewise-Linear-LUT",
        choices=[0, 1],
    )
    parser.add_argument(
        "--sum_technique",
        type=int,
        default=1,
        help="LNS Summation Technique, 0=Sequential-Adder and 1=Tree-Adder",
        choices=[0, 1],
    )
    parser.add_argument(
        "--lin_size",
        type=int,
        default=16,
        help="Size of the LUT for Piecewise-Linear Summation Approximation (only used if sum_approx=1)",
    )
    parser.add_argument(
        "--x_ub",
        type=int,
        default=16,
        help="Upper Bound of the LUT for Piecewise-Linear Summation Approximation (only used if sum_approx=1)",
    )

    parser.add_argument(
        "--sample_magnitude",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=" 0 = small, 1 = mixed, 2 = large: Sample magnitude used to compute the LNS Summation Approximation LUT (only used if sum_approx=1)",
    )

    parser.add_argument("--loss_scale", type=int, default=1, help="Loss Scaling Factor")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="L2 Regularization Coefficient"
    )
    parser.add_argument(
        "--leaky_flag", type=int, default=0, help="Enable/Disable Leaky Relu"
    )
    parser.add_argument(
        "--leaky_coeff", type=float, default=1e-3, help="Leaky Relu Slope"
    )
    parser.add_argument(
        "--save_logs", action="store_true", help="save logs, model and stats"
    )
    parser.add_argument(
        "--continue_from",
        default="n",
        type=str,
        help="directory to continue training from",
    )

    args = parser.parse_args()

    print(args.config)

    logdir = None

    if args.continue_from != "n":
        logdir = f"logs/{args.continue_from}"
        print(f"Resume training from logging directory: {logdir}")
        # read args from the configuration file in the logging directory
        with open(f"{logdir}/config.conf", "r") as file:
            new_device = args.device
            args = parse_config(f"{logdir}/config.conf")
            args.continue_from = logdir
            args.save_logs = True
            args.device = new_device
            args.lns_bits = args.TOTAL_BITS
            args.frac_bits = args.FRAC_BITS
            args.lin_size = args.LIN_SIZE
            args.x_ub = args.X_UB

    # Seed the RNG
    np.random.seed(args.seed)
    # Set device
    if "cuda" in args.device or args.device.isdigit():
        print(cp.cuda.runtime.getDeviceCount(), "GPUs detected")
        device_id = int(args.device) if args.device.isdigit() else int(args.device[-1])
        cp.cuda.Device(device_id).use()
        print("Using Device:", cp.cuda.Device())
        args.device = "gpu"
    else:
        args.device = "cpu"
        print("Using Device:", args.device)

    # important to import modules after setting device, e.g. in load_architecture, initialization of layers weights depends on device
    from net_arch import load_architecture
    from src.python.datautils import load_dataset, DataLoader
    from src.python.utils import calc_stats
    from src.python.layers import NeuralNetwork
    from src.python.optim import (
        Optimizer,
        SGD,
        Adam,
        Scheduler,
        CosineAnnealingWarmRestarts,
    )
    from src.python.tensor import set_lns_summation_mode

    # Set LNS Summation Mode
    set_lns_summation_mode(args.sum_approx, args.sum_technique, args.lns_bits)

    # Create Logging Directory
    if args.continue_from == "n" and args.save_logs:
        now = datetime.now()
        if args.arithmetic == "float":
            logdir = f"logs/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}-{args.sim_name}-{args.arithmetic}-{args.dataset}-{args.network}-{args.optim}"
        else:
            logdir = f"logs/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}-{args.sim_name}-{args.arithmetic}-T{args.lns_bits}-F{args.frac_bits}-L{args.lin_size}-Xub{args.x_ub}-Mag{args.sample_magnitude}_{args.dataset}"
        os.makedirs(logdir)
        print(f"Logging Directory: {logdir}")
        shutil.copy(f"configs/{args.config}", f"{logdir}/config.conf")
        with open(logdir + "/stats.txt", "w") as f_stats:
            f_stats.write(
                "Epoch\tTrain Acc\tTrain Loss\tValid Acc\tValid Loss\tMedian Last 10 Valid Acc"
            )
        np.random.seed(args.seed)
        cp.random.seed(
            args.seed
        )  # Note: GPU operations are not deterministic, even with a fixed seed!

    # Load Dataset and Create Dataloaders
    train_dataset, val_dataset, test_dataset, nClasses = load_dataset(
        args.dataset, args.arithmetic
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

    # Define or Load Model and Optimizer
    if args.continue_from != "n":
        model = NeuralNetwork.load_state(logdir)
        optimizer = Optimizer.load_state(logdir, model)
        scheduler = Scheduler.load_state(logdir)
        # load rng state
        with open(logdir + "/rng_state", "rb") as f:
            rng_state = pickle.load(f)
            np.random.set_state(rng_state["np_rng_state"])
            cp.random.set_random_state(rng_state["cp_rng_state"])
    else:
        # Load Model Architecture
        layers_dict = load_architecture(
            args.network, nClasses, train_dataset.img_shape, args.arithmetic
        )
        # Define Network
        model = NeuralNetwork(
            layers_dict, nClasses, args.arithmetic, logdir, args.save_logs
        )
        # Initialize Optimizer
        optimizer = init_optimizer(args, model, len(train_dataloader))
        # Initialize Scheduler
        scheduler = init_scheduler(args, optimizer)

    # Pack Various Training Parameters
    train_params = {
        "loss_scale": args.loss_scale,
        "leaky_flag": args.leaky_flag,
        "leaky_coeff": args.leaky_coeff,
    }

    # Start Training
    val_loss, val_acc = main_loop(
        model,
        train_dataloader,
        val_dataloader,
        args.epochs,
        optimizer,
        scheduler,
        args.device,
        train_params,
        args.save_logs,
        logdir,
    )

    # Clean up
    if args.save_logs:
        # delete model, optimizer and rng state
        os.remove(os.path.join(logdir, "model.pkl"))
        os.remove(os.path.join(logdir, "optim"))
        os.remove(os.path.join(logdir, "rng_state"))
        os.remove(os.path.join(logdir, "scheduler"))

    print("\nValidation Loss:")
    for ee in range(args.epochs):
        print(" {:.3f}".format(val_loss[ee]), end=",")
    print("\n\nValidation accuracy:")
    for ee in range(args.epochs):
        print(" {:.3f}".format(val_acc[ee]), end=",")
    print("\n")
