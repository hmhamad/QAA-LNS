from abc import ABC, abstractmethod
from src.python.layers import NeuralNetwork, AbstractLayer, ResidualBlock
from src.python.tensor import FloatTensor, ones_lns, zeros_tensor
import numpy as np
import pickle
from collections import OrderedDict
import math


class Optimizer(ABC):
    @abstractmethod
    def step(self, model: NeuralNetwork, train_params: dict):
        """
        Updates all trainable weights and biases of the model.
        :param model: An instance of a Network
        :param train_params: A dictionary of training parameters (loss_scale is relevant for optimizer)
        """
        pass

    @classmethod
    def load_state(self, log_dir: str, model: NeuralNetwork):
        """
        Loads the state of the optimizer from a log directory.
        :param log_dir: The path to the log directory
        :param model: An instance of a Network (needed to rename the layers in optimizer to match the model's state)
        """
        with open(f"{log_dir}/optim", "rb") as file:
            optim = pickle.load(file)
        optim.copy_layer_names(model.layers)
        return optim

    def save_state(self, log_dir: str):
        with open(f"{log_dir}/optim", "wb") as file:
            pickle.dump(self, file)


class Scheduler(ABC):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def get_lr(self):
        """
        This method should be overridden by all subclasses to return the current learning rate.
        """
        pass

    def step(self):
        """
        Update the learning rate of the optimizer.
        This method can be overridden by subclasses for more complex behavior.
        """
        lr = self.get_lr()
        self.optimizer.lr = lr

    @classmethod
    def load_state(self, log_dir: str):
        with open(f"{log_dir}/scheduler", "rb") as file:
            scheduler = pickle.load(file)
        return scheduler

    def save_state(self, log_dir: str):
        with open(f"{log_dir}/scheduler", "wb") as file:
            pickle.dump(self, file)


class SGD(Optimizer):
    def __init__(
        self,
        lr: float,
        mom: float,
        model: NeuralNetwork,
        epochs: int,
        n_batches: int,
        weight_decay: float,
        arithmetic: str,
    ):
        self.lr = lr
        self.mom = mom
        self.n_batches = n_batches
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.current_epoch = 0
        self.device = "cpu"
        self.arithmetic = arithmetic

        self.W_mom_est = {}
        self.b_mom_est = {}
        self.step_t = 1
        self.total_steps = epochs * n_batches

        self._init_momentum_estimates(model.layers)

    def _init_momentum_estimates(self, layers: OrderedDict):
        for layer in layers.values():
            if isinstance(layer, AbstractLayer):
                if hasattr(layer, "W"):
                    self.W_mom_est[layer] = zeros_tensor(
                        1, arithmetic=self.arithmetic, device=self.device
                    )
                    if layer.bias:
                        self.b_mom_est[layer] = zeros_tensor(
                            1, arithmetic=self.arithmetic, device=self.device
                        )
            elif isinstance(layer, ResidualBlock):
                self._init_momentum_estimates(
                    layer.layers
                )  # recursively initialize momentum estimates for layers inside the residual block
                if layer.shortcut:
                    self._init_momentum_estimates(layer.shortcut_layers)

    def step(self, model: NeuralNetwork, train_params: dict):
        self._update_weights(model.layers, train_params)
        self.step_t += 1
        self.current_epoch = self.step_t // self.n_batches

    def _update_weights(self, layers: OrderedDict, train_params: dict):
        for layer in layers.values():
            if isinstance(layer, AbstractLayer):
                if hasattr(layer, "W"):
                    # following pytorch convention for momentum estimation
                    if train_params["loss_scale"] != 1:
                        self.W_mom_est[layer] = (
                            self.W_mom_est[layer]
                            * self.mom
                            * train_params["loss_scale"]
                            + layer.dL_dW
                        )
                        if self.weight_decay != 0:
                            layer.W = (
                                layer.W * train_params["loss_scale"]
                                - (
                                    self.W_mom_est[layer]
                                    + layer.W
                                    * (self.weight_decay * train_params["loss_scale"])
                                )
                                * self.lr
                            )
                        else:
                            layer.W = (
                                layer.W * train_params["loss_scale"]
                                - self.W_mom_est[layer] * self.lr
                            )
                        # undo scaling
                        layer.W = layer.W / train_params["loss_scale"]
                        self.W_mom_est[layer] = (
                            self.W_mom_est[layer] / train_params["loss_scale"]
                        )
                    else:
                        self.W_mom_est[layer] = (
                            self.W_mom_est[layer] * self.mom + layer.dL_dW
                        )
                        if self.weight_decay != 0:
                            layer.W = (
                                layer.W
                                - (self.W_mom_est[layer] + layer.W * self.weight_decay)
                                * self.lr
                            )
                        else:
                            layer.W = layer.W - self.W_mom_est[layer] * self.lr

                    if layer.bias:
                        if train_params["loss_scale"] != 1:
                            self.b_mom_est[layer] = (
                                self.b_mom_est[layer]
                                * self.mom
                                * train_params["loss_scale"]
                                + layer.dL_db
                            )
                            if self.weight_decay != 0:
                                layer.b = (
                                    layer.b * train_params["loss_scale"]
                                    - (
                                        self.b_mom_est[layer]
                                        + layer.b
                                        * (
                                            self.weight_decay
                                            * train_params["loss_scale"]
                                        )
                                    )
                                    * self.lr
                                )
                            else:
                                layer.b = (
                                    layer.b * train_params["loss_scale"]
                                    - self.b_mom_est[layer] * self.lr
                                )
                            # undo scaling
                            layer.b = layer.b / train_params["loss_scale"]
                            self.b_mom_est[layer] = (
                                self.b_mom_est[layer] / train_params["loss_scale"]
                            )
                        else:
                            self.b_mom_est[layer] = (
                                self.b_mom_est[layer] * self.mom + layer.dL_db
                            )
                            if self.weight_decay != 0:
                                layer.b = (
                                    layer.b
                                    - (
                                        self.b_mom_est[layer]
                                        + layer.b * self.weight_decay
                                    )
                                    * self.lr
                                )
                            else:
                                layer.b = layer.b - self.b_mom_est[layer] * self.lr
            elif isinstance(layer, ResidualBlock):
                self._update_weights(
                    layer.layers, train_params
                )  # recursively update weights for layers inside the residual block
                if layer.shortcut:
                    self._update_weights(layer.shortcut_layers, train_params)

    def to(self, device: str):
        self.device = device
        for layer in self.W_mom_est.keys():
            self.W_mom_est[layer].to(device, inplace=True)
            if layer.bias:
                self.b_mom_est[layer].to(device, inplace=True)

    def copy_layer_names(self, layers: OrderedDict):
        model_layers_list = []
        for layer in layers.values():
            if isinstance(layer, AbstractLayer):
                if hasattr(layer, "W"):
                    model_layers_list.append(layer)
            elif isinstance(layer, ResidualBlock):
                for sub_layer in layer.layers.values():
                    if isinstance(sub_layer, AbstractLayer):
                        if hasattr(sub_layer, "W"):
                            model_layers_list.append(sub_layer)
                if layer.shortcut:
                    for sub_layer in layer.shortcut_layers.values():
                        if isinstance(sub_layer, AbstractLayer):
                            if hasattr(sub_layer, "W"):
                                model_layers_list.append(sub_layer)
        # Loop over both
        idx = 0
        for model_layer, optim_layer in zip(
            model_layers_list, list(self.W_mom_est.keys())
        ):
            self.W_mom_est[model_layer] = self.W_mom_est.pop(optim_layer)
            if model_layer.bias:
                self.b_mom_est[model_layer] = self.b_mom_est.pop(optim_layer)
            idx += 1


class Adam(Optimizer):
    def __init__(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        model: NeuralNetwork,
        epochs: int,
        n_batches: int,
        weight_decay: float,
        arithmetic: str,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.n_batches = n_batches
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.current_epoch = 0
        self.device = "cpu"
        self.arithmetic = arithmetic

        self.W_mom_est = {}
        self.b_mom_est = {}
        self.W_mom_est_sq = {}
        self.b_mom_est_sq = {}
        self.step_t = 1
        self.total_steps = epochs * n_batches

        self._init_momentum_estimates(model.layers)

    def _init_momentum_estimates(self, layers: OrderedDict):
        for layer in layers.values():
            if isinstance(layer, AbstractLayer):
                if hasattr(layer, "W"):
                    self.W_mom_est[layer] = zeros_tensor(
                        1, arithmetic=self.arithmetic, device=self.device
                    )
                    self.W_mom_est_sq[layer] = zeros_tensor(
                        1, arithmetic=self.arithmetic, device=self.device
                    )
                    if layer.bias:
                        self.b_mom_est[layer] = zeros_tensor(
                            1, arithmetic=self.arithmetic, device=self.device
                        )
                        self.b_mom_est_sq[layer] = zeros_tensor(
                            1, arithmetic=self.arithmetic, device=self.device
                        )
            elif isinstance(layer, ResidualBlock):
                self._init_momentum_estimates(layer.layers)
                if layer.shortcut:
                    self._init_momentum_estimates(layer.shortcut_layers)

        if self.arithmetic == "lns":
            self.float_approx = {}
            # TODO - move float computations to lns
            self.float_approx["one_minus_beta1"] = (
                FloatTensor(1 - self.beta1).to_lns().to(self.device)
            )
            self.float_approx["one_minus_beta2"] = (
                FloatTensor(1 - self.beta2).to_lns().to(self.device)
            )
            self.float_approx["one_minus_beta1_pow_t"] = (
                FloatTensor(np.array([1 - self.beta1**t for t in range(1, 100)]))
                .to_lns()
                .to(self.device)
            )
            self.float_approx["one_minus_beta2_pow_t"] = (
                FloatTensor(np.array([1 - self.beta2**t for t in range(1, 100)]))
                .to_lns()
                .to(self.device)
            )

    def step(self, model: NeuralNetwork, train_params: dict):
        self._update_weights(model.layers, train_params)
        self.step_t += 1
        self.current_epoch = self.step_t // self.n_batches

    def _update_weights(self, layers: OrderedDict, train_params: dict):
        for layer in layers.values():
            if isinstance(layer, AbstractLayer):
                if hasattr(layer, "W"):
                    # Scale down gradients
                    if train_params.get("loss_scale", False):
                        layer.dL_dW = layer.dL_dW / train_params["loss_scale"]
                    if self.arithmetic == "lns":
                        one_minus_beta1 = self.float_approx["one_minus_beta1"]
                        one_minus_beta2 = self.float_approx["one_minus_beta2"]
                        one_minus_beta1_pow_t = (
                            self.float_approx["one_minus_beta1_pow_t"][self.step_t - 1]
                            if self.step_t < 100
                            else ones_lns((1), device=self.device)
                            - self.beta1**self.step_t
                        )
                        one_minus_beta2_pow_t = (
                            self.float_approx["one_minus_beta2_pow_t"][self.step_t - 1]
                            if self.step_t < 100
                            else ones_lns((1), device=self.device)
                            - self.beta2**self.step_t
                        )

                        # Exponential moving averages
                        self.W_mom_est[layer] = (
                            self.W_mom_est[layer] * self.beta1
                            + layer.dL_dW * one_minus_beta1
                        )
                        self.W_mom_est_sq[layer] = (
                            self.W_mom_est_sq[layer] * self.beta2
                            + (layer.dL_dW**2) * one_minus_beta2
                        )

                        # Transient Compensation (remove initial bias towards smaller values)
                        W_mom_est_normalized = (
                            self.W_mom_est[layer] / one_minus_beta1_pow_t
                        )
                        W_mom_est_sq_normalized = (
                            self.W_mom_est_sq[layer] / one_minus_beta2_pow_t
                        )
                    else:
                        # Exponential moving averages
                        self.W_mom_est[layer] = self.W_mom_est[
                            layer
                        ] * self.beta1 + layer.dL_dW * (1 - self.beta1)
                        self.W_mom_est_sq[layer] = self.W_mom_est_sq[
                            layer
                        ] * self.beta2 + layer.dL_dW**2 * (1 - self.beta2)

                        # Transient Compensation (remove initial bias towards smaller values)
                        W_mom_est_normalized = self.W_mom_est[layer] / (
                            1 - self.beta1**self.step_t
                        )
                        W_mom_est_sq_normalized = self.W_mom_est_sq[layer] / (
                            1 - self.beta2**self.step_t
                        )

                    dL_dW_adam = (W_mom_est_normalized * self.lr) / (
                        W_mom_est_sq_normalized.sqrt() + self.eps
                    )
                    layer.W = layer.W - dL_dW_adam

                    if layer.bias:
                        if train_params.get("loss_scale", False):
                            layer.dL_db = layer.dL_db / train_params["loss_scale"]
                        if self.arithmetic == "lns":
                            self.b_mom_est[layer] = (
                                self.b_mom_est[layer] * self.beta1
                                + layer.dL_db * one_minus_beta1
                            )
                            self.b_mom_est_sq[layer] = (
                                self.b_mom_est_sq[layer] * self.beta2
                                + (layer.dL_db**2) * one_minus_beta2
                            )
                            b_mom_est_normalized = (
                                self.b_mom_est[layer] / one_minus_beta1_pow_t
                            )
                            b_mom_est_sq_normalized = (
                                self.b_mom_est_sq[layer] / one_minus_beta2_pow_t
                            )
                        else:
                            self.b_mom_est[layer] = self.b_mom_est[
                                layer
                            ] * self.beta1 + layer.dL_db * (1 - self.beta1)
                            self.b_mom_est_sq[layer] = self.b_mom_est_sq[
                                layer
                            ] * self.beta2 + layer.dL_db**2 * (1 - self.beta2)
                            b_mom_est_normalized = self.b_mom_est[layer] / (
                                1 - self.beta1**self.step_t
                            )
                            b_mom_est_sq_normalized = self.b_mom_est_sq[layer] / (
                                1 - self.beta2**self.step_t
                            )

                        dL_db_adam = (b_mom_est_normalized * self.lr) / (
                            b_mom_est_sq_normalized.sqrt() + self.eps
                        )
                        layer.b = layer.b - dL_db_adam

            elif isinstance(layer, ResidualBlock):
                self._update_weights(
                    layer.layers, train_params
                )  # recursively update weights for layers inside the residual block
                if layer.shortcut:
                    self._update_weights(layer.shortcut_layers, train_params)

    def to(self, device: str):
        self.device = device
        for layer in self.W_mom_est.keys():
            self.W_mom_est[layer].to(device, inplace=True)
            self.W_mom_est_sq[layer].to(device, inplace=True)
            if layer.bias:
                self.b_mom_est[layer].to(device, inplace=True)
                self.b_mom_est_sq[layer].to(device, inplace=True)

        if self.arithmetic == "lns":
            for item in self.float_approx.keys():
                self.float_approx[item].to(device, inplace=True)


class CosineAnnealingWarmRestarts(Scheduler):
    """
    Cosine annealing with restarts.
    code source: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts
    """

    def __init__(
        self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0
    ):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.eta_max = optimizer.lr
        self.T_cur = -1
        self.last_epoch = -1

    def step(self, epoch: float):
        """
        Updates the learning rate following the cosine annealing schedule with warm restarts.
        :param epoch: A float representing the current time in terms of epochs,
                          can be fractional to indicate progress within an epoch.
        """
        if epoch >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = epoch % self.T_0
            else:
                n = int(
                    math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult)
                )
                self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                    self.T_mult - 1
                )
                self.T_i = self.T_0 * self.T_mult ** (n)
        else:
            self.T_i = self.T_0
            self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        self.optimizer.lr = (
            self.eta_min
            + (self.eta_max - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )

    def get_lr(self):
        return (
            self.eta_min
            + (self.eta_max - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )
