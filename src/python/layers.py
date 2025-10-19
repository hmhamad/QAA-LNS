from abc import ABC, abstractmethod
import numpy as np
import cupy as cp
from src.python.utils import (
    relu,
    softmax,
    weight_init,
    patches_stride_trick,
    efficient_conv2d,
    dilate_array,
)
from src.python.tensor import FloatTensor, zeros_tensor, ones_tensor
import pickle
from collections import OrderedDict

from abc import ABC, abstractmethod


class AbstractLayer(ABC):
    """
    A base class representing an abstract layer in a neural network.

    This class provides the foundational structure and specifications for various types of layers in a neural network,
    enforcing the implementation of essential methods such as initialization of weights and biases, forward propagation,
    and backward propagation.
    """

    @abstractmethod
    def initialize_weights_and_biases(self, arithmetic):
        """
        Initializes the weights and biases of the layer.

        Parameters:
        - arithmetic (str): The arithmetic type, either 'float' or 'lns'.
        """
        pass

    @abstractmethod
    def forward(self, x, training_parameters):
        """
        Conducts forward propagation through the layer.

        Parameters:
        - x (numpy.ndarray): The input array.
        - training_parameters (dict): A dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: The layer's output after applying the activation function.
        """
        pass

    @abstractmethod
    def backward(self, x, dL_dout, training_parameters):
        """
        Conducts backward propagation through the layer, calculating the gradient of the loss with respect to weights,
        biases, and input.

        Parameters:
        - x (numpy.ndarray): The input array.
        - dL_dout (numpy.ndarray): The gradient backpropagated from the next layer, representing the derivative of the loss function with respect to this layer's output.
        - training_parameters (dict): A dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: The derivative of the loss function with respect to this layer's input, to be backpropagated to previous layers.
        """
        pass

    def to(self, device):
        """
        Moves the layer and its parameters to the specified device.

        Parameters:
        - device (str): The target device, e.g., 'cpu' or 'gpu'.
        """
        self.device = device
        if hasattr(self, "W"):
            self.W.to(device, inplace=True)
            if self.bias:
                self.b.to(device, inplace=True)
            if isinstance(self, BatchNormalizationLayer):
                self.moving_mu.to(device, inplace=True)
                self.moving_var.to(device, inplace=True)

    def set_training_mode(self):
        """
        Sets the layer to training mode.
        """
        self.mode = "train"

    def set_evaluation_mode(self):
        """
        Sets the layer to evaluation mode.
        """
        self.mode = "eval"


class FullyConnectedLayer(AbstractLayer):
    """
    Represents a fully connected layer in a neural network, followed by an activation function.

    Parameters:
    - n_out (int): The number of output neurons.
    - n_in (int): The number of input neurons.
    - act (str): Activation function. Supported: 'relu'
    - arithmetic (str): The arithmetic type, either 'float' or 'lns'.
    """

    def __init__(
        self,
        n_out: int,
        n_in: int,
        act: str | None = None,
        bias: bool = True,
        arithmetic: str = "float",
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.outputshape = n_out
        self.act = act
        self.bias = bias
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """ "
        Initializes layer's weights W (shape: (n_out, n_in)) and biases b (shape: (n_out, 1))
        """
        self.W = weight_init(self.n_in, self.n_out, (self.n_out, self.n_in))
        self.b = weight_init(self.n_in, self.n_out, (self.n_out, 1))
        self.W = FloatTensor(self.W)
        self.b = FloatTensor(self.b)
        if self.arithmetic == "lns":
            self.W = self.W.to_lns()
            self.b = self.b.to_lns()

    def forward(self, x, training_parameters):
        """
        Conducts forward propagation through the layer.

        Parameters:
        - x (numpy.ndarray): Input array of shape (n_in, batch_size).
        - training_parameters (dict): Dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: Layer output after applying activation function, of shape (n_out, batch_size).
        """
        self.x = x
        self.z = self.W @ self.x + self.b
        if self.act is None:
            out = self.z
        elif self.act == "relu":
            out = relu(self.z, training_parameters, self.arithmetic)
        return out

    def backward(self, dL_dout, training_parameters):
        """
        Conducts backward propagation through the layer, calculating the gradient of the loss with respect to weights,
        biases, and input.

        Parameters:
        - dL_dout (numpy.ndarray): Gradient backpropagated from next layer, of shape (n_out, batch_size).
        - training_parameters (dict): Dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: Derivative of the loss function with respect to this layer's input, of shape (n_in, batch_size).
        """
        batchsize = self.x.shape[1]

        if self.act == "relu":
            dL_dout[self.z <= 0] = zeros_tensor(
                1, arithmetic=self.arithmetic, device=self.device
            )

        dL_din = self.W.T @ dL_dout
        self.dL_dW = dL_dout @ self.x.T * (1 / batchsize)

        self.dL_db = dL_dout.sum(axis=1).reshape(self.n_out, 1) * (1 / batchsize)

        return dL_din


class FullyConnectedLayerWithSoftmax(AbstractLayer):
    """
    Represents a fully connected layer in a neural network, followed by the softmax function.

    Parameters:
    - n_out (int): The number of output neurons.
    - n_in (int): The number of input neurons.
    - activation_function (callable): The activation function.
    - arithmetic (str): The arithmetic type, either 'float' or 'lns'.
    """

    def __init__(self, n_out, n_in, bias=True, arithmetic="float"):
        self.n_in = n_in
        self.n_out = n_out
        self.arithmetic = arithmetic
        self.bias = bias
        self.outputshape = n_out
        self.device = "cpu"  # default device

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """ "
        Initializes layer's weights W (shape: (n_out, n_in)) and biases b (shape: (n_out, 1))
        """
        self.W = weight_init(self.n_in, self.n_out, (self.n_out, self.n_in))
        self.b = weight_init(self.n_in, self.n_out, (self.n_out, 1))
        self.W = FloatTensor(self.W)
        self.b = FloatTensor(self.b)
        if self.arithmetic == "lns":
            self.W = self.W.to_lns()
            self.b = self.b.to_lns()

    def forward(self, x, training_parameters):
        """
        Conducts forward propagation through the layer.

        Parameters:
        - x (numpy.ndarray): Input array of shape (n_in, batch_size).
        - training_parameters (dict): Dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: Layer output after applying activation function, of shape (n_out, batch_size).
        """
        self.x = x
        self.z = self.W @ self.x + self.b
        out, log_softmax_logits = softmax(self.z, self.arithmetic)
        self.log_softmax_logits = log_softmax_logits
        return out

    def backward(self, out, y, training_parameters):
        """
        Conducts backward propagation through the layer, calculating the gradient of the loss with respect to weights,
        biases, and input.

        Parameters:
        - out (numpy.ndarray): Output array after applying softmax of shape (n_out, batch_size).
        - y (numpy.ndarray): Ground truth labels of shape (n_out, batch_size).
        - training_parameters (dict): Dictionary containing training parameters, such as l2 regularization coefficient.

        Returns:
        - numpy.ndarray: Derivative of the loss function with respect to this layer's input, of shape (n_in, batch_size).
        """
        batchsize = self.x.shape[1]

        # if self.arithmetic=='lns':
        #     idx_true_class = y.argmax(axis=0) # find index of true class for each batch
        #     dL_dz = softmax(self.z,self.arithmetic,idx_true_class)
        #     # dL_dz = float_to_lns(out.to_float() - y.to_float())
        # else:

        dL_dz = out - y

        # Apply loss scaling
        if training_parameters["loss_scale"] != 1:
            dL_dz = dL_dz * training_parameters["loss_scale"]

        self.dL_dW = dL_dz @ self.x.T * (1 / batchsize)

        self.dL_db = dL_dz.sum(axis=1, keepdims=True) * (1 / batchsize)

        dz_dx = self.W
        dL_din = dz_dx.T @ dL_dz

        return dL_din


class MaxPooling2DLayer(AbstractLayer):
    """
    A layer implementing 2D max pooling operation.

    Parameters:
        inputshape (tuple): The shape of the input (height, width, nChannels).
        kernelsize (int): The size of the pooling window.
        stride (int): The stride of the pooling operation.
        padding (int): The negative infinity padding to be added on both sides
        outputshape (tuple): The shape of the output (new_height, new_width, nChannels).
        arithmetic (str): Number format, either 'float' or 'lns'.
    """

    def __init__(
        self,
        inputshape: tuple,
        kernelsize: int,
        stride: int | None = None,
        padding: int = 0,
        arithmetic: str = "float",
    ):
        self.kernelsize = kernelsize
        self.stride = stride if stride is not None else kernelsize
        self.padding = padding
        self.inputshape = inputshape
        self.nChannels = inputshape[2]
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

        new_height = (
            (self.inputshape[0] - self.kernelsize + 2 * self.padding) // self.stride
        ) + 1
        new_width = (
            (self.inputshape[1] - self.kernelsize + 2 * self.padding) // self.stride
        ) + 1
        self.outputshape = (new_height, new_width, self.nChannels)

    def initialize_weights_and_biases(self, arithmetic):
        pass  # no weights or biases

    def forward(self, input_image, training_parameters):
        """
        Performs forward propagation through the max pooling layer.

        Args:
            input_image (np.ndarray): A 4D array of shape (height, width, nChannels, batchsize).
            train_params (dict): A dictionary containing training parameters.

        Returns:
            np.ndarray: The output feature maps of shape (new_height, new_width, nChannels, batchsize).
        """
        self.input_image = input_image
        batchsize = self.input_image.shape[3]

        if self.padding != 0:
            self.input_image = self.input_image.pad(
                (
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                    (0, 0),
                ),
                constant_values="-inf",
            )

        # Using the technique of stacking image patches into a column matrix with stride trick (patches_stride_trick function)
        window_matrix = patches_stride_trick(
            self.input_image, self.kernelsize, self.stride
        )
        self.max_idx = window_matrix.argmax(axis=0)  # need for backprop
        max_vals = window_matrix[
            self.max_idx,
            cp.arange(self.max_idx.shape[0])[:, None, None],
            cp.arange(self.max_idx.shape[1])[None, :, None],
            cp.arange(self.max_idx.shape[2])[None, None, :],
        ]
        feature_maps = max_vals.reshape(
            self.outputshape[0], self.outputshape[1], self.nChannels, batchsize
        )

        return feature_maps

    def backward(self, dL_dout, training_parameters):
        """
        Backpropagates through the max pooling layer.

        Args:
            dL_dout (np.ndarray): The gradient backpropagated from the next layer of shape (new_height, new_width, nChannels, batchsize).
            train_params (dict): A dictionary containing training parameters.

        Returns:
            np.ndarray: The gradient of the loss function w.r.t. this layer's input of shape (height, width, nChannels, batchsize).
        """

        batchsize = self.input_image.shape[3]

        # Using the technique of stacking image patches into a column matrix with stride trick (patches_stride_trick function)
        dL_din = zeros_tensor(
            self.input_image.shape, arithmetic=self.arithmetic, device=self.device
        )

        n_windows = self.max_idx.shape[0]
        row_idx, col_idx = cp.unravel_index(cp.arange(n_windows), self.outputshape[:2])

        row_idx_max = (
            cp.floor(self.max_idx / self.kernelsize)
            + row_idx[:, cp.newaxis, cp.newaxis] * self.stride
        ).astype(int)
        col_idx_max = (
            cp.mod(self.max_idx, self.kernelsize)
            + col_idx[:, cp.newaxis, cp.newaxis] * self.stride
        )

        expanded_row_idx_y = row_idx[:, None, None]
        expanded_col_idx_y = col_idx[:, None, None]

        expand_channels_variable = np.arange(self.nChannels)[:, None]
        arange_batchsize = np.arange(batchsize)
        # Assign the gradients to the correct positions using advanced indexing
        if self.stride >= self.kernelsize:
            # non-overlapping windows
            dL_din[
                row_idx_max, col_idx_max, expand_channels_variable, arange_batchsize
            ] = dL_dout[
                expanded_row_idx_y,
                expanded_col_idx_y,
                expand_channels_variable,
                arange_batchsize,
            ]
        else:
            # overlapping windows
            for ww in range(n_windows):  # loop over each window (patch)
                dL_din[
                    row_idx_max[ww, :, :],
                    col_idx_max[ww, :, :],
                    expand_channels_variable,
                    arange_batchsize,
                ] += dL_dout[
                    row_idx[ww], col_idx[ww], expand_channels_variable, arange_batchsize
                ]

        # Remove padding if added during the forward pass
        if self.padding > 0:
            dL_din = dL_din[
                self.padding : -self.padding, self.padding : -self.padding, :, :
            ]

        return dL_din


class GlobalAveragePooling2DLayer(AbstractLayer):
    """
    A layer implementing 2D global average pooling operation.
    This layer performs global average pooling over the spatial dimensions
    of the input tensor.

    Parameters:
        inputshape (tuple): The shape of the input (height, width, nChannels).
        outputshape int: The shape of the output (nChannels).
        arithmetic (str): Number format, either 'float' or 'lns'.
    """

    def __init__(self, inputshape: tuple, arithmetic: str = "float"):
        self.inputshape = inputshape
        self.nChannels = inputshape[2]
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

        self.outputshape = self.nChannels

    def initialize_weights_and_biases(self, arithmetic):
        pass  # no weights or biases

    def forward(self, x, training_parameters):
        """
        Forward pass of the global average pooling layer.

        :param x: (np.array) of shape (height, width, nChannels, batchsize)
        :return: (np.array) of shape (nChannels, batchsize)
        """
        self.input_shape = x.shape
        out = x.mean(axis=(0, 1)).reshape(self.nChannels, self.input_shape[3])
        return out

    def backward(self, dL_dy, training_parameters):
        """
        Backward pass of the global average pooling layer.

        :param dL_dy: Gradient of the loss with respect to the output of this layer, shape (nChannels, batchsize)
        :return: Gradient of the loss with respect to the input of this layer, shape (height, width, nChannels, batchsize)
        """
        height, width, _, _ = self.input_shape
        dL_dy = dL_dy / (height * width)
        dL_dx = dL_dy.tile((height, width, 1, 1))
        return dL_dx


class Convolutional2DLayer(AbstractLayer):
    """
    Implements a single convolutional 2D layer.

    args:
        inputshape (tuple of ints): Shape of the input in the format (height, width, nChannels).
        nFilters (int): Number of filters.
        kernelsize (int): Size of one dimension of the applied square kernel.
        stride (int): Stride of the kernel.
        padding (str): Either 'same' or 'valid' or int. 'same' pads input so that output retains the same spatial dimensions, 'valid' performs no padding.
        act (str): Activation function. Supported: 'relu'
        arithmetic (str): Specifies the arithmetic to be used: 'float' for floating point, 'lns' for logarithmic number system.
    """

    def __init__(
        self,
        inputshape: tuple,
        nFilters: int,
        kernelsize: int,
        stride: int = 1,
        padding: int = 0,
        act: str | None = None,
        bias: bool = True,
        arithmetic: str = "float",
    ):
        self.nFilters = nFilters
        self.kernelsize = kernelsize
        self.stride = stride
        self.padding = padding
        self.act = act
        self.inputshape = inputshape
        self.nChannels = self.inputshape[2]
        self.bias = bias
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

        if self.padding == "valid":
            new_height = int(
                np.floor((self.inputshape[0] - self.kernelsize) / self.stride + 1)
            )
            new_width = int(
                np.floor((self.inputshape[1] - self.kernelsize) / self.stride + 1)
            )
        elif self.padding == "same":
            assert self.stride == 1, "Padding 'same' only supported for stride=1"
            new_height = self.inputshape[0]
            new_width = self.inputshape[1]
            pad_len = self.kernelsize - 1

            self.pad_top = pad_len // 2  # floor division
            self.pad_bottom = pad_len - self.pad_top
            self.pad_left = pad_len // 2  # floor division
            self.pad_right = pad_len - self.pad_left
        elif isinstance(self.padding, int):
            new_height = int(
                np.floor(
                    (self.inputshape[0] + 2 * self.padding - self.kernelsize)
                    / self.stride
                    + 1
                )
            )
            new_width = int(
                np.floor(
                    (self.inputshape[1] + 2 * self.padding - self.kernelsize)
                    / self.stride
                    + 1
                )
            )

            self.pad_top = self.padding
            self.pad_bottom = self.padding
            self.pad_left = self.padding
            self.pad_right = self.padding

        self.outputshape = (new_height, new_width, self.nFilters)

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """
        Initializes layer's weights W (shape: (kernelsize,kernelsize,nChannels,nFilters)) and biases b (shape: (nFilters,1))
        """
        fan_in = self.kernelsize * self.kernelsize * self.nChannels
        fan_out = self.kernelsize * self.kernelsize * self.nFilters
        self.W = weight_init(
            fan_in,
            fan_out,
            (self.kernelsize, self.kernelsize, self.nChannels, self.nFilters),
        )
        self.W = FloatTensor(self.W).to_arithmetic(self.arithmetic)
        if self.bias:
            self.b = weight_init(fan_in, fan_out, (self.nFilters, 1))
            self.b = FloatTensor(self.b).to_arithmetic(self.arithmetic)

    def forward(self, input_image, training_parameters):
        """
        Performs forward propagation through the convolutional layer and returns the output feature maps.

        Parameters:
            input_image (numpy array): A 4D array representing the input image,
                                        with shape (height, width, nChannels, batchsize).
            training_parameters (dict): A dictionary containing training parameters.

        Returns:
            numpy array: The output feature maps after applying the convolution and activation function,
                         with shape (new_height, new_width, nFilters, batchsize).
        """
        self.input_image = input_image
        if self.padding != 0 and self.padding != "valid":
            self.input_image = self.input_image.pad(
                (
                    (self.pad_top, self.pad_bottom),
                    (self.pad_left, self.pad_right),
                    (0, 0),
                    (0, 0),
                )
            )
        self.feature_maps = efficient_conv2d(
            self.W, self.input_image, self.stride, (2, 2), self.arithmetic
        )
        self.feature_maps = self.feature_maps.transpose(0, 1, 3, 2)
        self.feature_maps = self.feature_maps
        if self.bias:
            self.feature_maps += self.b
        if self.act is None:
            out = self.feature_maps
        elif self.act == "relu":
            out = relu(self.feature_maps, training_parameters, self.arithmetic)

        return out

    def backward(self, dL_dout, training_parameters):
        """
        Performs backpropagation through the convolutional layer, calculating gradients with respect to layer parameters and inputs.

        Parameters:
            dL_dout (numpy array): The gradient of the loss with respect to the output of this layer,
                                   with shape (new_height, new_width, nFilters, batchsize)
            training_parameters (dict): A dictionary containing training parameters.

        Returns:
            - numpy.ndarray: Derivative of the loss function with respect to this layer's input, of shape (height, width, nChannels, batchsize).
        """
        batchsize = self.input_image.shape[3]
        # Backprop through activation function
        if self.act == "relu":
            dL_dout[self.feature_maps <= 0] = zeros_tensor(
                1, arithmetic=self.arithmetic, device=self.device
            )

        # Computing dL_dW (kernelsize,kernelsize,nChannels,nFilters):
        # dL_dW is the stride=1 convolution of dilated output gradient dL_dout with input image
        dL_dout_dilated = dilate_array(dL_dout, self.stride)

        self.dL_dW = efficient_conv2d(
            dL_dout_dilated,
            self.input_image,
            stride=1,
            axes_sum=(3, 3),
            arithmetic=self.arithmetic,
        )

        if self.dL_dW.shape[0] != self.W.shape[0]:
            # when stride and kernel size don't perfectly fit the input image,
            # the rightmost column and bottom row of the image will not be used,
            # => need to artificially remove the rightmost column and bottom row
            # of the computed weight gradient matrix
            self.dL_dW = self.dL_dW[:-1, :-1, :, :]
        self.dL_dW *= 1 / batchsize

        # Computing dL_db (nFilters,1):
        if self.bias:
            self.dL_db = dL_dout.sum(axis=(0, 1, 3)).reshape(self.nFilters, 1)
            self.dL_db = self.dL_db * (1 / batchsize)

        # Computing dL_din (height,width,nChannels,batchsize) :
        # dL_din is the stride=1 "full" convolution of "180 deg rotated" kernel with dilated output gradient matrix dL_dout
        # Equivalently, it is the valid convolution of rotated kernel with "dilated and zero padded" output gradient matrix dL_dout
        # To rotated the kernel, flip vertically then flip horizontally (or vice-versa)
        padwidth = (
            (self.kernelsize - 1, self.kernelsize - 1),
            (self.kernelsize - 1, self.kernelsize - 1),
            (0, 0),
            (0, 0),
        )

        dL_dout_dilated_padded = dL_dout_dilated.pad(pad_width=padwidth)
        rotated_kernel = self.W.flip(axis=0)
        rotated_kernel = rotated_kernel.flip(axis=1)

        dL_din = efficient_conv2d(
            rotated_kernel,
            dL_dout_dilated_padded,
            stride=1,
            axes_sum=(3, 2),
            arithmetic=self.arithmetic,
            partition=True,
        )
        dL_din = dL_din.transpose(0, 1, 3, 2)

        if dL_din.shape[0] != self.input_image.shape[0]:
            # when stride and kernel size don't perfectly fit the input image,
            # the rightmost column and bottom row of the image will not be used,
            # => need to artificially add these column and row, and set their gradients to zero
            p = self.input_image.shape[0] - dL_din.shape[0]
            padwidth = ((0, p), (0, p), (0, 0), (0, 0))
            dL_din = dL_din.pad(pad_width=padwidth)

        if (
            self.padding != 0 and self.padding != "valid"
        ):  # unpad gradient matrix before sending it to previous layer
            dL_din = dL_din[
                self.pad_top : -self.pad_bottom, self.pad_left : -self.pad_right, :, :
            ]

        return dL_din


class DropoutLayer(AbstractLayer):
    """
    Implements a dropout layer, randomly deactivating a subset of neurons during training.

    Arguments:
    - p (float): Dropout probability, the fraction of input units to drop.
    - inputshape (tuple of ints): Shape of the input array.
    - arithmetic (str): The arithmetic method, either 'fp' for floating point or 'lns' for logarithmic number system.
    """

    def __init__(self, p, inputshape, arithmetic):
        self.p = p
        self.outputshape = inputshape
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

    def initialize_weights_and_biases(self):
        pass  # no weights or biases

    def forward(self, x, training_parameters):
        """
        Forward propagates through the dropout layer, applying dropout to the input.
        - mask (numpy array): Dropout mask used during training, needed for backpropagation. Shape is same as input.
        - In training mode, the output is scaled by 1/(1-p) to account for the fact that some neurons are deactivated.
        - In evaluation mode, the output is the same as the input.

        Parameters:
            x (numpy array): Input data of shape (height, width, nChannels, batchsize) or (n_out, batchsize), depending on the previous layer.
            training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.

        Returns:
            a (numpy array): Output after applying dropout, same shape as input.
        """
        if self.mode == "train":
            # generate random sequence using numpy first to ensure reproducibility !
            drop = np.random.binomial(1, 1 - self.p, x.shape)
            if self.device == "gpu":
                drop = cp.asarray(drop)

            self.mask = FloatTensor(drop).to_arithmetic(self.arithmetic).to(self.device)

            self.mask = self.mask / (1 - self.p)
            out = x * self.mask

        elif self.mode == "eval":
            out = x

        return out

    def backward(self, dL_dout, training_parameters):
        """
        Backward propagates through the dropout layer, computing the gradient of the loss function with respect to the input.

        Parameters:
            dL_dout (numpy array): Gradient of the loss function with respect to the output of this layer.
            training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.

        Returns:
            dL_din (numpy array): Gradient of the loss function with respect to the input of this layer.
        """
        dL_din = dL_dout * self.mask
        return dL_din


class FlattenLayer(AbstractLayer):
    """
    Class implementing a flatten layer
    Parameters:
        - inputshape (tuple of ints): Shape of the input array (height,width,nChannels,batchsize)
        - arithmetic (str): The arithmetic method, either 'float' for floating point or 'lns' for logarithmic number system.
    """

    def __init__(self, inputshape, arithmetic):
        self.inputshape = inputshape
        self.arithmetic = arithmetic
        self.outputshape = np.prod(inputshape)
        self.device = "cpu"  # default device

    def initialize_weights_and_biases(self):
        pass  # no weights or biases

    def forward(self, x, training_parameters):
        """
        Forward propagate through the flatten layer.
        Parameters:
            - x (numpy array): Input data of shape (height, width, nChannels, batchsize)
            - training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.
        Returns:
            - a (numpy array): Output after flattening, of shape (n_out, batchsize)
        """

        batchsize = x.shape[3]
        # out = x.reshape(-1,batchsize)
        out = x.transpose(2, 0, 1, 3).reshape(
            -1, batchsize
        )  # tranpose to match pytorch convention
        return out

    def backward(self, dL_dout, training_parameters):
        """
        Backward propagate through the flatten layer.
        Parameters:
            - dL_dout (numpy array): Gradient of the loss function with respect to the output of this layer (n_out, batchsize)
            - training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.
        Returns:
            - dL_din (numpy array): Gradient of the loss function with respect to the input of this layer (height, width, nChannels, batchsize)
        """
        batchsize = dL_dout.shape[1]
        # dL_din = dL_dout.reshape(self.inputshape[0],self.inputshape[1],self.inputshape[2],batchsize)
        dL_din = dL_dout.reshape(
            self.inputshape[2], self.inputshape[0], self.inputshape[1], batchsize
        ).transpose(
            1, 2, 0, 3
        )  # tranpose to match pytorch convention
        return dL_din


class BatchNormalizationLayer(AbstractLayer):
    """
    Implements a batch normalization layer. Supports both 2D and 4D input.
    Parameters:
        - inputshape (tuple of ints): Shape of the input array (height,width,nChannels,batchsize) or (n_out,batchsize)
        - momentum (float): Momentum for moving averages
        - eps (float): Epsilon for numerical stability
        - act (str): Activation function. Supported: 'relu'
        - arithmetic (str): The arithmetic method, either 'float' for floating point or 'lns' for logarithmic number system.
    """

    def __init__(
        self,
        inputshape: int | tuple,
        momentum: float = 0.1,
        eps: float = 1e-5,
        act: str | None = None,
        bias: bool = True,
        arithmetic: str = "float",
    ):
        self.inputshape = inputshape
        self.momentum = momentum
        self.eps = eps
        self.act = act
        self.bias = bias
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

        if np.isscalar(
            inputshape
        ):  # inputshape is a scalar if previous layer is flatten layer
            self.D = inputshape
        elif (
            len(inputshape) == 3
        ):  # inputshape is a 3D tuple if previous layer is convolutional/pooling layer
            self.D = inputshape[2]

        if self.arithmetic == "lns":
            self.one_minus_momentum = (
                FloatTensor(np.array([1 - self.momentum])).to(self.device).to_lns()
            )

        self.outputshape = inputshape

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """
        Initialize W (gamma), b (beta) and moving averages.
        """
        self.moving_mu = zeros_tensor(
            (self.D, 1), arithmetic=self.arithmetic, device=self.device
        )  # not learnable
        self.moving_var = ones_tensor(
            (self.D, 1), arithmetic=self.arithmetic, device=self.device
        )  # not learnable
        self.W = ones_tensor(
            (self.D, 1), arithmetic=self.arithmetic, device=self.device
        )  # gamma 1D column vector (name is W for consistency with other layers)
        self.b = zeros_tensor(
            (self.D, 1), arithmetic=self.arithmetic, device=self.device
        )  # beta  1D column vector

    def forward(self, x, training_parameters):
        """
        Forward propagate through the batch normalization layer.
        Parameters:
            - x (numpy array): Input data of shape (height, width, nChannels, batchsize) or (n_out, batchsize), depending on the previous layer.
            - training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.
        Returns:
            - (numpy array): Output after applying batch normalization, same shape as input.
        """
        batchsize = x.shape[3]
        self.x = x
        if not np.isscalar(self.outputshape):
            batchsize = x.shape[3]
            x = x.transpose(2, 0, 1, 3)
            x = x.reshape(self.D, -1)
        if self.mode == "train":
            mu = x.mean(axis=1).reshape(-1, 1)
            var = x.var(axis=1, mu=mu).reshape(-1, 1)
            unbiases_var = var * (x.shape[1] / (x.shape[1] - 1))
            self.moving_mu = self.moving_mu * (1 - self.momentum) + mu * self.momentum
            self.moving_var = (
                self.moving_var * (1 - self.momentum) + unbiases_var * self.momentum
            )
            xmu = x - mu
            sqrtvar = (var + self.eps).sqrt()
            inv_sqrtvar = sqrtvar.inv()
            xhat = xmu * inv_sqrtvar
            self.cache = {
                "xmu": xmu,
                "sqrtvar": sqrtvar,
                "inv_sqrtvar": inv_sqrtvar,
                "xhat": xhat,
            }
        elif self.mode == "eval":
            xhat = (x - self.moving_mu) / (self.moving_var + self.eps).sqrt()

        out = xhat * self.W + self.b
        if not np.isscalar(self.outputshape):
            out = out.reshape(
                self.D, self.outputshape[0], self.outputshape[1], batchsize
            )
            out = out.transpose(1, 2, 0, 3)

        if self.act is None:
            pass
        elif self.act == "relu":
            self.cache["pre_act"] = out
            out = relu(out, training_parameters, self.arithmetic)
        return out

    def backward(self, dL_dout, training_parameters):
        """
        Backward propagate through the batch normalization layer.
        Parameters:
            - dL_dout (numpy array): Gradient of the loss function with respect to the output of this layer (height, width, nChannels, batchsize) or (n_out, batchsize), depending on the previous layer.
            - training_parameters (dict): Dictionary containing training parameters such as l2 regularization coefficient.
        Returns:
            - (numpy array): Gradient of the loss function with respect to the input of this layer (height, width, nChannels, batchsize) or (n_out, batchsize), depending on the previous layer.
        """
        x = self.x
        inputshape = x.shape
        batchsize = x.shape[3]

        if self.act == "relu":
            dL_dout[self.cache["pre_act"] <= 0] = zeros_tensor(
                1, arithmetic=self.arithmetic, device=self.device
            )

        if len(dL_dout.shape) == 4:
            dL_dout = dL_dout.transpose(2, 0, 1, 3)
            dL_dout = dL_dout.reshape(self.D, -1)

        batchsize_2d = dL_dout.shape[1]

        self.dL_db = dL_dout.sum(axis=1, keepdims=True) * (1 / batchsize)
        self.dL_dW = (dL_dout * self.cache["xhat"]).sum(axis=1, keepdims=True) * (
            1 / batchsize
        )

        # Intermediate gradients calculations are often a source of redundancy and can be streamlined.
        # Use the chain rule to simplify the computation and reduce the number of operations.
        dL_dxhat = dL_dout * self.W

        # We can merge some of the computations here to avoid repetitive calculations.
        dxhat_dvar = (self.cache["xmu"] * (-0.5)) * self.cache["inv_sqrtvar"] ** 3
        dvar_dxmu = self.cache["xmu"] * 2.0 / batchsize_2d
        dL_dvar = (dL_dxhat * dxhat_dvar).sum(axis=1, keepdims=True)

        # Now we can compute dL_dxmu in one go, avoiding the split into two terms.
        dL_dxmu = dL_dxhat * self.cache["inv_sqrtvar"] + dL_dvar * dvar_dxmu

        # The gradient with respect to the mean can now be calculated.
        dL_dmu = -dL_dxmu.sum(axis=1, keepdims=True)

        # Finally, the gradient with respect to the input can be calculated.
        # Again, we avoid splitting into two terms.
        dL_din = dL_dxmu + dL_dmu / batchsize_2d

        if not np.isscalar(self.outputshape):
            dL_din = dL_din.reshape(self.D, inputshape[0], inputshape[1], inputshape[3])
            dL_din = dL_din.transpose(1, 2, 0, 3)
        return dL_din


class ResidualBlock:
    """
    Implements a Residual Block layer for a neural network. A Residual Block consists of a series of layers
    where the input is added back to the output.

    :param layers: OrderedDict, containing layers of the Residual Block
    :param expansion: int, multiplier for the output dimension of the block (default: 1)
    :param stride: int, stride of the convolutional layers in the block (default: 1)
    :param arithmetic: str, the arithmetic used in the layer ('float' or 'lns')
    """

    def __init__(
        self,
        layers: OrderedDict,
        expansion: int = 1,
        stride: int = 1,
        arithmetic: str = "float",
    ):
        self.layers = layers
        self.expansion = expansion
        self.stride = stride
        self.shortcut = False
        self.input_shape = self.layers[0].inputshape
        output_shape = self.layers[len(self.layers) - 1].outputshape
        if stride != 1 or self.input_shape[2] != expansion * output_shape[2]:
            self.shortcut = True
            self.shortcut_layers = OrderedDict()
            self.shortcut_layers[0] = Convolutional2DLayer(
                nFilters=output_shape[2] * expansion,
                kernelsize=1,
                stride=stride,
                act=None,
                arithmetic=arithmetic,
                inputshape=self.input_shape,
            )
            self.shortcut_layers[1] = BatchNormalizationLayer(
                inputshape=self.shortcut_layers[0].outputshape, arithmetic=arithmetic
            )

        self.outputshape = output_shape
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device

    def forward(self, x, training_params):
        """
        Forward pass for the Residual Block. Computes the output of the block for a given input.

        :param x: numpy.ndarray, input to the block
        :param training_params: dict, containing training hyperparameters and settings
        :return: numpy.ndarray, output of the block
        """
        identity = x
        for layer in self.layers.values():
            x = layer.forward(x, training_params)
        if self.shortcut:
            for layer in self.shortcut_layers.values():
                identity = layer.forward(identity, training_params)
        self.x = x + identity
        out = relu(self.x, training_params, self.arithmetic)
        return out

    def backward(self, dL_dout, training_params):
        """
        Backward pass for the Residual Block. Computes the gradients of the loss with respect to the input.

        :param dL_dout: numpy.ndarray, gradient of the loss with respect to the output of the block
        :param training_params: dict, containing training hyperparameters and settings
        :return: numpy.ndarray, gradient of the loss with respect to the input of the block
        """
        dL_dout[self.x <= 0] = zeros_tensor(
            1, arithmetic=self.arithmetic, device=self.device
        )

        dL_dout_original = dL_dout

        if self.shortcut:
            for layer in reversed(self.shortcut_layers.values()):
                dL_dout_original = layer.backward(dL_dout_original, training_params)

        for layer in reversed(self.layers.values()):
            dL_dout = layer.backward(dL_dout, training_params)

        dL_dout = dL_dout + dL_dout_original

        return dL_dout

    def to(self, device):
        """
        Sets the device (cpu or gpu) for the block.
        """
        self.device = device
        for layer in self.layers.values():
            layer.to(device)
        if self.shortcut:
            for layer in self.shortcut_layers.values():
                layer.to(device)

    def set_training_mode(self):
        """
        Sets the block to training mode.
        """
        self.mode = "train"
        for layer in self.layers.values():
            layer.set_training_mode()
        if self.shortcut:
            for layer in self.shortcut_layers.values():
                layer.set_training_mode()

    def set_evaluation_mode(self):
        """
        Sets the block to evaluation mode.
        """
        self.mode = "eval"
        for layer in self.layers.values():
            layer.set_evaluation_mode()
        if self.shortcut:
            for layer in self.shortcut_layers.values():
                layer.set_evaluation_mode()


class NeuralNetwork:
    """
    Class implementing a Convolutional Neural Network for Classification
    Parameters:
        - layers (OrderedDict): Dictionary containing the ordered layers of the network layers[0]...layers[nlayers-1] .
                       Should follow the format:
                        1) layers[0] is the first layer in the network.
                        2) Last layer is always a FullyConnectedLayerWithSoftmax() with number of units equal to nClasses
                        3) Each layers' input shape is the previous layer output shape
        - nClasses (int): Number of classes for classification
        - arithmetic (str): The arithmetic method, either 'float' for floating point or 'lns' for logarithmic number system.
        - logdir (str): Directory to save logs to.
        - save_logs (bool): Whether to save logs or not.

    """

    def __init__(
        self,
        layers: OrderedDict,
        nClasses: int,
        arithmetic: str,
        logdir: str | None,
        save_logs: bool,
    ):
        self.layers = layers
        self.nClasses = nClasses
        self.arithmetic = arithmetic
        self.device = "cpu"  # default device
        self.logdir = logdir

        self.nlayers = len(self.layers)
        self.save_logs = save_logs

    def forward(self, input_image, training_parameters):
        """
        Forward propagates through the network, returning the output of the last layer.
        """
        out = input_image
        for ll in range(self.nlayers):
            out = self.layers[ll].forward(out, training_parameters)
        return out

    def backward(self, out, label, training_parameters):
        """
        Backpropagates through the network, calculating gradients with respect to layer parameters and inputs.
        """
        dL_din = self.layers[self.nlayers - 1].backward(out, label, training_parameters)
        for ll in range(self.nlayers - 2, -1, -1):
            dL_din = self.layers[ll].backward(dL_din, training_parameters)

    def to(self, device):
        """
        Sets the device (cpu or gpu) for the network.
        """
        self.device = device
        for ll in range(self.nlayers):
            self.layers[ll].to(device)

    def train(self):
        """
        Sets the network to training mode.
        """
        for ll in range(self.nlayers):
            self.layers[ll].set_training_mode()

    def eval(self):
        """
        Sets the network to evaluation mode.
        """
        for ll in range(self.nlayers):
            self.layers[ll].set_evaluation_mode()

    def save_state(self, log_dir: str):
        """
        Saves the state of the network to a pickle file.
        """
        with open(f"{log_dir}/model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_state(cls, log_dir: str):
        """
        Loads the state of the network from a pickle file.
        """
        with open(f"{log_dir}/model.pkl", "rb") as f:
            return pickle.load(f)
