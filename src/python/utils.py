import numpy as np
from src.python.tensor import FloatTensor, LNSTensor, zeros_tensor, ones_tensor

# ******************** SECTION (1): Activation Functions and their derivatives ************************#


def relu(x, train_params, arithmetic):
    """
    Relu activation function.
    Inputs:
        - x: Matrix/Vector of floats [?D array]
    Outputs:
        - y: Matrix/Vector of floats [same shape as x]
    """
    y = x.maximum(0)
    if arithmetic == "lns":
        if train_params["leaky_flag"]:
            y[(x.s_x == 0) & (x.s_z == 0)] = (
                x[(x.s_x == 0) & (x.s_z == 0)] * train_params["leaky_coeff"]
            )
    else:
        if train_params["leaky_flag"]:
            y[x < 0] = x[x < 0] * train_params["leaky_coeff"]
    return y


def d_relu(x, train_params, arithmetic):
    """
    Derivative of Relu activation function.
    Inputs:
        - x: Matrix/Vector of floats [?D array]
    Outputs:
        - d: Matrix/Vector of floats [same shape as x]
    """
    if arithmetic == "lns":
        one_lns = ones_tensor(x.shape, arithmetic, device=x.device)
        d = zeros_tensor(x.shape, arithmetic, device=x.device)
        d[x > 0] = one_lns
        if train_params["leaky_flag"]:
            d[(x.s_x == 0) & (x.s_z == 0)] = train_params["leaky_coeff"]
    else:
        d = FloatTensor(x > 0)
        if train_params["leaky_flag"]:
            d[x < 0] = train_params["leaky_coeff"]
    return d


def softmax(x, arithmetic, idx_true_class=None):
    """
    Softmax activation function.
    Inputs:
        - x: Matrix/Vector of floats [(nClasses,nExamples) array]
    Outputs:
        - y: Matrix/Vector of floats [(nClasses,nExamples) array]
    """
    logc = -x.max(axis=0, keepdims=True)  # For numerical stability
    xc = (x + logc).exp_neg()
    y = xc / xc.sum(axis=0)

    logsumexp = (x + logc).exp_neg().sum(axis=0).loge()
    log_softmax_logits = (x + logc) - logsumexp
    return y, log_softmax_logits


# ******************** SECTION (2): Loss Functions and Statistics Calculation ************************#


def l2_cost(y_predicted, y_D):
    """
    Calculate l2 mse loss
    Inputs:
        - y_predicted: Predicted data labels [(nClasses,nExamples) array]
        - y_D: True data labels [(nClasses,nExamples) array]
    Outputs:
        - L: mse loss [float scalar]
    """
    L = np.sum(0.5 * (y_predicted - y_D) ** 2)
    return L


def ce_cost(y_predicted, y_D):
    """
    Calculate cross-entropy loss. One Hot Encoding assumed.
    Inputs:
        - y_predicted: Predicted data labels [(nClasses,nExamples) array]
        - y_D: True data labels [(nClasses,nExamples) array]
    Outputs:
        - L: cross-entropy loss [float scalar]
    """
    nExamples = y_predicted.shape[1]
    non_zero_idx = np.argmax(y_D, axis=0)
    y = y_predicted[non_zero_idx, np.arange(nExamples)]
    y = y + 1e-20  # avoid log(0)
    return np.sum(-np.log(y[y != 0])) / nExamples


def calc_stats(y_predicted, y_true, M, layers, train_params, arithmetic):
    """
    Calculate Classification Statistics: Cross entropy loss and Accuracy. One Hot Encoding assumed.
    Inputs:
        - y_true: True data labels [2D array (M,nExamples)]
        - y_predicted: Predicted data labels [2D array (M,nExamples)]
        - M: Number of Classes [int scalar]
        - layers: layers of the network, to use weights in regularization [dictionary]
        - l2_reg_coeff: l2_regularization coefficeint [float scalar]
    Outputs:
        - L: Cross-entropy loss [float scalar]
        - acc: Classifiaction accuracy [float scalar]
    """
    if arithmetic == "lns":
        y_predicted = y_predicted.to_float()
        y_true = y_true.to_float()

    if y_predicted.device == "gpu":
        y_predicted = y_predicted.to_cpu()
        y_true = y_true.to_cpu()

    y_predicted = y_predicted.x
    y_true = y_true.x

    loss = ce_cost(y_predicted, y_true)
    y_predicted_idx = np.argmax(y_predicted, axis=0)
    y_true_idx = np.argmax(y_true, axis=0)
    correct = (y_predicted_idx == y_true_idx).sum()
    return loss, correct


# *************** SECTION (3): Misc: One Hot Encoding, Dataloader, epoch_time, etc... *******************#


def OneHotEncode(y, M):
    """
    Perform One Hot encoding of Labels.
    Inputs:
        - y: label vector, [1D array of size N]
        - M: Number of Classes [integer]
    Outputs:
        - yhot: one hot encoded labels [2D array of shape (M,N)]
    """
    N = len(y)
    yhot = np.zeros((M, N))
    yhot[y, np.arange(0, N)] = 1
    return yhot


def weight_init(fan_in, fan_out, W_shape, kind="pytorch"):
    """
    Implemetns Xavier uniform initialization for layer weights
    Inputs:
        - fan_in: Number of input units to the weight tensor [int scalar]
        - fan_out: Number of output units to the weight tensor [int scalar]
        - W_shape: Shape of the weight array [tuple of ints]
        - kind: 'pytorch' or 'xavier' [string]
    Outputs:
        - W_init: Weights initialized according to Xavier uniform initialization [?D array of shape W_shape ]
    """
    if kind == "pytorch":
        W_init = np.random.uniform(
            -np.sqrt(1 / fan_in), np.sqrt(1 / fan_in), size=W_shape
        ).astype(np.float32)
    elif kind == "xavier":
        low = -np.sqrt(6 / (fan_in + fan_out))
        high = np.sqrt(6 / (fan_in + fan_out))
        W_init = np.random.uniform(low, high, size=W_shape).astype(np.float32)
    return W_init


def Shuffle(x, y):
    """
    Do a random shuffle of the dataset
    Inputs:
        - x: Data examples [4D array of shape (height,width,nChannels,nExamples)]
        - y: Data Labels  [2D array of shape (nClasses,nExamples)]
    Outputs:
        - x: Shuffled Data examples [4D array of shape (height,width,nChannels,nExamples)]
        - y: Shuffled Data Labels [2D array of shape (nClasses,nExamples)]
    """
    n = x.shape[-1]
    rand_idx = np.random.choice(n, n, replace=False)
    x = x[:, :, :, rand_idx]
    y = y[:, rand_idx]
    return x, y


def dilate_array(input_array: FloatTensor | LNSTensor, dilation_rate: int):
    """
    Perform a dilation operation on a 4D tensor along the first two dimensions.

    Args:
    input_array (np.ndarray): A 4D array to be dilated.
    dilation_rate (int): The rate of dilation. It determines the spacing
                         between the values in the dilated array. A dilation
                         rate of 1 means no dilation.

    Returns:
    np.ndarray: The dilated array.

    Note:
    - The dilation is applied to the first two dimensions of the array.
    - The values between the original values in the dilated array are filled with zeros.
    - If the dilation_rate is 1, the function returns the original array.
    """
    if dilation_rate == 1:
        return input_array

    if dilation_rate < 1 or not isinstance(dilation_rate, int):
        raise ValueError("Dilation rate must be an integer greater than or equal to 1.")

    h, w, _, _ = input_array.shape
    dilated_h, dilated_w = h + (h - 1) * (dilation_rate - 1), w + (w - 1) * (
        dilation_rate - 1
    )
    dilated_array = zeros_tensor(
        (dilated_h, dilated_w, input_array.shape[2], input_array.shape[3]),
        input_array.arithmetic,
        device=input_array.device,
    )

    dilated_array[::dilation_rate, ::dilation_rate, :, :] = input_array
    return dilated_array


# *************** SECTION (4): Im2col+Strides Tricks, Convolution Function *******************#


def patches_stride_trick(
    input_image: FloatTensor | LNSTensor,
    masksize: int,
    k_stride: int,
    axes_sum: tuple | None = None,
):
    """
    Extract all windows (patches) from an image and stack them as columns of a matrix, in preparation for convolution.
    Numpy allows to change the memory stride values of arrays, i.e. change how array is viewed and read in memory.
    This function uses this numpy feature to extract all patches from an image without using any for loop !
    We simply change the way numpy accesses the image in memory which creates a matrix containing all our patches.
    Inputs:
        - input_image: [4D array of shape (height,width,nChannels,nExamples)]
        - masksize: size of one dimensions of the applied mask [int scalar]  (mask is 2D array (masksize,masksize) )
        - stride: stride of the kernel [int scalar]
    Outputs:
        - patch_matrix: All image patches for convolution stacked into a matrix. [4D array of shape (masksize^2,nPatches,nChannels,nExamples)]
        where each patch of size (masksize,masksize) is stacked into  a vector of size (masksize^2)
    """
    n_channels, batch_size = input_image.shape[2], input_image.shape[3]

    num_patches_height = (input_image.shape[0] - masksize) // k_stride + 1
    num_patches_width = (input_image.shape[1] - masksize) // k_stride + 1

    if axes_sum is None:
        # for maxpooling
        output_shape = (
            masksize,
            masksize,
            num_patches_height,
            num_patches_width,
            n_channels,
            batch_size,
        )
        patch_matrix = input_image.as_strided(
            output_shape=output_shape, k_stride=k_stride, axes_sum=axes_sum
        )
        patch_matrix = patch_matrix.reshape(
            masksize * masksize,
            num_patches_height * num_patches_width,
            n_channels,
            batch_size,
        )
    else:
        # for convolution
        # Designing strides to directly produce the desired output shape (used in im2col)
        if axes_sum == (3, 2) or axes_sum == (2, 2):
            output_shape = (
                num_patches_height,
                num_patches_width,
                batch_size,
                masksize,
                masksize,
                n_channels,
            )
            patch_matrix = input_image.as_strided(
                output_shape=output_shape, k_stride=k_stride, axes_sum=axes_sum
            )
            patch_matrix = patch_matrix.reshape(
                num_patches_height * num_patches_width * batch_size,
                masksize * masksize * n_channels,
            )
        elif axes_sum == (3, 3):
            output_shape = (
                num_patches_height,
                num_patches_width,
                n_channels,
                masksize,
                masksize,
                batch_size,
            )
            patch_matrix = input_image.as_strided(
                output_shape=output_shape, k_stride=k_stride, axes_sum=axes_sum
            )
            patch_matrix = patch_matrix.reshape(
                num_patches_height * num_patches_width * n_channels,
                masksize * masksize * batch_size,
            )

    return patch_matrix


def efficient_conv2d(
    kernel: FloatTensor | LNSTensor,
    in_maps: FloatTensor | LNSTensor,
    stride: int,
    axes_sum: tuple,
    arithmetic: str,
    partition: bool = False,
):
    """
    Perform Efficient Convolution of 4D kernel with 4D in_maps.
    Convolution is performed over the first two dimenions of kernel and input, then sumed over axes_sum dimensions.
    Efficient refers to using two tricks:
        1) Extract windows (patches) from in_maps by manipulating numpy memory strides
        2) Perform convolution via im2col (image to column), i.e. via matrix multiplication
    Inputs:
        - kernel: [4D array of shape (kernelsize,kernelsize,?,?)]
        - in_maps:  [4D array of shape (height,width,?,?)]
        - stride: Convolution Stride [int scalar]
        - axes_sum: List [a,b] where a is the dimension to sum over in kernel and b is the dimension to sum over in in_maps,
          i.e. kernel.shape[a]==in_maps.shape[b] [list of ints [a,b]]
        - feature_maps: Output of convolution, last two dimensions are the ones not sumed over [4D array of shape (new_height, new_width, ?, ?)]
    """
    kernelsize = kernel.shape[0]
    kernel_shape = kernel.shape

    outputshape = np.array([0, 0])
    outputshape[0] = np.floor((in_maps.shape[0] - kernelsize) / stride + 1)
    outputshape[1] = np.floor((in_maps.shape[1] - kernelsize) / stride + 1)

    K_axis_remain = [2, 3]
    K_axis_remain.remove(axes_sum[0])
    K_axis_remain = K_axis_remain[0]

    kernel = kernel.transpose(0, 1, axes_sum[0], K_axis_remain).reshape(
        kernelsize**2 * kernel_shape[axes_sum[0]], kernel_shape[K_axis_remain]
    )

    P_axis_remain = [2, 3]
    P_axis_remain.remove(axes_sum[1])
    P_axis_remain = P_axis_remain[0]

    if partition:
        # pre-allocation
        feature_maps = zeros_tensor(
            (
                outputshape[0],
                outputshape[1],
                in_maps.shape[P_axis_remain],
                kernel_shape[K_axis_remain],
            ),
            arithmetic,
            device=in_maps.device,
        )
        num_partitions = 10
        for part in range(num_partitions):
            start_range = int(np.floor(part * in_maps.shape[3] / num_partitions))
            end_range = int(np.floor((part + 1) * in_maps.shape[3] / num_partitions))
            patches_matrix_part = patches_stride_trick(
                in_maps[:, :, :, start_range:end_range],
                kernelsize,
                stride,
                axes_sum=axes_sum,
            )
            feature_maps_part = patches_matrix_part @ kernel
            feature_maps[:, :, start_range:end_range, :] = feature_maps_part.reshape(
                (outputshape[0], outputshape[1], -1, kernel_shape[K_axis_remain])
            )
    else:
        patches_matrix = patches_stride_trick(
            in_maps, kernelsize, stride, axes_sum=axes_sum
        )
        feature_maps = patches_matrix @ kernel

    feature_maps = feature_maps.reshape(
        (
            outputshape[0],
            outputshape[1],
            in_maps.shape[P_axis_remain],
            kernel_shape[K_axis_remain],
        )
    )

    return feature_maps
