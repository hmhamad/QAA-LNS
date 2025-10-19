import numpy as np
import src.python.get_kernels as kernels
import cupy as cp

THREADS_PER_BLOCK = int(16)
block_config_1d = (THREADS_PER_BLOCK, 1, 1)
block_config_2d = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

binary_op_to_idx = {"add": 0, "mul": 1, "max": 2, "min": 3}
logical_op_to_idx = {"eq": 0, "gt": 1, "lt": 2, "ge": 3, "le": 4}
val_reduce_op_to_idx = {"sum": 0, "max": 1, "min": 2}
arg_reduce_op_to_idx = {"argmax": 0, "argmin": 1}

# Global variables
SUM_APPROX = 1  # 0: LUT, 1: LINEAR
SUM_TECHNIQUE = 1  # 0: block, 1: doubling
TOTAL_BITS = (
    32  # Total number of bits in the LNS representation (including sign and zero flag)
)
LNS_SIZE_IN_BYTES = 6  # 4 bytes for log, 1 byte for sign, 1 byte for zero


# Set the technique for all LNS operations involving summation
def set_lns_summation_mode(sum_approx, sum_technique, lns_bits):
    global SUM_APPROX, SUM_TECHNIQUE, TOTAL_BITS
    SUM_APPROX = sum_approx
    SUM_TECHNIQUE = sum_technique
    TOTAL_BITS = lns_bits


# empty, zeros, and ones
def empty_lns(shape: int | tuple, device: str = "gpu"):
    if np.isscalar(shape):
        shape = shape
    if device == "gpu":
        log = cp.empty(shape, dtype=np.int32)
        sign = cp.empty(shape, dtype=bool)
        zero = cp.empty(shape, dtype=bool)
    else:
        log = np.empty(shape, dtype=np.int32)
        sign = np.empty(shape, dtype=bool)
        zero = np.empty(shape, dtype=bool)
    return LNSTensor(log, sign, zero)


def zeros_lns(shape: int | tuple, device: str = "gpu"):
    if np.isscalar(shape):
        shape = shape
    MAX_POSS_INT = int((1 << ((TOTAL_BITS - 2) - 1)) - 1)
    if device == "gpu":
        log = cp.ones(shape, dtype=np.int32) * -MAX_POSS_INT
        sign = cp.ones(shape, dtype=bool)
        zero = cp.ones(shape, dtype=bool)
    else:
        log = np.ones(shape, dtype=np.int32) * -MAX_POSS_INT
        sign = np.ones(shape, dtype=bool)
        zero = np.ones(shape, dtype=bool)
    return LNSTensor(log, sign, zero)


def ones_lns(shape: int | tuple, device: str = "gpu"):
    if np.isscalar(shape):
        shape = shape
    if device == "gpu":
        log = cp.zeros(shape, dtype=np.int32)
        sign = cp.ones(shape, dtype=bool)
        zero = cp.zeros(shape, dtype=bool)
    else:
        log = np.zeros(shape, dtype=np.int32)
        sign = np.ones(shape, dtype=bool)
        zero = np.zeros(shape, dtype=bool)
    return LNSTensor(log, sign, zero)


def zeros_tensor(shape: int | tuple, arithmetic: str, device: str = "gpu"):
    if arithmetic == "lns":
        return zeros_lns(shape, device)
    else:
        if device == "gpu":
            return FloatTensor(cp.zeros(shape))
        else:
            return FloatTensor(np.zeros(shape))


def ones_tensor(shape: int | tuple, arithmetic: str, device: str = "gpu"):
    if arithmetic == "lns":
        return ones_lns(shape, device)
    else:
        if device == "gpu":
            return FloatTensor(cp.ones(shape))
        else:
            return FloatTensor(np.ones(shape))


class LNSTensor:
    def __init__(self, log, sign, zero, reorder: bool = False, copy: bool = False):
        # reorder: After an indexing operation, array can lose row-major order. This flag allows to reorder the array
        # in effect, making a new cheap copy of the array
        # reorder takes precedence over copy

        assert log.dtype == np.int32, "log must be a array of type np.int32"
        assert (
            sign.dtype == zero.dtype == bool
        ), "sign and zero must be a array of type bool"
        assert (
            log.shape == sign.shape == zero.shape
        ), "log, sign and zero must have the same shape"
        assert (
            type(log) == type(sign) == type(zero)
        ), "log, sign and zero must be of the same type, either np.ndarray for cpu or cp.ndarray for gpu"

        self.arithmetic = "lns"
        self.device = "gpu" if type(log) == cp.ndarray else "cpu"

        if self.device == "gpu" and reorder:
            self.log = cp.asarray(log, order="C")
            self.sign = cp.asarray(sign, order="C")
            self.zero = cp.asarray(zero, order="C")
        elif copy:
            self.log = log.copy()
            self.sign = sign.copy()
            self.zero = zero.copy()
        else:
            self.log = log
            self.sign = sign
            self.zero = zero

    def __repr__(self):
        if self.device == "cpu":
            return "CPU LNSTensor(log = {}, sign = {}, zero = {})".format(
                self.log, self.sign, self.zero
            )
        else:
            return "GPU LNSTensor(log = {}, sign = {}, zero = {})".format(
                self.log, self.sign, self.zero
            )

    @property
    def T(self):
        return LNSTensor(self.log.T, self.sign.T, self.zero.T, reorder=True)

    @property
    def F(self):
        return lns_to_float(self)

    @property
    def shape(self):
        return self.log.shape

    @property
    def size(self):
        return self.log.size

    @property
    def strides(self):
        return tuple((self.log.strides, self.sign.strides, self.zero.strides))

    @property
    def ndim(self):
        return self.log.ndim

    @property
    def dtype(self):
        return (self.log.dtype, self.sign.dtype, self.zero.dtype)

    def __getitem__(self, item):
        # numpy and cupy slicing doesn't actually copy the data, only changes the view
        # to use the new slice in cuda, the stride of the slice needs to be manually handled in cuda
        # Workaround: Make a fresh new array (additional overhead) by .copy()
        return LNSTensor(self.log[item], self.sign[item], self.zero[item], copy=True)

    def __setitem__(self, item, value):
        self.log[item] = value.log
        self.sign[item] = value.sign
        self.zero[item] = value.zero

    def __eq__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_logical_op(self, other, "eq")

    def __ne__(self, other):
        return ~self.__eq__(other)

    def __gt__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_logical_op(self, other, "gt")

    def __ge__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_logical_op(self, other, "ge")

    def __lt__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_logical_op(self, other, "lt")

    def __le__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_logical_op(self, other, "le")

    def __neg__(self):
        new_sign = ~self.sign
        return LNSTensor(self.log, new_sign, self.zero, copy=True)

    def __abs__(self):
        if self.device == "cpu":
            new_sign = np.ones(self.shape, dtype=bool)
        else:
            new_sign = cp.ones(self.shape, dtype=bool)
        return LNSTensor(self.log, new_sign, self.zero, copy=True)

    def __pow__(self, power):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_unary_op(self, "pow", power)

    def sqrt(self):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_unary_op(self, "sqrt")

    def exp_neg(self):
        if self.device == "cpu":
            pass
        else:
            return lns_unary_op(self, "exp_neg")

    def loge(self):
        if self.device == "cpu":
            pass
        else:
            return self.to_float().loge().to_lns()

    def __add__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "add")

    def __iadd__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "add")

    def __sub__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, -other, "add")

    def __isub__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, -other, "add")

    def __mul__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "mul")

    def __imul__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "mul")

    def __truediv__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        assert not cp.any(other.zero), "Divide by zero error"
        other_inverse = LNSTensor(-other.log, other.sign, other.zero)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other_inverse, "mul")

    def __itruediv__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)
        assert not cp.any(other.zero), "Divide by zero error"
        other_inverse = LNSTensor(-other.log, other.sign, other.zero)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other_inverse, "mul")

    def inv(self):
        assert not cp.any(self.zero), "Divide by zero error"
        return LNSTensor(-self.log, self.sign, self.zero)

    def maximum(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "max")

    def minimum(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to_lns().to(self.device)

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_elementwise_op(self, other, "min")

    def __matmul__(self, other):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_mat_mult(self, other)

    def __imatmul__(self, other):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_mat_mult(self, other)

    def reshape(self, *args):
        # Emulate Numpy Reshape function
        log_new = self.log.reshape(*args)
        sign_new = self.sign.reshape(*args)
        zero_new = self.zero.reshape(*args)
        if self.device == "cpu":
            return LNSTensor(log_new, sign_new, zero_new, copy=True)
        else:
            return LNSTensor(log_new, sign_new, zero_new, reorder=True)

    def transpose(self, *args):
        # Emulate Numpy Transpose function
        log_new = self.log.transpose(*args)
        sign_new = self.sign.transpose(*args)
        zero_new = self.zero.transpose(*args)
        if self.device == "cpu":
            return LNSTensor(log_new, sign_new, zero_new, copy=True)
        else:
            return LNSTensor(log_new, sign_new, zero_new, reorder=True)

    def flatten(self):
        return self.reshape(-1)

    def pad(self, pad_width, constant_values=0):
        # Emulate Numpy Pad function
        MAX_POSS_INT = int((1 << ((TOTAL_BITS - 2) - 1)) - 1)
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            if constant_values == 0:
                log_new = cp.pad(
                    self.log, pad_width, "constant", constant_values=-MAX_POSS_INT
                )
                sign_new = cp.pad(
                    self.sign, pad_width, "constant", constant_values=True
                )
                zero_new = cp.pad(
                    self.zero, pad_width, "constant", constant_values=True
                )
            elif constant_values == "-inf":
                log_new = cp.pad(
                    self.log, pad_width, "constant", constant_values=MAX_POSS_INT
                )
                sign_new = cp.pad(
                    self.sign, pad_width, "constant", constant_values=False
                )
                zero_new = cp.pad(
                    self.zero, pad_width, "constant", constant_values=False
                )
            return LNSTensor(log_new, sign_new, zero_new, reorder=True)

    def flip(self, axis):
        # Emulate Numpy Flip function
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            log_new = cp.flip(self.log, axis)
            sign_new = cp.flip(self.sign, axis)
            zero_new = cp.flip(self.zero, axis)
            return LNSTensor(log_new, sign_new, zero_new, reorder=True)

    def tile(self, reps):
        # Emulate Numpy Tile function
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            log_new = cp.tile(self.log, reps)
            sign_new = cp.tile(self.sign, reps)
            zero_new = cp.tile(self.zero, reps)
            return LNSTensor(log_new, sign_new, zero_new, reorder=True)

    def as_strided(
        self, output_shape: tuple, k_stride: int, axes_sum: tuple | None = None
    ):
        log_strides, sign_strides, zero_strides = self.strides
        if axes_sum is None:
            log_strides = (
                log_strides[0],
                log_strides[1],
                k_stride * log_strides[0],
                k_stride * log_strides[1],
                log_strides[2],
                log_strides[3],
            )
            sign_strides = (
                sign_strides[0],
                sign_strides[1],
                k_stride * sign_strides[0],
                k_stride * sign_strides[1],
                sign_strides[2],
                sign_strides[3],
            )
            zero_strides = (
                zero_strides[0],
                zero_strides[1],
                k_stride * zero_strides[0],
                k_stride * zero_strides[1],
                zero_strides[2],
                zero_strides[3],
            )
        elif axes_sum == (3, 2) or axes_sum == (2, 2):
            log_strides = (
                k_stride * log_strides[0],
                k_stride * log_strides[1],
                log_strides[3],
                log_strides[0],
                log_strides[1],
                log_strides[2],
            )
            sign_strides = (
                k_stride * sign_strides[0],
                k_stride * sign_strides[1],
                sign_strides[3],
                sign_strides[0],
                sign_strides[1],
                sign_strides[2],
            )
            zero_strides = (
                k_stride * zero_strides[0],
                k_stride * zero_strides[1],
                zero_strides[3],
                zero_strides[0],
                zero_strides[1],
                zero_strides[2],
            )
        elif axes_sum == (3, 3):
            log_strides = (
                k_stride * log_strides[0],
                k_stride * log_strides[1],
                log_strides[2],
                log_strides[0],
                log_strides[1],
                log_strides[3],
            )
            sign_strides = (
                k_stride * sign_strides[0],
                k_stride * sign_strides[1],
                sign_strides[2],
                sign_strides[0],
                sign_strides[1],
                sign_strides[3],
            )
            zero_strides = (
                k_stride * zero_strides[0],
                k_stride * zero_strides[1],
                zero_strides[2],
                zero_strides[0],
                zero_strides[1],
                zero_strides[3],
            )

        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            patch_matrix_log = cp.lib.stride_tricks.as_strided(
                self.log, shape=output_shape, strides=log_strides
            )
            patch_matrix_sign = cp.lib.stride_tricks.as_strided(
                self.sign, shape=output_shape, strides=sign_strides
            )
            patch_matrix_zero = cp.lib.stride_tricks.as_strided(
                self.zero, shape=output_shape, strides=zero_strides
            )
            return LNSTensor(
                patch_matrix_log, patch_matrix_sign, patch_matrix_zero, reorder=True
            )

    def copy(self):
        return LNSTensor(self.log, self.sign, self.zero, copy=True)

    def to_float(self, inplace=False):
        if inplace:
            self = lns_to_float(self)
            return None
        else:
            return lns_to_float(self)

    def to_arithmetic(self, arithmetic, inplace=False):
        if inplace:
            if arithmetic == "float":
                self.to_float(inplace=True)
            else:
                pass  # already in lns
            return None
        else:
            if arithmetic == "float":
                return self.to_float()
            else:
                return self  # already in lns

    def to(self, device, inplace=False):
        if device == "cpu":
            return self.to_cpu(inplace)
        elif device == "gpu":
            return self.to_gpu(inplace)
        else:
            raise ValueError("Device must be cpu or gpu")

    def to_gpu(self, inplace=False):
        if self.device == "gpu":  # already on gpu
            return None if inplace else self.copy()
        else:  # move to gpu
            if inplace:
                self.device = "gpu"
                self.log = cp.array(self.log)
                self.sign = cp.array(self.sign)
                self.zero = cp.array(self.zero)
                return None
            else:
                return LNSTensor(
                    cp.array(self.log), cp.array(self.sign), cp.array(self.zero)
                )

    def to_cpu(self, inplace=False):
        if self.device == "cpu":  # already on cpu
            return None if inplace else self.copy()
        else:  # move to cpu
            if inplace:
                self.device = "cpu"
                self.log = cp.asnumpy(self.log)
                self.sign = cp.asnumpy(self.sign)
                self.zero = cp.asnumpy(self.zero)
                return None
            else:
                return LNSTensor(
                    cp.asnumpy(self.log), cp.asnumpy(self.sign), cp.asnumpy(self.zero)
                )

    def sum(self, axis=None, keepdims=False):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_val_reduce(self, "sum", axis, keepdims)

    def max(self, axis=None, keepdims=False):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_val_reduce(self, "max", axis, keepdims)

    def min(self, axis=None, keepdims=False):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_val_reduce(self, "min", axis, keepdims)

    def argmax(self, axis=None, get=False):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_arg_reduce(self, "argmax", axis, get)

    def argmin(self, axis=None, get=False):
        if self.device == "cpu":
            pass  # TODO: implement cpu version
        else:
            return lns_arg_reduce(self, "argmin", axis, get)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            size = self.size
        elif isinstance(axis, int):
            size = self.shape[axis]
        else:
            size = 1
            for i in axis:
                size *= self.shape[i]
        return self.sum(axis, keepdims) / size

    def var(self, axis=None, mu=None, unbiased=False, keepdims=False):
        if mu is None:
            mu = self.mean(axis, keepdims=True)
        if unbiased:
            if axis is None:
                size = self.size
            elif isinstance(axis, int):
                size = self.shape[axis]
            else:
                size = 1
                for i in axis:
                    size *= self.shape[i]
            return ((self - mu) ** 2).sum(axis, keepdims) / (size - 1)
        else:
            return ((self - mu) ** 2).mean(axis, keepdims=keepdims)


class FloatTensor:
    def __init__(self, x, reorder=False, copy=False):
        # reorder: After an indexing operation, array can lose row-major order. This flag allows to reorder the array
        # in effect, making a new cheap copy of the array
        # reorder takes precedence over copy

        # if scalar, convert to array
        if np.isscalar(x):
            x = np.array(x, dtype=np.float32)

        assert (
            type(x) == np.ndarray or type(x) == cp.ndarray
        ), "x must be either a np.ndarray or cp.ndarray"

        # make dtype float32
        x = x.astype(np.float32)

        self.arithmetic = "float"
        self.device = "gpu" if type(x) == cp.ndarray else "cpu"

        if self.device == "gpu" and reorder:
            self.x = cp.asarray(x, order="C")
        elif copy:
            self.x = x.copy()
        else:
            self.x = x

    def __repr__(self):
        if self.device == "cpu":
            return "CPU FloatTensor({})".format(self.x)
        else:
            return "GPU FloatTensor({})".format(self.x)

    @property
    def T(self):
        if self.device == "cpu":
            return FloatTensor(self.x.T, copy=True)
        else:
            return FloatTensor(self.x.T, reorder=True)

    @property
    def shape(self):
        return self.x.shape

    @property
    def size(self):
        return self.x.size

    @property
    def strides(self):
        return self.x.strides

    @property
    def ndim(self):
        return self.x.ndim

    @property
    def dtype(self):
        return self.x.dtype

    def __getitem__(self, item):
        # numpy and cupy slicing doesn't actually copy the data, only changes the view
        # to use the new slice in cuda, the stride of the slice needs to be manually handled in cuda
        # Workaround: Make a fresh new array (additional overhead) by .copy()
        return FloatTensor(self.x[item], copy=True)

    def __setitem__(self, item, value):
        self.x[item] = value.x

    def __eq__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return self.x == other.x

    def __ne__(self, other):
        return ~self.__eq__(other)

    def __gt__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return self.x > other.x

    def __ge__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return self.x >= other.x

    def __lt__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return self.x < other.x

    def __le__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return self.x <= other.x

    def __neg__(self):
        return FloatTensor(-self.x)

    def __abs__(self):
        return FloatTensor(abs(self.x))

    def __pow__(self, power):
        return FloatTensor(self.x**power)

    def sqrt(self):
        if self.device == "cpu":
            return FloatTensor(np.sqrt(self.x))
        else:
            return FloatTensor(cp.sqrt(self.x))

    def __add__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return FloatTensor(self.x + other.x)

    def __iadd__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        self.x += other.x
        return self

    def __sub__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return FloatTensor(self.x - other.x)

    def __isub__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        self.x -= other.x
        return self

    def __mul__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return FloatTensor(self.x * other.x)

    def __imul__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        self.x *= other.x
        return self

    def __truediv__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        return FloatTensor(self.x / other.x)

    def __itruediv__(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        self.x /= other.x
        return self

    def inv(self):
        assert not cp.any(self.x == 0), "Divide by zero error"
        return FloatTensor(1 / self.x)

    def maximum(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        if self.device == "cpu":
            return FloatTensor(np.maximum(self.x, other.x))
        else:
            return FloatTensor(cp.maximum(self.x, other.x))

    def minimum(self, other):
        if np.isscalar(other):
            other = FloatTensor(other).to(self.device)
        if self.device == "cpu":
            return FloatTensor(np.minimum(self.x, other.x))
        else:
            return FloatTensor(cp.minimum(self.x, other.x))

    def __matmul__(self, other):
        if self.device == "cpu":
            return FloatTensor(np.matmul(self.x, other.x))
        else:
            return FloatTensor(cp.matmul(self.x, other.x))

    def __imatmul__(self, other):
        if self.device == "cpu":
            self.x = np.matmul(self.x, other.x)
        else:
            self.x = cp.matmul(self.x, other.x)
        return self

    def reshape(self, *args):
        if self.device == "cpu":
            return FloatTensor(self.x.reshape(*args))
        else:
            return FloatTensor(self.x.reshape(*args), reorder=True)

    def transpose(self, *args):
        if self.device == "cpu":
            return FloatTensor(self.x.transpose(*args))
        else:
            return FloatTensor(self.x.transpose(*args), reorder=True)

    def flatten(self):
        return self.reshape(-1)

    def pad(self, pad_width, constant_values=0):
        if constant_values == "-inf":
            constant_values = -np.inf
        if self.device == "cpu":
            return FloatTensor(
                np.pad(self.x, pad_width, "constant", constant_values=constant_values)
            )
        else:
            return FloatTensor(
                cp.pad(self.x, pad_width, "constant", constant_values=constant_values),
                reorder=True,
            )

    def flip(self, axis):
        if self.device == "cpu":
            return FloatTensor(np.flip(self.x, axis))
        else:
            return FloatTensor(cp.flip(self.x, axis), reorder=True)

    def tile(self, reps):
        if self.device == "cpu":
            return FloatTensor(np.tile(self.x, reps))
        else:
            return FloatTensor(cp.tile(self.x, reps), reorder=True)

    def as_strided(
        self, output_shape: tuple, k_stride: int, axes_sum: tuple | None = None
    ):
        if axes_sum is None:
            strides = (
                self.strides[0],
                self.strides[1],
                k_stride * self.strides[0],
                k_stride * self.strides[1],
                self.strides[2],
                self.strides[3],
            )
        elif axes_sum == (3, 2) or axes_sum == (2, 2):
            strides = (
                k_stride * self.strides[0],
                k_stride * self.strides[1],
                self.strides[3],
                self.strides[0],
                self.strides[1],
                self.strides[2],
            )
        elif axes_sum == (3, 3):
            strides = (
                k_stride * self.strides[0],
                k_stride * self.strides[1],
                self.strides[2],
                self.strides[0],
                self.strides[1],
                self.strides[3],
            )
        if self.device == "cpu":
            return FloatTensor(
                np.lib.stride_tricks.as_strided(
                    self.x, shape=output_shape, strides=strides
                )
            )
        else:
            return FloatTensor(
                cp.lib.stride_tricks.as_strided(
                    self.x, shape=output_shape, strides=strides
                ),
                reorder=True,
            )

    def copy(self):
        return FloatTensor(self.x, copy=True)

    def to(self, device, inplace=False):
        if device == "cpu":
            return self.to_cpu(inplace)
        elif device == "gpu":
            return self.to_gpu(inplace)
        else:
            raise ValueError("Device must be cpu or gpu")

    def to_gpu(self, inplace=False):
        if self.device == "gpu":  # already on gpu
            return None if inplace else self.copy()
        else:  # move to gpu
            if inplace:
                self.x = cp.array(self.x)
                self.device = "gpu"
                return None
            else:
                return FloatTensor(cp.array(self.x))

    def to_cpu(self, inplace=False):
        if self.device == "cpu":  # already on cpu
            return None if inplace else self.copy()
        else:  # move to cpu
            if inplace:
                self.device = "cpu"
                self.x = cp.asnumpy(self.x)
                return None
            else:
                return FloatTensor(cp.asnumpy(self.x))

    def sum(self, axis=None, keepdims=False):
        return FloatTensor(self.x.sum(axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        return FloatTensor(self.x.max(axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False):
        return FloatTensor(self.x.min(axis, keepdims=keepdims))

    def argmax(self, axis=None, get=False):
        return (
            (self.x.argmax(axis)).get()
            if (get and self.device == "gpu")
            else self.x.argmax(axis)
        )

    def argmin(self, axis=None, get=False):
        return (
            (self.x.argmin(axis)).get()
            if (get and self.device == "gpu")
            else self.x.argmin(axis)
        )

    def mean(self, axis=None, keepdims=False):
        return FloatTensor(self.x.mean(axis, keepdims=keepdims))

    def var(self, axis=None, mu=None, unbiased=False, keepdims=False):
        if mu is None:
            mu = self.mean(axis, keepdims=True)
        if unbiased:
            if axis is None:
                size = self.size
            elif isinstance(axis, int):
                size = self.shape[axis]
            else:
                size = 1
                for i in axis:
                    size *= self.shape[i]
            return ((self - mu) ** 2).sum(axis, keepdims) / (size - 1)
        else:
            return ((self - mu) ** 2).mean(axis, keepdims=keepdims)

    def to_lns(self, inplace=False):
        if inplace:
            self = float_to_lns(self)
            return None
        else:
            return float_to_lns(self)

    def to_arithmetic(self, arithmetic, inplace=False):
        if inplace:
            if arithmetic == "float":
                pass  # already in float
            else:
                self.to_lns(inplace=True)
            return None
        else:
            if arithmetic == "float":
                return self  # already in float
            else:
                return self.to_lns()

    def exp_neg(self):
        if self.device == "cpu":
            return FloatTensor(np.exp(self.x))
        else:
            return FloatTensor(cp.exp(self.x))

    def loge(self):
        if self.device == "cpu":
            return FloatTensor(np.log(self.x))
        else:
            return FloatTensor(cp.log(self.x))


def batch_concat(tensors: list[FloatTensor | LNSTensor], axis: int = -1):
    if type(tensors[0]) == FloatTensor:
        if tensors[0].device == "cpu":
            return FloatTensor(np.stack([t.x for t in tensors], axis=axis))
        else:
            return FloatTensor(cp.stack([t.x for t in tensors], axis=axis))
    else:
        if tensors[0].device == "cpu":
            return LNSTensor(
                np.stack([t.log for t in tensors], axis=axis),
                np.stack([t.sign for t in tensors], axis=axis),
                np.stack([t.zero for t in tensors], axis=axis),
            )
        else:
            return LNSTensor(
                cp.stack([t.log for t in tensors], axis=axis),
                cp.stack([t.sign for t in tensors], axis=axis),
                cp.stack([t.zero for t in tensors], axis=axis),
            )


# LNS->Float and Float->LNS
def float_to_lns(x_f: FloatTensor):
    x_size = x_f.size
    x_shape = x_f.shape

    if x_f.device == "cpu":
        # TODO: Implement CPU version (current version cpu->gpu->cpu)
        d_x_f = cp.asarray(x_f.x)

        x_lns = empty_lns(x_shape, device="gpu")

        NUM_BLOCKS = int((x_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
        grid_config = (NUM_BLOCKS, 1, 1)

        kernels.cuda_float_to_lns(
            grid_config,
            block_config_1d,
            (d_x_f, x_lns.log, x_lns.sign, x_lns.zero, np.int32(x_size)),
        )
        x_lns = x_lns.to_cpu()
    else:
        d_x_f = cp.asarray(x_f.x)

        x_lns = empty_lns(x_shape, device=x_f.device)

        NUM_BLOCKS = int((x_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
        grid_config = (NUM_BLOCKS, 1, 1)

        kernels.cuda_float_to_lns(
            grid_config,
            block_config_1d,
            (d_x_f, x_lns.log, x_lns.sign, x_lns.zero, np.int32(x_size)),
        )

    return x_lns


def lns_to_float(x_lns: LNSTensor):
    x_size = x_lns.size
    x_shape = x_lns.shape

    if x_lns.device == "cpu":
        # x_f = cp.asnumpy(d_x_f)
        pass  # Todo: Implement CPU version
    else:
        d_x_f = cp.empty(x_shape, dtype=np.float32)

        NUM_BLOCKS = int((x_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
        grid_config = (NUM_BLOCKS, 1, 1)

        kernels.cuda_lns_to_float(
            grid_config,
            block_config_1d,
            (x_lns.log, x_lns.sign, x_lns.zero, d_x_f, np.int32(x_size)),
        )

        x_f = FloatTensor(d_x_f)

    return x_f


# Logical Operators
def lns_logical_op(A, B, op):
    out_size = A.size if A.size > B.size else B.size
    out_shape = A.shape if A.size > B.size else B.shape
    out = cp.empty(
        out_shape, dtype=bool
    )  # need to send bool vec only save comparison result
    grid_config = (int((out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 1, 1)
    assert op in logical_op_to_idx, "Invalid logical operator"
    op_idx = logical_op_to_idx[op]
    kernels.cuda_lns_logical(
        grid_config,
        block_config_1d,
        (
            A.log,
            A.sign,
            A.zero,
            B.log,
            B.sign,
            B.zero,
            out,
            np.int32(op_idx),
            np.int32(A.size),
            np.int32(B.size),
            np.int32(out_size),
        ),
    )
    return out


# Elementwise Unary Operators (sqrt,pow)
def lns_unary_op(A, op, pow=None):
    out_size = A.size
    out_shape = A.shape
    out = empty_lns(out_shape)
    grid_config = (int((out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 1, 1)
    if op == "sqrt":
        kernels.cuda_lns_sqrt(
            grid_config,
            block_config_1d,
            (A.log, A.sign, A.zero, out.log, out.sign, out.zero, np.int32(A.size)),
        )
    elif op == "pow":
        kernels.cuda_lns_pow(
            grid_config,
            block_config_1d,
            (
                A.log,
                A.sign,
                A.zero,
                out.log,
                out.sign,
                out.zero,
                np.int32(pow),
                np.int32(A.size),
            ),
        )
    elif op == "exp_neg":
        assert not cp.any(
            A.sign & ~A.zero
        ), "LNS Exponential function only supports negative numbers"
        kernels.cuda_lns_exp_neg(
            grid_config,
            block_config_1d,
            (A.log, A.sign, A.zero, out.log, out.sign, out.zero, np.int32(A.size)),
        )
    else:
        raise ValueError(f"Operator {op} not supported")
    return out


# Elementwise Binary Operators (maximum,minimum,add,mul) with broadcasting
def lns_elementwise_op(A, B, op):  # Supports broadcasting
    A_log, B_log = cp.broadcast_arrays(A.log, B.log)
    A_log = A_log.copy()
    B_log = B_log.copy()
    A_sign, B_sign = cp.broadcast_arrays(A.sign, B.sign)
    A_sign = A_sign.copy()
    B_sign = B_sign.copy()
    A_zero, B_zero = cp.broadcast_arrays(A.zero, B.zero)
    A_zero = A_zero.copy()
    B_zero = B_zero.copy()

    out_shape = A_log.shape
    out_size = A_log.size
    out = empty_lns(out_shape)

    NUM_BLOCKS = int((out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    grid_config = (NUM_BLOCKS, 1, 1)

    assert op in binary_op_to_idx, f"Operator {op} not supported"
    op_idx = binary_op_to_idx[op]

    kernels.cuda_lns_binary_op(
        grid_config,
        block_config_1d,
        (
            A_log,
            A_sign,
            A_zero,
            B_log,
            B_sign,
            B_zero,
            out.log,
            out.sign,
            out.zero,
            np.int32(out_size),
            np.int32(op_idx),
            np.int32(SUM_APPROX),
        ),
    )

    return out


# Matrix Operations
def lns_mat_mult(A, B):
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    C = empty_lns((M, N))

    BLOCKS_ROWS = int((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    BLOCKS_COLS = int((M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    grid_config = (BLOCKS_ROWS, BLOCKS_COLS, 1)

    if SUM_APPROX == 0:  # LUT
        if SUM_TECHNIQUE == 0:  # block
            kernels.cuda_lns_mat_mul_block_lut(
                grid_config,
                block_config_2d,
                (
                    A.log,
                    A.sign,
                    A.zero,
                    B.log,
                    B.sign,
                    B.zero,
                    C.log,
                    C.sign,
                    C.zero,
                    np.int32(M),
                    np.int32(K),
                    np.int32(N),
                ),
            )
        elif SUM_TECHNIQUE == 1:  # doubling
            kernels.cuda_lns_mat_mul_doubling_lut(
                grid_config,
                block_config_2d,
                (
                    A.log,
                    A.sign,
                    A.zero,
                    B.log,
                    B.sign,
                    B.zero,
                    C.log,
                    C.sign,
                    C.zero,
                    np.int32(M),
                    np.int32(K),
                    np.int32(N),
                ),
            )
    elif SUM_APPROX == 1:  # LIN
        if SUM_TECHNIQUE == 0:  # block
            kernels.cuda_lns_mat_mul_block_lin(
                grid_config,
                block_config_2d,
                (
                    A.log,
                    A.sign,
                    A.zero,
                    B.log,
                    B.sign,
                    B.zero,
                    C.log,
                    C.sign,
                    C.zero,
                    np.int32(M),
                    np.int32(K),
                    np.int32(N),
                ),
            )
        elif SUM_TECHNIQUE == 1:  # doubling
            kernels.cuda_lns_mat_mul_doubling_lin(
                grid_config,
                block_config_2d,
                (
                    A.log,
                    A.sign,
                    A.zero,
                    B.log,
                    B.sign,
                    B.zero,
                    C.log,
                    C.sign,
                    C.zero,
                    np.int32(M),
                    np.int32(K),
                    np.int32(N),
                ),
            )
    else:
        raise ValueError("Technique not supported")

    return C


# Reductions


def iterate_over_axis(arr, axis):
    # Create a list of slice(None) which acts like a ':' in indexing
    slicers = [slice(None)] * arr.ndim

    # Iterate over the specified axis
    for i in range(arr.shape[axis]):
        slicers[axis] = i
        select = arr[tuple(slicers)]
        yield select


def lns_val_reduce(A, op, axis=None, keepdims=False):
    assert op in val_reduce_op_to_idx, f"Operator {op} not supported"

    if axis == None:  # sum all axes
        return lns_val_absolute_reduce(A, op, keepdims)
    elif axis == -1:
        axis = (A.ndim - 1,)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        assert isinstance(
            axis, tuple
        ), "axes must be None, an integer or a tuple of integers."

    # make sure axes are positive and in range
    assert all(0 <= ax < A.ndim for ax in axis), "axis out of range"
    # sort axes
    axis = sorted(axis)

    if axis == A.shape:  # sum all axes
        return lns_val_absolute_reduce(A, op)

    # calculate total number of elements of output array
    out_size = 1
    for i in range(A.ndim):
        if i not in axis:
            out_size *= A.shape[i]
    if keepdims:
        output_shape = tuple(
            dim if i not in axis else 1 for i, dim in enumerate(A.shape)
        )
    else:
        output_shape = tuple(dim for i, dim in enumerate(A.shape) if i not in axis)

    # if output is 2-dim (usually bias in a layer) and has relatively low size,
    # if also numer of elements to sum over is very large (e.g. one axis 32k or two axis each is 1k)
    # so it is faster to do a simple for loop and do an absolute reduction on each row
    num_total_reduce_elements = 1
    for ax in axis:
        num_total_reduce_elements *= A.shape[ax]

    if len(output_shape) <= 2 and num_total_reduce_elements > 1024 and out_size < 256:
        out = empty_lns(output_shape)
        remaining_axis = list(A.shape).index(output_shape[0])
        for i, selection in enumerate(iterate_over_axis(A, remaining_axis)):
            # for i in range(out_size):
            # sum over all axis except the remaining one
            out[i] = lns_val_absolute_reduce(selection, op, keepdims=True)
    else:
        out = lns_val_reduce_single_axis(A, op, axis[0], keepdims)
        for i, ax in enumerate(axis[1:]):
            # adjust axes to account for previous reductions
            ax -= i + 1
            out = lns_val_reduce_single_axis(out, op, ax, keepdims)

    return out


def lns_val_reduce_single_axis(A, op, axis, keepdims):
    shape = A.shape
    num_dims = A.ndim

    # calculate total number of elements of output array
    out_size = 1
    for i in range(num_dims):
        if i != axis:
            out_size *= shape[i]
    if keepdims:
        output_shape = tuple(dim if i != axis else 1 for i, dim in enumerate(shape))
    else:
        output_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)

    out = empty_lns(output_shape)

    NUM_BLOCKS = int((out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    grid_config = (NUM_BLOCKS, 1, 1)

    A_copy = (
        A.copy()
    )  # copy A to avoid modifying it if using recursive doubling summation
    op_idx = val_reduce_op_to_idx[op]

    kernels.cuda_lns_val_reduce_single_axis(
        grid_config,
        block_config_1d,
        (
            A_copy.log,
            A_copy.sign,
            A_copy.zero,
            out.log,
            out.sign,
            out.zero,
            cp.array(shape, dtype=np.int32),
            cp.int32(axis),
            cp.int32(num_dims),
            cp.int32(out_size),
            cp.int32(op_idx),
            cp.int32(SUM_APPROX),
            cp.int32(SUM_TECHNIQUE),
        ),
    )

    return out


def lns_val_absolute_reduce(g_in, op, keepdims):
    N = g_in.size
    out = empty_lns(1) if not keepdims else empty_lns((1,) * g_in.ndim)
    op_idx = val_reduce_op_to_idx[op]

    if SUM_TECHNIQUE == 1 or op == "max" or op == "min":  # Recursive Doubling
        g_in_temp = g_in.copy().flatten()  # copy input array to avoid modifying it
        g_out = empty_lns(N)
        current_size = N
        shared_mem = THREADS_PER_BLOCK * LNS_SIZE_IN_BYTES
        while current_size > 1:
            NUM_BLOCKS = int(
                (current_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
            )  # ceil(current_size/NUM_THREADS_PER_BLOCK)
            grid_config = (NUM_BLOCKS, 1, 1)
            kernels.cuda_lns_val_reduce_recursive_doubling_one_block(
                grid_config,
                block_config_1d,
                (
                    g_in_temp.log,
                    g_in_temp.sign,
                    g_in_temp.zero,
                    g_out.log,
                    g_out.sign,
                    g_out.zero,
                    np.int32(current_size),
                    cp.int32(op_idx),
                    cp.int32(SUM_APPROX),
                ),
                shared_mem=shared_mem,
            )
            g_in_temp[:current_size] = g_out[:current_size]
            current_size = NUM_BLOCKS

        out = g_in_temp[0]
    if SUM_TECHNIQUE == 0:  # Sequential
        kernels.cuda_lns_reduce_sequential(
            (1,),
            (1,),
            (
                g_in.log,
                g_in.sign,
                g_in.zero,
                out.log,
                out.sign,
                out.zero,
                cp.int32(N),
                cp.int32(SUM_APPROX),
            ),
        )
    return out


def lns_arg_reduce(A, op, axis=None, get=False):
    assert op in arg_reduce_op_to_idx, f"Operator {op} not supported"

    if axis == None:  # sum all axes
        return lns_arg_absolute_reduce(A, op, get)
    elif isinstance(axis, int):
        return lns_arg_reduce_single_axis(A, op, axis, get)
    else:
        raise TypeError("axis must be None or an integer.")


def lns_arg_reduce_single_axis(A, op, axis, get):
    shape = A.shape
    num_dims = A.ndim

    # calculate total number of elements of output array
    if axis is None:
        axis = -1  # -1 is a special value that indicates all axes
        out_size = 1
        output_shape = (1,)
    else:
        out_size = 1
        for i in range(num_dims):
            if i != axis:
                out_size *= shape[i]
        output_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)

    output_array = cp.empty(output_shape, dtype=np.int32)  # output array for indices

    NUM_BLOCKS = int((out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    grid_config = (NUM_BLOCKS, 1, 1)

    op_idx = arg_reduce_op_to_idx[op]
    kernels.cuda_lns_arg_reduce_single_axis(
        grid_config,
        block_config_1d,
        (
            A.log,
            A.sign,
            A.zero,
            output_array,
            cp.array(shape, dtype=np.int32),
            cp.int32(axis),
            cp.int32(num_dims),
            cp.int32(out_size),
            cp.int32(op_idx),
        ),
    )

    return (
        output_array.get() if get else output_array
    )  # get() copies the array from gpu to cpu


def lns_arg_absolute_reduce(g_in, op, get):
    # Recursive Doubling
    N = g_in.size
    op_idx = arg_reduce_op_to_idx[op]

    g_in_temp = g_in.copy().flatten()  # copy input array to avoid modifying it
    g_out = empty_lns(N)
    in_argidx = cp.arange(
        N, dtype=np.int32
    )  # array of indices, used to keep track of the index of the max/min value
    out_argidx = cp.arange(
        N, dtype=np.int32
    )  # array of indices, used to keep track of the index of the max/min value
    current_size = N
    shared_mem = THREADS_PER_BLOCK * LNS_SIZE_IN_BYTES
    while current_size > 1:
        NUM_BLOCKS = int(
            (current_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        )  # ceil(current_size/NUM_THREADS_PER_BLOCK)
        grid_config = (NUM_BLOCKS, 1, 1)
        kernels.cuda_lns_arg_reduce_recursive_doubling_one_block(
            grid_config,
            block_config_1d,
            (
                g_in_temp.log,
                g_in_temp.sign,
                g_in_temp.zero,
                g_out.log,
                g_out.sign,
                g_out.zero,
                in_argidx,
                out_argidx,
                cp.int32(current_size),
                cp.int32(op_idx),
            ),
            shared_mem=shared_mem,
        )
        g_in_temp[:current_size] = g_out[:current_size]
        in_argidx[:current_size] = out_argidx[:current_size]
        current_size = NUM_BLOCKS

    return (
        out_argidx[0].get() if get else out_argidx[0]
    )  # get() copies the array from gpu to cpu
