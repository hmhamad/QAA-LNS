import os
import cupy as cp

# Get the cuda binary files
file_dir = os.path.dirname(os.path.realpath(__file__))

# Load the LNS kernels
lns_lib_path = os.path.join(file_dir, "../c++/lns_lib.cubin")
print("Loading kernels from: " + lns_lib_path)
module = cp.RawModule(path=lns_lib_path)


# LNS->Float and Float->LNS
cuda_float_to_lns = module.get_function("cuda_float_to_lns")
cuda_lns_to_float = module.get_function("cuda_lns_to_float")
# Logical Operators
cuda_lns_logical = module.get_function("cuda_lns_logical")
# Elementwise Unary Operators
cuda_lns_sqrt = module.get_function("cuda_lns_sqrt")
cuda_lns_pow = module.get_function("cuda_lns_pow")
cuda_lns_exp_neg = module.get_function("cuda_lns_exp_neg")
# Elementwise Binary Operators
cuda_lns_binary_op = module.get_function("cuda_lns_binary_op")
# Matrix Operations
cuda_lns_mat_mul_block_lut = module.get_function("cuda_lns_mat_mul_block_lut")
cuda_lns_mat_mul_doubling_lut = module.get_function("cuda_lns_mat_mul_doubling_lut")
cuda_lns_mat_mul_block_lin = module.get_function("cuda_lns_mat_mul_block_lin")
cuda_lns_mat_mul_doubling_lin = module.get_function("cuda_lns_mat_mul_doubling_lin")
# Reduction Operations
cuda_lns_val_reduce_single_axis = module.get_function("cuda_lns_val_reduce_single_axis")
cuda_lns_arg_reduce_single_axis = module.get_function("cuda_lns_arg_reduce_single_axis")
cuda_lns_val_reduce_recursive_doubling_one_block = module.get_function(
    "cuda_lns_val_reduce_recursive_doubling_one_block"
)
cuda_lns_arg_reduce_recursive_doubling_one_block = module.get_function(
    "cuda_lns_arg_reduce_recursive_doubling_one_block"
)
cuda_lns_reduce_sequential = module.get_function("cuda_lns_reduce_sequential")
