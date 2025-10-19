#include "lns_gpu.h"
#include <assert.h>

// nvcc -Isrc/c++2a --cubin -arch sm_75 src/c++/lns_gpu.cu -o src/c++/lns_lib.cubin

#define zero_lns (lns){-MAX_POSS_INT, true, true}
#define one_lns (lns){0, true, false}
#define pos_inf_lns (lns){MAX_POSS_INT, true, false}
#define neg_inf_lns (lns){-MAX_POSS_INT, false, false}

// Device code

// LNS->Float and Float->LNS

__device__ lns cuda_float_to_lns_scalar(float fl_a) {
    /*
    Transform a float scalar into LNS format.
    */
    lns lns_a;
    lns_a.sign = (fl_a >= 0);
    lns_a.zero = (fl_a == 0.0);

    if (!lns_a.zero) {
        // Handling overflow
        if (fabs(fl_a) > MAX_POSS_FLOAT) {
            lns_a.log = MAX_POSS_INT;
        }
        // Handling underflow
        else if (fabs(fl_a) < RESOLUTION) {
            lns_a.log = -MAX_POSS_INT;
        }
        // Normal case
        else {
            lns_a.log = static_cast<int>(round(log2(fabs(fl_a)) * SCALE));
        }
    }
    return lns_a;
}

__device__ float cuda_lns_to_float_scalar(lns lns_a){
    /*
    Transform an LNS number back into float format.
    */
    if (lns_a.zero) {
        return 0.0;
    } else {
        float logValue = static_cast<float>(lns_a.log) / SCALE;
        float absValue = pow(2.0, logValue);
        return lns_a.sign ? absValue : -absValue;
    }
}

extern "C" __global__ void cuda_float_to_lns(float* x_f, int* x_log, bool* x_sign, bool* x_zero, int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){
        lns x_lns = cuda_float_to_lns_scalar(x_f[i]);
        x_log[i]  = x_lns.log; x_sign[i] = x_lns.sign; x_zero[i] = x_lns.zero;
    }
}

extern "C" __global__ void cuda_lns_to_float(int* x_log, bool* x_sign, bool* x_zero, float* x_f, int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){
        lns x_lns =  (lns) {x_log[i], x_sign[i], x_zero[i]};
        x_f[i] = cuda_lns_to_float_scalar(x_lns);
    }
}

// Logical Operators

__device__ bool cuda_lns_eq_scalar(const lns& lns_a, const lns& lns_b){
    /*
    Check if two LNS numbers are equal.
    */
    
    // If both numbers are zero, they are equal regardless of their log value
    if (lns_a.zero && lns_b.zero) {
        return true;
    }

    // If only one of them is zero, they are not equal
    if (lns_a.zero != lns_b.zero) {
        return false;
    }

    // For non-zero numbers, check if both sign and log values are equal
    return (lns_a.sign == lns_b.sign) && (lns_a.log == lns_b.log);
}

__device__ bool cuda_lns_gt_scalar(const lns& lns_a, const lns& lns_b){
    // Handle zero cases
    if (lns_a.zero) {
        return !lns_b.zero && !lns_b.sign; // a is zero, b is non-zero and negative
    }
    if (lns_b.zero) {
        return lns_a.sign; // b is zero, a is non-zero and positive
    }

    // Handle different sign cases
    if (lns_a.sign != lns_b.sign) {
        return lns_a.sign; // a is greater if it is positive and b is negative
    }

    // Both non-zero and same sign
    if (lns_a.sign) {
        // Both are positive, the one with the larger log value is greater
        return lns_a.log > lns_b.log;
    } else {
        // Both are negative, the one with the smaller log value is greater
        return lns_a.log < lns_b.log;
    }
}

__device__ bool cuda_lns_lt_scalar(const lns& lns_a, const lns& lns_b){
    return cuda_lns_gt_scalar(lns_b, lns_a);
}

__device__ bool cuda_lns_ge_scalar(const lns& lns_a, const lns& lns_b){
    return cuda_lns_eq_scalar(lns_a, lns_b) || cuda_lns_gt_scalar(lns_a, lns_b);
}

__device__ bool cuda_lns_le_scalar(const lns& lns_a, const lns& lns_b){
    return cuda_lns_eq_scalar(lns_a, lns_b) || cuda_lns_lt_scalar(lns_a, lns_b);
}

extern "C" __global__ void cuda_lns_logical(int* x_log, bool* x_sign, bool* x_zero, int* y_log, bool* y_sign, bool* y_zero, bool* out, int op_idx, int x_size, int y_size, int out_size) { 
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < out_size){
        lns x = (x_size==1) ? (lns){x_log[0], x_sign[0], x_zero[0]} : (lns){x_log[i], x_sign[i], x_zero[i]};
        lns y = (y_size==1) ? (lns){y_log[0], y_sign[0], y_zero[0]} : (lns){y_log[i], y_sign[i], y_zero[i]};
        if (op_idx == 0) // eq
            out[i] = cuda_lns_eq_scalar(x, y);
        else if (op_idx == 1) // gt
            out[i] = cuda_lns_gt_scalar(x, y);
        else if (op_idx == 2) // lt
            out[i] = cuda_lns_lt_scalar(x, y);
        else if (op_idx == 3) // ge
            out[i] = cuda_lns_ge_scalar(x, y);
        else if (op_idx == 4) // le
            out[i] = cuda_lns_le_scalar(x, y);
        else
            assert(false && "Invalid logical operator index");
    }
}

// Elementwise Binary Operators

__device__ lns cuda_lns_max_scalar(const lns& lns_a, const lns& lns_b) {
    return cuda_lns_gt_scalar(lns_a, lns_b) ? lns_a : lns_b;
}

__device__ lns cuda_lns_min_scalar(const lns& lns_a, const lns& lns_b) {
    return cuda_lns_lt_scalar(lns_a, lns_b) ? lns_a : lns_b;
}

__device__ lns cuda_lns_mul_scalar(const lns& lns_a, const lns& lns_b){
    lns result;

    // Handle zero cases
    if (lns_a.zero || lns_b.zero) {
        return zero_lns;
    }

    // Check for potential overflow and underflow before performing addition
    if (lns_a.log > 0 && lns_b.log > MAX_POSS_INT - lns_a.log) { // Overflow
        result.log = MAX_POSS_INT;
    } else if (lns_a.log < 0 && lns_b.log < -MAX_POSS_INT - lns_a.log) { // Underflow
        result.log = -MAX_POSS_INT;
    } else {
        // Perform addition if no overflow or underflow
        result.log = lns_a.log + lns_b.log;
    }

    result.sign = (lns_a.sign == lns_b.sign); // Determine the sign of the result
    result.zero = false; // The result cannot be zero unless one of the operands is zero

    return result;

}

__device__ lns cuda_lns_add_lut_scalar(const lns& lns_a, const lns& lns_b){
    if (lns_a.zero) return lns_b;
    if (lns_b.zero) return lns_a;

    lns lns_sum;
    
    lns_sum.sign = lns_a.log>lns_b.log ? lns_a.sign : lns_b.sign;

    int d = abs(lns_a.log-lns_b.log);
    int delta;
    int d_linear_idx;

    if (lns_a.sign==lns_b.sign){ // \delta_{+}
        lns_sum.zero = false;
        d_linear_idx = d >> DPLUS_TABLE_SHIFT; // only works when 1/R is a power of 2
        if (d_linear_idx > DPLUS_LUT_SIZE-1) {
            lns_sum.log = lns_a.log > lns_b.log ? lns_a.log : lns_b.log;
            return lns_sum; 
        }
        else {
            delta= d==0 ? SCALE : uniform_delp_val[d_linear_idx];
        }
    } 
    else { // \delta_{-}
        d_linear_idx = d >> DMINUS_TABLE_SHIFT; // only works when 1/R is a power of 2
        if (d_linear_idx > DMINUS_LUT_SIZE-1) {
            lns_sum.log = lns_a.log > lns_b.log ? lns_a.log : lns_b.log;
            return lns_sum; 
        }
        else if (d==0){
            lns_sum.zero = true;
            lns_sum.log = -MAX_POSS_INT-1;
            return lns_sum; 
        }
        else {
            delta =  uniform_delm_val[d_linear_idx];
        }
    }

    lns_sum.log = lns_a.log > lns_b.log ? lns_a.log : lns_b.log;
    
    // Check for potential overflow and underflow before performing addition
    if (lns_sum.log > 0 && delta > MAX_POSS_INT - lns_sum.log) { // Overflow
        lns_sum.log = MAX_POSS_INT;
    } else if (lns_sum.log < 0 && delta < -MAX_POSS_INT - lns_sum.log) { // Underflow
        lns_sum.log = -MAX_POSS_INT;
    } else {
        // Perform addition if no overflow or underflow
        lns_sum.log += delta;
    }

    return lns_sum;  
}

__device__ lns cuda_lns_add_lin_scalar(const lns& lns_a, const lns& lns_b){

    if (lns_a.zero) return lns_b;
    if (lns_b.zero) return lns_a;

    lns lns_sum;
    
    lns_sum.sign = lns_a.log>lns_b.log ? lns_a.sign : lns_b.sign;
    
    int d = abs(lns_a.log-lns_b.log);
    int delta = 0;
    int left = 0, right = LIN_SIZE, mid;

    if (lns_a.sign==lns_b.sign){ // \delta_{+}
        lns_sum.zero = false;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (linear_delp_state[mid] >= d) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (left < LIN_SIZE){
            delta = -(d >> linear_delp_slopes_bitshift[left]);
            delta += linear_delp_yintercepts[left];
        }
    } else { // \delta_{-}
        lns_sum.zero = d==0 ? true : false;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (linear_delm_state[mid] >= d) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (left < LIN_SIZE){
            delta = linear_delm_slopes_bitshift[left] < 0 ? (d >> -linear_delm_slopes_bitshift[left]) : (d << linear_delm_slopes_bitshift[left]);
            delta +=linear_delm_yintercepts[left];
        }

    }

    lns_sum.log = lns_a.log > lns_b.log ? lns_a.log : lns_b.log;
    
    // Check for potential overflow and underflow before performing addition
    if (lns_sum.log > 0 && delta > MAX_POSS_INT - lns_sum.log) { // Overflow
        lns_sum.log = MAX_POSS_INT;
    } else if (lns_sum.log < 0 && delta < -MAX_POSS_INT - lns_sum.log) { // Underflow
        lns_sum.log = -MAX_POSS_INT;
    } else {
        // Perform addition if no overflow or underflow
        lns_sum.log += delta;
    }
    
    return lns_sum;
}

extern "C" __global__ void cuda_lns_binary_op(int* x_log, bool* x_sign, bool* x_zero, int* y_log, bool* y_sign, bool* y_zero, int* out_log, bool* out_sign, bool* out_zero,
                                              int out_size, int op_idx, int sum_approx) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < out_size){
        lns x = (lns) {x_log[i], x_sign[i], x_zero[i]};
        lns y = (lns) {y_log[i], y_sign[i], y_zero[i]};
        lns out;

        if (op_idx==0){
            if (sum_approx == 0) // LUT approx
                out = cuda_lns_add_lut_scalar(x,y);
            else if (sum_approx == 1) // LIN approx
                out = cuda_lns_add_lin_scalar(x,y);
            else
                assert(false && "Invalid sum_approx");
        }
        else if (op_idx == 1) // multiplication
            out = cuda_lns_mul_scalar(x,y);
        else if (op_idx == 2) // maximum
            out = cuda_lns_max_scalar(x,y);
        else if (op_idx == 3) // minimum
            out = cuda_lns_min_scalar(x,y);
        else
            assert(false && "Invalid binary op_idx");

        out_log[i] = out.log; out_sign[i] = out.sign; out_zero[i] = out.zero;
    }
}

// Elementwise Unary Operators

__device__ lns cuda_lns_sqrt_scalar(const lns& lns_a){

    if (lns_a.zero){
        return zero_lns;
    }
    else {
        assert(lns_a.sign && "Attempting to compute square root of a negative number");
        lns lns_sqrt = (lns) {lns_a.log >> 1, true, false};
        return lns_sqrt;
    }
}

extern "C" __global__ void cuda_lns_sqrt(int* x_in_log, bool* x_in_sign, bool* x_in_zero, int* x_out_log, bool* x_out_sign, bool* x_out_zero, const int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
  
        lns x_in = (lns) {x_in_log[i], x_in_sign[i], x_in_zero[i]};
        lns x_out = cuda_lns_sqrt_scalar(x_in);
        x_out_log[i] = x_out.log; x_out_sign[i] = x_out.sign; x_out_zero[i] = x_out.zero;
    }
}

extern "C" __global__ void cuda_lns_pow(int* x_in_log, bool* x_in_sign, bool* x_in_zero, int* x_out_log, bool* x_out_sign, bool* x_out_zero, const int pow, const int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        lns x_in = (lns) {x_in_log[i], x_in_sign[i], x_in_zero[i]};
        lns x_pow = one_lns; // initialize to 1
        for (int j = 0; j < pow; j++) {
            x_pow = cuda_lns_mul_scalar(x_pow, x_in);
        }
        x_out_log[i] = x_pow.log; x_out_sign[i] = x_pow.sign; x_out_zero[i] = x_pow.zero;
    }

}

extern "C" __global__ void cuda_lns_exp_neg(int* x_in_log, bool* x_in_sign, bool* x_in_zero, int* x_out_log, bool* x_out_sign, bool* x_out_zero, const int N){
    /*
    Perform Exponential in LNS using Lookup-table for ***negative numbers only***
    */
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        if (x_in_zero[i] or x_in_log[i] < linear_exp_state[0]) { // exp of zero (or too small) is 1
            x_out_log[i] = one_lns.log; x_out_sign[i] = true; x_out_zero[i] = false;
            return;
        }
        lns x_exp = zero_lns; // initialize to 0
        int left = 0, right = EXP_LUT_SIZE, mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (linear_exp_state[mid] >= x_in_log[i]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (left < EXP_LUT_SIZE){
            // printf("left: %d, x_in_log: %d, slopes bitshift: %d, yintercepts: %d\n", left, x_in_log[i], linear_exp_slopes_bitshift[left], linear_exp_yintercepts[left]);
            x_exp.log = linear_exp_slopes_bitshift[left] < 0 ? -(x_in_log[i] >> -linear_exp_slopes_bitshift[left]) : -(x_in_log[i] << linear_exp_slopes_bitshift[left]);
            if (linear_exp_slopes_bitshift[left] > 0 && (((int) x_in_log[i]) << linear_exp_slopes_bitshift[left]) > MAX_POSS_INT){ // overflow 
                // printf("Overflow\n");
                x_exp.log = -MAX_POSS_INT;
            }
            else {
                x_exp.log += linear_exp_yintercepts[left];
            }
        }
        else {
            // printf("End x_in_log: %d, slopes bitshift: %d, yintercepts: %d\n", x_in_log[i], linear_exp_slopes_bitshift[EXP_LUT_SIZE-1], linear_exp_yintercepts[EXP_LUT_SIZE-1]);
            x_exp.log = linear_exp_slopes_bitshift[EXP_LUT_SIZE-1] < 0 ? -(x_in_log[i] >> -linear_exp_slopes_bitshift[EXP_LUT_SIZE-1]) : -(x_in_log[i] << linear_exp_slopes_bitshift[EXP_LUT_SIZE-1]);
            if (linear_exp_slopes_bitshift[EXP_LUT_SIZE-1] > 0 && (((int) x_in_log[i]) << linear_exp_slopes_bitshift[EXP_LUT_SIZE-1]) > MAX_POSS_INT){ // overflow 
                // printf("Overflow\n");
                x_exp.log = -MAX_POSS_INT;
            }
            else {
                x_exp.log += linear_exp_yintercepts[EXP_LUT_SIZE-1];
            }            
        }
        x_out_log[i] = x_exp.log; x_out_sign[i] = true; x_out_zero[i] = false;
    }
}

// Matrix Operations

__device__ void cuda_lns_vec_sum_recursive_doubling_single_thread_lut(lns* arr, int n) {
    
    if (n % 2 == 1) {
        arr[n-2] = cuda_lns_add_lut_scalar(arr[n-2],arr[n-1]);
    }

    for (int i=n/2; i > 0; i>>=1) {
        for (int j=0; j < i; j++) {
            arr[j] = cuda_lns_add_lut_scalar(arr[j], arr[j+i]);
        }
        if ((i % 2 == 1) && i>1) arr[i-2] = cuda_lns_add_lut_scalar(arr[i-2], arr[i-1]);
    }

}

__device__ void cuda_lns_vec_sum_recursive_doubling_single_thread_lin(lns* arr, int n) {
    
    if (n & 1) {
        arr[n-2] = cuda_lns_add_lin_scalar(arr[n-2],arr[n-1]);
    }

    for (int i=n/2; i > 0; i>>=1) {
        for (int j=0; j < i; j++) {
                arr[j] = cuda_lns_add_lin_scalar(arr[j], arr[j+i]);
        }
        if ((i & 1) && i>1) arr[i-2] = cuda_lns_add_lin_scalar(arr[i-2], arr[i-1]);
    }

}

extern "C" __global__ void cuda_lns_mat_mul_block_lut(int* A_log, bool* A_sign, bool* A_zero, int* B_log, bool* B_sign, bool* B_zero, int* C_log, bool* C_sign, bool* C_zero, const int M, const int K, const int N) {
    int tx = threadIdx.x; //row in block
    int ty = threadIdx.y; //col in block
    int row = blockIdx.y*BLOCK_SIZE + ty; //row in grid
    int col = blockIdx.x*BLOCK_SIZE+ tx; //col in grid

    lns local_c = zero_lns;

    __shared__ lns A_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ lns B_share[BLOCK_SIZE][BLOCK_SIZE];

    int NUM_BLOCKS = (BLOCK_SIZE+K-1)/BLOCK_SIZE; // ceil(K/BLOCK_SIZE)

    // block matrix multiplication
    for (int i=0; i < NUM_BLOCKS; i++) {
        if (i*BLOCK_SIZE+tx < K && row < M) {
            A_share[ty][tx] = (lns) {A_log[row*K + i*BLOCK_SIZE + tx], A_sign[row*K + i*BLOCK_SIZE + tx], A_zero[row*K + i*BLOCK_SIZE + tx]};
        } else {
            A_share[ty][tx] = zero_lns;
        }
        if (i*BLOCK_SIZE+ty < K && col < N) {
            B_share[ty][tx] = (lns) {B_log[(i*BLOCK_SIZE+ty)*N + col], B_sign[(i*BLOCK_SIZE+ty)*N + col], B_zero[(i*BLOCK_SIZE+ty)*N + col]};
        } else {
            B_share[ty][tx] = zero_lns;
        }

        __syncthreads();

        for (int j=0; j < BLOCK_SIZE; j++) {
                local_c =  cuda_lns_add_lut_scalar(local_c, cuda_lns_mul_scalar(A_share[ty][j], B_share[j][tx]));
        }

        __syncthreads();
    }

    if (row < M && col < N){
        C_log[row*N+col] = local_c.log;
        C_sign[row*N+col] = local_c.sign;
        C_zero[row*N+col] = local_c.zero;
    }
}

extern "C" __global__ void cuda_lns_mat_mul_block_lin(int* A_log, bool* A_sign, bool* A_zero, int* B_log, bool* B_sign, bool* B_zero, int* C_log, bool* C_sign, bool* C_zero, const int M, const int N, const int K) {
    int tx = threadIdx.x; //row in block
    int ty = threadIdx.y; //col in block
    int row = blockIdx.y*BLOCK_SIZE + ty; //row in grid
    int col = blockIdx.x*BLOCK_SIZE+ tx; //col in grid

    lns local_c = zero_lns;
    
    __shared__ lns A_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ lns B_share[BLOCK_SIZE][BLOCK_SIZE];

    int NUM_BLOCKS = (BLOCK_SIZE+N-1)/BLOCK_SIZE; // ceil(N/BLOCK_SIZE)

    // block matrix multiplication
    for (int i=0; i < NUM_BLOCKS; i++) {
        if (i*BLOCK_SIZE+tx < N && row < M) {
            A_share[ty][tx] = (lns) {A_log[row*N + i*BLOCK_SIZE + tx], A_sign[row*N + i*BLOCK_SIZE + tx], A_zero[row*N + i*BLOCK_SIZE + tx]};
        } else {
            A_share[ty][tx] = zero_lns;
        }
        if (i*BLOCK_SIZE+ty < N && col < K) {
            B_share[ty][tx] = (lns) {B_log[(i*BLOCK_SIZE+ty)*K + col], B_sign[(i*BLOCK_SIZE+ty)*K + col], B_zero[(i*BLOCK_SIZE+ty)*K + col]};
        } else {
            B_share[ty][tx] = zero_lns;
        }

        __syncthreads();

        for (int j=0; j < BLOCK_SIZE; j++) {
            local_c =  cuda_lns_add_lin_scalar(local_c, cuda_lns_mul_scalar(A_share[ty][j], B_share[j][tx]));
        }

        __syncthreads();
    }

    if (row < M && col < K){
        C_log[row*K+col] = local_c.log;
        C_sign[row*K+col] = local_c.sign;
        C_zero[row*K+col] = local_c.zero;
    }
}

extern "C" __global__ void cuda_lns_mat_mul_doubling_lut(int* A_log, bool* A_sign, bool* A_zero, int* B_log, bool* B_sign, bool* B_zero, int* C_log, bool* C_sign, bool* C_zero, const int M, const int N, const int K) {
    int tx = threadIdx.x; //row in block
    int ty = threadIdx.y; //col in block
    int row = blockIdx.y*BLOCK_SIZE + ty; //row in grid
    int col = blockIdx.x*BLOCK_SIZE + tx; //col in grid

    __shared__ lns A_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ lns B_share[BLOCK_SIZE][BLOCK_SIZE];
    int NUM_BLOCKS = (BLOCK_SIZE+N-1)/BLOCK_SIZE; // ceil(N/BLOCK_SIZE)
    int num_blocks_pow_2=0;
    int num_blocks_pow_3=0;

    // allocate global memory for mult_arr and final_arr
    // mult_arr is used to store the sum of each block
    // final_arr is used to store the sum of all blocks
    lns mult_arr [BLOCK_SIZE];
    lns temp_arr1 [BLOCK_SIZE];
    lns temp_arr2 [BLOCK_SIZE];
    lns temp_arr3 [BLOCK_SIZE]; // allows for up to (BLOCK_SIZE)^4 elements


    for (int i=0; i < NUM_BLOCKS; i++) {
        if (i*BLOCK_SIZE+tx < N && row < M) {
            A_share[ty][tx] = (lns) {A_log[row*N + i*BLOCK_SIZE + tx], A_sign[row*N + i*BLOCK_SIZE + tx], A_zero[row*N + i*BLOCK_SIZE + tx]};
        } else {
            A_share[ty][tx] = zero_lns;
        }
        if (i*BLOCK_SIZE+ty < N && col < K) {
            B_share[ty][tx] = (lns) {B_log[(i*BLOCK_SIZE+ty)*K + col], B_sign[(i*BLOCK_SIZE+ty)*K + col], B_zero[(i*BLOCK_SIZE+ty)*K + col]};
        } else {
            B_share[ty][tx] = zero_lns;
        }

        __syncthreads();

        // Element-Wise Multiplication
        for (int j=0; j < BLOCK_SIZE; j++) {
                mult_arr[j] =  cuda_lns_mul_scalar(A_share[ty][j],B_share[j][tx]);
        }

        // do recursive doubling to sum elements in vector mult_arr
        // for the current block of size BLOCK_SIZE
        cuda_lns_vec_sum_recursive_doubling_single_thread_lut(mult_arr, BLOCK_SIZE);

        // save the sum of the current block in temp_arr1
        temp_arr1[i%BLOCK_SIZE] = mult_arr[0];

        if ((i+1) % BLOCK_SIZE == 0){
            // do recursive doubling to sum elements in vector temp_arr1
            // for the current block of size BLOCK_SIZE
            cuda_lns_vec_sum_recursive_doubling_single_thread_lut(temp_arr1, BLOCK_SIZE);
            // save the sum of the current block in temp_arr2
            temp_arr2[num_blocks_pow_2] = temp_arr1[0];
            num_blocks_pow_2++;
        }

        if ((i+1) % (BLOCK_SIZE*BLOCK_SIZE) == 0){
            // do recursive doubling to sum elements in vector temp_arr2
            // for the current block of size BLOCK_SIZE
            cuda_lns_vec_sum_recursive_doubling_single_thread_lut(temp_arr2, BLOCK_SIZE);
            // save the sum of the current block in temp_arr3
            temp_arr3[num_blocks_pow_3] = temp_arr2[0];
            num_blocks_pow_3++; num_blocks_pow_2=0;
        }

        __syncthreads();
    }

    int n_remain_one_blocksize = NUM_BLOCKS % (BLOCK_SIZE);
    int n_remain_blocksize_pow_2 = num_blocks_pow_2 % (BLOCK_SIZE*BLOCK_SIZE);
    cuda_lns_vec_sum_recursive_doubling_single_thread_lut(temp_arr1, n_remain_one_blocksize);
    temp_arr2[num_blocks_pow_2] = n_remain_one_blocksize > 0 ? temp_arr1[0] : zero_lns;
    cuda_lns_vec_sum_recursive_doubling_single_thread_lut(temp_arr2, num_blocks_pow_2+1);
    temp_arr3[num_blocks_pow_3] = (n_remain_one_blocksize + n_remain_blocksize_pow_2) > 0 ? temp_arr2[0] : zero_lns;
    cuda_lns_vec_sum_recursive_doubling_single_thread_lut(temp_arr3, num_blocks_pow_3+1);

    if (row < M && col < K){
        C_log[row*K+col] = temp_arr3[0].log;
        C_sign[row*K+col] = temp_arr3[0].sign;
        C_zero[row*K+col] = temp_arr3[0].zero;
    }
}

extern "C" __global__ void cuda_lns_mat_mul_doubling_lin(int* A_log, bool* A_sign, bool* A_zero, int* B_log, bool* B_sign, bool* B_zero, int* C_log, bool* C_sign, bool* C_zero, const int M, const int N, const int K) {
    int tx = threadIdx.x; //row in block
    int ty = threadIdx.y; //col in block
    int row = blockIdx.y*BLOCK_SIZE + ty; //row in grid
    int col = blockIdx.x*BLOCK_SIZE + tx; //col in grid

    __shared__ lns A_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ lns B_share[BLOCK_SIZE][BLOCK_SIZE];
    int NUM_BLOCKS = (BLOCK_SIZE+N-1)/BLOCK_SIZE; // ceil(N/BLOCK_SIZE)
    int num_blocks_pow_2=0;
    int num_blocks_pow_3=0;

    // allocate global memory for mult_arr and final_arr
    // mult_arr is used to store the sum of each block
    // final_arr is used to store the sum of all blocks
    lns mult_arr [BLOCK_SIZE];
    lns temp_arr1 [BLOCK_SIZE];
    lns temp_arr2 [BLOCK_SIZE];
    lns temp_arr3 [BLOCK_SIZE]; // allows for up to (BLOCK_SIZE)^4 elements


    for (int i=0; i < NUM_BLOCKS; i++) {

        int a_idx = row*N + i*BLOCK_SIZE + tx;
        int b_idx = (i*BLOCK_SIZE+ty)*K + col;

        A_share[ty][tx] = (i*BLOCK_SIZE+tx < N && row < M) ? (lns) {A_log[a_idx], A_sign[a_idx], A_zero[a_idx]} : zero_lns;
        B_share[ty][tx] = (i*BLOCK_SIZE+ty < N && col < K) ? (lns) {B_log[b_idx], B_sign[b_idx], B_zero[b_idx]} : zero_lns;

        __syncthreads();

        // Element-Wise Multiplication
        for (int j=0; j < BLOCK_SIZE; j++) {
                mult_arr[j] =  cuda_lns_mul_scalar(A_share[ty][j],B_share[j][tx]);
        }

        // do recursive doubling to sum elements in vector mult_arr
        // for the current block of size BLOCK_SIZE
        cuda_lns_vec_sum_recursive_doubling_single_thread_lin(mult_arr, BLOCK_SIZE);

        // save the sum of the current block in temp_arr1
        temp_arr1[i%BLOCK_SIZE] = mult_arr[0];

        if ((i+1) % BLOCK_SIZE == 0){
            // do recursive doubling to sum elements in vector temp_arr1
            // for the current block of size BLOCK_SIZE
            cuda_lns_vec_sum_recursive_doubling_single_thread_lin(temp_arr1, BLOCK_SIZE);
            // save the sum of the current block in temp_arr2
            temp_arr2[num_blocks_pow_2] = temp_arr1[0];
            num_blocks_pow_2++;
        }

        if ((i+1) % (BLOCK_SIZE*BLOCK_SIZE) == 0){
            // do recursive doubling to sum elements in vector temp_arr2
            // for the current block of size BLOCK_SIZE
            cuda_lns_vec_sum_recursive_doubling_single_thread_lin(temp_arr2, BLOCK_SIZE);
            // save the sum of the current block in temp_arr3
            temp_arr3[num_blocks_pow_3] = temp_arr2[0];
            num_blocks_pow_3++; num_blocks_pow_2=0;
        }

        __syncthreads();
    }
    int n_remain_one_blocksize = NUM_BLOCKS % (BLOCK_SIZE);
    int n_remain_blocksize_pow_2 = num_blocks_pow_2 % (BLOCK_SIZE*BLOCK_SIZE);
    cuda_lns_vec_sum_recursive_doubling_single_thread_lin(temp_arr1, n_remain_one_blocksize);
    temp_arr2[num_blocks_pow_2] = n_remain_one_blocksize > 0 ? temp_arr1[0] : zero_lns;
    cuda_lns_vec_sum_recursive_doubling_single_thread_lin(temp_arr2, num_blocks_pow_2+1);
    temp_arr3[num_blocks_pow_3] = (n_remain_one_blocksize + n_remain_blocksize_pow_2) > 0 ? temp_arr2[0] : zero_lns;
    cuda_lns_vec_sum_recursive_doubling_single_thread_lin(temp_arr3, num_blocks_pow_3+1);

    if (row < M && col < K){
        C_log[row*K+col] = temp_arr3[0].log;
        C_sign[row*K+col] = temp_arr3[0].sign;
        C_zero[row*K+col] = temp_arr3[0].zero;
    }
}

// Reductions

extern "C" __global__ void sum_axis(const float* g_in, float* g_out, const int* in_shape, int axis, int in_num_dims, int out_total_elements) {
    int output_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (output_idx >= out_total_elements) return;
    
    int input_idx;
    int stride = 1;
    for (int i = in_num_dims - 1; i > axis; --i) {
        stride *= in_shape[i];
    }

    input_idx = (output_idx/stride)*(stride*in_shape[axis]) + (output_idx%stride);

    g_out[output_idx] = 0;
    for (int i = 0; i < in_shape[axis]; ++i) {
        g_out[output_idx] += g_in[input_idx];
        input_idx += stride;
    }
}

extern "C" __global__ void cuda_lns_val_reduce_single_axis(int* in_log, bool* in_sign, bool* in_zero, int* out_log, bool* out_sign, bool* out_zero,
                                                       const int* in_shape, int axis, int in_num_dims, int out_size, int op_idx,
                                                       int sum_approx, int technique) {
    
    int output_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (output_idx >= out_size) return;
    
    int input_idx;
    int input_stride = 1;
    for (int i = in_num_dims - 1; i > axis; --i) {
        input_stride *= in_shape[i];
    }

    input_idx = (output_idx/input_stride)*(input_stride*in_shape[axis]) + (output_idx%input_stride);
    
    lns out;
    // op_idx 0: sum, 1: max, 2: min
    if (op_idx==0) // init to 0 for sum
        out = zero_lns;
    else if (op_idx==1) // init to -inf for max
        out = neg_inf_lns;
    else if (op_idx==2) // init to inf for min
        out = pos_inf_lns;
    else
        assert(false && "op_idx must be 0, 1, 2, 3, or 4");

    if (op_idx == 0) { // sum

        if (technique == 0) { //sequential
            if (sum_approx == 0) { // lut approximation
                for (int i = 0; i < in_shape[axis]; ++i) {
                    lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
                    out = cuda_lns_add_lut_scalar(out, in);
                    input_idx += input_stride;
                }
            }
            else if (sum_approx == 1) { // linear approximation
                for (int i = 0; i < in_shape[axis]; ++i) {
                    lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
                    out = cuda_lns_add_lin_scalar(out, in);
                    input_idx += input_stride;
                }
            }
            else
                assert(false && "sum_approx must be 0 or 1");
        }

        else if (technique == 1) { // recursive doubling using single thread (with input_stride for input_idx)
            
            if (sum_approx == 0) { // lut approximation
            
                int n = in_shape[axis]; // final element [n-1] is input_idx+(n-1)*input_stride
                int idx;
                if (n == 1){
                    out = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
                }
                else if (n % 2 == 1) {
                    idx = input_idx + (n-2)*input_stride;
                    out = cuda_lns_add_lut_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                  (lns){in_log[idx + input_stride], in_sign[idx + input_stride], in_zero[idx + input_stride]}
                                                  );
                    in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                }

                for (int i=n/2; i > 0; i>>=1) {
                    for (int j=0; j < i; j++) {
                        idx = input_idx + j*input_stride;
                        out = cuda_lns_add_lut_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                      (lns){in_log[idx + i*input_stride], in_sign[idx + i*input_stride], in_zero[idx + i*input_stride]}
                                                      );
                        in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                        
                    }
                    if ((i % 2 == 1) && i>1) {
                        idx = input_idx + (i-2)*input_stride;
                        out = cuda_lns_add_lut_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                      (lns){in_log[idx + input_stride], in_sign[idx + input_stride], in_zero[idx + input_stride]}
                                                      );
                        in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                        
                    }
                }
            }
            else if (sum_approx == 1) { // linear approximation
                int n = in_shape[axis]; // final element [n-1] is input_idx+(n-1)*input_stride
                int idx;
                if (n == 1){
                    out = (lns){in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
                }
                else if (n % 2 == 1) {
                    idx = input_idx + (n-2)*input_stride;
                    out = cuda_lns_add_lin_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                  (lns){in_log[idx + input_stride], in_sign[idx + input_stride], in_zero[idx + input_stride]}
                                                  );
                    in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                }

                for (int i=n/2; i > 0; i>>=1) {
                    for (int j=0; j < i; j++) {
                        idx = input_idx + j*input_stride;
                        out = cuda_lns_add_lin_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                      (lns){in_log[idx + i*input_stride], in_sign[idx + i*input_stride], in_zero[idx + i*input_stride]}
                                                      );
                        in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                        
                    }
                    if ((i % 2 == 1) && i>1) {
                        idx = input_idx + (i-2)*input_stride;
                        out = cuda_lns_add_lin_scalar((lns){in_log[idx], in_sign[idx], in_zero[idx]},
                                                      (lns){in_log[idx + input_stride], in_sign[idx + input_stride], in_zero[idx + input_stride]}
                                                      );
                        in_log[idx] = out.log; in_sign[idx] = out.sign; in_zero[idx] = out.zero;
                        
                    }
                }
            }
            else
                assert(false && "sum_approx must be 0 or 1");
        }
    }
    else if (op_idx == 1) { // max
        for (int i = 0; i < in_shape[axis]; ++i) {
            lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
            out = cuda_lns_max_scalar(out, in);
            input_idx += input_stride;
        }
    }
    else if (op_idx == 2) { // min
        for (int i = 0; i < in_shape[axis]; ++i) {
            lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
            out = cuda_lns_min_scalar(out, in);
            input_idx += input_stride;
        }
    }
    else
        assert(false && "op_idx must be 0, 1, 2");


    out_log[output_idx] = out.log; 
    out_sign[output_idx] = out.sign;
    out_zero[output_idx] = out.zero;
}

extern "C" __global__ void cuda_lns_arg_reduce_single_axis(int* in_log, bool* in_sign, bool* in_zero, int* arg_idx, const int* in_shape, int axis, int in_num_dims, int out_size, int op_idx) {
    unsigned int output_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (output_idx >= out_size) return;

    lns out;
    // op_idx 0: argmax, 1: argmin
    if (op_idx==0) // init to -inf for argmax
        out = neg_inf_lns;
    else if (op_idx==1) // init to inf for argmin
        out = pos_inf_lns;
    else
        assert(false && "op_idx must be 0 or 1");

    int input_idx = 0;
    int stride = 1;
    if (axis!=-1) { // if axis==-1, then we are reducing to a scalar
        for (int i = in_num_dims - 1; i > axis; --i) {
            stride *= in_shape[i];
        }
        input_idx = (output_idx/stride)*(stride*in_shape[axis]) + (output_idx%stride);
    }

    if (op_idx == 0) { // argmax
        int max_idx = -1;
        for (int i = 0; i < in_shape[axis]; ++i) {
            lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
            if (cuda_lns_gt_scalar(in, out)) {
                max_idx = i;
                out = in;
            }
            input_idx += stride;
        }
        arg_idx[output_idx] = max_idx;
    }
    else if (op_idx == 1) { // argmin
        int min_idx = -1;
        for (int i = 0; i < in_shape[axis]; ++i) {
            lns in = (lns) {in_log[input_idx], in_sign[input_idx], in_zero[input_idx]};
            if (cuda_lns_lt_scalar(in, out)) {
                min_idx = i;
                out = in;
            }
            input_idx += stride;
        }
        arg_idx[output_idx] = min_idx;
    }

}


extern "C" __global__ void cuda_lns_val_reduce_recursive_doubling_one_block(int* g_idata_log, bool* g_idata_sign, bool* g_idata_zero, int* g_odata_log, bool* g_odata_sign, bool* g_odata_zero, int N,
                                                       int op_idx, int sum_approx) {
    extern __shared__ lns sdata[]; // size allocated on invocation in the 3rd argument of the kernel
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){
        sdata[tid].log = g_idata_log[i];
        sdata[tid].sign = g_idata_sign[i];
        sdata[tid].zero = g_idata_zero[i];
    }
    else {
        if (op_idx==0)
            sdata[tid] = zero_lns; // set to zero for sum
        else if (op_idx==1)
            sdata[tid] = neg_inf_lns; // set to -inf for max
        else if (op_idx==2)
            sdata[tid] = pos_inf_lns; // set to inf for min
        else
            assert(false && "op_idx must be 0, 1, or 2");
    }

    __syncthreads();
    // do reduction in shared mem
    if (op_idx == 0){ //sum
        if (sum_approx==0) { // lut approx
            for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if (tid % (2*s) == 0) {
                    sdata[tid] = cuda_lns_add_lut_scalar(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
        }
        else if (sum_approx==1) { // lin approx
            for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if (tid % (2*s) == 0) {
                    sdata[tid] = cuda_lns_add_lin_scalar(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
        }
        else 
            assert (false && "sum_idx must be 0 or 1");
    }

    else if (op_idx == 1){ //max
        for(unsigned int s=1; s < blockDim.x; s *= 2) {
            if (tid % (2*s) == 0) {
                sdata[tid] = cuda_lns_max_scalar(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
    }
    else if (op_idx == 2){ //min
        for(unsigned int s=1; s < blockDim.x; s *= 2) {
            if (tid % (2*s) == 0) {
                sdata[tid] = cuda_lns_min_scalar(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
    }

    // write result for this block to global mem
    if (tid == 0){
        g_odata_log[blockIdx.x] = sdata[0].log;
        g_odata_sign[blockIdx.x] = sdata[0].sign;
        g_odata_zero[blockIdx.x] = sdata[0].zero;
    }
}

extern "C" __global__ void cuda_lns_arg_reduce_recursive_doubling_one_block(int* g_idata_log, bool* g_idata_sign, bool* g_idata_zero,
                                                                             int* g_odata_log, bool* g_odata_sign, bool* g_odata_zero,
                                                                             int* in_arg_idx, int* out_arg_idx, int N, int op_idx) {
    extern __shared__ lns sdata[]; // size allocated on invocation in the 3rd argument of the kernel
    __shared__ int sdata_argidx[BLOCK_SIZE*sizeof(int)];
    
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N){
        sdata[tid].log = g_idata_log[i];
        sdata[tid].sign = g_idata_sign[i];
        sdata[tid].zero = g_idata_zero[i];
        sdata_argidx[tid] = in_arg_idx[i];
    }
    else {
        sdata_argidx[tid] = in_arg_idx[0]; // will be overwritten 
        if (op_idx==0)
            sdata[tid] = neg_inf_lns; // set to -inf for argmax
        else if (op_idx==1)
            sdata[tid] = pos_inf_lns; // set to inf for argmin
        else
            assert(false && "op_idx must be 0 or 1");
    }


    __syncthreads();
    // do reduction in shared mem
    if (op_idx == 0){ //argmax
        for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if (tid % (2*s) == 0) {
                    if (cuda_lns_lt_scalar(sdata[tid], sdata[tid + s])) {
                        sdata[tid] = sdata[tid + s];
                        sdata_argidx[tid] = sdata_argidx[tid + s];
                    }
                }
                __syncthreads();
            }
        }

    else if (op_idx == 1){ //argmin
        for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if (tid % (2*s) == 0) {
                    if (cuda_lns_gt_scalar(sdata[tid], sdata[tid + s])) {
                        sdata[tid] = sdata[tid + s];
                        sdata_argidx[tid] = sdata_argidx[tid + s];
                    }
                }
                __syncthreads();
            }
    }
    // write result for this block to global mem
    if (tid == 0){
        g_odata_log[blockIdx.x] = sdata[0].log;
        g_odata_sign[blockIdx.x] = sdata[0].sign;
        g_odata_zero[blockIdx.x] = sdata[0].zero;
        out_arg_idx[blockIdx.x] = sdata_argidx[0];
    }
}


extern "C" __global__ void cuda_lns_reduce_sequential(int* g_idata_log, bool* g_idata_sign, bool* g_idata_zero, int* sum_log, bool* sum_sign, bool* sum_zero, int N,
                                        int sum_approx) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid == 0 ){
        lns sum = zero_lns; // init to zero
        if (sum_approx == 0) { // lut
            for (int i=0; i < N; i++){
                sum = cuda_lns_add_lut_scalar(sum, (lns) {g_idata_log[i],g_idata_sign[i],g_idata_zero[i]});
            }
        }
        else if (sum_approx == 1) { // lin
            for (int i=0; i < N; i++){
                sum = cuda_lns_add_lin_scalar(sum,(lns) {g_idata_log[i],g_idata_sign[i],g_idata_zero[i]});
            }
        }

        sum_log[0] = sum.log;
        sum_sign[0] = sum.sign;
        sum_zero[0] = sum.zero;
    }
    
}