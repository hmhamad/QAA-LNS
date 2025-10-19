#ifndef UTILS_H
#define UTILS_H

#include<random>
#include<chrono> /* to seed rng*/

#define BLOCK_SIZE 16

extern unsigned seed;
extern std::default_random_engine generator;

void init_zero_matrix(float* A, const int M, const int N);

void init_const_matrix(float* A, const int M, const int N, const float val);

void init_uniform_matrix(float* A, const int M, const int N, const float low, const float high);

void init_gaussian_matrix(float* A, const int M, const int N, const float mean, const float std_dev);

#endif
