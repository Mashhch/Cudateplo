#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <device_functions.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define N (32*32)


__device__ __host__ double f(double x, double t) {
	return 2 * x*t + (1 + tanh(x - t) - 2 * powf(tanh(x - t), 2)) / cosh(x - t);
}

__host__ __device__ double u0(double x, double t) {
	return 1 / cosh(x - t) + x * powf(t, 2);
}

__host__ __device__ double gamma1(double t) {
	return powf(t, 2) + (1 + tanh(t)) / cosh(t);
}

__host__ __device__ double gamma2(double t) {
	return powf(t, 2) + 1 / cosh(1 - t);
}

__host__ __device__ double fi(double x) {
	return 1 / cosh(x);
}

__host__ __device__ double * alpha_func() {
	double alpha_[2] = { 1, 0 };
	return alpha_;
}

__host__ __device__ double* beta_func() {
	double beta[2] = { 1, 1 };
	return  beta;
}


__host__ double* method_progonki(double* a, double* b, double* c, double* d, int n) {
	double *A, *B, *y;
	A = (double*)malloc(sizeof(double)*n);
	B = (double*)malloc(sizeof(double)*n);
	y = (double*)malloc(sizeof(double)*n);

	A[0] = -c[0] / b[0];
	B[0] = d[0] / b[0];

	for (int i = 1; i < n - 1; i++) {
		A[i] = -c[i] / (b[i] + a[i] * A[i - 1]);
		A[n - 1] = 0;
	}
	for (int i = 1; i < n; i++) {
		B[i] = (d[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1]);
	}
	y[n - 1] = B[n - 1];
	for (int i = n - 2; i >= 0; i--) {
		y[i] = B[i] + A[i] * y[i + 1];
	}
	return y;
}



__host__ void next_2_ord_0n(double* a, double* b, double* c, double* d, double* prev, double tau, double sigma, double t, double* x, double h, int n) {
	double alpha[2] = { 1, 0 };
	double beta[2] = { 1, 1 };
	int a_const = 1;

	a[0] = 0;
	c[n - 1] = 0;

	if (alpha[0] == 0) {
		b[0] = beta[0];
		d[0] = gamma1(t*tau);
	}
	else
	{
		b[0] = 1 - pow(a_const, 2) * tau / (pow(h, 2) * 2) * (-2 + beta[0] * 2 * h / alpha[0]);
		c[0] = -pow(a_const, 2) * tau / pow(h, 2);
		d[0] = prev[0] + pow(a_const, 2) * tau / (2 * pow(h, 2))*(-gamma1(t*tau) * 2 * h / alpha[0] + prev[1] - 2 * prev[0] + prev[1]
			- (gamma1((t - 1)*tau) - beta[0] * prev[0]) * 2 * h / alpha[0]) + tau * f(x[0], (t - 0.5)*tau);
	}
	if (alpha[1] == 0) {
		a[n - 1] = 0;
		b[n - 1] = beta[1];
		d[n - 1] = gamma2(t*tau);
	}
	else {
		d[n - 1] = prev[0] + pow(a_const, 2) * tau / (2 * pow(h, 2))*(gamma2(t*tau) * 2 * h / alpha[1] + prev[n - 2] - 2 * prev[n - 1] + prev[n - 2]
			+ (gamma2((t - 1)*tau) - beta[1] * prev[0]) * 2 * h / alpha[1]) + tau * f(x[n - 1], (t - 0.5)*tau);
		b[n - 1] = 1 - pow(a_const, 2) * tau / (pow(h, 2) * 2)*(-2 - beta[1] * 2 * h / alpha[1]);
		a[n - 1] = -pow(a_const, 2) * tau / pow(h, 2);
	}
	return;
}

__global__ void next_2_ord(double *threads_, double *a, double *b, double *c, double *d, double* prev, double tau, double sigma, double t, double* x, double h, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double a_const = 1;

	if (i < n-1 && i>0) {
		a[i] = tau * powf(a_const, 2) * sigma / powf(h, 2);
		//printf("my %f \n", a[i]);
		b[i] = -1 - 2 * tau * powf(a_const, 2) * sigma / powf(h, 2);
		c[i] = tau * powf(a_const, 2) * sigma / powf(h, 2);
		d[i] = -prev[i]- tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * powf(a_const, 2) / powf(h, 2)) * (prev[i + 1] - 2 * prev[i] + prev[i - 1]);
	}
}

int main()
{
	double *a, *b, *c, *d, *prev_2, *dev_a, *dev_b, *dev_c, *dev_d, *dev_prev_2, *x, *dev_x, *t, dev_t, *threads, *dev_threads;

	unsigned int mem_size = sizeof(double)*N;


	a = (double*)malloc(mem_size);
	b = (double*)malloc(mem_size);
	c = (double*)malloc(mem_size);
	d = (double*)malloc(mem_size);
	threads = (double*)malloc(mem_size);
	prev_2 = (double*)malloc(sizeof(double) * N);

	cudaMalloc((void**)&dev_threads, mem_size);
	cudaMalloc((void**)&dev_a, mem_size);
	cudaMalloc((void**)&dev_b, mem_size);
	cudaMalloc((void**)&dev_c, mem_size);
	cudaMalloc((void**)&dev_d, mem_size);
	cudaMalloc((void**)&dev_prev_2, mem_size);



	double x_left = 0;
	double x_right = 1;
	double a_const = 1;
	double t0 = 0;
	double T = 1;
	double sigma = 0.5;
	int Nx = N;
	int Nt = 21;
	double h = (double)1 / (Nx - 1);
	double tau = (double)1 / (Nt - 1);
	int x_size = sizeof(double) * Nx;
	x = (double*)malloc(x_size);
	//printf("lol %d", sizeof(x)/sizeof(x[0]));
	t = (double*)malloc(sizeof(double) * Nt);
	double* u0_ = (double*)malloc(sizeof(double) * Nx);
	
	cudaMalloc((void**)&dev_t, sizeof(double) * Nt);
	cudaMalloc((void**)&dev_x, x_size);

	double* next_2 = (double*)malloc(sizeof(double) * Nx);
	double* errors_ = (double*)malloc(sizeof(double) * Nx);

	for (int i = 0; i < Nx; i++) {
		a[i] = 0;
		b[i] = 0;
		c[i] = 0;
		d[i] = 0;
		threads[i] = 0;
	}

	

	for (int i = 0; i < Nx; i++) {
		x[i] = x_left + i * h;
	}
	for (int i = 0; i < Nt; i++) {
		t[i] = t0 + i * tau;
	}
	for (int i = 0; i < Nx; i++) {
		u0_[i] = u0(x[i], t[Nt - 1]);
		//printf("%f \n", x[i]);
		//printf("%f \n", t[Nt - 1]);
		//printf("%d \n", u0_[i]);
	}

	for (int i = 0; i < Nx; i++) {
		prev_2[i] = fi(x[i]);
		//printf("lol %f \n", prev_2[i]);
	}

	cudaMemcpy(dev_x, x, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_threads, threads, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d, d, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_prev_2, prev_2, mem_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(prev_2, dev_prev_2, mem_size, cudaMemcpyDeviceToHost);


	for (int i = 1; i < Nt; i++) {

		//next_2_ord_n0 <<<1,1>>> (dev_a, dev_b, dev_c, dev_d, dev_prev_2, tau, sigma, i, x, h, N);
		next_2_ord << <32, 32 >> > (dev_threads, dev_a, dev_b, dev_c, dev_d, dev_prev_2, tau, sigma, i, dev_x, h, N);

		cudaMemcpy(a, dev_a, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(b, dev_b, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(c, dev_c, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(d, dev_d, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(threads, dev_threads, mem_size, cudaMemcpyDeviceToHost);

		for (int k = 0; k < Nx; k++) {
			printf("daaaa %f \n", d[k]);
		}

		next_2_ord_0n(a, b, c, d, prev_2, tau, sigma, i, x, h, N);
		next_2 = method_progonki(a, b, c, d, N);

		for (int k = 0; k < Nx; k++) {
			printf("lol %f \n", next_2[k]);
			prev_2[k] = next_2[k];
		}

		cudaMemcpy(dev_prev_2, prev_2, mem_size, cudaMemcpyHostToDevice);
	}

	double max_err;
	for (int j = 0; j < Nx; j++) {
		errors_[j] = abs(next_2[j] - u0_[j]);
	}
	for (int j = 0; j < Nx; j++) {
		if (j == 0) max_err = errors_[j];
		if (j > 0 & errors_[j] > errors_[j - 1]) max_err = errors_[j];
		printf("%f \n", errors_[j]);
	}
	printf("max err 2 porydok: %f", max_err);

	return 0;
}