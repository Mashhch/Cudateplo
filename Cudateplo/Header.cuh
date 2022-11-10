cudaMemcpy(dev_A, A, mem_size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_B, B, mem_size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_C, C, mem_size, cudaMemcpyHostToDevice);

reversed_method_progonki << <128, 128 >> > (dev_A, dev_B, dev_C, dev_y, N);

cudaMemcpy(y, dev_y, mem_size, cudaMemcpyDeviceToHost);
//printf("lol %f \n", next_2[i]);

for (int i = 0; i < Nx; i++) {
	prev_2[i] = y[i];
	printf("lol %f \n", prev_2[i]);
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <device_functions.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#define N (32*32)


double f(double x, double t) {
	return 2 * x*t + (1 + tanh(x - t) - 2 * pow(tanh(x - t), 2)) / cosh(x - t);
}

double u0(double x, double t) {
	return 1 / cosh(x - t) + x * pow(t, 2);
}

double gamma1(double t) {
	return pow(t, 2) + (1 + tanh(t)) / cosh(t);
}

double gamma2(double t) {
	return pow(t, 2) + 1 / cosh(1 - t);
}

double fi(double x) {
	return 1 / cosh(x);
}

void method_progonki(double* a, double* b, double* c, double* d, double* A, double* B, double* C, double* y, int n) {

	A[0] = -c[0] / b[0];
	B[0] = d[0] / b[0];
	A[n - 1] = 0;

	for (int i = 1; i < n - 1; i++) {
		A[i] = -c[i] / (b[i] + a[i] * A[i - 1]);

	}
	for (int i = 1; i < n; i++) {
		B[i] = (d[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1]);
	}

	y[n - 1] = B[n - 1];
	for (int i = n - 2; i >= 0; i--) {
		y[i] = B[i] + A[i] * y[i + 1];

	}

	//for (int i = 0; i < n; i++) {
	//	printf("yy %f \n", y[i]);	
	//}

	return;
}

__global__ void reversed_method_progonki(double* A, double* B, double* C, double* y, int n) {
	y[n - 1] = B[n - 1];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	while (n - 2 - tid > 0 & tid < n) {
		y[n - 2 - tid] = B[tid] + A[tid] * y[tid + 1];
	}
}



void next_2_ord(double *a, double *b, double *c, double *d, double* prev, double tau, double sigma, double t, double* x, double h, int n) {
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
	for (int i = 1; i < n - 1; i++) {
		a[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		b[i] = -1 - 2 * tau * pow(a_const, 2) * sigma / pow(h, 2);
		c[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		d[i] = -prev[i] - tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * pow(a_const, 2) / pow(h, 2)) * (
			prev[i + 1] - 2 * prev[i] + prev[i - 1]);
	}
	//for (int i = 0; i < n; i++) {
		//printf("lol %f \n", b[i]);
	//}

	return;

}



int main() {
	double *a, *b, *c, *d, *y, *dev_a, *dev_b, *dev_c, *dev_y, *dev_d, *dev_A, *dev_B, *dev_C, *A, *B, *C;

	unsigned int mem_size = sizeof(double)*N;

	cudaMalloc((void**)&dev_a, mem_size);
	cudaMalloc((void**)&dev_b, mem_size);
	cudaMalloc((void**)&dev_c, mem_size);
	cudaMalloc((void**)&dev_d, mem_size);
	cudaMalloc((void**)&dev_A, mem_size);
	cudaMalloc((void**)&dev_B, mem_size);
	cudaMalloc((void**)&dev_C, mem_size);
	cudaMalloc((void**)&dev_y, mem_size);

	a = (double*)malloc(mem_size);
	b = (double*)malloc(mem_size);
	c = (double*)malloc(mem_size);
	d = (double*)malloc(mem_size);
	y = (double*)malloc(mem_size);
	A = (double*)malloc(mem_size);
	B = (double*)malloc(mem_size);
	C = (double*)malloc(mem_size);

	double x_left = 0;
	double x_right = 1;
	double a_const = 1;
	double t0 = 0;
	double T = 1;
	double sigma = 0.5;


	//int Nx = (int)(x_right - x_left) / h + 1;
	int Nx = 128 * 128;
	int Nt = 21;
	double h = (double)1 / (Nx - 1);
	double tau = (double)1 / (Nt - 1);
	int x_size = sizeof(double) * Nx;
	double* x = (double*)malloc(x_size);
	double* t = (double*)malloc(sizeof(double) * Nt);
	double* u0_ = (double*)malloc(sizeof(double) * Nx);
	double* prev_2 = (double*)malloc(sizeof(double) * Nx);
	double* next_2 = (double*)malloc(sizeof(double) * Nx);
	double* errors_ = (double*)malloc(sizeof(double) * Nx);
	for (int i = 0; i < Nx; i++) {
		x[i] = x_left + i * h;
		//printf("xx %f \n", x[i]);
	}
	for (int i = 0; i < Nt; i++) {
		t[i] = t0 + i * tau;
		//printf("tt%f \n", t[i]);
	}
	for (int i = 0; i < Nx; i++) {
		u0_[i] = u0(x[i], t[Nt - 1]);
		//printf("%f \n", x[i]);
		//printf("%f \n", t[Nt - 1]);
		//printf("uu %f \n", u0_[i]);
	}
	for (int i = 0; i < Nx; i++) {
		prev_2[i] = fi(x[i]);
		//printf("prev_2 %lf \n", prev_2[i]);
	}
	for (int i = 1; i < Nt; i++) {
		next_2_ord(a, b, c, d, prev_2, tau, sigma, i, x, h, N);

		method_progonki(a, b, c, d, A, B, C, y, N);

	}

	for (int i = 0; i < Nx; i++) {
		prev_2[i] = y[i];
		printf("lol %lf \n", prev_2[i]);
	}






}



__global__ void next_2_ord_(double *threads_, double *a, double *b, double *c, double *d, double* prev, double tau, double sigma, double t, double* x, double h, int n) {

	double alpha[2] = { 1, 0 };
	double beta[2] = { 1, 1 };
	double a_const = 1;
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


	int i = threadIdx.x + blockIdx.x*blockDim.x;
	threads_[i] = 32;
	//printf("kenl %d \n", i);

	while (i >= 1 & i < n - 1) {
		a[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		//printf("my %f \n", a[i]);
		b[i] = -1 - 2 * tau * pow(a_const, 2) * sigma / pow(h, 2);
		c[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		d[i] = -prev[i] - tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * pow(a_const, 2) / pow(h, 2)) * (
			prev[i + 1] - 2 * prev[i] + prev[i - 1]);
	}

	return;

}