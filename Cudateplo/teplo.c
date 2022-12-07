#include <iostream>
#include <math.h>
#include <stdio.h>


float f(float x, float t) {
	return 2 * x*t + (1 + tanh(x - t) - 2 * pow(tanh(x - t), 2)) / cosh(x - t);
}

float u0(float x, float t) {
	return 1 /cosh(x - t) + x * pow(t,2);
}

float gamma1(float t) {
	return pow(t, 2) + (1 + tanh(t)) / cosh(t);
}

float gamma2(float t) {
	return pow(t,2) + 1 / cosh(1 - t);
}

float fi(float x) {
	return 1 / cosh(x);
}

float * alpha_func() {
	float alpha_[2] = { 1, 0 };
	return alpha_;
}

float* beta_func() {
	float beta[2] = { 1, 1};
	return  beta;
}


float* method_progonki(float* a, float* b, float* c, float* f, int n){
	float *A, *B, *y;
	A = (float*)malloc(sizeof(float)*n);
	B = (float*)malloc(sizeof(float)*n);
	y = (float*)malloc(sizeof(float)*n);

	A[0] = -c[0] / b[0];
	B[0] = f[0] / b[0];

	for (int i = 1; i < n - 1; i++) {
		A[i] = -c[i] / (b[i] + a[i] * A[i - 1]);
		A[n-1] = 0;
	}
	for (int i = 1; i < n; i++) {
		B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1]);
	}
	y[n-1] = B[n-1];
	for (int i = n - 2; i >= 0; i--) {
		y[i] = B[i] + A[i] * y[i + 1];
	}
	return y;
}


float* next_2_ord(float* prev, float tau, float sigma, float t, float* x, float h, int n) {
	float alpha[2] = { 1, 0 };
	float beta[2] = { 1, 1 };
	int a_const = 1;

	float *a, *b, *c, *d;
	a = (float*)malloc(sizeof(float) * n);
	b = (float*)malloc(sizeof(float) * n);
	c = (float*)malloc(sizeof(float) * n);
	d = (float*)malloc(sizeof(float) * n);
	a[0] = 0;
	c[n - 1] = 0;

	if (alpha[0] == 0) {
		b[0] = beta[0];
		d[0] = gamma1(t*tau);
	}
	else
	{
		b[0] = 1 -pow(a_const, 2) * tau / (pow(h,2) * 2) * (-2 + beta[0] * 2 * h / alpha[0]);
		c[0] = -pow(a_const,2) * tau / pow(h,2);
		d[0] = prev[0] + pow(a_const, 2) * tau / (2 * pow(h, 2))*(-gamma1(t*tau) * 2 * h / alpha[0] + prev[1] - 2 * prev[0] + prev[1]
			- (gamma1((t - 1)*tau) - beta[0] * prev[0]) * 2 * h / alpha[0]) + tau * f(x[0], (t - 0.5)*tau);
	}
	if (alpha[1] == 0) {
		a[n - 1] = 0;
		b[n-1] = beta[1];
		d[n-1] = gamma2(t*tau);
	}
	else {
		d[n-1] = prev[0] + pow(a_const, 2) * tau / (2 * pow(h, 2))*(gamma2(t*tau) * 2 * h / alpha[1] + prev[n-2] - 2 * prev[n-1] + prev[n-2]
			+ (gamma2((t - 1)*tau) - beta[1] * prev[0]) * 2 * h / alpha[1]) + tau * f(x[n-1], (t - 0.5)*tau);
		b[n-1] = 1 - pow(a_const, 2) * tau / (pow(h, 2) * 2)*(-2 - beta[1] * 2 * h / alpha[1]);
		a[n-1] = -pow(a_const, 2) * tau / pow(h, 2);
	}
	for (int i = 1; i < n - 1; i++) {
		a[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		b[i] = -1 - 2 * tau * pow(a_const, 2) * sigma / pow(h, 2);
		c[i] = tau * pow(a_const, 2) * sigma / pow(h, 2);
		d[i] = -prev[i] - tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * pow(a_const, 2) / pow(h, 2)) * (
			prev[i + 1] - 2 * prev[i] + prev[i - 1]);
	}

	for (int i = 0; i < 2; i++) {
		printf("alpha %f \n", alpha[i]);
	}
	for (int i = 0; i < n; i++) {
		printf("aaaa %f \n", b[i]);
	}
	float* ret = method_progonki(a, b, c, d, n);
	return ret;

}


int main()
{
	float x_left = 0;
	float x_right = 1;
	float a_const = 1;
	float t0 = 0;
	float T = 1;
	float sigma = 0.5;
	float tau = 0.05;
	float h = 0.05;
	int Nt = (int)(x_right - x_left) / h+1;
	int Nx = int((T - t0) / tau)+1;
	int x_size = sizeof(float) * Nx;
	float* x = (float*)malloc(x_size);
	//printf("lol %d", sizeof(x)/sizeof(x[0]));
	float* t = (float*)malloc(sizeof(float) * Nt);
	float* u0_ = (float*)malloc(sizeof(float) * Nx);
	float* prev_2 = (float*)malloc(sizeof(float) * Nx);
	float* next_2 = (float*)malloc(sizeof(float) * Nx);
	float* errors_ = (float*)malloc(sizeof(float) * Nx);

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
	for (int i = 1; i < Nt; i++) {
		next_2 = next_2_ord(prev_2, tau, sigma, i, x, h, Nx);
		printf("lol %f \n", next_2[i]);
		for (int i = 0; i < Nx; i++)
			prev_2[i] = next_2[i];
	}
	
	for (int i = 1; i < Nx; i++) {
		prev_2[i] = next_2[i];
	}
	float max_err;
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