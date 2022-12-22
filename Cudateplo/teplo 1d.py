import math
import matplotlib.pyplot as plt
import numpy as np


def f(x, t):
    return 2*x*t+ (1+np.tanh(x-t) - 2*(np.tanh(x-t))**2)/np.cosh(x-t)

def u0(x, t):
    return 1/np.cosh(x-t) +x*t**2

def gamma1(t):
    return t**2 + (1+np.tanh(t))/np.cosh(t)

def gamma2(t):
    return t**2 + 1/np.cosh(1-t)

def fi(x):
    return 1/np.cosh(x)

def coefficients():
    alpha = [1, 0]
    beta = [1, 1]
    return alpha, beta

def method_progonki(a,b,c,f, n):
    A = np.zeros(n)
    B = np.zeros(n)
    y = np.zeros(n)

    A[0] = -c[0] / b[0]
    B[0] = f[0] / b[0]

    for i in range(1, n - 1):
        A[i] = -c[i] / (b[i] + a[i] * A[i - 1])
    A[-1] = 0
    for i in range(1, n):
        B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1])

    y[-1] = B[-1]
    for i in reversed(range(n-1)):
        y[i] = B[i] + A[i] * y[i + 1]
    return y

def next_1_ord(prev, tau, sigma, t, x, h):
    alpha, beta = coefficients()
    a_const = 1

    a = np.zeros(len(x))
    b = np.zeros(len(x))
    c = np.zeros(len(x))
    d = np.zeros(len(x))

    d[0] = gamma1(t*tau)
    d[-1] = gamma2(t*tau)

    b[0] = -(alpha[0]/h) + beta[0]
    c[0] = alpha[0]/h

    b[-1] = (alpha[1] / h) + beta[1]
    a[-1] = -alpha[1]/h

    for i in range(1, len(x)-1):
        a[i] = tau * a_const**2* sigma / h**2
    for i in range(1, len(x)-1):
        b[i] = -1 - 2 * tau * a_const**2 * sigma/ h**2
    for i in range(1, len(x)-1):
        c[i] = tau * a_const**2* sigma / h**2
    for i in range(1, len(x)-1):
        d[i] = - prev[i] -tau * f(x[i], (t - 0.5)*tau) +(sigma-1)*(tau * a_const**2 / h ** 2)*(prev[i+1] - 2*prev[i] + prev[i-1])

    return method_progonki(a,b,c,d,len(x))


def next_2_ord(prev, tau, sigma, t, x, h):
    alpha, beta = coefficients()
    a_const = 1

    a = np.zeros(len(x))
    b = np.zeros(len(x))
    c = np.zeros(len(x))
    d = np.zeros(len(x))

    if alpha[0] == 0:
        b[0] = beta[0]
        d[0] = gamma1(t*tau)
    else:
        b[0] = 1 - a_const ** 2 * tau / (h ** 2 * 2) * (-2 + beta[0] * 2 * h / alpha[0])
        c[0] = - a_const ** 2 * tau / h ** 2
        d[0] = prev[0]+a_const**2*tau/(2*h**2)*(-gamma1(t*tau)*2*h/alpha[0] + prev[1] - 2*prev[0]+prev[1]
                                                - (gamma1((t-1)*tau)-beta[0]*prev[0])*2*h/alpha[0]) + tau*f(x[0], (t - 0.5)*tau)

    if alpha[1] == 0:
        b[-1] = beta[1]
        d[-1] = gamma2(t*tau)
    else:
        d[-1] = prev[0]+a_const**2*tau/(2*h**2)*(gamma2(t*tau)*2*h/alpha[1] + prev[-2] - 2*prev[-1]+prev[-2]
                                                 + (gamma2((t-1)*tau)-beta[1]*prev[0])*2*h/alpha[1]) + tau*f(x[-1], (t - 0.5)*tau)
        b[-1] = 1 - a_const**2*tau/(h**2*2)*(-2 - beta[1]*2*h/alpha[1])
        a[-1] = - a_const**2*tau/h**2

    for i in range(1, len(x) - 1):
        a[i] = tau * a_const ** 2 * sigma / h ** 2
    for i in range(1, len(x) - 1):
        b[i] = -1 - 2 * tau * a_const ** 2 * sigma / h ** 2
    for i in range(1, len(x) - 1):
        c[i] = tau * a_const ** 2 * sigma / h ** 2
    for i in range(1, len(x) - 1):
        d[i] = - prev[i] - tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * a_const ** 2 / h ** 2) * (
                    prev[i + 1] - 2 * prev[i] + prev[i - 1])

    return method_progonki(a, b, c, d, len(x))


x_left = 0
x_right = 1
a_const = 1
t0 = 0
T = 1
sigma = 0.5
tau = 0.05
h = 1/(32*32-1)
x = np.arange(x_left, x_right+h, h)
t = np.arange(t0, T+tau,tau)
u0_ = [ u0(x[k], t[-1]) for k in range(len(x))]

prev_1 = [fi(x[i]) for i in range(len(x))]
next_1 = [0 for i in range(len(x))]
for i in range(1, len(t)):
    next_1 = next_1_ord(prev_1, tau, sigma, i, x, h)
    prev_1 = next_1.copy()

y1 = [abs(next_1[k]-u0_[k]) for k in range (len(next_1))]
er1 = max(y1)
print("Максимальаня ошибка 1 порядка: ", er1)

prev_2 = [fi(x[i]) for i in range(len(x))]
next_2 = [0 for i in range(len(x))]
for i in range(1, len(t)):
    next_2 = next_2_ord(prev_2, tau, sigma, i, x, h)
    prev_2 = next_2.copy()

y2 = [abs(next_2[k]-u0_[k]) for k in range (len(next_1))]
er2 = max(y2)
print("Максимальаня ошибка 2 порядка: ", er2)


plt.figure(1)
plt.subplot(211)
plt.plot(x, u0_, color='yellow', label='Истинное значение')
plt.plot(x, next_1, color='green', label='1 порядка')
plt.title(' график функции')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(x, y1, color='yellow', label='Ошибка 1 порядка')
plt.plot(x, [h]*len(x), color='green', label='h')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.subplot(211)
plt.plot(x, u0_, color='yellow', label='Истинное значение')
plt.plot(x, next_2, color='brown', label='2 порядка')
plt.title(' график функции')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(x, y2, color='yellow', label='Ошибка 2 порядка')
plt.plot(x, [h**2]*len(x), color='green', label='h^2')
plt.legend()
plt.grid(True)


hrange = [0.05, 0.005, 0.025, 0.125, 0.1, 0.002]
h_log = np.zeros(len(hrange))
error_1 = np.zeros(len(hrange))
error_2 = np.zeros(len(hrange))
for j in range(len(hrange)):
    h = hrange[j]
    tau = h
    t_ = np.arange(t0, T+tau, tau)
    x_ = np.arange(x_left, x_right+h, h)
    prev_1_ = [fi(x_[i]) for i in range(len(x_))]
    next_1_ = [0 for i in range(len(x_))]
    prev_2_ = [fi(x_[i]) for i in range(len(x_))]
    next_2_ = [0 for i in range(len(x_))]
    h_log[j] = np.log(h)
    u0_1 = [u0(x_[k], t_[-1]) for k in range(len(x_))]
    for i in range(1, len(t_)):
        next_1_ = next_1_ord(prev_1_, tau, sigma, i, x_, h)
        prev_1_ = next_1_.copy()
        next_2_ = next_2_ord(prev_2_, tau, sigma, i, x_, h)
        prev_2_ = next_2_.copy()
    y_1 = [abs(next_1_[k] - u0_1[k]) for k in range(len(x_))]
    error_1[j] = np.log(max(y_1))
    y_2 = [abs(next_2_[k] - u0_1[k]) for k in range(len(x_))]
    error_2[j] = np.log(max(y_2))

plt.figure(3)
plt.plot(h_log, error_1, color='yellow', label='1 порядка')
plt.plot(h_log, error_2, color='brown', label='2 порядка')
plt.title('Логарифмическая ошибка')
plt.legend()
plt.grid(True)
plt.show()