import numpy as np

np.seterr(all='ignore')
from matplotlib import pyplot as plt
import math


def f(x, y):
    return -2 * y + 4 * x


def exact_f(x):
    return 2 * x - 1 + math.exp(-2 * x)


def solve_ivp_euler(x0, y0, x_max, number_of_steps):
    # x0 = 0
    # y0 = 0
    # x_max = 3
    # number_of_steps = 11
    step = (x_max - x0) / (number_of_steps - 1)
    x = np.linspace(x0, x_max, number_of_steps)
    y = np.zeros([number_of_steps])
    exact_y = np.zeros([number_of_steps])
    local_error = np.zeros([number_of_steps])

    y[0] = y0
    exact_y[0] = y0
    local_error[0] = 0

    global_error = 0


    for i in range(1, number_of_steps):
        y[i] = step * f(x[i - 1], y[i - 1]) + y[i - 1]
        exact_y[i] = exact_f(x[i])
        local_error[i] = abs(exact_y[i] - y[i])
        if local_error[i] > global_error:
            global_error = local_error[i]

    return x, y, exact_y, local_error, global_error


x, y, exact_y, local_error, global_error = solve_ivp_euler(0, 0, 5, 10)

# global_error = np.zeros([number_of_steps])
# print(exact_f(x0))

plt.plot(x, y, color='red')
plt.plot(x, exact_y, color='grey')
# plt.plot(x, np.zeros([number_of_steps]), color='black')
plt.xlabel("X")
plt.ylabel("approximated value")
plt.ylim(bottom=0, top=10)
plt.title("Euler's method")
plt.show()

plt.plot(x, local_error, color='red')
plt.xlabel("X")
plt.ylabel("Local error")
plt.ylim(bottom=0, top=10)
plt.title("Local error on each step for Euler's method")
plt.show()



