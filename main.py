import numpy as np
# np.seterr(all='ignore')
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


def solve_ivp_with_eulers_method(x0, y0, x_max, number_of_steps):
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


def solve_ivp_with_improved_eulers_method(x0, y0, x_max, number_of_steps):
    step = (x_max - x0) / (number_of_steps - 1)
    x = np.linspace(x0, x_max, number_of_steps)
    y = np.zeros([number_of_steps])
    exact_y = np.zeros([number_of_steps])
    local_error = np.zeros([number_of_steps])
    y[0] = y0
    exact_y[0] = y0
    local_error[0] = 0
    global_error = 0
    k = [0, 0]
    for i in range(1, number_of_steps):
        k[0] = f(x[i - 1], y[i - 1])
        k[1] = f(x[i - 1] + step, y[i - 1] + step * k[0])
        y[i] = y[i - 1] + step / 2 * (k[0] + k[1])
        exact_y[i] = exact_f(x[i])
        local_error[i] = abs(exact_y[i] - y[i])
        if local_error[i] > global_error:
            global_error = local_error[i]

    return x, y, exact_y, local_error, global_error


def solve_ivp_with_runge_kutta_method(x0, y0, x_max, number_of_steps):
    step = (x_max - x0) / (number_of_steps - 1)
    x = np.linspace(x0, x_max, number_of_steps)
    y = np.zeros([number_of_steps])
    exact_y = np.zeros([number_of_steps])
    local_error = np.zeros([number_of_steps])
    y[0] = y0
    exact_y[0] = y0
    local_error[0] = 0
    global_error = 0
    k = [0, 0, 0, 0]
    for i in range(1, number_of_steps):
        k[0] = f(x[i - 1], y[i - 1])
        k[1] = f(x[i - 1] + step / 2, y[i - 1] + step / 2 * k[0])
        k[2] = f(x[i - 1] + step / 2, y[i - 1] + step / 2 * k[1])
        k[3] = f(x[i - 1] + step, y[i - 1] + step * k[2])
        y[i] = y[i - 1] + step / 6 * (k[0] + 2*k[1] + 2*k[2] + k[3])
        exact_y[i] = exact_f(x[i])
        local_error[i] = abs(exact_y[i] - y[i])
        if local_error[i] > global_error:
            global_error = local_error[i]

    return x, y, exact_y, local_error, global_error


def global_errors(x0, y0, x_max):
    g1 = np.zeros([25])
    g2 = np.zeros([25])
    g3 = np.zeros([25])
    x = np.zeros([25])
    for i in range(5, 30):
        x[i-5] = i
        _, _, _, _, g1[i-5] = solve_ivp_with_eulers_method(x0, y0, x_max, i)
        _, _, _, _, g2[i-5] = solve_ivp_with_improved_eulers_method(x0, y0, x_max, i)
        _, _, _, _, g3[i-5] = solve_ivp_with_runge_kutta_method(x0, y0, x_max, i)
    return x,g1,g2,g3



x0 = 0
y0 = 0
x_max = 3
number_of_steps = 4

x, y1, exact_y, local_error1, _ = solve_ivp_with_eulers_method(x0, y0, x_max, number_of_steps)
_, y2, _, local_error2, _ = solve_ivp_with_improved_eulers_method(x0, y0, x_max, number_of_steps)
_, y3, _, local_error3, _ = solve_ivp_with_runge_kutta_method(x0, y0, x_max, number_of_steps)

step_axes, g1,g2,g3 = global_errors(x0, y0, x_max)

plt.plot(x, y1, color='red')
plt.plot(x, y2, color='green')
plt.plot(x, y3, color='blue')
plt.plot(x, exact_y, color='black')
# plt.plot(x, np.zeros([number_of_steps]), color='black')
plt.xlabel("X")
plt.ylabel("approximated value")
plt.ylim(bottom=0, top=10)
plt.title("Euler's method")
plt.show()

plt.plot(x, local_error1, color='red')
plt.plot(x, local_error2, color='green')
plt.plot(x, local_error3, color='blue')
plt.xlabel("X")
plt.ylabel("Local error")
plt.ylim(bottom=0, top=3)
plt.title("Local error on each step for Euler's method")
plt.show()

plt.plot(step_axes, g1, color='red')
plt.plot(step_axes, g2, color='green')
plt.plot(step_axes, g3, color='blue')
plt.xlabel("X")
plt.ylabel("Glboal errors")
plt.ylim(bottom=0, top=0.6)
plt.title("Global error on steps [5;30]")
plt.show()


global_errors(x0, y0, x_max)
