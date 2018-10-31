import numpy as np
np.seterr(all='ignore')
from matplotlib import pyplot as plt
import math

#TODO find point of discontinuity in general case
def check_discontinuity_points(x,xi,xf):
    bool = 0
    n = int((xf - xi) // (math.pi/2))
    n2 = int((xi) // math.pi)
    print(n2)
    first_as = (n2)*math.pi - math.pi/2
    # while first_as <= x0:
    #     first_as+=math.pi
    print(first_as, x0)
    print("xi: ",xi, " xf: ",xf, " n: ",n)
    for i in range(-n,n):
        current_x = math.pi*(1/2 + i)
        if current_x < xi: continue
        if current_x > xf: break
        # if (abs(current_x - x)) <

def f(x, y):
    return y * y * y * y * np.cos(x) - y * np.tan(x)

def exact_f(x):
    return (-3/4*np.sin(x) - 9/8*x*(np.cos(x)**-3) - 9/16*np.sin(2*x)*(np.cos(x)**-3) + np.cos(x)**-3)**3



x0 = -4
y0 = 1
x_max = 8
number_of_steps = 101
step = (x_max - x0) / (number_of_steps - 1)
x = np.linspace(x0, x_max, number_of_steps)
y = np.zeros([number_of_steps])
exact_y = np.zeros([number_of_steps])

#print(exact_f(x0))

y[0] = y0
for i in range(1,number_of_steps):
    y[i] = step*f(x[i-1],y[i-1]) + y[i-1]

as_points = []
for i in range(0, number_of_steps):
    # if (x[i])#
    #if (math.pi/2 - x[i-1] < step or 3*math.pi/2 - x[i-1] < step):
        exact_y[i] = exact_f(x[i])


plt.plot(x[:20],exact_y[:20],color = 'red')
plt.plot(x[20:55], exact_y[20:55], color = 'red')
plt.plot(x[60:], exact_y[60:], color = 'red')

plt.xlabel("X")
plt.ylabel("Y")
plt.ylim(bottom= -10000, top = 10000)
plt.title("Euler's method")
# plt.show()

check_discontinuity_points(x0, x0, x_max)

