import numpy as np
np.seterr(all='ignore')
from matplotlib import pyplot as plt
import math


def f(x, y):
    return -2 * y + 4 * x

def exact_f(x):
    return 2*x - 1 + math.exp(-2*x)



x0 = 0
y0 = 0
x_max = 3
number_of_steps = 11
step = (x_max - x0) / (number_of_steps - 1)
x = np.linspace(x0, x_max, number_of_steps)
y = np.zeros([number_of_steps])
exact_y = np.zeros([number_of_steps])

# print(exact_f(x0))

y[0] = y0
for i in range(1,number_of_steps):
    y[i] = step*f(x[i-1],y[i-1]) + y[i-1]

for i in range(0, number_of_steps):
        exact_y[i] = exact_f(x[i])


plt.plot(x,y,color = 'red')
plt.plot(x,exact_y,color = 'pink')
plt.plot(x,np.zeros([number_of_steps]),color = 'black')

plt.xlabel("X")
plt.ylabel("Y")
plt.ylim(bottom= 0, top = 10)
plt.title("Euler's method")
plt.show()


