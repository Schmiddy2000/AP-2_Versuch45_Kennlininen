import numpy as np
from matplotlib import pyplot as plt

x_lin = np.linspace(-25, 25, 250)
a = 0.01262
b = 0.06771


def f_x3(x): return 15.778 + 1247.672 * (x ** 2)


def f_arctan(x): return (x ** 2 + 1) / (a * (x ** 2 + 1) + b)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_lin, f_x3(x_lin), label=r'$y = a + bx^2$')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_lin, f_arctan(x_lin), label=r'$y = \frac{x^2 + 1}{a (x^2 + 1) + b}$')
plt.legend()

plt.show()
