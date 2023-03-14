import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Define the model function
def modelX3_func(x_f, a_f, b_f):
    return a_f * x_f + b_f * x_f ** 7


def modelArctan_func(x_f, a_f, b_f):
    return a_f * x_f + b_f * np.arctan(x_f)


# Generate some example data
# x = np.linspace(-5, 5, 100)
# y = 2 * x + 1.5 * x ** 3 + np.random.normal(scale=0.5, size=100)

I2n = np.array([0.025, 0.035, 0.055, 0.068, 0.089, 0.099, 0.122, 0.144, 0.158, 0.174, 0.186, 0.195, 0.2])
U2n = np.array([0.13, 0.2, 0.61, 1.05, 1.38, 2.24, 3.36, 4.42, 5.21, 6.15, 6.93, 7.57, 7.88])

I2n_errors = (I2n * 0.015) + 0.002
U2n_errors = (U2n * 0.005) + 0.02

I2n_reversed = -I2n[::-1]
U2n_reversed = -U2n[::-1]

I2p = np.array([0.025, 0.05, 0.056, 0.063, 0.074, 0.104, 0.125, 0.138, 0.153, 0.163, 0.186, 0.193, 0.2])
U2p = np.array([0.132, 0.442, 0.637, 0.879, 1.278, 2.46, 3.47, 4.14, 4.97, 5.55, 6.93, 7.4, 7.9])

U2p_small = np.array([0.132, 0.442, 0.637, 0.879, 1.278])
U2p_big = np.array([2.46, 3.47, 4.14, 4.97, 5.55, 6.93, 7.4, 7.9])

U2p_small_errors = (U2p_small * 0.005) + 0.002
U2p_big_errors = (U2p_big * 0.005) + 0.02

I2p_errors = (I2p * 0.015) + 0.002
U2p_errors = np.concatenate((U2p_small_errors, U2p_big_errors))


y = np.concatenate((-U2n[::-1], U2p))
x = np.concatenate((-I2n[::-1], I2p))

y_errors = np.concatenate((U2n_errors[::-1], U2p_errors))
x_errors = np.concatenate((I2n_errors[::-1], I2p_errors))

# Fit the data to the model function
# p0 = [1, 5]     # Initial guesses for a and b
popt, pcov = curve_fit(modelX3_func, x, y)
# popt, pcov = curve_fit(modelArctan_func, x, y)

# Extract the fit parameters
a = popt[0]
b = popt[1]

print('a:', a, '\nb:', b)

# Generate the best-fit curve
y_fit = modelX3_func(x, a, b)
# y_fit = modelArctan_func(x, a, b)

plt.figure(figsize=(12, 5))
# Plot the data and the best-fit curve
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $U$ gegen $I$')
plt.xlabel('Stromstärke $I$ in [A]')
plt.ylabel('Spannung $U$ in [V]')

plt.scatter(I2n_reversed, U2n_reversed, label='Messwerte (negativ)', marker='x', color='b')
plt.scatter(I2p, U2p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(x, y, yerr=y_errors, xerr=x_errors, fmt='none',
             capsize=3, label='Fehler', color='black')

plt.plot(x, y_fit, label='Best fit der Form \n $y=ax + bx^3$', color='black', linestyle='--')
plt.legend()

# plt.savefig('KubischerBestFit.png', dpi=300)
plt.show()
