import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import BasicLinRegFunctions as lr

# Data:

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

x1 = np.concatenate((-U2n[::-1], U2p))
y1 = np.concatenate((-I2n[::-1], I2p))

x1_errors = np.concatenate((U2n_errors[::-1], U2p_errors))
y1_errors = np.concatenate((I2n_errors[::-1], I2p_errors))


# Define the model function
def model(x, a, b):
    return a * x + b * np.arctan(x)
    # return a * x + b * x ** 3


# sigma1 = np.vstack((y1_errors, x1_errors))

# sigma1 = np.concatenate((y1_errors, x1_errors)).reshape(y1.shape)
sigma2 = np.vstack((y1_errors, x1_errors)).T

print(sigma2)

# Fit the model to the data using curve_fit, including the errors
popt, pcov = curve_fit(model, x1, y1, sigma=y1_errors, absolute_sigma=True)

# Compute the standard errors for the fitted parameters
perr = np.sqrt(np.diag(pcov))

# Print the fitted parameters and their standard errors
print('a =', lr.r(popt[0]), '+/-', lr.r(perr[0]))
print('b =', lr.r(popt[1]), '+/-', lr.r(perr[1]))

upperBestFit = model(x1, popt[0] + perr[0], popt[1] - perr[1])
lowerBestFit = model(x1, popt[0] - perr[0], popt[1] + perr[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $I$ gegen $U$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Spannung $U$ in [V]')

plt.scatter(U2n_reversed, I2n_reversed, label='Messwerte (negativ)', marker='x', color='b')
plt.scatter(U2p, I2p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(x1, y1, yerr=y1_errors, xerr=y1_errors, fmt='none',
             capsize=3, label='Fehler', color='black')

plt.fill_between(x1, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(x1, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

plt.plot(x1, model(x1, *popt), label='Best fit der Form \n $y=ax + bx^3$', color='black', linestyle='--')
plt.plot()
plt.legend()

plt.savefig('ArctanBestFit.png', dpi=300)
plt.show()

# Plot the data and the fitted function
# plt.figure(figsize=(12, 5))
#
# plt.errorbar(x1, y1, xerr=x1_errors, yerr=y1_errors, fmt='none', label='Data')
# plt.plot(x1, model(x1, *popt), 'r-', label='Fit')
# plt.legend()
# plt.show()
