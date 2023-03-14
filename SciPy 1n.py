import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import data as d


# Data:


def modelX(x, m): return m * x


def modelXC(x, m, c): return x * m + c


popt, p_cov = curve_fit(modelXC, -d.U1n[::-1], -d.I1n[::-1], sigma=-d.U1n_errors[::-1], absolute_sigma=True)

p_err = np.sqrt(np.diag(p_cov))

# Print the fitted parameters and their standard errors
print('m =', lr.r(popt[0]), '+/-', lr.r(p_err[0]))
print('c =', lr.r(popt[1]), '+/-', lr.r(p_err[1]))

upperBestFit = modelXC(-d.U1n[::-1], popt[0] + p_err[0], popt[1] - p_err[1])
lowerBestFit = modelXC(-d.U1n[::-1], popt[0] - p_err[0], popt[1] + p_err[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $I$ gegen $U$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Spannung $U$ in [V]')

plt.fill_between(-d.U1n[::-1], upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(-d.U1n[::-1], upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

plt.scatter(-d.U1n[::-1], -d.I1n[::-1], label='Messwerte (negativ)', marker='x', color='b')
# plt.scatter(d.U1p, d.I1p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(-d.U1n[::-1], -d.I1n[::-1], xerr=d.U1n_errors[::-1], yerr=d.I1n_errors[::-1], fmt='none',
             capsize=3, label='Fehler', color='black')

plt.plot(-d.U1n[::-1], modelXC(-d.U1n[::-1], *popt), label='Best fit der Form \n $y=ax + bx^3$', color='black', linestyle='--')
plt.plot()
plt.legend()

plt.savefig('LinearFitnVT1_mit_C.png', dpi=300)
plt.show()

# LinearFitVT1_ohne_C.png --> m = 0.01485 +/- 0.00141
# LinearFitVT1_mit_C.png --> m = 0.01481 +/- 0.00214; c = 0.00022 +/- 0.00831
