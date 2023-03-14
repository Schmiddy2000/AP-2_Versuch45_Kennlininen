import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import data as d


# Data:


def modelX(x, m, c): return m * (x + c) ** 2


def modelXC(x, m, c): return x * m + c


k = (abs(min(d.I3n)) + abs(max(d.I3n))) / (abs(min(d.U3n)) + abs(max(d.U3n)))
print(1/k)

I3_neu = -d.U3n[::-1] / ((-d.U3n[::-1]/-d.I3n[::-1]) - 220)


fake_err = np.zeros(len(d.I3n))

combinedSigma = np.sqrt(d.I3n_errors ** 2 + (k * d.U3n_errors) ** 2)
combinedSigma2 = np.sqrt((d.I3n_errors ** 2 + d.U3n_errors ** 2))

popt, p_cov = curve_fit(modelX, -d.U3n[::-1], I3_neu, sigma=combinedSigma, absolute_sigma=True)

p_err = np.sqrt(np.diag(p_cov))

# Print the fitted parameters and their standard errors
print('a =', lr.r(popt[0]), '+/-', lr.r(p_err[0]))
print('This means R =', lr.r(1 / popt[0]))
print('With an error of +', lr.r(1 / popt[0] - 1 / (popt[0] + p_err[0])), 'and -',
      lr.r((1 / (popt[0] - p_err[0])) - 1 / popt[0]))
print('c =', lr.r(popt[1]), '+/-', lr.r(p_err[1]))

upperBestFit = modelX(-d.U3n[::-1], popt[0] + p_err[0], popt[1] - p_err[1])
lowerBestFit = modelX(-d.U3n[::-1], popt[0] - p_err[0], popt[1] + p_err[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie '+ '$I_{LED}$' + ' gegen $U$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Spannung $U$ in [V]')

plt.fill_between(-d.U3n[::-1], upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(-d.U3n[::-1], upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

plt.scatter(-d.U3n[::-1], I3_neu, label='Messwerte (negativ)', marker='x', color='b')
# plt.scatter(d.U1p, d.I3p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(-d.U3n[::-1], I3_neu, xerr=d.U3n_errors[::-1], yerr=d.I3n_errors[::-1], fmt='none',
             capsize=6, label='Fehler', color='black', alpha=0.7)

plt.plot(-d.U3n[::-1], modelX(-d.U3n[::-1], *popt), label='Best fit der Form \n $y=a (x+c)^2$', color='black', linestyle='--')
plt.grid()
plt.legend()

plt.savefig('LinearFitVT3_I_LED.png', dpi=300)
plt.show()

# Ohne c:
# m = 0.00283 +/- 1e-05
# This means R = 353.55356
# With an error of + 0.99393 and - 0.99955

# mit c:
# m = 0.00433 +/- 3e-05
# This means R = 230.76668
# With an error of + 1.5254 and - 1.54583
# c = 0.01156 +/- 0.00021

# mit c (x^2):
# m = -0.00031 +/- 0.0
# This means R = -3182.13863
# With an error of + 21.51635 and - 21.22927
# c = -0.00205 +/- 0.00013

# mit m * (x-c):
# m = 0.00433 +/- 3e-05
# This means R = 230.76668
# With an error of + 1.5254 and - 1.54583
# c = -2.66874 +/- 0.03245
