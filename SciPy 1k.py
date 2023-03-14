import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import data as d


# Data:


def modelX(x, m): return m * x


def modelXC(x, m, c): return x * m + c


k = (abs(min(d.I1)) + abs(max(d.I1))) / (abs(min(d.U1)) + abs(max(d.U1)))
print(1/k)

combinedSigma = np.sqrt(d.I1_errors ** 2 + (k * d.U1_errors) ** 2)
combinedSigma2 = np.sqrt((d.I1_errors ** 2 + d.U1_errors ** 2))

popt, p_cov = curve_fit(modelX, d.U1, d.I1, sigma=combinedSigma, absolute_sigma=True)

p_err = np.sqrt(np.diag(p_cov))

# Print the fitted parameters and their standard errors
print('m =', lr.r(popt[0]), '+/-', lr.r(p_err[0]))
print('This means R =', lr.r(1 / popt[0]))
print('With an error of +', lr.r(1 / popt[0] - 1 / (popt[0] + p_err[0])), 'and -',
      lr.r((1 / (popt[0] - p_err[0])) - 1 / popt[0]))
# print('c =', lr.r(popt[1]), '+/-', lr.r(p_err[1]))

upperBestFit = modelX(d.U1, popt[0] + p_err[0])  # , popt[1] - p_err[1])
lowerBestFit = modelX(d.U1, popt[0] - p_err[0])  # , popt[1] + p_err[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $I$ gegen $U$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Spannung $U$ in [V]')

plt.fill_between(d.U1, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(d.U1, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

plt.scatter(-d.U1n[::-1], -d.I1n[::-1], label='Messwerte (negativ)', marker='x', color='b')
plt.scatter(d.U1p, d.I1p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(d.U1, d.I1, xerr=d.U1_errors, yerr=d.I1_errors, fmt='none',
             capsize=6, label='Fehler', color='black', alpha=0.7)

plt.plot(d.U1, modelX(d.U1, *popt), label='Best fit der Form \n $y=mx$', color='black', linestyle='--')
plt.grid()
plt.legend()

plt.savefig('LinearFitVT1_ohne_C.png', dpi=300)
plt.show()
