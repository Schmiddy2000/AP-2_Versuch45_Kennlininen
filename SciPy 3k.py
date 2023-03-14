import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import data as d


# Data:


def modelX(x, m, c): return m * x # (-1/(x - c))


def modelXC(x, m, c): return x * m + c


k = (abs(min(d.I3p)) + abs(max(d.I3p))) / (abs(min(d.U3p)) + abs(max(d.U3p)))
print(1/k)

fake_err = np.zeros(len(d.I3p))

combinedSigma = np.sqrt(d.I3p_errors ** 2 + (k * d.U3p_errors) ** 2)
combinedSigma2 = np.sqrt((d.I3p_errors ** 2 + d.U3p_errors ** 2))

print(combinedSigma, combinedSigma2)

popt, p_cov = curve_fit(modelX, d.U3p, d.I3p, sigma=combinedSigma, absolute_sigma=True)

p_err = np.sqrt(np.diag(p_cov))

# Print the fitted parameters and their standard errors
print('m =', (popt[0]), '+/-', (p_err[0]))
print('This means R =', lr.r(1 / popt[0]))
print('With an error of +', lr.r(1 / popt[0] - 1 / (popt[0] + p_err[0])), 'and -',
      lr.r((1 / (popt[0] - p_err[0])) - 1 / popt[0]))
# print('c =', lr.r(popt[1]), '+/-', lr.r(p_err[1]))

upperBestFit = modelX(d.U3p, popt[0] + p_err[0], popt[1] - p_err[1])
lowerBestFit = modelX(d.U3p, popt[0] - p_err[0], popt[1] + p_err[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $I$ gegen $U$')
plt.ylabel('Stromstärke $I$ in [$\mu$A]')
plt.xlabel('Spannung $U$ in [V]')

plt.fill_between(d.U3p, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(d.U3p, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

plt.scatter(d.U3p, d.I3p, label='Messwerte (negativ)', marker='x', color='b')
# plt.scatter(d.U1p, d.U3pp, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(d.U3p, d.I3p, xerr=d.U3p_errors, yerr=d.I3p_errors, fmt='none',
             capsize=6, label='Fehler', color='black', alpha=0.7)

plt.plot(d.U3p, modelX(d.U3p, *popt), label='Best fit der Form \n $y=mx$', color='black', linestyle='--')
plt.grid()
plt.legend()

plt.savefig('LinearFitVT3_positiv.png', dpi=300)
plt.show()
