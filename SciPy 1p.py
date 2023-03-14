import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import data as d


# Data:


def modelX(x, m): return m * x


def modelXC(x, m, c): return x * m + c


combinedSigma = np.sqrt(d.I1p_errors**2 + d.U1p_errors**2)

print(combinedSigma)

popt, p_cov = curve_fit(modelX, d.U1p, d.I1p, sigma=combinedSigma, absolute_sigma=True)

p_err = np.sqrt(np.diag(p_cov))

# Print the fitted parameters and their standard errors
print('m =', lr.r(popt[0]), '+/-', lr.r(p_err[0]))
print('This means R =', lr.r(1/popt[0]))
print('With an error of +', lr.r(1/popt[0] - 1/(popt[0] + p_err[0])), 'and -',
      lr.r((1/(popt[0] - p_err[0])) - 1/popt[0]))
# print('c =', lr.r(popt[1]), '+/-', lr.r(p_err[1]))

upperBestFit = modelX(d.U1p, popt[0] + p_err[0]) # , popt[1] - p_err[1])
lowerBestFit = modelX(d.U1p, popt[0] - p_err[0]) # , popt[1] + p_err[1])

# Plot the data and the best-fit curve

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.title('Kennlinie $I$ gegen $U$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Spannung $U$ in [V]')

plt.fill_between(d.U1p, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(d.U1p, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                 interpolate=True, color='pink', alpha=0.5)

# plt.scatter(d.U1p, d.I1p, label='Messwerte (negativ)', marker='x', color='b')
plt.scatter(d.U1p, d.I1p, label='Messwerte (positiv)', marker='x', color='r')

plt.errorbar(d.U1p, d.I1p, xerr=d.U1p_errors, yerr=d.I1p_errors, fmt='none',
             capsize=3, label='Fehler', color='black')

plt.plot(d.U1p, modelX(d.U1p, *popt), label='Best fit der Form \n $y=ax + bx^3$', color='black', linestyle='--')
plt.plot()
plt.legend()

plt.savefig('LinearFitpVT1_ohne_C.png', dpi=300)
plt.show()

# LinearFitpVT1_mit_C.png --> m = 0.01489 +/- 0.00503; c = -0.00027 +/- 0.01948
# LinearFitpVT1_ohne_C.png --> m = 0.01483 +/- 0.00305

# I1p_error:
# m = 0.01484 +/- 0.0002
# This means R = 67.36837536483017
# With an error of + 0.9096670518981256 and - 0.9349151449103204

# U1p_error:
# m = 0.01483 +/- 0.00305
# This means R = 67.4103
# With an error of + 11.48218 and - 17.41479

# Combined 1p_error:
# m = 0.01483 +/- 0.00305
# This means R = 67.41009
# With an error of + 11.50398 and - 17.46502
