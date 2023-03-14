import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import BasicLinRegFunctions as lr
import babaLinreg as bl
import data as d

# Data:

I4n = np.array([0.43, 0.92, 1.84, 1.99, 2.32, 1.87, 2.17, 2.48, 2.7, 3.14]) # * 1e-3
U4n = np.array([0.9, 2.47, 3.16, 5.41, 6.1, 6.1, 6.68, 7.54, 9.94, 9.94])

I4p = np.array([0.3, 0.7, 1, 1.4, 1.7, 2.1, 2.5, 3.1, 3.3, 4.3, 5.15]) # * 1e-3
U4p = np.array([0.86, 1.44, 2.11, 2.92, 3.55, 4.3, 5.13, 6.16, 7.01, 8.4, 9.94])

I4 = np.concatenate((-I4n[::-1], I4p))
U4 = np.concatenate((-U4n[::-1], U4p))

I4n_errors = (I4n * 0.015) + (2 * 1e-1)
I4p_errors = (I4p * 0.015) + (2 * 1e-1)

U4n_errors = (U4n * 0.015) + 0.02
U4p_errors = (U4p * 0.015) + 0.02

I4n = -I4n[::-1]
U4n = -U4n[::-1]

I4n_errors = I4n_errors[::-1]
U4n_errors = U4n_errors[::-1]

B1 = np.array([2.5, 2.4, 2.4, 2.6, 3.2, 4.7, 12.5, 33.5]) * 1e-3
B2 = np.array([2.3, 2.3, 2.7, 3.2, 4, 6.7, 16, 48]) * 1e-3
B3 = np.array([2.2, 2.3, 2.9, 3.5, 4.8, 8.4, 18.5, 70]) * 1e-3

L1 = np.array([1, 0.75, 0.5, 0.2, 0.1, 0.05, 0.02, 0.001]) + 0.015
L2 = np.array([1, 0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.001]) + 0.015

L1_errors = L1 * 0.1
L2_errors = L2 * 0.1

B1_errors = (B1 * 0.015) + (2 * 1e-4)
B2_errors = (B2 * 0.015) + (2 * 1e-4)
B3_errors = (B3 * 0.015) + (2 * 1e-4)


print(I4n.size, U4n.size, I4p.size, U4p.size)


def modelX(x, m, c): return m * 1/((x + c)**2) + c


k1 = (abs(min(B1)) + abs(max(B1))) / (abs(min(L1)) + abs(max(L1)))
k2 = (abs(min(B2)) + abs(max(B2))) / (abs(min(L2)) + abs(max(L2)))
k3 = (abs(min(B3)) + abs(max(B3))) / (abs(min(L2)) + abs(max(L2)))
# print(1/k)

fake_err = np.zeros(len(U4n))

combinedSigma = np.sqrt(L1_errors ** 2 + (k1 * B1_errors) ** 2)
combinedSigma2 = np.sqrt(L2_errors ** 2 + (k2 * B2_errors) ** 2)
combinedSigma3 = np.sqrt(L2_errors ** 2 + (k3 * B3_errors) ** 2)

popt1, p_cov1 = curve_fit(modelX, L1, B1, sigma=combinedSigma, absolute_sigma=True)
popt2, p_cov2 = curve_fit(modelX, L2, B2, sigma=combinedSigma2, absolute_sigma=True)
popt3, p_cov3 = curve_fit(modelX, L2, B3, sigma=combinedSigma3, absolute_sigma=True)


p_err1 = np.sqrt(np.diag(p_cov1))
p_err2 = np.sqrt(np.diag(p_cov2))
p_err3 = np.sqrt(np.diag(p_cov3))

# Print the fitted parameters and their standard errors
print('\nm1 =', lr.r(popt1[0]), '+/-', lr.r(p_err1[0]))
print('This means R1 =', lr.r(1 / popt1[0]))
print('With an error of +', lr.r(1 / popt1[0] - 1 / (popt1[0] + p_err1[0])), 'and -',
      lr.r((1 / (popt1[0] - p_err1[0])) - 1 / popt1[0]))
print('c =', lr.r(popt1[1]), '+/-', lr.r(p_err1[1]))

print('\nm1 =', lr.r(popt2[0]), '+/-', lr.r(p_err2[0]))
print('This means R1 =', lr.r(1 / popt2[0]))
print('With an error of +', lr.r(1 / popt2[0] - 1 / (popt2[0] + p_err2[0])), 'and -',
      lr.r((1 / (popt2[0] - p_err2[0])) - 1 / popt2[0]))
print('c =', lr.r(popt2[1]), '+/-', lr.r(p_err2[1]))

print('\nm1 =', lr.r(popt3[0]), '+/-', lr.r(p_err3[0]))
print('This means R1 =', lr.r(1 / popt3[0]))
print('With an error of +', lr.r(1 / popt3[0] - 1 / (popt3[0] + p_err3[0])), 'and -',
      lr.r((1 / (popt3[0] - p_err3[0])) - 1 / popt3[0]))
print('c =', lr.r(popt3[1]), '+/-', lr.r(p_err3[1]))

upperBestFit1 = modelX(L1, popt1[0] + p_err1[0], popt1[1] - p_err1[1])
lowerBestFit1 = modelX(L1, popt1[0] - p_err1[0], popt1[1] + p_err1[1])

upperBestFit2 = modelX(L2, popt2[0] + p_err2[0], popt2[1] - p_err2[1])
lowerBestFit2 = modelX(L2, popt2[0] - p_err2[0], popt2[1] + p_err2[1])

upperBestFit3 = modelX(L2, popt3[0] + p_err3[0], popt3[1] - p_err3[1])
lowerBestFit3 = modelX(L2, popt3[0] - p_err3[0], popt3[1] + p_err3[1])

plt.figure(figsize=(14.5, 6))
plt.subplot(1, 3, 1)
plt.title('Verhalten von $I$ gegen $d$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Entfernung $d$ zur Lichtquelle in [m]')
plt.scatter(L1, B1, marker='x', color='b')
plt.ylim(0, 1.05 * max(modelX(L1, *popt1)))
plt.plot(L1, modelX(L1, *popt1), label='Helligkeitsstufe 1', color='b', linestyle='--')
plt.errorbar(L1, B1, xerr=L1_errors, yerr=B1_errors, fmt='none', capsize=6, label='Fehler', color='black',
             capthick=0.85)
plt.fill_between(L1, upperBestFit1, lowerBestFit1, where=lowerBestFit1 > upperBestFit1,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(L1, upperBestFit1, lowerBestFit1, where=upperBestFit1 >= lowerBestFit1,
                 interpolate=True, color='pink', alpha=0.5)
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.title('Verhalten von $I$ gegen $d$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Entfernung Lichtquelle in [m]')
plt.ylim(0, 1.05 * max(modelX(L2, *popt2)))
plt.scatter(L2, B2, marker='x', color='g')
plt.plot(L2, modelX(L2, *popt2), label='Helligkeitsstufe 2', color='g', linestyle='--')
plt.errorbar(L2, B2, xerr=L2_errors, yerr=B2_errors, fmt='none', capsize=6, label='Fehler', color='black',
             capthick=0.85)
plt.fill_between(L2, upperBestFit2, lowerBestFit2, where=lowerBestFit2 > upperBestFit2,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(L2, upperBestFit2, lowerBestFit2, where=upperBestFit2 >= lowerBestFit2,
                 interpolate=True, color='pink', alpha=0.5)
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.title('Verhalten von $I$ gegen $d$')
plt.ylabel('Stromstärke $I$ in [A]')
plt.xlabel('Entfernung Lichtquelle in [m]')
plt.ylim(0, 1.05 * max(modelX(L2, *popt3)))
plt.plot(L2, modelX(L2, *popt3), label='Helligkeitsstufe 4', color='r', linestyle='--')
plt.errorbar(L2, B3, xerr=L2_errors, yerr=B3_errors, fmt='none', capsize=6, label='Fehler', color='black',
             capthick=0.85)
plt.fill_between(L2, upperBestFit3, lowerBestFit3, where=lowerBestFit3 > upperBestFit3,
                 interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
plt.fill_between(L2, upperBestFit3, lowerBestFit3, where=upperBestFit3 >= lowerBestFit3,
                 interpolate=True, color='pink', alpha=0.5)
plt.scatter(L2, B3, marker='x', color='r')
plt.grid()
plt.legend()

# plt.fill_between(U4n, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
#                  interpolate=True, color='pink', alpha=0.5, label='Konfidenzband')
# plt.fill_between(U4n, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
#                  interpolate=True, color='pink', alpha=0.5)

# # plt.scatter(L1, B1, marker='x', color='r')
# plt.scatter(L2, B2, marker='x', color='g')
# plt.scatter(L2, B3, marker='x', color='b')
#
# # plt.scatter(U4n, I4n, label='Messwerte (negativ)', marker='x', color='b')
#
# # plt.errorbar(U4n, I4n, xerr=U4n_errors, yerr=I4n_errors, fmt='none',
#             #  capsize=6, label='Fehler', color='black', alpha=0.7)
#
# plt.plot(L2, modelX(L2, *popt2), label='Best fit der Form \n $y=mx$', color='g', linestyle='--')
# plt.plot(L2, modelX(L2, *popt3), label='Best fit der Form \n $y=mx$', color='b', linestyle='--')
# plt.grid()
# plt.legend()

# plt.subplot(1, 2, 1)
# bl.babaLinreg(U4n, U4n, title='Negativ', yLabel='Strom', xLabel='Spannung')
#
# plt.subplot(1, 2, 2)
# bl.babaLinreg(U4n, U4n, title='positiv', yLabel='Strom', xLabel='Spannung')

# bl.babaLinreg(U4, I4, title='positiv', yLabel='Strom', xLabel='Spannung')

plt.savefig('VT4_Helleigkeitsabhaengigkeit.png', dpi=300)
plt.show()

# mx negativ:
# m = 0.32967 +/- 0.01335
# This means R = 3.03334
# With an error of + 0.11808 and - 0.12805

# mx positiv:
# m = 0.49764 +/- 0.01444
# This means R = 2.00947
# With an error of + 0.05665 and - 0.06004

print('\n')

lr.GetRWithErrors(U4p, U4p_errors, I4p, I4p_errors)
