import numpy as np
from matplotlib import pyplot as plt

import babaLinreg as bl
import BasicLinRegFunctions as lr

# Data:

relativeZero = 1e-9

# -- 1:
I1n = np.array([0.01, 0.017, 0.032, 0.047, 0.063, 0.078, 0.095, 0.110, 0.12, 0.148])
U1n = np.array([0.74, 1.2, 2.18, 3.21, 4.25, 5.28, 6.43, 7.4, 8.1, 9.93])

I1p = np.array([0.013, 0.021, 0.034, 0.044, 0.06, 0.074, 0.094, 0.102, 0.114, 0.148])
U1p = np.array([0.87, 1.45, 2.31, 2.96, 4.04, 5.03, 6.34, 6.89, 7.64, 9.92])

I1 = np.concatenate((-I1n[::-1], I1p))
U1 = np.concatenate((-U1n[::-1], U1p))

I1n_errors = (I1n * 0.015) + 0.002
U1n_errors = (U1n * 0.005) + 0.02

I1p_errors = (I1p * 0.015) + 0.002
U1p_errors = (U1p * 0.005) + 0.02

I1_errors = np.concatenate((I1n_errors, I1p_errors))
U1_errors = np.concatenate((U1n_errors, U1p_errors))

# -- 2:
I2n = np.array([0.025, 0.035, 0.055, 0.068, 0.089, 0.099, 0.122, 0.144, 0.158, 0.174, 0.186, 0.195, 0.2])
U2n = np.array([0.13, 0.2, 0.61, 1.05, 1.38, 2.24, 3.36, 4.42, 5.21, 6.15, 6.93, 7.57, 7.88])

I2p = np.array([0.025, 0.05, 0.056, 0.063, 0.074, 0.104, 0.125, 0.138, 0.153, 0.163, 0.186, 0.193, 0.2])
U2p = np.array([0.132, 0.442, 0.637, 0.879, 1.278, 2.46, 3.47, 4.14, 4.97, 5.55, 6.93, 7.4, 7.9])

# -- 3:
I3n = np.array([0.0026, 0.0747, 1.175, 1.602, 4.75, 6.48, 7.74, 8.73, 11.25, 15.42, 16.95, 19.62, 22.1, 25.8, 31.7])
I3n = I3n * 0.001  # account for mA
U3n = np.array([2.29, 2.49, 2.88, 3.03, 3.69, 4.12, 4.43, 4.67, 5.29, 6.30, 6.66, 7.31, 7.72, 8.59, 9.95])

I3p = np.array([0.05, 0.09, 0.13, 0.21, 0.30, 0.45, 0.52, 0.61, 0.70, 0.85, 0.90, 0.99]) * 1e-6
U3p = np.array([0.52, 0.93, 1.35, 2.11, 3.01, 4.5, 5.22, 6.12, 6.98, 8.47, 9.01, 9.93])

I3n_small1 = np.array([0.0026, 0.0747]) * 0.001
I3n_small2 = np.array([1.175, 1.602]) * 0.001
I3n_small3 = np.array([4.75, 6.48, 7.74, 8.73, 11.25, 15.42, 16.95, 19.62]) * 0.001
I3n_big = np.array([22.1, 25.8, 31.7]) * 0.001

# Was sind hier die richtigen + ... Ableseungenauigkeiten?
I3n_small1_errors = (I3n_small1 * 0.008) + (6 * 1e-7)  # * ???
I3n_small2_errors = (I3n_small2 * 0.008) + (2 * 1e-6)
I3n_small3_errors = (I3n_small3 * 0.008) + (2 * 1e-5)
I3n_big_errors = (I3n_big * 0.015) + (2 * 1e-4)

I3n_errors = np.concatenate((I3n_small1_errors, I3n_small2_errors, I3n_small3_errors, I3n_big_errors))
U3n_errors = (U3n * 0.015) + 0.02

I3p_errors = (I3p * 0.02) + (0.06 * 1e-6)
U3p_errors = (U3p * 0.015) + 0.02

I3 = np.concatenate((-I3n[::-1], I3p))
U3 = np.concatenate((-U3n[::-1], U3p))

I3_errors = np.concatenate((I3n_errors, I3p_errors))
U3_errors = np.concatenate((U3n_errors, U3p_errors))

In_array = [I1n, I2n, I3n]
Ip_array = [I1p, I2p, I3p]
Un_array = [U1n, U2n, U3n]
Up_array = [U1p, U2p, U3p]

I1_array = [I1n, I1p]
U1_array = [U1n, U1p]

I1_error_array = [I1n_errors, I1p_errors]
U1_error_array = [U1n_errors, U1p_errors]

I2_array = [I2n, I2p]
U2_array = [U2n, U2p]

I3_array = [I3n, I3p]
U3_array = [U3n, U3p]

I3_error_array = [I3n_errors, I3p_errors]
U3_error_array = [U3n_errors, U3p_errors]

print('Neuer Bestwert für R1n:')
lr.GetRWithErrors(U1n, U1n_errors, I1n, I1n_errors)
print('\nNeuer Bestwert für R1p:')
lr.GetRWithErrors(U1p, U1p_errors, I1p, I1p_errors)
print('\nNeuer Bestwert für R1k:')
lr.GetRWithErrors(U1, U1_errors, I1, I1_errors)

# plt.figure(figsize=(12, 8))
#
#
# for i in range(len(In_array)):
#     plt.subplot(2, 2, i + 1)
#     plt.title('Versuchsteil ' + str(i + 1))
#     plt.xlabel('U')
#     plt.ylabel('I')
#     plt.scatter(Un_array[i], In_array[i], marker='x')
#
# plt.show()

print('\nBerechnung durch die Steigung:')
bl.babaLinreg(U3p, I3p, printOutput=True)

np = ['n', 'p']
br = ['b', 'r']
switchSignArray = [True, False]

plt.figure(figsize=(12, 4.1))

# for i in range(0, len(I3_array)):
#     plt.subplot(1, 2, i + 1)
#     print('Für U/I' + str(3) + np[i] + ':')
#     bl.babaLinreg(U3_array[i], I3_array[i], title='Kennlinie $I$ gegen $U$', xLabel='Spannung $U$ in [V]\n',
#                   yLabel='Stromstärke $I$ in [A]', language='g', printOutput=True, xError=U3_error_array[i],
#                   yError=I3_error_array[i], markerColor=br[i], switchSigns=switchSignArray[i])

# bl.CombinedBabaLinreg(U1_array, I1_array, title='Kennlinie $I$ gegen $U$', xLabel='Spannung $U$ in [V]\n',
#                       yLabel='Stromstärke $I$ in [A]', language='g', printOutput=True, xError_arr=U1_error_array,
#                       yError_arr=I1_error_array)

# plt.savefig('Kennlinien1combined.png', dpi=300)
# plt.show()

# print(mySlope, mySlope.size)
# print(U1n, U1n.size)
#
# plt.scatter(U1n, I1n)
# plt.plot(U1n, mySlope)
# plt.show()
