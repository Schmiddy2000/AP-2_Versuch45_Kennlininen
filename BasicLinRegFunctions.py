import numpy as np


def gerade(x, m, c): return m * x + c
def m_linreg(x, y): return (len(x) * sum(x*y) - sum(x) * sum(y)) / (len(x) * sum(x*x) - sum(x)**2)
def c_linreg(x, y): return (sum(y) * sum(x*x) - sum(x*y) * sum(x)) / (len(x) * sum(x*x) - sum(x)**2)
def streumaß(x, y): return np.sqrt((1/(len(x)-2)) * sum((y-(c_linreg(x, y)+m_linreg(x, y)*x))**2))
def sc_linreg(x, y): return streumaß(x, y) * np.sqrt(sum(x**2) / (len(x) * sum(x**2) - sum(x)**2))
def sm_linreg(x, y): return streumaß(x, y) * np.sqrt(len(x) / (len(x) * sum(x**2) - sum(x)**2))
def r(x): return round(x, 5)


def outputStd(data):
    mean = np.mean(data)
    variance = np.mean((data - mean) ** 2)
    std_dev = np.sqrt(variance)
    print("The standard deviation is", std_dev)
    return std_dev


def std(data): return np.sqrt(np.mean((data - np.mean(data)) ** 2))


def outputCorrelatedStd(data1, data2):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    variance = np.mean((data1 - mean1) * (data2 - mean2))
    std_dev = np.sqrt(variance)
    print("The correlated standard deviation is", std_dev)
    return std_dev


def outputCorrelationCoefficient(data1, data2):
    variance1 = outputStd(data1)
    variance2 = outputStd(data2)
    variance12 = outputCorrelatedStd(data1, data2)

    return variance12 / (variance1 * variance2)


def empiricalCovariance(data1, data2):
    pass


def linregVariance(bestfit, data1, data2):
    pass


def GetRWithErrors(V, dV, A, dA):
    newR = V / A
    uncertainties = np.sqrt(((V * dA) / (A ** 2)) ** 2 + (dV / A) ** 2)
    weights = 1 / np.square(uncertainties)
    weighted_avg = np.average(newR, weights=weights)
    new_uncertainty = np.sqrt(1 / np.sum(weights))

    # print('R-Werte:', newR)
    # print('Unsicherheiten auf R:', uncertainties)
    # print('Gewichtungen:', weights)
    print('This means R =', r(weighted_avg))
    print('With an error dR =', r(new_uncertainty))
    return '.'


