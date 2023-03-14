# Thoughts:
# - Maybe add a language switch to 'babaLinreg'

import numpy as np
from matplotlib import pyplot as plt

import BasicLinRegFunctions as blf


def r(x): return round(x, 5)


sampleX = np.array([1, 2, 3, 4, 5])
sampleY = np.array([1, 2.1, 3.1, 3.4, 5])


# 'babaLinreg' should output all data needed to immediately crate a plot with a bestfit curve and a confidence band
def babaLinreg(x, y, xError=None, yError=None, title=None, xLabel=None, yLabel=None, language=None, printOutput=None,
               markerColor=None, switchSigns=None):
    # taking care of the optionals:
    if title is None:
        title = 'Untitled'
    if xLabel is None:
        xLabel = 'x-axis'
    if yLabel is None:
        yLabel = 'y-axis'
    if markerColor is None:
        markerColor = 'b'
    if switchSigns:
        x = -x
        y = -y

    if language == 'g':
        bestFitLabel = 'Ausgleichsgerade'
        confidenceBandLabel = 'Konfidenzband'
        valueLabel = 'Messwerte'
    else:
        bestFitLabel = 'best fit'
        confidenceBandLabel = 'confidence band'
        valueLabel = 'values'

    if markerColor == 'b':
        valueLabel += ' (negativ)'
    if markerColor == 'r':
        valueLabel += ' (positiv)'

    # Basic Linreg stuff:
    slope = blf.m_linreg(x, y)
    offset = blf.c_linreg(x, y)
    bestFit = blf.gerade(x, slope, offset)

    dispersionSlope = blf.sm_linreg(x, y)
    dispersionOffset = blf.sc_linreg(x, y)

    upperBestFit = blf.gerade(x, slope + dispersionSlope, offset - dispersionOffset)
    lowerBestFit = blf.gerade(x, slope - dispersionSlope, offset + dispersionOffset)

    if printOutput:
        print('The slope m =', r(slope))
        print('The offset c =', r(offset))
        print('Therefore R =', r(1/slope))
        #  maybe also add the std and so forth...
        print('Dispersion slope:', r(dispersionSlope))
        print('The new lower R is:', r(1 / (slope + dispersionSlope)))
        print('The new upper R is:', r(1 / (slope - dispersionSlope)))
        print('This means the uncertainty is +', r((1/slope) - 1/(slope + dispersionSlope)), 'and -',
              r((1 / (slope - dispersionSlope)) - (1/slope)))

    # if printOutput:

    # Plot the data
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(x, bestFit, ls='-.', c='gray', label=bestFitLabel)
    plt.scatter(x, y, marker='x', label=valueLabel, color=markerColor)
    # if xError or yError is not None:
    plt.errorbar(x, y, yerr=yError, xerr=xError, fmt='none',
                 capsize=3, label='Fehler', color='black')
    plt.fill_between(x, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                     interpolate=True, color='pink', alpha=0.5, label=confidenceBandLabel)
    plt.fill_between(x, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                     interpolate=True, color='pink', alpha=0.5)
    plt.legend()


def CombinedBabaLinreg(x_arr, y_arr, xError_arr=None, yError_arr=None, title=None, xLabel=None, yLabel=None, language=None,
                       printOutput=None):
    # taking care of the optionals:
    if title is None:
        title = 'Untitled'
    if xLabel is None:
        xLabel = 'x-axis'
    if yLabel is None:
        yLabel = 'y-axis'

    if language == 'g':
        bestFitLabel = 'Ausgleichsgerade'
        confidenceBandLabel = 'Konfidenzband'
        valueLabel = 'Messwerte'
    else:
        bestFitLabel = 'best fit'
        confidenceBandLabel = 'confidence band'
        valueLabel = 'values'

    nX = -x_arr[0]
    pX = x_arr[1]
    nY = -y_arr[0]
    pY = y_arr[1]

    newX = np.concatenate((nX, pX))
    newY = np.concatenate((nY, pY))

    if xError_arr is None:
        print('None')
    else:
        newXError = np.concatenate((xError_arr[0], xError_arr[1]))
        newYError = np.concatenate((yError_arr[0], yError_arr[1]))

    # Basic Linreg stuff:
    slope = blf.m_linreg(newX, newY)
    offset = blf.c_linreg(newX, newY)
    bestFit = blf.gerade(newX, slope, offset)

    nBestFit = blf.gerade(nX, slope, offset)
    pBestFit = blf.gerade(pX, slope, offset)

    if printOutput:
        print('The slope m =', r(slope))
        print('The offset c =', r(offset))
        print('Therefore R =', r(1 / slope))
        #  maybe also add the std and so forth...

    dispersionSlope = blf.sm_linreg(newX, newY)
    dispersionOffset = blf.sc_linreg(newX, newY)

    upperBestFit = blf.gerade(newX, slope + dispersionSlope, offset - dispersionOffset)
    lowerBestFit = blf.gerade(newX, slope - dispersionSlope, offset + dispersionOffset)

    # Calculate residues:
    nResidues = nY - nBestFit
    pResidues = pY - pBestFit

    # Plot the data
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(newX, bestFit, ls='-.', c='gray', label=bestFitLabel)
    plt.scatter(pX, pY, marker='x', label='positive Messwerte', color='r')
    plt.scatter(nX, nY, marker='x', label='negative Messwerte', color='b')
    # if xError or yError is not None:
    plt.errorbar(newX, newY, yerr=newYError, xerr=newXError, fmt='none',
                 capsize=3, label='Fehler', color='black')
    plt.fill_between(newX, upperBestFit, lowerBestFit, where=lowerBestFit > upperBestFit,
                     interpolate=True, color='pink', alpha=0.5, label=confidenceBandLabel)
    plt.fill_between(newX, upperBestFit, lowerBestFit, where=upperBestFit >= lowerBestFit,
                     interpolate=True, color='pink', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Darstellung der Residuen')
    plt.plot([min(newX), max(newX)], [0, 0], color='black')
    plt.scatter(pX, pResidues, label='Residuen (positiv)', marker='x', color='r')
    plt.scatter(nX, nResidues, label='Residuen (negativ)', marker='x', color='b')
    plt.legend()
