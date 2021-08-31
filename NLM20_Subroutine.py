import string
import warnings
from math import *

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy

from NLM20_find_lam3 import *

warnings.simplefilter('ignore')

def determinant(P2, lam3_root, lam, beta, wavelength):
    """ Return determinant of 6x6 coefficient matrix

    Parameters
    ----------
    P_2 : float
        pressure in 2 direction (normalized by mu_f)
    lam3_root : float
        value of stretch in 3 direction
    lam : float
        values of compression in 1 direction
    beta : float
        stiffness ratio (film/substrate)
    wavelength : float
        wavelength (normalized by H_f)

    Returns
    -------
    dd : float
        determinant of coefficient matrix.

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    # L = wavelength
    # LAMBDA = 1. + lam ** 4.

    lam3=lam3_root
    lam_1=lam
    KH = 2.*pi/wavelength
    P0=1/(lam_1**2.*lam3**2.)+P2/(lam_1*lam3)
    AA = numpy.zeros((6, 6), dtype='float64')

    try:
        AA[0][0] = + 1.
        AA[0][1] = - 1.
        AA[0][2] = + (lam_1**2.) * lam3
        AA[0][3] = - (lam_1**2.) * lam3
        AA[0][4] = + 1.
        AA[0][5] = + (lam_1**2.) * lam3

        AA[1][0] = - 1.
        AA[1][1] = - 1.
        AA[1][2] = - 1.
        AA[1][3] = - 1.
        AA[1][4] = + 1.
        AA[1][5] = + 1.

        AA[2][0] =   ( 2./(lam_1**2.*lam3**2.)+P2/(lam_1*lam3) ) * exp(-KH/(lam_1**2.*lam3) )
        AA[2][1] =   ( 2./(lam_1**2.*lam3**2.)+P2/(lam_1*lam3) ) * exp( KH/(lam_1**2.*lam3) )
        AA[2][2] =   (lam_1**2.+ P0 ) * exp(-KH )
        AA[2][3] =   (lam_1**2.+ P0 ) * exp( KH )
        AA[2][4] =   0.
        AA[2][5] =   0.

        AA[3][0] = - (lam_1 ** 2. + P0) * exp(-KH / (lam_1 ** 2. * lam3))
        AA[3][1] = (lam_1 ** 2. + P0) * exp(KH / (lam_1 ** 2. * lam3))
        AA[3][2] = - (2 / lam3 + P2 * lam_1) * exp(-KH)
        AA[3][3] = (2 / lam3 + P2 * lam_1) * exp(KH)
        AA[3][4] = 0.
        AA[3][5] = 0.

        AA[4][0] = beta * (1 / (lam_1 ** 2. * lam3 ** 2.) + P0)
        AA[4][1] = beta * (1 / (lam_1 ** 2. * lam3 ** 2.) + P0)
        AA[4][2] = beta * (lam_1 ** 2. + P0)
        AA[4][3] = beta * (lam_1 ** 2. + P0)
        AA[4][4] = - (1. / (lam_1 ** 2. * lam3 ** 2.) + P0)
        AA[4][5] = - (lam_1 ** 2. + P0)

        AA[5][0] = - beta * (lam_1 ** 2. + 1 / (lam_1 ** 2. * lam3 ** 2.) + P2 / (lam_1 * lam3))
        AA[5][1] = beta * (lam_1 ** 2. + 1 / (lam_1 ** 2. * lam3 ** 2.) + P2 / (lam_1 * lam3))
        AA[5][2] = - beta * (1 / lam3 + P0 * lam_1 ** 2. * lam3)
        AA[5][3] = beta * (1 / lam3 + P0 * lam_1 ** 2. * lam3)
        AA[5][4] = - (lam_1 ** 2. + P0)
        AA[5][5] = - (1 / lam3 + P0 * lam_1 ** 2. * lam3)

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam1 = ", lam1)
    except (OverflowError):
        dd = None
        # print("overflow at lam1 = ", lam1)
    return dd

def Ridder(P_2, P_3, lam3_root_a, lam3_root_b, a, b, beta, wavelength, tol):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    P_2 : float
        pressure in 2 direction (normalized by mu_f)
    P_3 : float
        pressure in 3 direction, P_3 = P_2 * 1.3
    lam3_root_a, lam3_root_b : float
        values of stretch in 3 direction corresponding to upper and lower brackets of lam1, a and b, respectively
    a, b : float
        upper and lower brackets of lambda for Ridders' method
    beta : float
        stiffness ratio (film/substrate)
    wavelength : float
        wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance

    Returns
    -------
    x_ridder : float
        value of axial compression, lambda, that satisfies Eq. 4.11
    i : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    nmax = 50

    fa = determinant(P_2, lam3_root_a, a, beta, wavelength)
    fb = determinant(P_2, lam3_root_b, b, beta, wavelength)

    if fa == 0.0:
        # print("lower bracket is root")
        return a, 0
    if fb == 0.0:
        # print("upper bracket is root")
        return b, 0
    if fa * fb > 0.0:
        # print("Root is not bracketed")
        # print(P2, beta, "if fa*fb > 0.0: ")
        return None, None

    for i in range(nmax):
        c = 0.5 * (a + b)

        lam3_root_c = find_lam3_1(P_2, P_3, lam3_root_a, tol, c)
        fc = determinant(P_2, lam3_root_c, c, beta, wavelength)


        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x_ridder = c + dx
        lam3_ridder = find_lam3_1(P_2, P_3, lam3_root_c, tol, x_ridder)

        fx = determinant(P_2, lam3_ridder, x_ridder, beta, wavelength)

        if x_ridder > 0.999:
            tol = 1.e-12
        # check for convergence
        if i > 0:
            if abs(x_ridder - xOld) < tol * max(abs(x_ridder), 1.0):
                return x_ridder, i
        xOld = x_ridder

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x_ridder
                fb = fx
            else:
                a = x_ridder
                fa = fx
        else:
            a = c
            b = x_ridder
            fa = fc
            fb = fx

    res = abs(x_ridder - xOld) / max(abs(x_ridder), 1.0)

    print("Too many iterations, res = {res}".format(res=res))
    # print(P2, beta, "end")
    return None, nmax


def find_critical_values(P_2, P_3, lam1s, lam3s, beta, wavelengths, npts, plotroots, findroots, printoutput, tol):
    """ Finds critical strain for each specified wavelength

    Parameters
    ----------
    P_2 : float
        pressure in 2 direction (normalized by mu_f)
    P_3 : float
        pressure in 3 direction, P_3 = P_2 * 1.3
    lam1s : list of floats
        list of values of compression in 1 direction
    lam3s : list of floats
        list of values of stretch in 3 direction
    beta : float
        stiffness ratio (film/substrate)
    wavelengths : list of floats
        list of wavelengths for which to calculate determinant
    npts : int
        number of strain values to consider when checking for existence of roots
    plotroots : boolean
        whether or not to plot lines showing positive or negative value at all npts for each wavelength
    findroots : boolean
        whether or not to find the values of each root (set to False and plotroots to True to see root plots)
    printoutput : boolean
        whether or not to print every root found at every wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance

    Returns
    -------
    strains : list of floats
        list of all critical strains which satisfy Eq. 4.11 for one beta and one P_2 values.

    Notes
    -----
    Called by NLM20.py
    """

    lam_min = lam1s[0]
    lam_max = lam1s[99]
    strains = []

    for wavelength in wavelengths:

        [rootexists, a, c ,lam3_root_a, lam3_root_c] = check_roots(P_2, lam3s, beta, wavelength, lam_min, lam_max, npts, plotroots)

        if findroots:
            strains = find_roots(P_2, P_3, lam3_root_a, lam3_root_c, beta, wavelength, strains, rootexists, a, c, printoutput, tol)

    if findroots:
        return strains


def check_roots(P_2, lam3s, beta, wavelength, lam_min, lam_max, npts, plotroots):
    """ Calculates the value and/or sign of the determinant at every lambda 

    Parameters
    ----------
    P_2 : float
        pressure in 2 direction (normalized by mu_f)
    lam3s : list of floats
        list of values of compression in 3 direction
    beta : float
        stiffness ratio (film/substrate)
    wavelength : float
        wavelength
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    plotroots : boolean
        plot lines showing positive or negative value at all npts

    Returns
    -------
    rootexists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    a : float
        lower bracket
    c : float
        upper bracket
    lam3_root_a, lam3_root_c : float
        values of stretch in 3 direction corresponding to upper and lower brackets of lam1, a and c, respectively
    """

    lam1s = numpy.linspace(lam_min, lam_max, npts)[::-1]
    lam3s = lam3s[::-1]
    dds = numpy.zeros(npts, dtype='float64')
    dds_abs = numpy.zeros(npts, dtype='float64')

    rootexists = False
    a = 0.  # lower bracket
    c = 0.  # upper bracket
    lam3_root_a = 0.
    lam3_root_c = 0.

    # move backwards, in order of decreasing strain and report values bracking highest root
    for i in range(npts):

        lam = lam1s[i]
        lam3 = lam3s[i]
        dds[i] = determinant(P_2, lam3, lam, beta, wavelength)

        # if encountering NaN before finding root:
        if isnan(dds[i]) and not rootexists:
            dds[i] = 0.
            rootexists = False
            a = 0.
            c = 0.

        # otherwise, step through
        if i > 0 and not rootexists:
            dds_abs[i] = dds[i] / abs(dds[i])
            # if encountering root
            if dds[i] * dds[i - 1] < 0.:  # sign change
                rootexists = True
                a = lam1s[i]
                c = lam1s[i - 1]
                lam3_root_a = lam3s[i]
                lam3_root_c = lam3s[i - 1]
                break

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -100000, 100000])
        plt.xlabel('$\lambda_1$')
        plt.ylabel('energy')
        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lam1s, dds, color='b', linestyle='-')

    return rootexists, a, c, lam3_root_a, lam3_root_c


def find_roots(P_2, P_3, lam3_root_a, lam3_root_c, beta, wavelength, strains, rootexists, a, c, printoutput, tol):
    """ Calculates the critical strain for each specified wavelength

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_root_a, lam3_root_c : float
            values of stretch in 3 direction corresponding to upper and lower brackets of lam1, a and c, respectively
        beta : float
            stiffness ratio (layer/matrix)
        wavelength : float
            wavelength (normalized by H_l)
        strains : list of floats
             list of all critical strains which satisfy Eq. 24 for one beta and one H_m values.
        rootexists : boolean
            boolean value indicating whether or not a root (sign change) was detected
        a : float
            lower bracket
        c : float
            upper bracket
        printoutput : boolean
            whether or not to print every root found at every wavelength
        tol : float
            tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance

        Returns
        -------
        strains : list of floats
            list of all critical strains which satisfy Eq. 4.11 for one beta and one H_m values.

        Notes
        -----
        returns compressive strain (1 - lam) at which buckling occurs for the given parameters
        """

    if printoutput:
        print("\nx = %0.2f, a = %f, c = %f" % (wavelength, a, c))

    if rootexists:

        [lam1, n] = Ridder(P_2, P_3, lam3_root_a, lam3_root_c, a, c, beta, wavelength, tol)
        if printoutput: print("lam = %0.5f, n = %d" % (lam1, n))
    else:  # no root means that system is infinitely stable
        lam1 = 0.
        if printoutput: print("wavelength = %0.2f, no root" % wavelength)

    strains.append(1. - lam1)

    return strains


def find_threshold_values(wavelengths, crit_strains, i, j, thresh_wavelengths, thresh_strains):
    """ Finds threshold critical strain and corresponding threshold wavelength

    Parameters
    ----------
    wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength
    j : integer
        column counter of crit_strains.  j = P2
    i : integer
        row counter of crit_strains. i = beta
    thresh_wavelengths :
        critical wavelength (corresponding to critical strain)
    thresh_strains :
        minimum critical strain
    Returns
    -------
    thresh_wavelengths : ndarray float
        critical wavelength (corresponding to critical strain)
    thresh_strains : ndarray float
        minimum critical strains corresponding to specific values of P_2s and betas
    
    Notes
    -----
    If there is no true minimum, it returns the zero for the wavelength and the zero wavelength strain
    """

    index = 0

    if crit_strains[0] == min(crit_strains):
        start = False
    else:
        start = True

    for x in range(1, len(wavelengths) - 1):
        if crit_strains[x] == 1.0 and start:
            index = x + 1
        elif crit_strains[x] == 1.0:
            index = index
        elif crit_strains[x] < crit_strains[index]:
            index = x
            start = False

    # remove no-root results
    strains_masked = numpy.ma.masked_greater(crit_strains, 0.999)
    if max(strains_masked) - min(strains_masked) < 0.001:
        index = 0

    # if very small wavelength is dominating, check to see if infinity is equally favorable
    if index > 0.95 * len(wavelengths):
        if abs(crit_strains[0] - crit_strains[index]) < 0.001:
            index = 0

    thresh_wavelengths[i, j] = wavelengths[index]
    thresh_strains[i, j] = crit_strains[index]

    return thresh_wavelengths, thresh_strains
