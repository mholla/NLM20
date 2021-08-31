import numpy
import sympy as sy
import matplotlib.pyplot as plt
from math import *
import string
import warnings
warnings.simplefilter('ignore')

def find_lam3s(P_2, P_3, lam1s, lam3s, tol):
    """ Finds lam3s values which satisfy equation 2.8

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam1s : list of floats
            list of values of compression in 1 direction
        lam3s : list of floats
            list of values of compression in 3 direction
        tol : float
            tolerance for Newton-Raphson's method; solution will be returned when the absolute value of the function is below the tolerance

        Returns
        -------
        lam3s : list of floats
        list of all lam3 values which satisfy Eq. 2.8 for one P_2, P_3, and lam1 values.

        Notes
        -----
        Called by NLM20.py
        """

    for i in range(len(lam1s)):
        lam1 = lam1s[i]
        if i == 0:
            init_lam3 = init_lam3_guess(P_2, P_3, lam1)
        lam3 = newtons_method(f, df, P_2, P_3, init_lam3, tol, lam1)
        init_lam3 = lam3
        lam3s[i] = lam3
    return lam3s

def init_lam3_guess(P_2, P_3, lam1):
    """ Finds initial guess of lam3 which is needed to begin the Newton-Raphson root finding procedure

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam1s : float
            values of compression in 1 direction

        Returns
        -------
        init_lam3 : float
            initial guess of the root of equation 2.8

        """
    trial_lam3s = numpy.linspace(0., 50, 100)
    fs = numpy.zeros(len(trial_lam3s), dtype='float64')
    fs_abs = numpy.zeros(len(trial_lam3s), dtype='float64')
    init_lam3 = 0.
    for i in range(len(trial_lam3s)):
        first_lam3 = trial_lam3s[i]
        fs[i] = f(P_2, P_3, first_lam3, lam1)
        # otherwise, step through
        if i > 0:
            fs_abs[i] = fs[i] / abs(fs[i])
            # if encountering root
            if fs[i] * fs[i - 1] < 0.:  # sign change
                init_lam3 = trial_lam3s[i]
                return init_lam3
    return init_lam3

def newtons_method(f, df, P_2, P_3, lam3_n, tol, lam1):
    """ Finds lam3 (root) in polynomial Eq. 2.8

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_n : float
            values of compression in 3 direction. initial guess for begining the root finding process
        tol : float
            tolerance for Newton-Raphson's method; solution will be returned when the absolute value of the function is below the tolerance
        lam1 : floats
            value of compression in 1 direction
        Returns
        -------
        lam3_np1 : float
            values of compression in 3 direction. The root found by Newton-Raphson's method

        """
    err = dlam3(f, P_2, P_3, lam3_n, lam1)
    count = 0
    lam3_np1 = lam3_n
    while err > tol:
        lam3_n = lam3_np1
        lam3_np1 = lam3_n - f(P_2, P_3, lam3_n, lam1) / df(P_2, P_3, lam3_n, lam1)
        err = dlam3(f, P_2, P_3, lam3_np1, lam1)
        count = count + 1
        if count > 6:
            return lam3_np1
    return lam3_np1

# Newton–Raphson
def f(P_2, P_3, lam3_n, lam1):
    """ Calculates f = Eq. 2.8

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_n : float
            values of compression in 3 direction.
        lam1 : floats
            value of compression in 1 direction
        Returns
        -------
        lam3_np1 : float
            value of the function f

        """
    return lam3_n ** 4 + P_3 * lam3_n ** 3 - 1 / lam1 ** 2 - P_2 * lam3_n / lam1

def df(P_2, P_3, lam3_n, lam1):
    """ Calculates the derivative of f with respect to lam3

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_n : float
            values of compression in 3 direction.
        lam1 : floats
            value of compression in 1 direction
        Returns
        -------
        lam3_np1 : float
            value of the function f

        """
    return 4. * lam3_n ** 3 + 3. * P_3 * lam3_n ** 2 - P_2 / lam1

def dlam3(f, P_2, P_3, lam3_n, lam1):
    """ Calculates the the absolute value of the function f Eq. 2.8

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_n : float
            values of compression in 3 direction.
        lam1 : floats
            value of compression in 1 direction
        Returns
        -------
        abs(0 - f(P_2, P_3, lam3_n, lam1)) : float
            value of the function f

        """
    return abs(0 - f(P_2, P_3, lam3_n, lam1))


def find_lam3_1(P_2, P_3, lam3_root_c, tol, x_ridder):
    """ Finds lam3s values corresponding to x_ridder

        Parameters
        ----------
        P_2 : float
            pressure in 2 direction (normalized by mu_f)
        P_3 : float
            pressure in 3 direction, P_3 = P_2 * 1.3
        lam3_root_c: float
            values of stretch in 3 direction corresponding to x_ridder
        tol : float
            tolerance for Newton-Raphson's method; solution will be returned when the absolute value of the function is below the tolerance
        x_ridder: float
            value of compression in 1 direction

        Returns
        -------
        lam3s : float
            lam3 value which satisfy Eq. 2.8 for one P_2, P_3, and lam1 values.

        Notes
        -----
        Called by NLM20_Subroutine.py
        """
    lam33 = newtons_method(f, df, P_2, P_3, lam3_root_c, tol, x_ridder)
    return lam33

#
if __name__ == '__main__':
    stop = 1
#

