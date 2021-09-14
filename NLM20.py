import string
import warnings
from math import *
import numpy

from NLM20_Subroutine import *
from NLM20_find_lam3 import *

warnings.simplefilter('ignore')

def solve(params, output_options, fig_9):
    """ Calculations of threshold values for each wrinkling mode

    Parameters
    ----------
    params : list
                P_2s : list of floats
                    list of pressures in 2 direction (normalized by mu_f)
                betas : list of floats
                    list of stiffness ratios (film/substrate) for which to calculate determinant
                wavelengths : list of floats
                    list of wavelengths (normalized by H_f) for which to calculate determinant
                lam1s : list of floats
                    list of values of compression in 1 direction
                tol : float
                    tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
                npts : int
                    number of points between lam_min and lam_max at which to calculate determinant
    output_options : list
                plotroots : boolean
                    whether or not to plot lines showing positive or negative value at all npts for each wavelength
                findroots : boolean
                    whether or not to find the values of each root (set to False and plotroots to True to see root plots)
                printoutput : boolean
                    whether or not to print every root found at every wavelength
    fig_9 : boolean
        whether or not calculate the results for figure 9

    Returns
    -------
    None


    Notes
    -----
    "critical_strain_fig_5_plane.txt" and "critical_strain_fig_5_Uni.txt" are provided for figure 5 a and b.


    """

    [P_2s, betas, wavelengths, lam1s, tol, npts] = params
    [findroots, plotroots, printoutput] = output_options

    critical_strains = numpy.zeros((len(wavelengths), int(int(len(betas)) * int(len(P_2s) ))))
    thresh_strains = numpy.zeros((len(betas), int(len(P_2s) )))
    thresh_wavelengths = numpy.zeros((len(betas), int(len(P_2s) )))

    column = 0

    for i in range(len(betas)):

        for j in range(len(P_2s)):

            beta = betas[i]
            P_2 = P_2s[j]
            P_3 = P_2 * 1.3

            lam3s = numpy.zeros(npts)

            lam3s = find_lam3s(
                P_2,
                P_3,
                lam1s,
                lam3s,
                tol
            )


            crit_strains = find_critical_values(
                P_2,
                P_3,
                lam1s,
                lam3s,
                beta,
                wavelengths,
                npts,
                plotroots,
                findroots,
                printoutput,
                tol
            )

            critical_strains[:, column] = crit_strains[:]
            column = column + 1
            [thresh_wavelengths, thresh_strains] = find_threshold_values(
                wavelengths,
                crit_strains,
                i,
                j,
                thresh_wavelengths,
                thresh_strains
            )

    # save all critical strains (1000x(40x40) matrix)
    # each row corresponds to a certain value of normalized wavelength
    # for every 40 columns (number of P_2 values) the value of beta changes
    if fig_9:
        numpy.savetxt('threshold_strain_{name}.txt'.format(name = 'fig_9'),
                      thresh_strains, fmt='%.8f')
    if not fig_9:
        numpy.savetxt('critical_strain.txt',
                      critical_strains, fmt='%.8f')

        # save threshold values for every combination of beta and P_2
        numpy.savetxt('threshold_strain.txt',
                      thresh_strains, fmt='%.8f')
        # numpy.savetxt('threshold_wavelength.txt',
        #               thresh_wavelengths, fmt='%.8f')

if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################
    npts = 100  # (number of points between 0.1 and 1.1 to look for the existence of roots)
    n_wavelengths = 1000
    n_betas = 40
    n_P2 = 40
    tol = 1.e-12  # this is a default value

    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths)  # "L" in the paper
    wavelengths = wavelengths[::-1]  # start at right end of the graph (where the distance between roots is larger)
    lam1s = numpy.linspace(0.01, 0.99999, npts) # compression in direction 1
    # lam3s = numpy.zeros(npts) # stretch in direction 3

    P2_a = numpy.linspace(0.1, 1, int(n_P2 / 2))
    P2_b = numpy.linspace(1.1, 4, int(n_P2 / 2))
    P_2s = numpy.concatenate((P2_a, P2_b), axis=0) # pressures in 2 direction (normalized by mu_f)
    # round some values of H_ms array for certain figures
    P_2s_rounded = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.8,2,2.2,2.5,3,3.5,4]  # Rounded P_2 values
    ind_rounded = [2,4,6,9,11,13,15,17,19,21,22,25,26,27,29,33,36,39]
    for i in range(18):
        P_2s[ind_rounded[i]] = P_2s_rounded[i]
    P_3_P_2_ratio = [0,0.8,1.3,2] # P_3_f / P_2_f
    betas_1 = numpy.linspace(0.1, 0.9, int(n_betas / 2))  # softer layer (beta < 1)
    betas_2 = numpy.linspace(1.1, 4, int(n_betas / 2))  # stiffer layer (beta > 1)
    betas = numpy.concatenate((betas_1, betas_2), axis=0)

    # round some values in betas array for certain figures
    beta_rounded = numpy.concatenate((numpy.linspace(0.1, 0.9, 9), numpy.linspace(1.1, 4, 9)), axis=0)
    ind_rounded = [0, 3, 5, 7, 10, 12, 14, 17, 19, 20, 22, 25, 27, 30, 32, 34, 37, 39]
    for i in range(18):
        betas[ind_rounded[i]] = beta_rounded[i]

    params = [P_2s, betas, wavelengths, lam1s, tol, npts]

    # parameters for output
    findroots = True  # only set to false for troubleshooting, using plotroots below
    plotroots = False  # save plot of absolute value of determinant at each n_wavelengths
    printoutput = False  # print every root found at every n_wavelengths
    fig_9 = False
    output_options = [findroots, plotroots, printoutput]

    solve(params, output_options, fig_9)

    # Calculate the threshold strains for figure_9
    betas = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 12, 14, 16, 18, 20, 23, 25,
             30, 33, 35, 40, 43, 45, 50, 53, 60, 65, 70, 75, 80, 100]
    params = [P_2s, betas, wavelengths, lam1s, tol, npts]
    fig_9 = True
    solve(params, output_options, fig_9)

    stop = 1 # ignore this line

