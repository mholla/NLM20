import string
import warnings
from math import *
import numpy
from NLM20_plot_subroutines import *

if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################
    npts = 100  # (number of points between 0.1 and 1.1 to look for the existence of roots)
    n_wavelengths = 1000
    n_betas = 40
    n_P2 = 40
    tol = 1.e-12

    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths)  # "L" in the paper
    wavelengths = wavelengths[::-1]  # start at right end of the graph (where the distance between roots is larger)
    lam1s = numpy.linspace(0.01, 0.999, npts)
    lam3s = numpy.zeros(npts)

    P2_a = numpy.linspace(0.1, 1, int(n_P2 / 2))
    P2_b = numpy.linspace(1.1, 4, int(n_P2 / 2))
    P_2s = numpy.concatenate((P2_a, P2_b), axis=0) # normalized pressure : P_2_f / mu_f
    # choose some values of H_ms array for certain figures
    ind_rounded_p = [2, 4, 6, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 27, 29, 32, 36, 37, 38, 39]

    P_3_P_2_ratio = [0, 0.8, 1.3, 2] # P_3_f / P_2_f
    betas_1 = numpy.linspace(0.1, 0.9, int(n_betas / 2))  # softer layer (beta < 1)
    betas_2 = numpy.linspace(1.1, 4, int(n_betas / 2))  # stiffer layer (beta > 1)
    betas = numpy.concatenate((betas_1, betas_2), axis=0)
    # choose some values in betas array for certain figures
    ind_rounded_b = [0, 3, 5, 7, 10, 12, 14, 17, 19, 20, 22, 25, 27, 30, 32, 33, 35, 39]

    [critical_strains, crit_strain_fig_5_plane, crit_strain_fig_5_Uni, thresh_strains, thresh_strains_fig9] = read_data()

    plot_fig3(P_3_P_2_ratio, lam1s, lam3s, tol)
    plot_fig4(thresh_strains, P_2s, betas)
    plot_fig5(critical_strains, crit_strain_fig_5_plane, crit_strain_fig_5_Uni, wavelengths)
    plot_fig6(critical_strains, wavelengths, ind_rounded_b)
    plot_fig7(critical_strains, wavelengths, ind_rounded_b)
    plot_fig8(thresh_strains, betas, ind_rounded_p)
    plot_fig9(thresh_strains_fig9)

    stop = 1
