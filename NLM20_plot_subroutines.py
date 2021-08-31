import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy
from NLM20_find_lam3 import *

def read_data():
    # load the results for each parts and concatenate them.
    critical_strains = numpy.loadtxt('critical_strain.txt')
    crit_strain_fig_5_plain = numpy.loadtxt('critical_strain_fig_5_plain.txt')
    crit_strain_fig_5_Uni = numpy.loadtxt('critical_strain_fig_5_Uni.txt')
    thresh_strains = numpy.loadtxt('threshold_strain.txt')
    thresh_strains_fig9 = numpy.loadtxt('threshold_strain_fig9.txt')
    return critical_strains, crit_strain_fig_5_plain, crit_strain_fig_5_Uni, thresh_strains, thresh_strains_fig9

def plot_fig3(P_3_P_2_ratio, lam1s, lam3s, tol):

    cmap = plt.get_cmap('Spectral')
    new_cmap = truncate_colormap(cmap, 0.45, 1)
    P_2s = [0.2, 0.6, 1, 1.4, 1.8, 2, 2.5, 3, 3.5, 4]
    n_lines = len(P_2s)
    fig3, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
    fig3.subplots_adjust(hspace=0.1, wspace=0.1)
    for k in range(len(P_3_P_2_ratio)):
        if k == 0:
            i = 0
            j = 0
        if k == 1:
            i = 0
            j = 1
        if k == 2:
            i = 1
            j = 0
        if k == 3:
            i = 1
            j = 1
        ax[i, j].set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0.45, 1, n_lines)))
        for b in range(len(P_2s)):
            P_2 = P_2s[b]
            P_3 = P_2 * P_3_P_2_ratio[k]

            lam3s = find_lam3s(
                P_2,
                P_3,
                lam1s,
                lam3s,
                tol
            )
            uniaxial_lam3s = 1 / numpy.sqrt(lam1s)

            ax[i, j].set_xlim(0.4, 1)
            ax[i, j].set_ylim(1, 2)
            if k == 0:
                ax[i, j].set_ylabel('$\mathit{\\lambda_3}$', rotation=90, fontsize=20, labelpad=15)
            if k == 2:
                ax[i, j].set_ylabel('$\mathit{\\lambda_3}$', rotation=90, fontsize=20, labelpad=15)
                ax[i, j].set_xlabel('$\mathit{\\lambda_1}$', rotation=0, fontsize=20, labelpad=15)
            if k == 3:
                ax[i, j].set_xlabel('$\mathit{\\lambda_1}$', rotation=0, fontsize=20, labelpad=15)
            ax[i, j].plot(lam1s, lam3s, linestyle='-', linewidth=1)
            ax[i, j].yaxis.set_tick_params(labelsize=15, size=5)
            ax[i, j].xaxis.set_tick_params(labelsize=15, size=5)
            ax[i, j].minorticks_on()
        ax[i, j].plot(lam1s, uniaxial_lam3s, color='gray', linestyle='--', linewidth=3)
    #
    ax3 = fig3.add_axes([0.91, 0.11, 0.015, 0.77])
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.8, 2, 2.2, 2.8, 3, 3.2, 3.8, 4]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=new_cmap, norm=norm, orientation='vertical')
    ax3.text(2, -0.3, '$ \mathit{P_2/\\mu_f} $', rotation=0, fontsize=20)
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    plt.savefig('fig_3.png', dpi=500)

def plot_fig4(thresh_strains, P_2s, betas):

    cmap = plt.get_cmap('Spectral')
    thresh_strains = numpy.transpose(thresh_strains)
    new_cmp = truncate_colormap(cmap, 1, 0.5)
    fig4 = plt.figure(2, figsize=(20, 10))
    ax = fig4.add_subplot(111)
    ax.contourf(betas, P_2s, thresh_strains, 20, cmap=new_cmp, vmax=1, vmin=0, interpolation='none')
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.axvline(x=1, color='w', linewidth=6)
    ax.axhline(y=2.0, color='w', linewidth=6, linestyle='-')
    ax.set_xlim(0.1, 4)
    ax.set_ylim(0, 4)
    ax.yaxis.set_tick_params(labelsize=20, size=5)
    ax.xaxis.set_tick_params(labelsize=20, size=5)
    plt.xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=30)
    plt.ylabel('$ \mathit{\dfrac{P_2}{\\mu_f}}  $', rotation=0, fontsize=30, labelpad=20)
    ax3 = fig4.add_axes([0.91, 0.11, 0.015, 0.77])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=new_cmp, norm=norm, orientation='vertical')
    ax3.yaxis.set_tick_params(labelsize=20, size=5)
    ax3.minorticks_on()
    cb1.set_label('$ \mathit{\\epsilon^{th}_c} $', fontsize=30, rotation=0)
    plt.savefig('fig_4.png', dpi=300)

def plot_fig5(crit_strain_fig_5, crit_strain_fig_5_plain, crit_strain_fig_5_Uni, wavelengths):

    crit_strain_fig_5 = forming_crit_strain_fig5(crit_strain_fig_5)

    n_lines = 8
    fig_5 = plt.figure(1, figsize=(18, 7))
    ax = fig_5.add_subplot(121)
    ax.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0, 1, n_lines)))
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    plt.xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=15)
    plt.ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=15, labelpad=15)
    plt.gca().set_ylim(0., 1)
    plt.gca().set_xlim(0.1, 1000.)
    for i in range(8):
        plt.semilogx(wavelengths, crit_strain_fig_5_plain[:, i * 2], linestyle='--')
    for i in range(8):
        plt.semilogx(wavelengths, crit_strain_fig_5[:, i * 2], linestyle='-')
    #
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.18, hspace=None)
    #
    ax2 = fig_5.add_subplot(122)
    ax2.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0, 1, n_lines)))
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    plt.xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    plt.ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)
    plt.gca().set_ylim(0., 1)
    plt.gca().set_xlim(0.1, 1000.)
    for i in range(8):
        plt.semilogx(wavelengths, crit_strain_fig_5_Uni[:, i * 2], linestyle='-.')
    for i in range(8):
        plt.semilogx(wavelengths, crit_strain_fig_5[:, i * 2], linestyle='-')
    #

    bound_1 = numpy.linspace(0.1, 0.9, 10)
    bound_2 = numpy.linspace(1.1, 4, 10)
    bounds = numpy.concatenate((bound_1, bound_2), axis=0)
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    #
    ax3 = fig_5.add_axes([0.91, 0.11, 0.015, 0.77])
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cm.Spectral, norm=norm, orientation='vertical')
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    cb1.set_label('$ \mathit{\\beta} $', fontsize=20, rotation=0)
    cb1.set_ticks([0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4])
    # plt.savefig('fig_5ab.png', dpi=300)

    # plot the inset
    n_lines = 4
    fig_5a_inset = plt.figure(2, figsize=(13, 8))
    ax4 = fig_5a_inset.add_subplot(111)
    ax4.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0, 1, n_lines)))
    ax4.minorticks_on()
    ax4.yaxis.set_ticks_position('both')
    ax4.yaxis.set_tick_params(labelsize=40, size=7)
    ax4.xaxis.set_tick_params(labelsize=40, size=7)
    ax4.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter(False))
    ax4.set_xticks([0.1, 1, 5])
    ax4.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter(False))
    ax4.set_yticks([0.45, 0.46])
    plt.gca().set_ylim(0.445, 0.46)
    plt.gca().set_xlim(0.1, 5.)
    ax4.text(0.5, 0.44332, '$ \mathit{\\bar L_c} $', rotation=0, fontsize=50)
    ax4.text(0.07, 0.452, '$ \mathit{\\epsilon_c} $', rotation=0, fontsize=50)

    for i in range(4):
        plt.semilogx(wavelengths, crit_strain_fig_5_plain[:, i * 2], linestyle='--', linewidth=5)

    ax4.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0, 1, n_lines)))
    for i in range(4):
        plt.semilogx(wavelengths, crit_strain_fig_5[:, i * 2], linestyle='-', linewidth=5)
    plt.gca().set_ylim(0.445, 0.46)
    plt.gca().set_xlim(0.1, 5.)
    plt.savefig('fig_5a_inset.png', dpi=300)

def plot_fig6(crit_strain_fig_6, wavelengths, ind_rounded_b):

    [crit_strain_a, crit_strain_b] = forming_crit_strain_fig6(crit_strain_fig_6, ind_rounded_b)

    fig6, (ax1, ax2) = plt.subplots(1, 2)
    fig6.set_figheight(8)
    fig6.set_figwidth(20)
    ax1.set_prop_cycle('color', plt.cm.Spectral([0.5, 0.45, 0.4, 0.37, 0.33, 0.3, 0.25, 0.2, 0.1]))
    ax1.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.yaxis.set_ticks_position('both')
    ax1.yaxis.set_tick_params(labelsize=15, size=5)
    ax1.xaxis.set_tick_params(labelsize=15, size=5)
    ax1.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax1.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)

    for j in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
        ax1.set_xlim([0., 1000])
        ax1.set_ylim([0., 1])
        ax1.semilogx(wavelengths, crit_strain_a[:, j], linestyle='-', linewidth=2)
    #
    ax2.set_prop_cycle('color', plt.cm.Spectral([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 1]))
    ax2.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax2.xaxis.set_minor_locator(minorLocator)
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax2.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)

    for j in range(9):
        ax2.set_xlim([0., 1000])
        ax2.set_ylim([0., 1])
        ax2.semilogx(wavelengths, crit_strain_b[:, j], linestyle='-', linewidth=2)
    #
    ax3 = fig6.add_axes([0.91, 0.11, 0.015, 0.77])
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.46, 1.8, 2.2, 2.55, 2.9, 3.3, 3.6, 4]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cm.Spectral, norm=norm, orientation='vertical')
    cb1.set_label('$ \mathit{\\beta} $', fontsize=20, rotation=0)
    cb1.set_ticks([0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4])
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    plt.savefig('fig_6ab.png', dpi=300)

def plot_fig7(crit_strain_fig_7, wavelengths, ind_rounded_b):

    [crit_strain_a, crit_strain_b] = forming_crit_strain_fig7(crit_strain_fig_7, ind_rounded_b)

    fig7, (ax1, ax2) = plt.subplots(1, 2)
    fig7.set_figheight(8)
    fig7.set_figwidth(20)
    ax1.set_prop_cycle('color', plt.cm.Spectral([0.1, 0.2, 0.25, 0.3, 0.35, 0.35, 0.4, 0.45, 0.5]))
    ax1.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.yaxis.set_ticks_position('both')
    ax1.yaxis.set_tick_params(labelsize=15, size=5)
    ax1.xaxis.set_tick_params(labelsize=15, size=5)
    ax1.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax1.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)

    for j in range(9):
        ax1.set_xlim([0., 1000])
        ax1.set_ylim([0., 1])
        ax1.semilogx(wavelengths, crit_strain_a[:, j], linestyle='-', linewidth=2)

    ax2.set_prop_cycle('color', plt.cm.Spectral([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 1]))
    ax2.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax2.xaxis.set_minor_locator(minorLocator)
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\bar L_c} $', rotation=0, fontsize=20)
    ax2.set_ylabel('$ \mathit{\\epsilon_c} $', rotation=0, fontsize=20, labelpad=15)

    for j in range(9):
        ax2.set_xlim([0., 1000])
        ax2.set_ylim([0., 1])
        ax2.semilogx(wavelengths, crit_strain_b[:, j], linestyle='-', linewidth=2)

    ax3 = fig7.add_axes([0.91, 0.11, 0.015, 0.77])
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.46, 1.8, 2.2, 2.55, 2.9, 3.3, 3.5, 4]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cm.Spectral, norm=norm, orientation='vertical')
    cb1.set_label('$ \mathit{\\beta} $', fontsize=20, rotation=0)
    cb1.set_ticks([0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4])
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    plt.savefig('fig_7ab.png', dpi=300)

def plot_fig8(thresh_strains, betas, ind_rounded_p):

    [thr_strain_a, thr_strain_b] = forming_thresh_strain_fig8(thresh_strains, ind_rounded_p)
    fig8, (ax1, ax2) = plt.subplots(1, 2)
    fig8.set_figheight(8)
    fig8.set_figwidth(20)
    ax1.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0.45, 0.8, 12)))
    ax1.minorticks_on()
    minorLocator = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.yaxis.set_ticks_position('both')
    ax1.yaxis.set_tick_params(labelsize=15, size=5)
    ax1.xaxis.set_tick_params(labelsize=15, size=5)
    ax1.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20, labelpad=15)
    ax1.set_ylabel('$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20, labelpad=15)
    ax1.set_xlim([0., 4])
    ax1.set_ylim([0., 1])
    for j in range(12):
        ax1.axvline(x=1, color='gray', linewidth=2, linestyle='--')
        if j == 3:
            ax1.plot(betas, thr_strain_a[:, j], linestyle='-.', linewidth=4)
        else:
            ax1.plot(betas, thr_strain_a[:, j], linestyle='-', linewidth=2)
    #
    ax2.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0.8, 1, 8)))
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20, labelpad=15)
    ax2.set_ylabel('$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20, labelpad=15)
    ax2.set_xlim([0., 4])
    ax2.set_ylim([0., 1])
    for j in range(6):
        ax2.axvline(x=1, color='gray', linewidth=2, linestyle='--')
        if j == 4:
            ax2.plot(betas, thr_strain_b[:, j], linestyle='-.', linewidth=4)
        else:
            ax2.plot(betas, thr_strain_b[:, j], linestyle='-', linewidth=2)
    #
    ax3 = fig8.add_axes([0.91, 0.11, 0.015, 0.77])
    cmap = plt.get_cmap('Spectral')
    newcmp = truncate_colormap(cmap, 0.45, 1)
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.8, 2, 2.2, 2.8, 3, 3.2, 3.8, 4]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=newcmp, norm=norm, orientation='vertical')
    ax3.text(2, 0.500, '$ \mathit{P_2/\\mu_f} $', rotation=0, fontsize=20)
    cb1.set_ticks([0, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    plt.savefig('fig_8ab.png', dpi=300)

def plot_fig9(thresh_strains_fig9):
    [thr_strain_a, thr_strain_b] = forming_thresh_strain_fig9(thresh_strains_fig9)
    beta_main = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 12, 14, 16, 18, 20, 23, 25,
                 30, 33, 35, 40, 43, 45, 50, 53, 60, 65, 70, 75, 80, 100]
    fig9, (ax1, ax2) = plt.subplots(1, 2)
    fig9.set_figheight(8)
    fig9.set_figwidth(20)
    ax1.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0.45, 0.8, 5)))
    ax1.minorticks_on()
    ax1.yaxis.set_tick_params(labelsize=15, size=5)
    ax1.xaxis.set_tick_params(labelsize=15, size=5)
    ax1.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20, labelpad=15)
    ax1.set_ylabel('$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20, labelpad=15)
    for j in range(5):
        ax1.set_xlim([1, 100])
        ax1.set_ylim([0., 1])
        ax1.semilogx(beta_main, thr_strain_a[:, j], linestyle='-', linewidth=2)

    ax2.set_prop_cycle('color', plt.cm.Spectral(numpy.linspace(0.8, 1, 5)))
    ax2.minorticks_on()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_tick_params(labelsize=15, size=5)
    ax2.xaxis.set_tick_params(labelsize=15, size=5)
    ax2.set_xlabel('$ \mathit{\\beta} $', rotation=0, fontsize=20, labelpad=15)
    ax2.set_ylabel('$ \mathit{\\epsilon^{th}_c} $', rotation=0, fontsize=20, labelpad=15)

    for j in range(5):
        ax2.set_xlim([1, 100])
        ax2.set_ylim([0., 1])
        ax2.semilogx(beta_main, thr_strain_b[:, j], linestyle='-', linewidth=2)

    ax3 = fig9.add_axes([0.91, 0.11, 0.015, 0.77])
    cmap = plt.get_cmap('Spectral')
    newcmp = truncate_colormap(cmap, 0.45, 1)
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.8, 2, 2.2, 2.8, 3, 3.2, 3.8, 4]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=newcmp, norm=norm, orientation='vertical')
    ax3.text(2, 0.50, '$ \mathit{P_2/\\mu_f} $', rotation=0, fontsize=20)
    cb1.set_ticks([0, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax3.yaxis.set_tick_params(labelsize=15, size=5)
    plt.savefig('fig_9ab.png', dpi=300)

def forming_crit_strain_fig5(critical_strains):
# separate the critical strains corresponding to P_2=0.5 and beta=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    ind_rounded_b= [0, 3, 5, 6, 7, 10, 12, 14, 17, 19, 20, 25, 27, 32, 34, 37]
    crit_strain_fig_5 = numpy.zeros((1000, 16))

    column_counter = 0
    for i in ind_rounded_b:
        crit_strain_fig_5[:, column_counter] = critical_strains[:, 9 + 40 * i]
        column_counter = column_counter + 1

    return crit_strain_fig_5

def forming_crit_strain_fig6(critical_strains, ind_rounded_b):
# separate the critical strains corresponding to P_2=0.5 and beta=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    crit_strain_fig6 = numpy.zeros((1000, 18))
    crit_strain_a = numpy.zeros((1000, 9))
    crit_strain_b = numpy.zeros((1000, 9))
    column_counter = 0
    for i in ind_rounded_b:
        crit_strain_fig6[:, column_counter] = critical_strains[:, 8 + 40 * i]
        column_counter = column_counter + 1

    for i in range(9):
        crit_strain_a[:, i] = crit_strain_fig6[:, i]
    for i in range(9):
        crit_strain_b[:, i] = crit_strain_fig6[:, i + 9]

    return crit_strain_a, crit_strain_b

def forming_crit_strain_fig7(critical_strains, ind_rounded_b):
# separate the critical strains corresponding to P_2=0.5 and beta=[0.1 to 4]
    crit_strain_fig7 = numpy.zeros((1000, 18))
    crit_strain_a = numpy.zeros((1000, 9))
    crit_strain_b = numpy.zeros((1000, 9))
    column_counter = 0
    for i in ind_rounded_b:
        crit_strain_fig7[:, column_counter] = critical_strains[:, 36 + 40 * i]
        column_counter = column_counter + 1

    for i in range(9):
        crit_strain_a[:, i] = crit_strain_fig7[:, i]
    for i in range(9):
        crit_strain_b[:, i] = crit_strain_fig7[:, i + 9]

    return crit_strain_a, crit_strain_b

def forming_thresh_strain_fig8(thresh_strains, ind_rounded_p):
# separate the critical strains corresponding to P_2=3.5 and beta=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    thr_strain_a = numpy.zeros((40, 12))
    thr_strain_b = numpy.zeros((40, 7))
    counter = 0
    counter2 = 0
    for i in ind_rounded_p:
        if i < 26:
            thr_strain_a[:, counter] = thresh_strains[:, i]
            counter = counter + 1
        else:
            thr_strain_b[:, counter2] = thresh_strains[:, i]
            counter2 = counter2 + 1
    return thr_strain_a, thr_strain_b

def forming_thresh_strain_fig9(thresh_strains_fig9):
# separate the critical strains corresponding to P2 = 0.5 and beta=[]
    ind_rounded_p = [2, 9, 13, 17, 21, 22, 28, 33, 36, 39]
    thr_strain_a = numpy.zeros((40, 5))
    thr_strain_b = numpy.zeros((40, 5))
    counter = 0
    counter2 = 0
    for i in ind_rounded_p:
        if i < 22:
            thr_strain_a[:, counter] = thresh_strains_fig9[:, i]
            counter = counter + 1
        else:
            thr_strain_b[:, counter2] = thresh_strains_fig9[:, i]
            counter2 = counter2 + 1
    return thr_strain_a, thr_strain_b

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(numpy.linspace(minval, maxval, n)))
    return new_cmap