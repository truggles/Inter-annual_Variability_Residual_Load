#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

from scipy.stats import rankdata, normaltest
from collections import OrderedDict
import pickle
from glob import glob
import os
from shutil import copy2
import copy
import calendar

import socket

app = 'Users' if socket.gethostname() == 'truggles-ciw' else 'home'
sys.path.append(f"/{app}/truggles/Inter-annual_Variability_Residual_Load")
from helpers import return_file_info_map


def plot_matrix_slice(region, plot_base, ms, idx_range, alt_idx, save_name, range_name):

    print(f"Plotting: {save_name}")

    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()#figsize=(4.5, 4))

    for name, m in ms.items():
        print(name)
        if range_name == 'solar':
            ax.plot( m[idx_range[0]:idx_range[1], alt_idx], label=name )
        if range_name == 'wind':
            ax.plot( m[alt_idx, idx_range[0]:idx_range[1]*2], label=name )

    ax.set_ylim( 0, ax.get_ylim()[1] )

    ax.set_xlabel(f"{range_name} generation\n(% mean load)")
    opp = 'wind' if range_name == 'solar' else 'solar'
    plt.title(f"{opp} generation {alt_idx} (% mean load)")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.8, top=0.9)

    plt.legend()
    plt.savefig(f"{plot_base}/{region}_{save_name}_{range_name}.{TYPE}")
    plt.clf()



def plot_matrix_thresholds(region, plot_base, mx_ary, solar_values, wind_values, save_name, titles=['',]):

    print(f"Plotting: {save_name}")

    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    cols = len(mx_ary)
    fig, axs = plt.subplots(figsize=(2.8*cols, 3.2), ncols=cols, sharey=True)

    # Contours
    min_and_max = []
    if 'RL_mean' in save_name:
        n_levels = np.arange(0,200,10)
        c_fmt = '%3.0f'
        ylab = "$\mu$ peak residual load\n(% mean load)"
        min_and_max = [100, 170]
    elif 'RL_std' in save_name:
        n_levels = np.arange(0,15,0.5)
        c_fmt = '%1.1f'
        ylab = "$\sigma$ peak residual load\n(% mean load)"
        min_and_max = [4, 10]
    elif 'RL_50pct' in save_name:
        n_levels = np.arange(0,50,2.5)
        c_fmt = '%1.1f'
        ylab = "mid-50% range peak residual load\n(% mean load)"
    elif 'RL_95pct' in save_name:
        n_levels = np.arange(0,50,5)
        c_fmt = '%1.1f'
        ylab = "mid-95% range peak residual load\n(% mean load)"
    elif 'RL_Mto97p5pct' in save_name:
        n_levels = np.arange(0,50,2.5)
        c_fmt = '%1.1f'
        ylab = "50-97.5% range peak residual load\n(% mean load)"
    elif 'PL_mean' in save_name:
        n_levels = np.arange(-100,200,20)
        c_fmt = '%3.0f'
        ylab = "$\mu$ residual load of peak\nload hours (% mean load)"
    elif 'PL_std' in save_name:
        n_levels = np.arange(0,100,5)
        c_fmt = '%1.0f'
        ylab = "$\sigma$ residual load of peak\nload hours (% mean load)"
    elif '_mean' in save_name: # else if so gets solar and wind means
        n_levels = np.arange(0,200,10)
        c_fmt = '%3.0f'
        app = 'wind' if 'wind' in save_name else 'solar'
        ylab = f"mean {app} capacity factor\n(during peak hours)"
        if '_solar' in save_name:
            min_and_max = [0, 70]
        if '_wind' in save_name:
            min_and_max = [0, 50]

    elif '_inter' in save_name:
        n_levels = np.arange(0,20,1)
        c_fmt = '%3.1f'
        ylab = "inter-annual variability\n(% mean load)"
        min_and_max = [3, 10]
        if 'Nom' in save_name:
            n_levels = np.arange(-3,3,.2)
            c_fmt = '%3.2f'
            min_and_max = [-3, 3]
    elif '_intra' in save_name:
        n_levels = np.arange(0,20,1)
        c_fmt = '%3.1f'
        ylab = "intra-annual variability\n(% mean load)"
        min_and_max = [2, 8]
    elif 'QuadR' in save_name:
        n_levels = np.arange(-10,10,1)
        c_fmt = '%1.1f'
        ylab = "$\sigma$ residual load of peak\nload hours (% mean load)"
        #min_and_max = [-6, 1]

    # Clip colormap before yellow high values so white
    # contour text shows up.
    cmapBig = matplotlib.cm.get_cmap('plasma', 512)
    top = 0.85
    cmapShort = matplotlib.colors.ListedColormap(cmapBig(np.linspace(0.0, top, int(512*top))))

    ims = []
    css = []
    cnt = 0
    for matrix, title in zip(mx_ary, titles):
        if len(min_and_max) == 0:
            im = axs[cnt].imshow(matrix, interpolation='none', origin='lower', cmap=cmapShort)
        else:
            im = axs[cnt].imshow(matrix, interpolation='none', origin='lower', vmin=min_and_max[0], vmax=min_and_max[1], cmap=cmapShort)
        ims.append(im)

        cs = axs[cnt].contour(matrix, n_levels, colors='w')
        css.append(cs)
        # inline labels
        axs[cnt].clabel(cs, inline=1, fontsize=12, fmt=c_fmt)

        wind_labs, solar_labs = [], []
        for v in wind_values:
            if int(v*4)==v*4:
                wind_labs.append(f"{int(v*100)}%")
            else:
                wind_labs.append('')
        for v in solar_values:
            if int(v*4)==v*4:
                solar_labs.append(f"{int(v*100)}%")
            else:
                solar_labs.append('')
        axs[cnt].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 25, 50, 75, 100]))
        axs[cnt].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
        axs[cnt].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 25, 50, 75, 100]))
        axs[cnt].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
        #my_ticks = np.arange(0, 101, 5)
        #my_labs = ['' for _ in range(len(my_ticks))]
        #my_labs[0] = '0%'
        #my_labs[5] = '25%'
        #my_labs[10] = '50%'
        #my_labs[15] = '75%'
        #my_labs[20] = '100%'
        #axs[cnt].set_xticks(my_ticks, my_labs)
        #axs[cnt].set_yticks(my_ticks, my_labs)
        ##axs[cnt].set_xticks(range(len(wind_values)), wind_labs, rotation=90)
        plt.setp( axs[cnt].xaxis.get_majorticklabels(), rotation=40 )
        # Adding X label below with one big overlaid axis
        #axs[cnt].set_xlabel("wind generation\n(% mean load)")
        if cnt == 0:
            axs[cnt].set_ylabel("solar generation\n(% mean load)")
        else:
            solar_labs = ['' for i in range(len(solar_values))]
        #axs[cnt].set_yticks(range(len(solar_values)), solar_labs)
        if cnt + 1 == len(axs):
            #cbar = axs[cnt].figure.colorbar(im)
            #cbar.ax.set_ylabel(ylab)
            #dec = 0
            #cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))

            fig.subplots_adjust(right=0.87)
            cbar_ax = fig.add_axes([0.88, 0.265, 0.02, 0.9 - 0.265])
            cbar = fig.colorbar(ims[-1], cax=cbar_ax)
            cbar.ax.set_ylabel(ylab)
            dec = 0
            cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))

        axs[cnt].set_title(title)
        cnt += 1
    #plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.25, right=0.88, top=0.9)
    plt.subplots_adjust(left=0.105, bottom=0.265, top=0.9)

    # SHARED X AXIS LABEL
    # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots/53172335
    # add a big axis, hide frame
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("wind generation (% mean load)", labelpad=21)

    plt.savefig(f"{plot_base}/{region}_{save_name}.{TYPE}")
    plt.clf()


def plot_matrix_thresholds_sq(region, plot_base, mx_ary, solar_values, wind_values, save_name, titles=['',]):

    print(f"Plotting: {save_name}")

    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(figsize=(7, 6), ncols=2, nrows=2, sharey=True, sharex=True)

    # Contours
    min_and_max = []
    if 'RL_mean' in save_name:
        n_levels = np.arange(0,200,10)
        c_fmt = '%3.0f'
        ylab = "$\mu$ peak residual load\n(% mean load)"
        min_and_max = [100, 170]
    elif 'RL_std' in save_name:
        n_levels = np.arange(0,15,0.5)
        c_fmt = '%1.1f'
        ylab = "$\sigma$ peak residual load\n(% mean load)"
        min_and_max = [4, 10]
    elif 'RL_50pct' in save_name:
        n_levels = np.arange(0,50,2.5)
        c_fmt = '%1.1f'
        ylab = "mid-50% range peak residual load\n(% mean load)"
    elif 'RL_95pct' in save_name:
        n_levels = np.arange(0,50,5)
        c_fmt = '%1.1f'
        ylab = "mid-95% range peak residual load\n(% mean load)"
    elif 'RL_Mto97p5pct' in save_name:
        n_levels = np.arange(0,50,2.5)
        c_fmt = '%1.1f'
        ylab = "50-97.5% range peak residual load\n(% mean load)"
    elif 'PL_mean' in save_name:
        n_levels = np.arange(-100,200,20)
        c_fmt = '%3.0f'
        ylab = "$\mu$ residual load of peak\nload hours (% mean load)"
    elif 'PL_std' in save_name:
        n_levels = np.arange(0,100,5)
        c_fmt = '%1.0f'
        ylab = "$\sigma$ residual load of peak\nload hours (% mean load)"
    elif '_mean' in save_name: # else if so gets solar and wind means
        n_levels = np.arange(0,200,10)
        c_fmt = '%3.0f'
        app = 'wind' if 'wind' in save_name else 'solar'
        ylab = f"$\mu$ {app} capacity factor\n(during peak residual load hours)"
        if '_solar' in save_name:
            min_and_max = [0, 70]
        if '_wind' in save_name:
            min_and_max = [0, 50]

    elif '_inter' in save_name:
        n_levels = np.arange(0,20,1)
        c_fmt = '%3.1f'
        ylab = "inter-annual variability (% mean load)"
        min_and_max = [3, 10]
        if 'Nom' in save_name:
            n_levels = np.arange(-3,3,.2)
            c_fmt = '%3.2f'
            min_and_max = [-3, 3]
    elif '_intra' in save_name:
        n_levels = np.arange(0,20,1)
        c_fmt = '%3.1f'
        ylab = "intra-annual variability\n(% mean load)"
        min_and_max = [2, 8]
    elif 'QuadR' in save_name:
        n_levels = np.arange(-10,10,1)
        c_fmt = '%1.1f'
        ylab = "$\sigma$ residual load of peak\nload hours (% mean load)"
        #min_and_max = [-6, 1]

    # Clip colormap before yellow high values so white
    # contour text shows up.
    cmapBig = matplotlib.cm.get_cmap('plasma', 512)
    top = 0.85
    cmapShort = matplotlib.colors.ListedColormap(cmapBig(np.linspace(0.0, top, int(512*top))))

    ims = []
    css = []
    cnt = 0
    for matrix, title in zip(mx_ary, titles):
        if len(min_and_max) == 0:
            im = axs.flatten()[cnt].imshow(matrix, interpolation='none', origin='lower', cmap=cmapShort)
        else:
            im = axs.flatten()[cnt].imshow(matrix, interpolation='none', origin='lower', vmin=min_and_max[0], vmax=min_and_max[1], cmap=cmapShort)
        ims.append(im)

        cs = axs.flatten()[cnt].contour(matrix, n_levels, colors='w')
        css.append(cs)
        # inline labels
        axs.flatten()[cnt].clabel(cs, inline=1, fontsize=12, fmt=c_fmt)

        wind_labs, solar_labs = [], []
        for v in wind_values:
            if int(v*4)==v*4:
                wind_labs.append(f"{int(v*100)}%")
            else:
                wind_labs.append('')
        for v in solar_values:
            if int(v*4)==v*4:
                solar_labs.append(f"{int(v*100)}%")
            else:
                solar_labs.append('')
        axs.flatten()[cnt].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 25, 50, 75, 100]))
        axs.flatten()[cnt].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
        axs.flatten()[cnt].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 25, 50, 75, 100]))
        axs.flatten()[cnt].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
        #my_ticks = np.arange(0, 101, 5)
        #my_labs = ['' for _ in range(len(my_ticks))]
        #my_labs[0] = '0%'
        #my_labs[5] = '25%'
        #my_labs[10] = '50%'
        #my_labs[15] = '75%'
        #my_labs[20] = '100%'
        #axs.flatten()[cnt].set_xticks(my_ticks, my_labs)
        #axs.flatten()[cnt].set_yticks(my_ticks, my_labs)
        ##axs.flatten()[cnt].set_xticks(range(len(wind_values)), wind_labs, rotation=90)
        plt.setp( axs.flatten()[cnt].xaxis.get_majorticklabels(), rotation=40 )
        # Adding X label below with one big overlaid axis
        #axs.flatten()[cnt].set_xlabel("wind generation\n(% mean load)")
        #if cnt == 0:
        #    axs.flatten()[cnt].set_ylabel("solar generation (% mean load)")
        #else:
        if cnt != 0:
            solar_labs = ['' for i in range(len(solar_values))]
        #axs.flatten()[cnt].set_yticks(range(len(solar_values)), solar_labs)
        if cnt + 1 == len(axs):
            #cbar = axs.flatten()[cnt].figure.colorbar(im)
            #cbar.ax.set_ylabel(ylab)
            #dec = 0
            #cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.86, 0.16+.05, 0.02, 0.94 - .16 -.1])
            cbar = fig.colorbar(ims[-1], cax=cbar_ax)
            cbar.ax.set_ylabel(ylab)
            dec = 0
            cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))

        axs.flatten()[cnt].set_title(title)
        cnt += 1
    #plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.25, right=0.88, top=0.9)
    plt.subplots_adjust(left=0.11, bottom=0.16, top=0.94)

    # SHARED X AXIS LABEL
    # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots/53172335
    # add a big axis, hide frame
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("wind generation (% mean load)", labelpad=25)
    ax.set_ylabel("solar generation (% mean load)", labelpad=10.5)

    plt.savefig(f"{plot_base}/{region}_{save_name}.{TYPE}")
    plt.clf()

    ## Make empty plots
    #plt.close()
    #fig, ax = plt.subplots()
    #m_nan = copy.deepcopy(matrix)
    #for i in range(len(m_nan)):
    #    for j in range(len(m_nan[i])):
    #        m_nan[i][j] = np.nan
    #cb_range = [np.min(matrix), np.max(matrix)]
    #im = ax.imshow(m_nan,interpolation='none',origin='lower',vmin=cb_range[0],vmax=cb_range[1])
    #plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    #plt.yticks(range(len(solar_values)), solar_labs)
    #plt.xlabel("wind generation\n(% mean load)")
    #plt.ylabel("solar generation\n(% mean load)")
    #cbar = ax.figure.colorbar(im)
    #cbar.ax.set_ylabel(ylab)
    #cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))
    #plt.title(f"")
    ##plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.25, right=0.88, top=0.97)
    #plt.savefig(f"{plot_base}/{region}_{save_name}_empty.{TYPE}")
    #plt.clf()

# Default regions for running `./make_basic_scan_plot.py`
region = 'CONUS'
region = 'NYISO'
region = 'ERCOT'
#region = 'PJM'


print(f"\nRunning {sys.argv[0]}")
print(f"Input arg list {sys.argv}")

if len(sys.argv) > 1:
    region = sys.argv[1]


if len(sys.argv) > 2:
    DATE = sys.argv[2]
else:
    DATE = '20210115v1'


if len(sys.argv) > 3:
    METHOD = sys.argv[3]
else:
    METHOD = "Nom"


if len(sys.argv) > 4:
    HOURS_PER_YEAR = int(sys.argv[4])
else:
    HOURS_PER_YEAR = 20

if len(sys.argv) > 5:
    N_YEARS = int(sys.argv[5])
else:
    N_YEARS = -1


if len(sys.argv) > 6:
    extra = sys.argv[6]


print(f"Region: {region}")
print(f"Date: {DATE}")
print(f"Method: {METHOD}")
print(f"Peak Hours: {HOURS_PER_YEAR}")


### HERE

TYPE = 'png'
TYPE = 'pdf'



# Define scan space by "Total X Generation Potential" instead of installed Cap
solar_max = 1.
wind_max = 1.
steps = 101

solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)

plot_base = f'plots/_plots_{steps}x{steps}_{extra}'
if not os.path.exists(plot_base):
    os.makedirs(plot_base)



mapper = OrderedDict()
mapper['nom'] = [DATE, 'NOM', '', region]
mapper['detrend'] = [DATE, 'DT', '', region]
#mapper['TMY'] = [DATE, 'TMY', '_TMY', region]
#mapper['plus1'] = [DATE, 'PLUS1', '', region]

if region == 'ALL':
    mapper = OrderedDict()
    mapper['ERCOT'] = [DATE, 'DT', '', 'ERCOT']
    mapper['PJM'] = [DATE, 'DT', '', 'PJM']
    mapper['NYISO'] = [DATE, 'DT', '', 'NYISO']
    mapper['France'] = [DATE, 'DT', '', 'FR']



ms = OrderedDict() # Matrices
ms['RL_mean'] = OrderedDict()
ms['wind_mean'] = OrderedDict()
ms['solar_mean'] = OrderedDict()
ms['inter'] = OrderedDict()

for name, info in mapper.items():
    print(name, info)
    pkl_file = f'pkls/pkl_{info[0]}{info[1]}_{steps}x{steps}_{info[3]}_hrs{HOURS_PER_YEAR}_nYrs{N_YEARS}{info[2]}'
    
    
    
    print(f"Opening {pkl_file}.pkl")
    pickle_in = open(f'{pkl_file}.pkl','rb')
    study_regions = pickle.load(pickle_in)
    
    
    
    m_rl_mean = []
    #m_rl_mean, m_rl_std = [], [] # Mean Residual Load, STD RL
    #m_rl_50pct, m_rl_95pct = [], [] # Other spreads for RL
    #m_rl_Mto97p5pct = []
    m_w_mean, m_s_mean = [], [] # mean wind CFs, mean solar CFs
    #m_pl_mean, m_pl_std = [], [] # Mean residual load of the original peak load values
    #intra, inter = [], [] # Interannual vs. intra-annual variability check
    inter = [] # Interannual variability check
    for i, solar_install_cap in enumerate(solar_gen_steps):
        solar_gen = solar_gen_steps[i]
        m_rl_mean.append([])
        #m_rl_std.append([])
        #m_rl_50pct.append([])
        #m_rl_95pct.append([])
        #m_rl_Mto97p5pct.append([])
        m_w_mean.append([])
        m_s_mean.append([])
        #m_pl_mean.append([])
        #m_pl_std.append([])
        #intra.append([])
        inter.append([])
        for j, wind_install_cap in enumerate(wind_gen_steps):
            wind_gen = wind_gen_steps[j]
            rls = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][0]
            m_rl_mean[i].append(np.mean(rls)*100)
            #m_rl_std[i].append(np.std(rls)*100)
            #m_rl_50pct[i].append( (np.percentile(rls, 75) - np.percentile(rls, 25))*100)
            #m_rl_95pct[i].append( (np.percentile(rls, 97.5) - np.percentile(rls, 2.5))*100)
            #m_rl_Mto97p5pct[i].append( (np.percentile(rls, 97.5) - np.mean(rls))*100)
            w_cfs = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1]
            s_cfs = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2]
            m_w_mean[i].append(np.mean(w_cfs)*100)
            m_s_mean[i].append(np.mean(s_cfs)*100)
            #pls = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][3]
            #m_pl_mean[i].append(np.mean(pls)*100)
            #m_pl_std[i].append(np.std(pls)*100)
            #intra[i].append(np.mean(study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][4])*100.)
            inter[i].append(np.std(study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][5])*100.)

    ms['inter'][name] = np.array(inter)
    ms['RL_mean'][name] = np.array(m_rl_mean)
    ms['solar_mean'][name] = np.array(m_s_mean)
    ms['wind_mean'][name] = np.array(m_w_mean)

#plot_matrix_thresholds(region, plot_base, m_rl_mean, solar_gen_steps, wind_gen_steps, f'top_20_RL_mean')
#plot_matrix_thresholds(region, plot_base, m_rl_std, solar_gen_steps, wind_gen_steps, f'top_20_RL_std')
#plot_matrix_thresholds(region, plot_base, m_w_mean, solar_gen_steps, wind_gen_steps, f'top_20_wind_mean')
#plot_matrix_thresholds(region, plot_base, m_s_mean, solar_gen_steps, wind_gen_steps, f'top_20_solar_mean')
#plot_matrix_thresholds(region, plot_base, intra, solar_gen_steps, wind_gen_steps, f'top_20_intra')
#plot_matrix_thresholds(region, plot_base, inter, solar_gen_steps, wind_gen_steps, f'top_20_inter')



# Normal plots
#plot_matrix_thresholds(region, plot_base, ms['nom'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_NOM', f'{region}: annual norm.')
#if 'TMY' in mapper.keys():
#    plot_matrix_thresholds(region, plot_base, ms['TMY'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_TMY')
#    plot_matrix_thresholds(region, plot_base, ms['TMY'] - ms['nom'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_TMY-Nom')
#if 'plus1' in mapper.keys():
#    plot_matrix_thresholds(region, plot_base, ms['plus1'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_Plus1')
#    plot_matrix_thresholds(region, plot_base, ms['plus1'] - ms['nom'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_Rand-Nom')
#if 'detrend' in mapper.keys():
#    plot_matrix_thresholds(region, plot_base, ms['detrend'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_DT', f'{region}: detrended')
#    plot_matrix_thresholds(region, plot_base, ms['detrend'] - ms['nom'], solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_inter_DT-Nom', f'{region}: detrended - annual norm.')


#idx_range = [0, 50]
#for alt_idx in [0, 25, 50]:
#    for resource in ['wind', 'solar']:
#        plot_matrix_slice(region, plot_base, ms, idx_range, alt_idx, f'alt_idx_{alt_idx}', resource)


# Testing array of inputs
if region == 'ALL':
    #ms_ary = [ ms['nom'], ms['detrend'],]# ms['detrend'] - ms['nom'] ]
    #titles = [ f'{region}: annual norm.', f'{region}: detrended',]# f'{region}: detrended - annual norm.' ]
    for k1, v1 in ms.items():
        ms_ary, titles = [], []
        for k, v in v1.items():
            titles.append(k)
            ms_ary.append(v)
        plot_matrix_thresholds(region, plot_base, ms_ary, solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_{k1}_ALL', titles)
        if 'inter' in k1:
            plot_matrix_thresholds_sq(region, plot_base, ms_ary, solar_gen_steps, wind_gen_steps, f'top_{HOURS_PER_YEAR}_{k1}_ALL_sq', titles)





