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

sys.path.append("/home/truggles/Inter-annual_Variability_Residual_Load")
from helpers import return_file_info_map




def get_peak_demand_hour_indices(df):

    df_tmp = df.sort_values(by=['demand'], ascending=False)
    return df_tmp.iloc[:20:].index






def get_dem_wind_solar(im, DEM_METHOD, RES_METHOD):

    rep = '.csv'
    if RES_METHOD == 'TMY':
        rep = '_TMY2.csv' # if wanting TMY, this will change the file name
    rep_load = '.csv'
    if DEM_METHOD == 'DT':
        rep_load = '_expDT.csv'
    demand = pd.read_csv('../'+im['demand'][0].replace('.csv', rep_load), header=im['demand'][1])
    wind = pd.read_csv('../'+im['wind'][0].replace('.csv', rep), header=im['wind'][1])
    solar = pd.read_csv('../'+im['solar'][0].replace('.csv', rep), header=im['solar'][1])

    return demand, wind, solar


def get_renewable_fraction(year, wind_install_cap, solar_install_cap, im, demand, wind, solar, zero_negative=True):
    
    d_profile = demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ].values
    w_profile = wind.loc[ wind[im['wind'][3]] == year, im['wind'][2] ].values
    s_profile = solar.loc[ solar[im['solar'][3]] == year, im['solar'][2] ].values

    final_profile = d_profile - wind_install_cap * w_profile - solar_install_cap * s_profile
    if zero_negative:
        final_profile = np.where(final_profile >= 0, final_profile, 0.)

    return np.mean(d_profile) - np.mean(final_profile)





def plot_matrix_thresholds(region, plot_base, matrix, solar_values, wind_values, save_name):

    print(f"Plotting: {save_name}")

    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()#figsize=(4.5, 4))

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
        ylab = "inter-annual variability\n(% mean load)"
        min_and_max = [3, 10]
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

    if len(min_and_max) == 0:
        im = ax.imshow(matrix, interpolation='none', origin='lower', cmap=cmapShort)
    else:
        im = ax.imshow(matrix, interpolation='none', origin='lower', vmin=min_and_max[0], vmax=min_and_max[1], cmap=cmapShort)

    cs = ax.contour(matrix, n_levels, colors='w')
    # inline labels
    ax.clabel(cs, inline=1, fontsize=12, fmt=c_fmt)

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
    plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    plt.yticks(range(len(solar_values)), solar_labs)
    plt.xlabel("wind generation\n(% mean load)")
    plt.ylabel("solar generation\n(% mean load)")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(ylab)
    dec = 0
    #if region == 'NYISO' or '_inter' in save_name:
    #    dec = 1
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=dec))
    plt.title(f"")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.14, bottom=0.25, right=0.88, top=0.97)


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


def get_avg_CF(dfs, name, im):
    to_avg = []
    for year, df in dfs.items():
        avg = np.mean(df[name])
        #print(year, len(df.index), name, avg)
        to_avg.append(avg)
    return np.mean(to_avg)


def get_annual_CF(df, name, im, year):

    return np.mean(df.loc[ df[im[name][3]] == year, im[name][2] ].values)







def get_annual_df(year, df, tgt, im, DEM_METHOD):

    df2 = df.loc[ df[ im[tgt][3]] == year ].copy()

    # Normalize
    if tgt == 'demand' and DEM_METHOD != 'DT':
        df2.loc[:, im[tgt][2]] = df2.loc[:, im[tgt][2]]/np.mean(df2.loc[:, im[tgt][2]])
    return df2


def return_ordered_df(demand, wind, solar, im):

    #rank_mthd='ordinal'
    rank_mthd='min'
    to_map = OrderedDict()
    to_map['month'] = demand['month'].values
    to_map['day'] = demand['day'].values
    to_map['hour'] = demand['hour'].values
    to_map['demand'] = demand[im['demand'][2]].values
    to_map['wind'] = wind[im['wind'][2]].values
    to_map['solar'] = solar[im['solar'][2]].values

    df = pd.DataFrame(to_map)
    return df


def get_range(vect):
    return np.max(vect) - np.min(vect)
        
# for 94% confidence in achieving X reliability goal (94% = 15years/16year based on ERCOT)
def get_2nd_highest(vect):
    vect.sort()
    return vect[-2] - np.mean(vect) # 2nd from end.  Sort defaults to ascending order.



# Return the position of the integrated threshold based on total demand.
# Total demand is normalized and is 8760 or 8784 based on leap years.
# Integrate down from the max values.
def get_integrated_threshold(vals, threshold_pct):

    int_threshold = len(vals) * (1. - threshold_pct)
    int_tot = 0.
    hours = 0
    prev_val = vals[-1] # to initialize
    for i, val in enumerate(reversed(vals)):
        current = hours * (prev_val - val)
        #print(f"{i} --- Running total: {round(int_tot,5)}   Hours: {hours}   Current val {round(val,5)}   To add? {round(current,5)}")
        if current + int_tot < int_threshold:
            hours += 1
            prev_val = val
            int_tot += current
            continue

        # Else, we overshoot the target
        # Find location between values which would meet target
        tot_needed = int_threshold - int_tot
        # Find fraction of 'current' needed
        frac = tot_needed / current
        # Return that frac as a distance between val i and val i-1 (going bkwards)
        dist = prev_val - val
        to_return = prev_val - dist * frac
        #print(f"  === tot_needed {tot_needed}   frac {frac}   dist {dist}   to_return {to_return}")
        return to_return



def load_duration_curve_and_PDF_plots(dfs, save_name, wind_install_cap, solar_install_cap, cnt, base, gens=[0, 0], threshold_pcts=[]):

    matplotlib.rcParams.update({'font.size': 14})
    plt.close()
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,5))
    #fig.suptitle(f'Dem - wind CF x {round(wind_install_cap,2)} - solar CF x {round(solar_install_cap,2)}')
    axs[0].set_title(f'Residual Load Duration Curve')
    axs[1].set_title(f'PDF of Residual Load')

    good_max = 0
    threshold_vals = []
    lw = 1.5
    for year, df in dfs.items():
        mod_df = df
        mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
        mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)
        axs[0].plot(np.linspace(0,100,len(mod_df.index)), mod_df['mod_dem']*100, linestyle='-', linewidth=lw)
        to_bins = np.linspace(-10*100,2*100,601)
        n, bins, patches = axs[1].hist(mod_df['mod_dem']*100, to_bins, orientation='horizontal', histtype=u'step', color=axs[0].lines[-1].get_color(), linewidth=lw)
        if np.max(n) > good_max:
            good_max = np.max(n)

        # If threshold_pcts has unique values
        for i, t in enumerate(threshold_pcts):
            vals = df['demand'].values - df['solar'].values * solar_install_cap - df['wind'].values * wind_install_cap
            vals.sort()
            pct = get_integrated_threshold(vals, t)
            if i == 0:
                #print(f"Adding threshold lines for {t}")
                threshold_vals.append(pct)
                axs[0].plot(np.linspace(-0.1,100,10), [pct*100 for _ in range(10)], color=axs[0].lines[-1].get_color(), linestyle='-', linewidth=0.5) 
                axs[1].plot(np.linspace(0,1000,10), [pct*100 for _ in range(10)], color=axs[0].lines[-1].get_color(), linestyle='-', linewidth=0.5) 
    if len(threshold_vals) > 0:
        plt.text(0.55, 0.53, f'range: {(round(np.max(threshold_vals) - np.min(threshold_vals),3))*100}%\n$\sigma$: {round(np.std(threshold_vals)*100,2)}%',
                horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, fontsize=14)


    plt.subplots_adjust(wspace=0.4)
    #axs[0].yaxis.grid(True)
    axs[0].set_xlim(-0.5, 100)
    #axs[0].set_ylim(0, axs[0].get_ylim()[1])
    axs[0].set_ylim(0, 200)
    axs[0].set_ylabel('residual load\n(% mean load)')
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    axs[0].set_xlabel('operating duration (% of year)')
    #axs[1].yaxis.grid(True)
    axs[1].set_ylabel('residual load\n(% mean load)')
    axs[1].set_xlabel('hours / bin')
    axs[1].set_xlim(0, good_max * 1.2)
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    #axs[1].set_ylim(0, axs[1].get_ylim()[1])
    axs[1].set_ylim(0, 200)
    axs[1].yaxis.set_tick_params(labelleft=True)
    plt.savefig(f"{base}/{save_name}_LDC_and_PDF_cnt{cnt:05}_solarGen{str(round(gens[1],4)).replace('.','p')}_windGen{str(round(gens[0],4)).replace('.','p')}.{TYPE}")


def PDF_plots(dfs, save_name, wind_install_cap, solar_install_cap, cnt, base, gens=[0, 0], threshold_pcts=[]):

    matplotlib.rcParams.update({'font.size': 14})
    plt.close()
    fig, ax = plt.subplots()

    good_max = 0
    threshold_vals = []
    lw = 1.5
    for year, df in dfs.items():
        mod_df = df
        mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
        mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)
        to_bins = np.linspace(0,200,101)
        n, bins, patches = ax.hist(mod_df['mod_dem']*100, to_bins, histtype=u'step', linewidth=lw)
        if np.max(n) > good_max:
            good_max = np.max(n)



    plt.subplots_adjust(wspace=0.4)
    ax.set_xlabel('residual load\n(% mean load)')
    ax.set_ylabel('hours / bin')
    #ax.set_ylim(0, good_max * 1.2)
    ax.set_ylim(0, 600)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 )
    ax.set_xlim(0, 200)
    ax.xaxis.set_tick_params(labelleft=True)
    plt.tight_layout()
    plt.savefig(f"{base}/{save_name}_PDF_cnt{cnt:05}_solarGen{str(round(gens[1],4)).replace('.','p')}_windGen{str(round(gens[0],4)).replace('.','p')}.{TYPE}")


def box_to_regression(rl_vects, years):

    rl_vals = []
    yrs = []
    #Unpack rl_vects and years into 1D arrays
    for group, yr in zip(rl_vects, years):
        for item in group:
            yrs.append(yr)
            rl_vals.append(item)


    slope, intercept, r_value, p_value, std_err = stats.linregress(yrs, rl_vals)
    print("slope, intercept, r_value, p_value, std_err")
    print(f"{slope}, {intercept}, {r_value}, {p_value}, {std_err}")

    # Make calibrated new_rls
    new_rls = np.array(rl_vals) - intercept - slope*np.array(yrs) + np.mean(rl_vals)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(yrs, new_rls)
    #print("slope, intercept, r_value, p_value, std_err")
    #print(f"{slope2}, {intercept2}, {r_value2}, {p_value2}, {std_err2}")
    
    new_rls_map = []
    hrs = len(rl_vects[0])
    for i, val in enumerate(new_rls):
        if i%hrs==0:
            new_rls_map.append([])
        new_rls_map[-1].append(val)

    return slope, intercept, r_value, p_value, std_err, new_rls_map


def plot_rl_box(rl_vects, years, save_name, wind_install_cap, solar_install_cap, cnt, base, gens=[0, 0], **kwargs):

    #detrend_peak: slope, intercept, r_value, p_value, std_err, new_rls_map = box_to_regression(rl_vects, years)
    plt.close()
    matplotlib.rcParams.update({'font.size': 16.5})
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7.5,6),
                       gridspec_kw={
                           'width_ratios': [3, 1]})

    #fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,5))
    #medianprops = dict(linestyle='-', linewidth=2.5)
    medianprops = dict(linestyle='-', linewidth=0.0)
    meanprops = dict(linestyle='-', linewidth=2.5, color='C1', markersize=0)
    #bplot = axs[0].boxplot(rl_vects, whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops)
    bplot = axs[0].boxplot(rl_vects, whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops, meanline=True, showmeans=True, meanprops=meanprops)
    #axs[0].set_xtick([i for i in range(1, len(years)+1)], years, rotation=90)
    axs[0].set_xticklabels(years)
    plt.setp(axs[0].get_xticklabels(), rotation=90)
    axs[0].yaxis.grid(True)
    axs[0].set_ylabel('peak residual load\n(% mean load)')
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    axs[0].set_ylim(0.8, 2.)
    if 'hourly' in kwargs:
        axs[0].set_ylim(0.8, 2.2)
    #detrend_peak: axs[0].plot(np.array(years) - years[0] + 1, intercept + slope*np.array(years), 'r', label='fitted line')
    #axs[0].set_ylim(1, 2)
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
    stds = []
    mus = []
    for row in rl_vects:
        stds.append(np.std(row))
        mus.append(np.mean(row))

    if not 'hourly' in kwargs:
        axs[0].plot( np.linspace(0.5, len(rl_vects) + 0.5, 100 ), np.ones(100), 'k--', label='mean load')
        axs[0].legend(loc='lower center')
    
    #bplot2 = axs[1].boxplot(np.array(rl_vects).flatten(), whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops)
    bplot2 = axs[1].boxplot(np.array(rl_vects).flatten(), whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops, meanline=True, showmeans=True, meanprops=meanprops)
    axs[1].set_xticklabels([f'{years[0]}-{years[-1]}',])
    axs[1].yaxis.grid(True)
    #axs[0].yaxis.grid(True)
    for patch in bplot2['boxes']:
        patch.set_facecolor('lightblue')

    if not 'hourly' in kwargs:
        axs[1].plot( np.linspace(0.5, 1.5, 100 ), np.ones(100), 'k--', label='mean load')

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat')
    
    mt = r'$\mu_{RL}$'
    st = r'$\sigma_{tot}$'
    st2 = r'$\sigma_{intra}$'
    st3 = r'$\sigma_{inter}$'
    #textstr = '\n'.join((
    #    f'{mt} = {np.mean(np.array(rl_vects).flatten())*100:.0f}%        {st} = {np.std(np.array(rl_vects).flatten())*100:.2f}%',
    #    f'{st2} = {np.std(mus)*100:.2f}%    {st3} = {np.mean(stds)*100:.2f}%',
    #    ))

    # place a text box in upper left in axes coords
    #axs[0].text(2, 1.98, textstr, fontsize=16.5,
    #        verticalalignment='top', bbox=props)

    textstr1 = '\n'.join((
        #f'{st2} = {np.mean(stds)*100:.3g}%',
        f'{st3} = {np.std(mus)*100:.3g}%',
        ))
    textstr2 = '\n'.join((
        f'{mt} = {np.mean(np.array(rl_vects).flatten())*100:.3g}%',
        #f'{st} = {np.std(np.array(rl_vects).flatten())*100:.3g}%',
        ))


    plus = 0 if not 'hourly' in kwargs else 0.95
    axs[0].text(5, 1.25+plus, textstr1, fontsize=18,
        verticalalignment='top', bbox=props)
    axs[1].text(0.35, 1.25+plus, textstr2, fontsize=18,
        verticalalignment='top', bbox=props)

    f_wind = r'f$_{wind}$'
    f_solar = r'f$_{solar}$'
    plt.suptitle(f"{region}: {f_wind} = {int(round(gens[0],4)*100)}%, {f_solar} = {int(round(gens[1],4)*100)}%")
    #plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.93)
    plt.subplots_adjust(left=0.18, bottom=0.13, right=0.97, top=0.90)
    plt.savefig(f"{base}/{save_name}_RL_box_cnt{cnt:05}_solarGen{str(round(gens[1],4)).replace('.','p')}_windGen{str(round(gens[0],4)).replace('.','p')}.{TYPE}")
    #detrend_peak: return new_rls_map

    #n, bins, patches = ax.hist(peak_load_original, n_bins, range=[min_x, 2], alpha=0.5, label=f'peak load: $\mu$ = {np.mean(peak_load_original):.2f} $\sigma$ = {np.std(peak_load_original):.3f}')

# Make box plots showing the wind and solar CFs for the top X thresholds
def make_box_plots(dfs, save_name, wind_install_cap, solar_install_cap, box_thresholds, cnt, base, gens=[0, 0]):

    to_plot = [[] for _ in range(len(box_thresholds)*2)]
    for i, threshold in enumerate(box_thresholds):
        for year, df in dfs.items():
            mod_df = df
            mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
            mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)
            for j, idx in enumerate(mod_df.index):
                to_plot[i].append(mod_df.loc[idx, 'wind'])
                to_plot[len(box_thresholds)+i].append(mod_df.loc[idx, 'solar'])
                if j == threshold: break

    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(5,5))
    ax.yaxis.grid(True)
    #ax.set_title(f'Dem - wind CF x {round(wind_install_cap,2)} - solar CF x {round(solar_install_cap,2)}: whiskers at 5%/95%')
    medianprops = dict(linestyle='-', linewidth=2.5)
    bplot = ax.boxplot(to_plot, whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops)
    x_labels = []
    for val in box_thresholds:
        x_labels.append(f'Wind:\nTop {val} Hours')
    for val in box_thresholds:
        x_labels.append(f'Solar:\nTop {val} Hours')
    plt.xticks([i for i in range(1, len(box_thresholds)*2+1)], x_labels, rotation=30)
    ax.set_ylabel('resource capacity factors')
    ax.set_ylim(0, 1)
    plt.title(f"solar generation: {int(round(gens[1],4)*100)}%")
    plt.tight_layout()
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
    plt.savefig(f"{base}/{save_name}_CFs_box_plot_cnt{cnt:05}_solarGen{str(round(gens[1],4)).replace('.','p')}_windGen{str(round(gens[0],4)).replace('.','p')}.{TYPE}")


def make_threshold_hist(vect, save_name, cnt, base, gens):

    ary = np.array(vect)*100
    mean = np.mean(ary)
    std = np.std(ary)
    #for bin_w in [.1, .25, .5, 1, 2, 2.5, 5]:
    for bin_w in [1,]:
        plt.close()
        matplotlib.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots()
        n1, bins1, patches1 = ax.hist(ary, np.arange(150, 185, bin_w), facecolor='k', alpha=0.5, label='threshold\npositions') # histtype=u'step', linewidth=4)
        y_lim = ax.get_ylim()[1]
        ax.plot(np.ones(10)*(mean), np.linspace(0, y_lim*1.2, 10), 'r--', label=f'mean: {round(mean,1)}%') # histtype=u'step', linewidth=4)
        ax.plot(np.ones(10)*(mean+std), np.linspace(0, y_lim*1.2, 10), 'b--', label=f'$\sigma$: {round(std,1)}%') # histtype=u'step', linewidth=4)
        # Below values are w.r.t. to ERCOT's 2019 mean load of 44 GW
        #ax.plot(np.ones(10)*(mean), np.linspace(0, y_lim*1.2, 10), 'r--', label=f'mean: {round(mean,1)}% (73 GW)') # histtype=u'step', linewidth=4)
        #ax.plot(np.ones(10)*(mean+std), np.linspace(0, y_lim*1.2, 10), 'b--', label=f'$\sigma$: {round(std,1)}% (1.5 GW)') # histtype=u'step', linewidth=4)
        ax.plot(np.ones(10)*(mean-std), np.linspace(0, y_lim*1.2, 10), 'b--', label='_nolabel_') # histtype=u'step', linewidth=4)

        plt.legend()
        ax.set_xlim(150,180)
        ax.set_ylim(0,6)
        ax.set_xlabel(f'Demand - VRE\n(% Mean Demand)')
        ax.set_ylabel(f'Entries / Bin')
        plt.tight_layout()
        plt.savefig(f"{base}/{save_name}_threshold_hist_{bin_w}_cnt{cnt:05}_solarGen{str(round(gens[1],4)).replace('.','p')}_windGen{str(round(gens[0],4)).replace('.','p')}.{TYPE}")




# Return top 20 values for: residual load (rls), wind/solar CFs (w/s_cfs), and
# residual load values for peak X hours (pls)
def get_top_X_per_year(hours_per_year, years_used, dfs, peak_indices, wind_install_cap, solar_install_cap, PRINT=False):
    first = True
    rl_vects = []
    #for year, df in dfs.items():
    for year in years_used:
        df = dfs[year]
        df['RL'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
        df_sort = df.sort_values(by=['RL'], ascending=False)

        if PRINT and year == 2017:
            return df_sort.iloc[:hours_per_year:].index.tolist()


        rls_tmp = df_sort.iloc[:hours_per_year:]['RL'].values
        rl_vects.append(list(rls_tmp))
        if first:
            first = False
            rls = df_sort.iloc[:hours_per_year:]['RL'].values
            w_cfs = df_sort.iloc[:hours_per_year:]['wind'].values
            s_cfs = df_sort.iloc[:hours_per_year:]['solar'].values
            hours = df_sort.iloc[:hours_per_year:]['hour'].values
            pls = df_sort.loc[ peak_indices[year], 'RL' ].values
        else:
            rls = np.append(rls, df_sort.iloc[:hours_per_year:]['RL'].values)
            w_cfs = np.append(w_cfs, df_sort.iloc[:hours_per_year:]['wind'].values)
            s_cfs = np.append(s_cfs, df_sort.iloc[:hours_per_year:]['solar'].values)
            hours = np.append(hours, df_sort.iloc[:hours_per_year:]['hour'].values)
            pls = np.append(pls, df_sort.loc[ peak_indices[year], 'RL' ].values)
    stds = []
    mus = []
    maxs = []
    for row in rl_vects:
        stds.append(np.std(row))
        mus.append(np.mean(row))
        maxs.append(np.max(row))
    return rls, w_cfs, s_cfs, pls, rl_vects, stds, mus, hours, maxs







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
    DEM_METHOD = sys.argv[3]
else:
    DEM_METHOD = "DT" # DT (detrended is now the default method)


if len(sys.argv) > 4:
    RES_METHOD = sys.argv[4]
else:
    RES_METHOD = "NOM"


if len(sys.argv) > 5:
    HOURS_PER_YEAR = int(sys.argv[5])
else:
    HOURS_PER_YEAR = 20

if len(sys.argv) > 6:
    N_YEARS = int(sys.argv[6])
else:
    N_YEARS = -1


if len(sys.argv) > 7:
    TEST_SENSITIVITY = True
else:
    TEST_SENSITIVITY = False

print(f"Region: {region}")
print(f"Date: {DATE}")
print(f"Demand Method: {DEM_METHOD}")
print(f"Resource Method: {RES_METHOD}")
print(f"Peak Hours: {HOURS_PER_YEAR}")
print(f"N Years: {N_YEARS}")
print(f"Test Sensitivity: {TEST_SENSITIVITY}")



### HERE

TYPE = 'png'
TYPE = 'pdf'

assert(DEM_METHOD in ['NOM', 'DT']), f"You selected an invalide Demand method (DEM_METHOD): {DEM_METHOD}"
assert(RES_METHOD in ['NOM', 'TMY', 'PLUS1']), f"You selected an invalide Resource method (RES_METHOD): {RES_METHOD}"

########################### METHODS ##########################
#                                                            #
### Different Demand Methods (DEM_METHODS) ###               #
# NOM = annual normalization                                 #
# DT = exponential detrending of load data                   #
#                                                            #
### Different Resource Methods (RES_METHODS) ###             #
# NOM = nominal wind and solar resources                     #
# TMY = wind and solar profile averaged over many years      #
# PLUS1 = wind and solar profiles from the following year    #
##############################################################

if RES_METHOD == 'TMY':
    print("You must first create the TMY resource files with 'prep_TMY_wind_and_solar_profiles.ipynb'")

# These have now been committed b/c they are the default method
#if DEM_METHOD == 'DT':
#    print("You must first create the DT load files with 'sensitivity_analysis/detrend_load_data.ipynb'")

test_ordering = True
#test_ordering = False
make_plots = True
make_plots = False


# Define scan space by "Total X Generation Potential" instead of installed Cap
solar_max = 1.
wind_max = 1.
steps = 101
if TEST_SENSITIVITY:
    solar_max = 0.5
    wind_max = 0.5
    steps = 51

solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)

app = ''
if N_YEARS > 0:
    app += f'_nYrs{N_YEARS}'
plot_base = f'plots/plots_{DATE}_{steps}x{steps}_{region}_hrs{HOURS_PER_YEAR}{app}'
if not os.path.exists(plot_base):
    os.makedirs(plot_base)

pkl_file = f'pkls/pkl_{DATE}_{steps}x{steps}_{region}_hrs{HOURS_PER_YEAR}{app}'

im = return_file_info_map(region)

years = im['years']

# Check 1) if N_YEARS test, and 2) if the request number is greater than the length
# for this region
if N_YEARS > 0 and N_YEARS > len( years ):
    print(f"Skipping N_YEARS = {N_YEARS} for {region} with usable years length = {len(years)}")
    exit()

if test_ordering:
    demand, wind, solar = get_dem_wind_solar(im, DEM_METHOD, RES_METHOD)
    dfs = OrderedDict()
    peak_indices = {}
    print(f"Number of years scanned: {len(years)}")
    #years = [y for y in range(2005, 2009)]

    years_used = []
    #for iii, year in enumerate(years):
    for iii, year in enumerate(reversed(years)):

        if iii >= N_YEARS:
            continue

        years_used.append(year)

        resource_year = year
        # Use an alternate resource year if this is selected
        if RES_METHOD == 'PLUS1':
            if calendar.isleap(year):
                resource_year += 4
                if resource_year > years[-1]:
                    resource_year = years[0]
                    while not calendar.isleap(resource_year):
                        resource_year += 1
            else:
                resource_year += 1
                if resource_year > years[-1]:
                    resource_year = years[0]
                while calendar.isleap(resource_year):
                    resource_year += 1
            print(f"Demand year {year}; resource year {resource_year}; Demand is leap {calendar.isleap(year)}")

        d_yr = get_annual_df(year, demand, 'demand', im, DEM_METHOD)
        w_yr = get_annual_df(resource_year, wind, 'wind', im, DEM_METHOD)
        s_yr = get_annual_df(resource_year, solar, 'solar', im, DEM_METHOD)
        d_yr.reset_index()
        w_yr.reset_index()
        s_yr.reset_index()
        dfs[year] = return_ordered_df(d_yr, w_yr, s_yr, im)
        peak_indices[year] = get_peak_demand_hour_indices(dfs[year])

    years_used.sort()
    print(f"Keys {dfs.keys()}")
    avg_wind_CF = get_avg_CF(dfs, 'wind', im)
    avg_solar_CF = get_avg_CF(dfs, 'solar', im)
    print(f"Avg wind CF: {avg_wind_CF}")
    print(f"Avg solar CF: {avg_solar_CF}")
    wind_cap_steps = np.linspace(0, wind_max/avg_wind_CF, steps)
    solar_cap_steps = np.linspace(0, solar_max/avg_solar_CF, steps)
    print("Wind cap increments:", wind_cap_steps)
    print("Solar cap increments:", solar_cap_steps)






    mapper = OrderedDict()
    for i, solar_install_cap in enumerate(solar_cap_steps):
        solar_gen = solar_gen_steps[i]
        mapper[str(round(solar_gen,2))] = OrderedDict()
    cnt = 0
    rl_cnt = 0
    for j, wind_install_cap in enumerate(wind_cap_steps):
        wind_gen = wind_gen_steps[j]
        print(f"Wind cap {wind_install_cap}, wind gen {wind_gen}")
        for i, solar_install_cap in enumerate(solar_cap_steps):
            cnt += 1
            solar_gen = solar_gen_steps[i]
            if cnt%100 == 0:
                print(f" --- {cnt}, wind gen {wind_gen} solar gen {solar_gen}")



            # Get top 20 peak residual load hours for each combo
            hours_per_year = HOURS_PER_YEAR
            rls, w_cfs, s_cfs, pls, rl_vects, stds, mus, hours, maxs = get_top_X_per_year(hours_per_year, years_used, dfs, peak_indices, wind_install_cap, solar_install_cap)
            mapper[str(round(solar_gen,2))][str(round(wind_gen,2))] = [rls, w_cfs, s_cfs, pls, stds, mus, maxs]



            #if i%20==0 and j%20==0:
            #if i<16 and j==0:
            #if (i<31 and j==0) or (i==30 and j<51) or (i==0 and j==50):
            #if (j<26 and i==0) or (j==25 and i<26) or (j==0 and i==25):
            #if (i<21 and j==0) or (i==20 and j<21) or (i==0 and j==20):
            if (i==0 and j==0) or (i==25 and j==0) or (i==0 and j==25) or (i==25 and j==25) or (i==50 and j==0) or (i==0 and j==50) or (i==50 and j==50):
                vals = get_top_X_per_year(hours_per_year, years_used, dfs, peak_indices, wind_install_cap, solar_install_cap, True)
                vals.sort()
                print(i, j, vals)
                cnts = []
                prev = -1
                length = 1
                for val in vals:
                    if prev == -1:
                        prev = val
                    elif val == prev + 1:
                        length += 1
                        prev = val
                    else:
                        cnts.append(length)
                        length = 1
                        prev = val
                cnts.append(length)
                #if i%2!=0 or j%2!=0:
                #    continue
                rl_cnt += 1
                #PDF_plots(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, cnt, plot_base, [wind_gen, solar_gen])
                plot_rl_box(rl_vects, years_used, f'ordering_{region}', wind_install_cap, solar_install_cap, rl_cnt, plot_base, [wind_gen, solar_gen])
                #detrend_peak: new_rls_map = plot_rl_box(rl_vects, years, f'ordering_{region}', wind_install_cap, solar_install_cap, rl_cnt, plot_base, [wind_gen, solar_gen])
                #detrend_peak: plot_rl_box(new_rls_map, years, f'ordering_{region}_calib', wind_install_cap, solar_install_cap, rl_cnt, plot_base, [wind_gen, solar_gen])
                #box_thresholds = [20,]
                ##make_box_plots(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, box_thresholds, cnt, plot_base, [wind_gen, solar_gen])



            #vect_range, vect_std, vect_mean, vect_2nd_from_top, p_val = make_ordering_plotsX(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, thresholds, int_thresholds, cnt, plot_base, [wind_gen, solar_gen])
            #mapper[str(round(solar_gen,2))][str(round(wind_gen,2))] = [vect_range, vect_std, vect_mean, vect_2nd_from_top, p_val]

            #if wind_gen == 0:
            #    box_thresholds = [20,]
            #    make_box_plots(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, box_thresholds, cnt, plot_base, [wind_gen, solar_gen])



    #print("Solar Wind max_range 100th_range")
    #for solar, info in mapper.items():
    #    for wind, vals in info.items():
    #        print(solar, wind, vals)
        
    pickle_file = open(f'{pkl_file}.pkl', 'wb')
    pickle.dump(mapper, pickle_file)
    pickle_file.close()

    ## Sort plots
    #tgt = plot_base+'/ordering*'

if make_plots:
    print("\nMAKE PLOTS\n")
    pickle_in = open(f'{pkl_file}.pkl','rb')
    study_regions = pickle.load(pickle_in)



    m_rl_mean, m_rl_std = [], [] # Mean Residual Load, STD RL
    m_rl_50pct, m_rl_95pct = [], [] # Other spreads for RL
    m_rl_Mto97p5pct = []
    m_w_mean, m_s_mean = [], [] # mean wind CFs, mean solar CFs
    m_pl_mean, m_pl_std = [], [] # Mean residual load of the original peak load values
    intra, inter = [], [] # Interannual vs. intra-annual variability check
    for i, solar_install_cap in enumerate(solar_gen_steps):
        solar_gen = solar_gen_steps[i]
        m_rl_mean.append([])
        m_rl_std.append([])
        m_rl_50pct.append([])
        m_rl_95pct.append([])
        m_rl_Mto97p5pct.append([])
        m_w_mean.append([])
        m_s_mean.append([])
        m_pl_mean.append([])
        m_pl_std.append([])
        intra.append([])
        inter.append([])
        for j, wind_install_cap in enumerate(wind_gen_steps):
            wind_gen = wind_gen_steps[j]
            if i == 0 and j == 0:
                peak_load_original = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][3]
            rls = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][0]
            m_rl_mean[i].append(np.mean(rls)*100)
            m_rl_std[i].append(np.std(rls)*100)
            m_rl_50pct[i].append( (np.percentile(rls, 75) - np.percentile(rls, 25))*100)
            m_rl_95pct[i].append( (np.percentile(rls, 97.5) - np.percentile(rls, 2.5))*100)
            m_rl_Mto97p5pct[i].append( (np.percentile(rls, 97.5) - np.mean(rls))*100)
            w_cfs = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1]
            s_cfs = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2]
            m_w_mean[i].append(np.mean(w_cfs)*100)
            m_s_mean[i].append(np.mean(s_cfs)*100)
            pls = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][3]
            m_pl_mean[i].append(np.mean(pls)*100)
            m_pl_std[i].append(np.std(pls)*100)
            intra[i].append(np.mean(study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][4])*100.)
            inter[i].append(np.std(study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][5])*100.)
            #if (i%20==0 and j%20==0) or (i<16 and j==0):
            #if (i<31 and j==0) or (i==30 and j<51) or (i==0 and j==50):
            #    if i%2==1 or j%2==1:
            #        continue
            #    save_name = f'triple_hist_cnt{str(int(i*len(wind_gen_steps)+j))}_w{str(round(wind_install_cap,2)).replace(".","p")}_s{str(round(solar_install_cap,2)).replace(".","p")}'

    intra_ary = np.array( intra ).flatten()
    inter_ary = np.array( inter ).flatten()
    mean_ary = np.array( m_rl_mean ).flatten()
    save_csv = True

    quad = np.sqrt( np.power(np.array( intra ), 2) + np.power( np.array( inter ), 2) )
    quad_R = (quad / np.array( m_rl_std ) - 1.) * 100.
    quad_ary = quad.flatten()
    std_ary = np.array( m_rl_std ).flatten()

    print(f"\n\nDummy chi2 = {np.sum( np.power((quad_ary - std_ary), 2) / std_ary )}")

    plot_matrix_thresholds(region, plot_base, m_rl_mean, solar_gen_steps, wind_gen_steps, f'top_20_RL_mean')
    plot_matrix_thresholds(region, plot_base, m_rl_std, solar_gen_steps, wind_gen_steps, f'top_20_RL_std')
    #plot_matrix_thresholds(region, plot_base, m_rl_50pct, solar_gen_steps, wind_gen_steps, f'top_20_RL_50pct')
    #plot_matrix_thresholds(region, plot_base, m_rl_95pct, solar_gen_steps, wind_gen_steps, f'top_20_RL_95pct')
    #plot_matrix_thresholds(region, plot_base, m_rl_Mto97p5pct, solar_gen_steps, wind_gen_steps, f'top_20_RL_Mto97p5pct')
    plot_matrix_thresholds(region, plot_base, m_w_mean, solar_gen_steps, wind_gen_steps, f'top_20_wind_mean')
    plot_matrix_thresholds(region, plot_base, m_s_mean, solar_gen_steps, wind_gen_steps, f'top_20_solar_mean')
    ##plot_matrix_thresholds(region, plot_base, m_pl_mean, solar_gen_steps, wind_gen_steps, f'top_20_PL_mean')
    ##plot_matrix_thresholds(region, plot_base, m_pl_std, solar_gen_steps, wind_gen_steps, f'top_20_PL_std')
    plot_matrix_thresholds(region, plot_base, intra, solar_gen_steps, wind_gen_steps, f'top_20_intra')
    plot_matrix_thresholds(region, plot_base, inter, solar_gen_steps, wind_gen_steps, f'top_20_inter')
    #plot_matrix_thresholds(region, plot_base, quad, solar_gen_steps, wind_gen_steps, f'top_20_RL_stdQ')
    #plot_matrix_thresholds(region, plot_base, quad_R, solar_gen_steps, wind_gen_steps, f'top_20_QuadR')



    # idx 0 = solar idx 1 = wind
    for k, m in {'intra':intra, 'inter':inter, 'var':m_rl_std}.items():
        print(f"{k},{round(m[0][0],2)}%,{round(m[25][0],2)}%,{round(m[25][0] - m[0][0],2)}%,{round(m[0][25],2)}%,{round(m[0][25] - m[0][0],2)}%,{round(m[25][25],2)}%,{round(m[25][25] - m[0][0],2)}%")

    #for int_threshold in int_thresholds:
    #    thresholds.append(int_threshold)
    #for t, threshold in enumerate(thresholds):
    #    print(threshold)

    #    # Std dev of dem - wind - solar
    #    matrix = []
    #    for i, solar_install_cap in enumerate(solar_gen_steps):
    #        matrix.append([])
    #        for j, wind_install_cap in enumerate(wind_gen_steps):
    #            val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1][t]
    #            matrix[i].append(val*100)
    #    ary = np.array(matrix)
    #    plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'threshold_std_{threshold:03}')

    #    # Needed dispatchable+storage
    #    matrix = []
    #    for i, solar_install_cap in enumerate(solar_gen_steps):
    #        matrix.append([])
    #        for j, wind_install_cap in enumerate(wind_gen_steps):
    #            val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
    #            matrix[i].append(val)
    #    ary = np.array(matrix)
    #    plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'needed_dispatchablePlusStorage_{threshold:03}')

    #    # Needed overbuild in dispatchable+storage
    #    matrix = []
    #    for i, solar_install_cap in enumerate(solar_gen_steps):
    #        matrix.append([])
    #        for j, wind_install_cap in enumerate(wind_gen_steps):
    #            val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1][t]
    #            val /= study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
    #            matrix[i].append(val)
    #    ary = np.array(matrix)
    #    plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'overbuild_dispatchablePlusStorage_{threshold:03}')

    #    # Needed overbuild in dispatchable+storage 94% conf (based on ERCOT)
    #    matrix = []
    #    for i, solar_install_cap in enumerate(solar_gen_steps):
    #        matrix.append([])
    #        for j, wind_install_cap in enumerate(wind_gen_steps):
    #            val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][3][t]
    #            val /= study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
    #            matrix[i].append(val)
    #    ary = np.array(matrix)
    #    plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'overbuild_95pct_dispatchablePlusStorage_{threshold:03}')

    #    # Norm test p_val (null = distribution is based on normal)
    #    matrix = []
    #    for i, solar_install_cap in enumerate(solar_gen_steps):
    #        matrix.append([])
    #        for j, wind_install_cap in enumerate(wind_gen_steps):
    #            val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][4][t]
    #            matrix[i].append(val)
    #    ary = np.array(matrix)
    #    plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'norm_pval_{threshold:03}')

