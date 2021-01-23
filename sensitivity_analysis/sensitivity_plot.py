import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
from glob import glob


def get_info(pkl_file, solar_gen_steps, wind_gen_steps, region, HOURS_PER_YEAR, 
            regs, solar, wind, hours, years, inter, intra, std, mean):
    pickle_in = open(f'{pkl_file}','rb')
    if 'nYrs' not in pkl_file:
        n_years = -1 
    else:
        info = pkl_file.split('_')
        for i in info:
            if 'nYrs' in i:
                n_years = int( i.replace('nYrs', '').replace('.pkl', '') )
    study_regions = pickle.load(pickle_in)
    
    for i, solar_gen in enumerate(solar_gen_steps):
        if solar_gen not in solar_vals:
            continue
        for j, wind_gen in enumerate(wind_gen_steps):
            if wind_gen not in wind_vals:
                continue
            print(region, HOURS_PER_YEAR, n_years, solar_gen, wind_gen)
            wind_gen = wind_gen_steps[j]
            rls = study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][0]
    
            # Fill a new row
            regs.append(region)
            solar.append(solar_gen)
            wind.append(wind_gen)
            hours.append(HOURS_PER_YEAR)
            years.append(n_years)
            inter.append(np.std(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][5])*100.)
            intra.append(np.mean(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][4])*100)
            std.append(np.std(rls)*100)
            mean.append(np.mean(rls)*100)





regions = ['ERCOT','PJM','NYISO','FR']
solar_vals = [0., 0.25, 0.5]
wind_vals = [0., 0.25, 0.5]


make_summary = True
make_summary = False

plot = True
#plot = False

n_hours_test = True
#n_hours_test = False

n_years_test = True
n_years_test = False

assert(n_hours_test != n_years_test), "Cannot have to tests analyzed simultaneously"

if n_hours_test:
    s_name = 'n_hours'
    DATE = '20210122v3DT'
    solar_max = 0.5
    wind_max = 0.5
    steps = 51
    # plotting params
    x_min = 0
    x_max = 200
    y_max1 = 10.5
    y_max2 = 4
    y_min2 = -4
    x_lab = "peak hours"
    x_var = 'hours'
    n_loc = 50
    thresh = 10
    thresh_lab = '10 hours'
if n_years_test:
    s_name = 'n_years'
    DATE = '20210122v2DT'
    solar_max = 0.5
    wind_max = 0.5
    steps = 51
    # plotting params
    x_min = 0
    x_max = 20
    y_max1 = 12.5
    y_max2 = 6
    y_min2 = -6
    x_lab = "number of years"
    x_var = 'years'
    n_loc = 5
    thresh = 10
    thresh_lab = '10 years'



solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)

if make_summary:
    regs = []
    solar = []
    wind = []
    hours = []
    years = []
    inter = []
    intra = []
    std = []
    mean = []
    
    for region in regions:


        if n_hours_test:
            for HOURS_PER_YEAR in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 75, 100, 200]:
                pkl_file = f'../main_analysis/pkls/pkl_{DATE}_{steps}x{steps}_{region}_hrs{HOURS_PER_YEAR}_nYrs10.pkl'
                get_info(pkl_file, solar_gen_steps, wind_gen_steps, region, HOURS_PER_YEAR, 
                        regs, solar, wind, hours, years, inter, intra, std, mean)
    
    
        if n_years_test:
            files = glob(f'../main_analysis/pkls/pkl_{DATE}_{steps}x{steps}_{region}_hrs*_nYrs*.pkl')
            HOURS_PER_YEAR = -1
            for pkl_file in files:

                get_info(pkl_file, solar_gen_steps, wind_gen_steps, region, HOURS_PER_YEAR, 
                        regs, solar, wind, hours, years, inter, intra, std, mean)
    
    
    df = pd.DataFrame({
        'region' : regs,
        'solar' : solar,
        'wind' : wind,
        'hours' : hours,
        'years' : years,
        'inter' : inter,
        'intra' : intra,
        'std' : std,
        'mean' : mean,
        })

    df.to_csv(f'summary_{s_name}.csv', index=False)

if not plot:
    exit()



solar_vals = [0., 0.25]#, 0.5]
wind_vals = [0., 0.25]#, 0.5]

df = pd.read_csv(f'summary_{s_name}.csv')
multi = df.set_index(['region', 'solar', 'wind']).sort_index()


#for region in regions:
#    fig, ax = plt.subplots()
#    
#    for solar in solar_vals:
#        for wind in wind_vals:
#            ax.plot(multi.loc[(region, solar, wind)][x_var], multi.loc[(region, solar, wind)]['inter'], label=f'{region} s{solar}:w{wind}')
#
#    ax.set_ylim(0, ax.get_ylim()[1])
#    plt.legend()
#    plt.savefig(f'plots/{region}.png')
#
#for region in regions:
#    scale = 4.5
#    fig, ax = plt.subplots(figsize=(1*scale,1*scale))
#    plt.axhline(0, color='black')
#    
#    for solar in solar_vals:
#        for wind in wind_vals:
#            if solar == 0 and wind == 0:
#                continue
#            s = int(solar*100)
#            w = int(wind*100)
#            #ax.plot(multi.loc[(region, solar, wind)][x_var].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'(0% wind, 0% solars) - ({solar}% solar, {wind}% wind)')
#            ax.plot(multi.loc[(region, solar, wind)][x_var].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'load - RL({s}% solar, {w}% wind)')
#    ax.set_ylim(-3, 4)
#    ax.set_xlim(x_min, x_max)
#    ax.set_ylabel(r"$\Delta$ inter-annual variability"+"\n(% mean annual load)")
#    ax.tick_params(axis='y', right=True, left=True)
#    ylim = ax.get_ylim()
#    ax.plot(np.ones(100)*10, np.linspace(ylim[0], ylim[1],100), label="selected threshold")
#    #ax.plot(np.ones(100)*20, np.linspace(ylim[0], ylim[1],100))
#    #ax.set_ylim(0, ax.get_ylim()[1])
#    plt.title(region)
#    plt.legend()
#    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.9)
#    plt.savefig(f'plots/{region}_diff.png')







fig, axs = plt.subplots(figsize=(8, 6), ncols=4, nrows=2, sharex=True)
for i, region in enumerate(regions):

    y_min = 0
    axs[0][i].axhline(0, color='gray', linewidth=1)
    for wind in wind_vals:
        for solar in solar_vals:
            s = int(solar*100)
            w = int(wind*100)


            if n_hours_test:
                if solar == 0 and wind == 0:
                    lab = 'Load'
                    axs[0][i].plot(multi.loc[(region, solar, wind)][x_var].values, multi.loc[(region, solar, wind)]['inter'].values, '-k', label=lab)
                else:
                    lab = f'RL({w}% wind, {s}% solar)'
                    axs[0][i].plot(multi.loc[(region, solar, wind)][x_var].values, multi.loc[(region, solar, wind)]['inter'].values, label=lab)


            if n_years_test:
                if solar == 0 and wind == 0:
                    lab = 'Load'
                    axs[0][i].scatter(multi.loc[(region, solar, wind)][x_var].values, multi.loc[(region, solar, wind)]['inter'].values, c='k', label=lab)
                else:
                    lab = f'RL({w}% wind, {s}% solar)'
                    axs[0][i].scatter(multi.loc[(region, solar, wind)][x_var].values, multi.loc[(region, solar, wind)]['inter'].values, label=lab)

    axs[0][i].set_xlim(x_min, x_max)
    if region == 'ERCOT':
        axs[0][i].set_ylabel(r"inter-annual variability"+"\n(% mean annual load)")
    axs[0][i].set_ylim(y_min, y_max1)
    axs[0][i].tick_params(axis='y', right=True, left=True)
    axs[0][i].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    if region != 'ERCOT':
        axs[0][i].set(yticklabels=[])
    axs[0][i].plot(np.ones(100)*thresh, np.linspace(y_min, y_max1, 100), '--r', label=f"threshold = {thresh_lab}")
    axs[0][i].set_title(region)
    if region == 'FR':
        if n_hours_test:
            vert = 0.25
            horiz = .7
            p = {}
        if n_years_test:
            vert = 0.22
            horiz = .9
            p = {'size': 8}
        axs[0][i].legend(loc='center', framealpha = 0.9, bbox_to_anchor=(horiz, vert), prop=p)
    
    axs[1][i].axhline(0, color='gray', linewidth=1)
    for wind in wind_vals:
        for solar in solar_vals:
            if solar == 0 and wind == 0:
                continue
            s = int(solar*100)
            w = int(wind*100)
            if n_hours_test:
                axs[1][i].plot(multi.loc[(region, solar, wind)][x_var].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'Load - RL({w}% wind, {s}% solar)')
            if n_years_test:
                axs[1][i].scatter(multi.loc[(region, solar, wind)][x_var].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'Load - RL({w}% wind, {s}% solar)')
            axs[1][i].set_xlabel(x_lab)
    axs[1][i].set_xlim(x_min, x_max)
    axs[1][i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(n_loc))
    if region == 'ERCOT':
        axs[1][i].set_ylabel(r"$\Delta$ inter-annual variability"+"\n(% mean annual load)")
    axs[1][i].set_ylim(y_min2, y_max2)
    axs[1][i].tick_params(axis='y', right=True, left=True)
    axs[1][i].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    if region != 'ERCOT':
        axs[1][i].set(yticklabels=[])
    axs[1][i].plot(np.ones(100)*thresh, np.linspace(y_min2, y_max2, 100), '--r', label=f"threshold = {thresh_lab}")
    axs[1][i].set_xlim(1, axs[1][i].get_xlim()[1])
    axs[1][i].set_xscale("log")
    axs[1][i].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 20, 40, 50, 60, 70, 80, 90, 100, 200])
    #plt.title(region)
    if region == 'FR':
        if n_hours_test:
            vert = 0.22
            horiz = 0.5
            p = {'size': 9}
        if n_years_test:
            vert = 0.21
            horiz = 0.52
            p = {'size': 9}
        axs[1][i].legend(loc='center', framealpha = 0.9, bbox_to_anchor=(horiz, vert), prop=p)
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94)
    plt.savefig(f'plots/all_sensitivity_{s_name}.pdf')

