import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib


DATE = '20201230v1'
solar_max = 1.
wind_max = 1.
steps = 101
solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)


regions = ['ERCOT','PJM','NYISO','FR']
solar_vals = [0., 0.25]
wind_vals = [0., 0.25, 0.5]
wind_vals = [0., 0.25]
    
make_summary = True
make_summary = False

plot = True
#plot = False

if make_summary:
    regs = []
    solar = []
    wind = []
    hours = []
    inter = []
    intra = []
    std = []
    mean = []
    
    for region in regions:
        for HOURS_PER_YEAR in [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 200]:
            pkl_file = f'pkls/pkl_{DATE}_{steps}x{steps}_{region}_hrs{HOURS_PER_YEAR}'
            pickle_in = open(f'{pkl_file}.pkl','rb')
            study_regions = pickle.load(pickle_in)
        
            for i, solar_gen in enumerate(solar_gen_steps):
                if solar_gen not in solar_vals:
                    continue
                for j, wind_gen in enumerate(wind_gen_steps):
                    if wind_gen not in wind_vals:
                        continue
                    print(region, HOURS_PER_YEAR, solar_gen, wind_gen)
                    wind_gen = wind_gen_steps[j]
                    rls = study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][0]
    
                    # Fill a new row
                    regs.append(region)
                    solar.append(solar_gen)
                    wind.append(wind_gen)
                    hours.append(HOURS_PER_YEAR)
                    inter.append(np.std(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][5])*100.)
                    intra.append(np.mean(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][4])*100)
                    std.append(np.std(rls)*100)
                    mean.append(np.mean(rls)*100)
    
    
                    #m_rl_mean[i].append(np.mean(rls)*100)
                    #m_rl_std[i].append(np.std(rls)*100)
                    ##m_rl_50pct[i].append( (np.percentile(rls, 75) - np.percentile(rls, 25))*100)
                    ##m_rl_95pct[i].append( (np.percentile(rls, 97.5) - np.percentile(rls, 2.5))*100)
                    ##m_rl_Mto97p5pct[i].append( (np.percentile(rls, 97.5) - np.mean(rls))*100)
                    #intra[i].append(np.mean(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][4])*100.)
                    #inter[i].append(np.std(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][5])*100.)
        
    
    
    df = pd.DataFrame({
        'region' : regs,
        'solar' : solar,
        'wind' : wind,
        'hours' : hours,
        'inter' : inter,
        'intra' : intra,
        'std' : std,
        'mean' : mean,
        })
    df.to_csv('summary_n_hour.csv', index=False)

if not plot:
    exit()

df = pd.read_csv('summary_n_hour.csv')
multi = df.set_index(['region', 'solar', 'wind']).sort_index()


#for region in regions:
#    fig, ax = plt.subplots()
#    
#    for solar in solar_vals:
#        for wind in wind_vals:
#            ax.plot(multi.loc[(region, solar, wind)]['hours'], multi.loc[(region, solar, wind)]['inter'], label=f'{region} s{solar}:w{wind}')
#
#    ax.set_ylim(0, ax.get_ylim()[1])
#    plt.legend()
#    plt.savefig(f'plotsX/{region}.png')
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
#            #ax.plot(multi.loc[(region, solar, wind)]['hours'].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'(0% wind, 0% solars) - ({solar}% solar, {wind}% wind)')
#            ax.plot(multi.loc[(region, solar, wind)]['hours'].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'load - RL({s}% solar, {w}% wind)')
#    ax.set_ylim(-3, 4)
#    ax.set_xlim(0, 200)
#    ax.set_ylabel(r"$\Delta$ inter-annual variability"+"\n(% mean annual load)")
#    ax.tick_params(axis='y', right=True, left=True)
#    ylim = ax.get_ylim()
#    ax.plot(np.ones(100)*10, np.linspace(ylim[0], ylim[1],100), label="selected threshold")
#    #ax.plot(np.ones(100)*20, np.linspace(ylim[0], ylim[1],100))
#    #ax.set_ylim(0, ax.get_ylim()[1])
#    plt.title(region)
#    plt.legend()
#    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.9)
#    plt.savefig(f'plotsX/{region}_diff.png')







fig, axs = plt.subplots(figsize=(8, 6), ncols=4, nrows=2, sharex=True)
for i, region in enumerate(regions):

    y_max = 10.5
    y_min = 0
    axs[0][i].axhline(0, color='gray', linewidth=1)
    for wind in wind_vals:
        for solar in solar_vals:
            s = int(solar*100)
            w = int(wind*100)
            if solar == 0 and wind == 0:
                lab = 'Load'
                axs[0][i].plot(multi.loc[(region, solar, wind)]['hours'].values, multi.loc[(region, solar, wind)]['inter'].values, '-k', label=lab)
            else:
                lab = f'RL({w}% wind, {s}% solar)'
                axs[0][i].plot(multi.loc[(region, solar, wind)]['hours'].values, multi.loc[(region, solar, wind)]['inter'].values, label=lab)
            axs[0][i].set_xlabel("peak hours")
    axs[0][i].set_xlim(0, 200)
    if region == 'ERCOT':
        axs[0][i].set_ylabel(r"inter-annual variability"+"\n(% mean annual load)")
    axs[0][i].set_ylim(y_min, y_max)
    axs[0][i].tick_params(axis='y', right=True, left=True)
    axs[0][i].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    if region != 'ERCOT':
        axs[0][i].set(yticklabels=[])
    axs[0][i].plot(np.ones(100)*10, np.linspace(y_min, y_max, 100), '--r', label="threshold = 10 hours")
    axs[0][i].set_title(region)
    if region == 'FR':
        horiz = .7
        vert = 0.25
        axs[0][i].legend(loc='center', framealpha = 1.0, bbox_to_anchor=(horiz, vert))
    
    y_max = 4
    y_min = -4
    axs[1][i].axhline(0, color='gray', linewidth=1)
    for wind in wind_vals:
        for solar in solar_vals:
            if solar == 0 and wind == 0:
                continue
            s = int(solar*100)
            w = int(wind*100)
            axs[1][i].plot(multi.loc[(region, solar, wind)]['hours'].values, (multi.loc[(region, 0, 0)]['inter'].values - multi.loc[(region, solar, wind)]['inter'].values), label=f'Load - RL({w}% wind, {s}% solar)')
            axs[1][i].set_xlabel("peak hours")
    axs[1][i].set_xlim(0, 200)
    if region == 'ERCOT':
        axs[1][i].set_ylabel(r"$\Delta$ inter-annual variability"+"\n(% mean annual load)")
    axs[1][i].set_ylim(y_min, y_max)
    axs[1][i].tick_params(axis='y', right=True, left=True)
    axs[1][i].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    if region != 'ERCOT':
        axs[1][i].set(yticklabels=[])
    axs[1][i].plot(np.ones(100)*10, np.linspace(y_min, y_max, 100), '--r', label="threshold = 10 hours")
    #plt.title(region)
    if region == 'FR':
        horiz = 0.4
        vert = 0.22
        axs[1][i].legend(loc='center', framealpha = 1.0, bbox_to_anchor=(horiz, vert))
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94)
    plt.savefig(f'plotsX/all_sensitivity.pdf')

