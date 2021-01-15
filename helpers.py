
def return_file_info_map(region):
    #assert(region in ['CONUS', 'ERCOT', 'NYISO', 'TEXAS'])

    info_map = { # region : # f_path, header rows
        'ERCOT': { # New files June 2020
            'demand': ['data/ERCOT_mem_1998-2019.csv', 0, 'demand (MW)', 'year'],
            'wind': ['data/20200624v4_ERCO_2018_mthd3_1990-2019_wind.csv', 0, 'w_cfs', 'year'],
            'solar': ['data/20200624v4_ERCO_2018_mthd3_1990-2019_solar.csv', 0, 's_cfs', 'year'],
            'years' : [y for y in range(2003, 2020)],
        },
        'NYISO': { # New files June 2020
            'demand': ['data/NYISO_demand_unnormalized.csv', 0, 'demand (MW)', 'year'],
            'wind': ['data/20200624v4_NYIS_2018_mthd3_1990-2019_wind.csv', 0, 'w_cfs', 'year'],
            'solar': ['data/20200624v4_NYIS_2018_mthd3_1990-2019_solar.csv', 0, 's_cfs', 'year'],
            'years' : [y for y in range(2004, 2020)],
        },
        'PJM': { # New files June 2020
            'demand': ['data/PJM_mem_1993-2019.csv', 0, 'demand (MW)', 'year'],
            'wind': ['data/20200624v4_PJM_2018_mthd3_1990-2019_wind.csv', 0, 'w_cfs', 'year'],
            'solar': ['data/20200624v4_PJM_2018_mthd3_1990-2019_solar.csv', 0, 's_cfs', 'year'],
            'years' : [y for y in range(2006, 2020)],
        },
        'FR': { # New files Dec 2020
            'demand': ['data/FR_demand_unnormalized.csv', 0, 'demand (MW)', 'year'],
            'wind': ['data/20201230v3_FR_mthd3_1980-2019_wind.csv', 0, 'w_cfs', 'year'],
            'solar': ['data/20201230v3_FR_mthd3_1980-2019_solar.csv', 0, 's_cfs', 'year'],
            'years' : [y for y in range(2008, 2018)],
        }
    }
    return info_map[region]
