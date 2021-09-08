# Main analysis

## Running analysis

Uncomment the section title ``Running default methods`` in `process.sh` and:

```
source process.sh
```

Comment out code after running.

Alternatively, grab the already produced results files saved at: https://zenodo.org/record/5495664

## Plotting results

Uncomment the section titled ``Plotting`` in `process.sh`, make sure DATE, HOURS, and YEARS are aligned with what was previously processed, and:

```
source process.sh
```

Comment out code after running.

## Input files

All input files are located in `main_analysis/data`.

## Values used in main analysis figures

Located in `./fig_data`

 * `inter` = inter-annual variability of peak load hours (fig. 5)
 * `RL_mean` = mean residual load value from the 10 peak hours and 10 years (fig. 4)
 * `solar_mean` = mean solar CF from the 10 peak hours and 10 years (fig. 8)
 * `wind_mean` = mean wind CF from the 10 peak hours and 10 years (fig. 9)
 * `RL_max` = single max residual load value across all ten years

