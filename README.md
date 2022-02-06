# Inter-annual_Variability_Residual_Load
Collecting code used in paper on inter-annual variability of peak residual load:

Please cite: 

Tyler H. Ruggles and Ken Caldeira, Wind and solar generation may reduce the inter-annual variability of peak residual load in certain electricity systems, Applied Energy, (2022) https://doi.org/10.1016/j.apenergy.2021.117773.

## Structure

 * The main analysis workflow is contained in `main_analysis` with a detailed README file for reconstructing the exact figures in the main paper
 * Documentation for downloading, cleaning, and detrending the historical load data in `load_data`
 * Defined shapefiles used for selecting the wind, solar, and temperature data from MERRA-2 found in `shape_files`
 * Resuling load, wind, solar, and temperature data in `data`
 * Documentation for analyzing results of the sensitivity analysis found in `sensitivity_analysis`
 * Documentation for the degree day and peak load temperature analysis found in `temperature_analysis`
 * Documentation of the demand response data and calculations found in `demand_response`
 * Documentation of the wind and solar capacity factor algorithms in `other_figs`

## Running the main analysis

 * See `main_analysis/README.md`

## Results files

 * The data values used in the main analysis figures can be found in the csv files in `main_analysis/fig_data/`
 * Larger results files containing additional data are located in Zenodo: https://zenodo.org/record/5495664
