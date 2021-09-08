# Demand response

Results are discussed in `4.3. Demand response programs` of the main paper with details presented in `S.17. Historical demand response` in the Supplementary material.

## Annual Electric Power Industry Report, Form EIA-861 detailed data files

This directory downloads and extracts the relevant info from EIA-861.

See: https://www.eia.gov/electricity/data/eia861/

 * Demand-Side Management (2001–2012) — This file, compiled from data collected on both Form EIA-861 and, for time-based rate programs, Form EIA-861S, contains information on electric utility demand-side management programs, including energy efficiency and load management effects and expenditures. Beginning in 2007, it also contains the number of customers in time-based rate programs.
 * Demand Response (2013 forward) — This file, compiled from data collected on Form EIA-861 only, contains the number of customers enrolled, energy savings, potential and actual peak savings, and associated costs.

## Utility to BA mapping

From 2013 onwards, `Sales_Ult_Cust_2019.xlsx` provides a mapping of utility to BA. This is need b/c the hourly data is per BA.

## EIA Demand Response of peak load post

https://www.eia.gov/todayinenergy/detail.php?id=38872

## Download historical data

Use `download.sh` to download and extract the data for analysis (the used files are saved in GitHub in './data/all_DR`

## Mapping Utility to BA

Use `extract_DR_by_BA.ipynb` to produce `mapping.csv` (mapping of utility to BA) and `DR_output.csv` which maps the relevant utilities to PJM, ERCOT, and NYISO.

## Plotting results

Use `plot_DR_by_BA.ipynb`
