# Producing historical load + forecast EV load profiles

We have taken EV profiles from the NREL Electrification Futures Studies. https://data.nrel.gov/submissions/126 Specifically, we are using the "EFSLoadProfile_High_Rapid.zip" file which produces a 2.5 GB file that should be placed in this directory.

Mai, Trieu, Paige Jadun, Jeffrey Logan, Colin McMillan, Matteo Muratori, Daniel Steinberg, Laura Vimmerstedt, Ryan Jones, Benjamin Haley, and Brent Nelson. 2018. Electrification Futures Study: Scenarios of Electric Technology Adoption and Power Consumption for the United States. National Renewable Energy Laboratory. NREL/TP-6A20-71500. https://doi.org/10.2172/1459351.

## Creating altered profiles
Use the notebook `electrification_of_demand_profiles.ipynb` which requires that one downloads NREL's data as documented in the notebook:

This will produce profiles in the `/data` directory appended with `_plusEVX.csv` that can be used in the baseline analysis.

## Results
See section `S.15. Electrification of vehicles` of the Supplementary material for results from this sensitivity analysis.
