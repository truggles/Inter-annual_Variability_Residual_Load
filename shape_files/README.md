# Shape file defining terriroty perimeters and the resource CF regions

## France has a single file

`selected_mask_outfile_FR.nc`

## The US BAs have a shared file

PJM, ERCO, NYIS based on 2018 territories.

`selected_mask_outfile_US_BAS.nc`

## Naming conventions

The mask naming conventions adhere to the defaults in: https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/master/get_regional_CF_time_series_on_linux-mac/step1p2_Select_grids_for_interested_regions.py (private) and are:
 * method 1 = full geographic area
 * method 2 = please ignore
 * method 3 = selected region based on cells representing the top 25% of wind or solar resources, defined by decadal mean capacity factor

It should be possible to dump the contents of these files using `ncdump` if you have installed `Netcdf4`: `conda install -c anaconda netcdf4`

Files also saved in my gDrive: https://drive.google.com/drive/u/1/folders/1q1_vR-NOqyntXckEM9EhTwYtDjIoGInf
