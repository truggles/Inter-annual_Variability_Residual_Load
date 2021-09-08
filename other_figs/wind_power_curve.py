##########################################
# Wind power curve
# See code in:
# https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/master/get_global_CF_time_series/step0_get_windCF.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  


def power_curve(x):

    # Cut in, ramp, and cut out speeds
    # https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/4c1cb17f45dbb192bcde626c9dcea726b02e5548/get_global_CF_time_series/step0_get_windCF.py#L34-L36
    u_ci = 3 # cut in speed in m/s
    u_r = 12 # rated speed in m/s
    u_co = 25 # cut out speed in m/s

    # power curve from:
    # https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/4c1cb17f45dbb192bcde626c9dcea726b02e5548/get_global_CF_time_series/step0_get_windCF.py#L73-L76
    if x <= u_ci:
        return 0.
    if x <= u_r:
        return (x**3) / (u_r**3)
    if x <= u_co:
        return 1.
    if x > u_co:
        return 0.

# From the NREL Wind Toolkit
# https://www.nrel.gov/docs/fy14osti/61714.pdf, Figure 1, Table 2
def power_curve_T1(x):
    if x <= 3:
        return 0
    if x <= 4:
        return .0043
    if x <= 5:
        return .0323
    if x <= 6:
        return .0771
    if x <= 7:
        return .1426
    if x <= 8:
        return .2329
    if x <= 9:
        return .3528
    if x <= 10:
        return .5024
    if x <= 11:
        return .6732
    if x <= 12:
        return .8287
    if x <= 13:
        return .9264
    if x <= 14:
        return .9774
    if x <= 15:
        return .9946
    if x <= 16:
        return .999
    if x <= 17:
        return .9999
    if x <= 25:
        return 1
    else:
        return 0

def power_curve_T2(x):
    if x <= 3:
        return 0
    if x <= 4:
        return .0052
    if x <= 5:
        return .0423
    if x <= 6:
        return .1031
    if x <= 7:
        return .1909
    if x <= 8:
        return .3127
    if x <= 9:
        return .4731
    if x <= 10:
        return .6693
    if x <= 11:
        return .8554
    if x <= 12:
        return .9641
    if x <= 13:
        return .9942
    if x <= 14:
        return .9994
    if x <= 25:
        return 1
    else:
        return 0

def power_curve_T3(x):
    if x <= 3:
        return 0
    if x <= 4:
        return .0054
    if x <= 5:
        return .053
    if x <= 6:
        return .1351
    if x <= 7:
        return .2508
    if x <= 8:
        return .4033
    if x <= 9:
        return .5952
    if x <= 10:
        return .7849
    if x <= 11:
        return .9178
    if x <= 12:
        return .9796
    if x <= 23:
        return 1
    else:
        return 0

# Power curves taken from NREL's wind database on 18 June 2021:
# Class I: https://github.com/NREL/turbine-models/blob/master/Onshore/IEC_Class1_Normalized_Industry_Composite.csv
# Class II: https://nrel.github.io/turbine-models/WTK_Validation_IEC-2_normalized.html
# Class III: https://github.com/NREL/turbine-models/blob/master/Onshore/WTK_Validation_IEC-3_normalized.csv
# Documentation: https://nrel.github.io/turbine-models/
# Reference:
# King, J., A. Clifton, and B.-M. Hodge. 2014. Validation of Power Output for the WIND Toolkit. Golden, CO: National Renewable Energy Laboratory. NREL/TP-5D00-61714. https://www.nrel.gov/docs/fy14osti/61714.pdf.
def get_NREL_power_curve(Class):
    assert( Class in ["I", "II", "III"])
    df = pd.read_csv(f"NREL_data/WTK_IEC_Class{Class}.csv")
    print(df.head())
    return df



speed = np.linspace(0, 30, 1000)
power = [power_curve(x) for x in speed]
#power1 = [power_curve_T1(x) for x in speed]
#power2 = [power_curve_T2(x) for x in speed]
#power3 = [power_curve_T3(x) for x in speed]

df1 = get_NREL_power_curve("I")
df2 = get_NREL_power_curve("II")
df3 = get_NREL_power_curve("III")

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(speed, power, 'k-', label="GE 1.6-100: as modeled")
ax.plot(df1["Wind Speed [m/s]"], df1["Power [-]"], ls="--", label="NREL Wind Toolkit: Class I")
ax.plot(df2["Wind Speed [m/s]"], df2["Power [-]"], ls="--", label="NREL Wind Toolkit: Class II")
ax.plot(df3["Wind Speed [m/s]"], df3["Power [-]"], ls="--", label="NREL Wind Toolkit: Class III")
#ax.plot(speed, power, 'k-', label="_nolabel_")
ax.set_ylabel("wind turbine power output\n(normalized)")
ax.set_xlabel("wind speed (m/s)")
lims = ax.get_ylim()
ax.set_ylim(lims[0], 2)
plt.legend()
plt.tight_layout()
plt.savefig("wind_power_curve.pdf")

