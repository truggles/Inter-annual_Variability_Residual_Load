##########################################
# Wind power curve
# See code in:
# https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/master/get_global_CF_time_series/step0_get_windCF.py


import numpy as np
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

speed = np.linspace(0, 35, 1000)
power = [power_curve(x) for x in speed]

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(speed, power, label="normalized power output")
ax.set_ylabel("wind turbine power output\n(normalized)")
ax.set_xlabel("wind speed (m/s)")
lims = ax.get_ylim()
ax.set_ylim(lims[0], 1.4)
plt.legend()
plt.tight_layout()
plt.savefig("wind_power_curve.pdf")

