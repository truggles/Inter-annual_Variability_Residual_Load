### Originally by Lei Duan: leiduan@carnegiescience.edu

#########################################################
# Code shows the impact of temperature on solar PV 
# production and thus capacity factor. The hardcoded
# equations here are taken from the Carnegie Science
# solar PV CF code: 
# https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/master/get_global_CF_time_series/step0_get_solarCF.py 
#########################################################

import numpy as np
import matplotlib.pyplot as plt  


# These constants from Ninja's code
# https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/4c1cb17f45dbb192bcde626c9dcea726b02e5548/get_global_CF_time_series/step0_get_solarCF.py#L62-L63
k_1, k_2, k_3, k_4, k_5, k_6 = -0.017162, -0.040289, -0.004681, 0.000148, 0.000169, 0.000005



# Actual CF equation:
# https://github.com/carnegie/Create_Wind_and_Solar_Resource_Files/blob/4c1cb17f45dbb192bcde626c9dcea726b02e5548/get_global_CF_time_series/step0_get_solarCF.py#L291-L297
# This is from:
# Huld, Thomas, Ralph Gottschalg, Hans Georg Beyer, and Marko Topič. 2010. “Mapping the Performance of PV Modules, Effects of Module Type and Data Averaging.” Solar Energy 84 (2): 324–38.
# Equation 2 in paper
def test_t(t, r=1000):
    T_mod_STC = 25
    T_mod = t + cT * r # eq 3
    a = T_mod - T_mod_STC
    g = r / 1000
    scf = g * ( 1 + k_1*np.log(g) + k_2*(np.log(g))**2 + a*(k_3+k_4*(np.log(g)) + k_5*(np.log(g))**2) + k_6*(a**2) ) 
    return (scf)


R = 1000 # W/m2
cT = 0.035 # "Based on measurements performed at JRC Ispra, the temperature coefficient of the c-Si module was set to cT = 0.035 C W1 m2"

t = np.arange(50)-25
b = test_t(t, R)
slope = ((b[-1]-b[0])/b[0]*100)/(t[0]-t[-1]) 
print (f"Slope: {round(slope,3)}% / degree C")

fig, ax = plt.subplots(figsize=(4,3))
T_mod = t + cT * R
ax.plot(T_mod, b, label="normalized power output")
#for ti, bi in zip(t, b):
#    print(ti, bi)
ax.set_ylabel("PV panel power output\n(normalized)")
ax.set_xlabel("module temperature (C)")
lims = ax.get_ylim()
ax.set_ylim(lims[0], lims[1]*1.05)
plt.legend()
plt.tight_layout()
plt.savefig("panel_performance_vs_temp.pdf")
