#This script loads a python dictionary containing the results from mc_moving_source.py
#It bins the results into a detectability map in distance & initial temperature
#Then, it calculates the detection efficiency from the map, and computes the upper limit given no detections

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rcParams
from matplotlib.pyplot import rc
import matplotlib
from scipy.signal import convolve2d
def setup_plot_env():
    #Set up figure
    #Plotting parameters
    fig_width = 8   # width in inches
    fig_height = 7  # height in inches
    fig_size =  [fig_width, fig_height]
    rcParams['font.family'] = 'serif'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelsize'] = 20
    rcParams['font.size'] = 20
    rcParams['axes.titlesize'] =16
    rcParams['legend.fontsize'] = 16
    rcParams['xtick.labelsize'] =18
    rcParams['ytick.labelsize'] =18
    rcParams['figure.figsize'] = fig_size
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6
    rcParams['xtick.minor.size'] = 3
    rcParams['ytick.minor.size'] = 3
    rcParams['figure.subplot.left'] = 0.16
    rcParams['figure.subplot.right'] = 0.92
    rcParams['figure.subplot.top'] = 0.90
    rcParams['figure.subplot.bottom'] = 0.12
    rcParams['text.usetex'] = True
    rc('text.latex', preamble=r'\usepackage{amsmath}')
setup_plot_env()

critical_chi2_value = 11.3
cutoff_temperature = 16.36 #GeV
MET_start = 239902981.0 #Time limits for 3FGL
MET_end = 365467563.0
elapsed_seconds = MET_end-MET_start
year_length = 86400.0*365.0
min_dist = 0.01
max_dist = 0.08
min_temp = 5.0
max_temp = 60.0
nbins = 30
distances = np.linspace(min_dist-0.5*(max_dist-min_dist)/(nbins-1), max_dist+0.5*(max_dist-min_dist)/(nbins-1), nbins+1)
temps = np.linspace(min_temp-0.5*(max_temp-min_temp)/(nbins-1), max_temp+0.5*(max_temp-min_temp)/(nbins-1), nbins+1)

def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))
def weighted_avg(array, weights):
    #weights = np.arange(1, len(my_array)+1, 1.0)**index
    return sum(weights*array)/sum(weights)


file = open('detectability_results_new.pk1','rb')
g = pickle.load(file)
file.close()

detected_map = np.zeros((len(temps)-1, len(distances)-1))
spectral_map = np.zeros((len(temps)-1, len(distances)-1))
motion_map = np.zeros((len(temps)-1, len(distances)-1))
trials = np.zeros((len(temps)-1,len(distances)-1))
ts_map = np.zeros((len(temps)-1,len(distances)-1))
my_arr=np.zeros((len(g)))
my_arr2=np.zeros((len(g)))

i = 0
j = 0
for entry in g:
    #if 'Source_Detected' in entry  and 'Temperature' in entry and 'Distance' in entry and 'Spectral_Match' in entry and 'MIN_CHI2' in entry and 'GLON' in entry and 'GLAT' in entry and 'V_Recovered' in entry and 'V_True' in entry and 'TS' in entry:
    row = np.argmax(np.histogram(entry['Temperature'], temps)[0])
    column = np.argmax(np.histogram(entry['Distance'], distances)[0])
    trials[row, column] += 1.0
    if  entry['Source_Detected']:
        detected_map[row,column] += 1.0
    if  entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected']:
        spectral_map[row,column] += 1.0
    if  entry['Sig']>4.0 and entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected'] and entry['V_Recovered']>0.0:
        motion_map[row, column] += 1.0
    if entry['V_Recovered']>0.0 and entry['Motion_Detected'] and entry['V_Recovered']>0.0 and entry['TS']>50:
        my_arr[i] = 1.0*entry['V_Recovered']/np.abs(1.0*entry['V_True'])
        if my_arr[i]>0.5 and my_arr[i]<1.5:
            my_arr2[i] = np.sqrt(entry['TS'])
        i += 1



detected_map *= 1./trials
spectral_map *= 1./trials
motion_map *= 1./trials

print motion_map
print "Total simulated PBHs: " + str(sum(sum(trials)))

fig = plt.figure()
fontProperties = {'family':'serif', 'weight': 'light', 'size': 20}
"""
ax = fig.add_subplot(131)
ax.imshow(detected_map, interpolation='none', extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]')
plt.xlabel('Distance [pc]')
plt.title('Point Source Detectability')

ax2 = fig.add_subplot(132)
ax2.imshow(spectral_map, interpolation='none',extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]')
plt.xlabel('Distance [pc]')
plt.title('Spectral Matching')
"""
ax3 = fig.add_subplot(111)
kernel = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0], [1.0, 1.0,1.0]])/9.0
#motion_map = convolve2d(motion_map,kernel)[1:-1, 1:-1]
mappable = ax3.imshow(motion_map, interpolation='none',cmap='viridis',extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]',fontProperties)
plt.xlabel('Distance [pc]',fontProperties)
#plt.title('Proper Motion Sensitivity')
cb = plt.colorbar(mappable)
cb.set_label('Fraction Detected')
print "spectral efficiency: " + str(100.0*sum(sum(spectral_map))/sum(sum(detected_map)))+"%"
print 'Motion efficiency: ' + str(100.0*sum(sum(motion_map))/sum(sum(spectral_map)))+"%"
#Calculate fraction of PBHs that are above the cutoff temperature (given a distribution N(T)~T^3)
x = np.linspace(min(temps),max(temps),100)
cutoff_row = np.argmin(np.abs(x-cutoff_temperature))
fraction = get_integral(x[cutoff_row:],x[cutoff_row:]**-4.0)/get_integral(x,x**-4.0)

new_temps = np.linspace(min_temp, max_temp, nbins)
new_dists = np.linspace(min_dist, max_dist, nbins)

limit = np.zeros((len(distances)-1))
for column in range(len(distances)-1):
    limit[column] = weighted_avg(motion_map[:,column], new_temps**-4.0)
print limit
volume = (4/3)*np.pi*max_dist**3

efficiency = weighted_avg(limit, new_dists**2.0)
print "efficiency: " + str(efficiency)
years = (elapsed_seconds/year_length)
the_limit = fraction*4.6/(volume*years*efficiency)
print "limit: " + str(the_limit)

ax3.set_xticklabels(ax3.get_xticks(), fontProperties)
ax3.set_yticklabels(ax3.get_yticks(), fontProperties)


#plt.savefig('detectability_map.png',bbox_inches='tight')
plt.show()
