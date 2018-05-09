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
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rc('text.latex', preamble=r'\usepackage{amsmath}')
setup_plot_env()

critical_chi2_value = 11.3
cutoff_temperature = 16.36 #GeV
MET_start = 239902981.0 #Time limits for 3FGL
MET_end = 365467563.0
elapsed_seconds = MET_end-MET_start
year_length = 86400.0*365.0
min_dist = 0.01
max_dist = 0.08#0.#0.04862069#
min_temp = 5.0
max_temp = 60.0
nbins_dist = 30#17#16
nbins = 30
distances = np.linspace(min_dist-0.5*(max_dist-min_dist)/(nbins_dist-1), max_dist+0.5*(max_dist-min_dist)/(nbins_dist-1), nbins_dist+1)
temps = np.linspace(min_temp-0.5*(max_temp-min_temp)/(nbins-1), max_temp+0.5*(max_temp-min_temp)/(nbins-1), nbins+1)

def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))
def weighted_avg(array, weights):
    #weights = np.arange(1, len(my_array)+1, 1.0)**index
    return sum(weights*array)/sum(weights)

def load_data(data_dict):
    detected_map = np.zeros((len(temps)-1, len(distances)-1))
    spectral_map = np.zeros((len(temps)-1, len(distances)-1))
    motion_map = np.zeros((len(temps)-1, len(distances)-1))
    trials = np.zeros((len(temps)-1,len(distances)-1))

    for entry in data_dict[:int(len(data_dict)/2.0)]:
        #if 'Source_Detected' in entry  and 'Temperature' in entry and 'Distance' in entry and 'Spectral_Match' in entry and 'MIN_CHI2' in entry and 'GLON' in entry and 'GLAT' in entry and 'V_Recovered' in entry and 'V_True' in entry and 'TS' in entry:
        if entry['Distance']>=min_dist and entry['Distance']<=max_dist and np.abs(entry['GLAT'])>10.0:
            row = np.argmax(np.histogram(entry['Temperature'], temps)[0])
            column = np.argmax(np.histogram(entry['Distance'], distances)[0])
            trials[row, column] += 1.0
            if  entry['Source_Detected']:
                detected_map[row,column] += 1.0
            if  entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected']:
                spectral_map[row,column] += 1.0
            if  entry['Sig']>3.6 and entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected'] and entry['V_Recovered']>0.0:
                motion_map[row, column] += 1.0
    detected_map *= 1./trials
    spectral_map *= 1./trials
    motion_map *= 1./trials

    return detected_map, spectral_map, motion_map, trials


def plot_map(data,plt_name):
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
    mappable = ax3.imshow(data, interpolation='none',cmap='viridis',extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
    plt.gca().invert_yaxis()
    plt.ylabel('Temperature [GeV]',fontProperties)
    plt.xlabel('Distance [pc]',fontProperties)
    #plt.title('Proper Motion Sensitivity')
    cb = plt.colorbar(mappable)
    cb.set_label('Fraction Detected')
    #print "spectral efficiency: " + str(100.0*sum(sum(spectral_map))/sum(sum(detected_map)))+"%"
    #print 'Motion efficiency: ' + str(100.0*sum(sum(motion_map))/sum(sum(spectral_map)))+"%"

    ax3.set_xticklabels(ax3.get_xticks(), fontProperties)
    ax3.set_yticklabels(ax3.get_yticks(), fontProperties)

    plt.savefig('detectability_map_'+str(plt_name)+'.png',bbox_inches='tight')
    plt.show()

def calculate_limit(map):
    #Calculate fraction of PBHs that are above the cutoff temperature (given a distribution N(T)~T^3)
    x = np.linspace(min(temps),max(temps),100)
    cutoff_row = np.argmin(np.abs(x-cutoff_temperature))
    fraction = get_integral(x[cutoff_row:],x[cutoff_row:]**-4.0)/get_integral(x,x**-4.0)

    new_temps = np.linspace(min_temp, max_temp, nbins)
    new_dists = np.linspace(min_dist, max_dist, nbins_dist)

    limit = np.zeros((len(distances)-1))
    for column in range(len(distances)-1):
        limit[column] = weighted_avg(map[:,column], new_temps**-4.0)
    volume = (4./3)*np.pi*max_dist**3

    efficiency = weighted_avg(limit, new_dists**2.0)
    years = (elapsed_seconds/year_length)

    #Geometric factor to account for the fact that we cut out 10 degrees around the galactic plane
    geometric_factor = 1.2104

    #Statistical factors:
    #0 Detections, 99% confidence
    stat_factor = 4.61
    the_limit = geometric_factor*fraction*stat_factor/(volume*years*efficiency)
    print "limit given 0 detections: " + str(the_limit)

    #1 Detection, 99% confidence
    stat_factor = 6.64
    the_limit = geometric_factor*fraction*stat_factor/(volume*years*efficiency)
    print "limit given 1 detection: " + str(the_limit)

def main():
    file = open('detectability_results.pk1','rb')
    g = pickle.load(file)
    file.close()
    file = open('detectability_results_low.pk1','rb')
    f = pickle.load(file)
    file.close()
    file = open('detectability_results_high.pk1','rb')
    h = pickle.load(file)
    file.close()
    print g[0]
    i = 0
    low_hist = np.zeros((10000))
    med_hist = np.zeros((10000))
    high_hist = np.zeros((10000))
    low_true = np.zeros((10000))
    med_true = np.zeros((10000))
    high_true = np.zeros((10000))
    high_c = np.zeros((10000))
    low_c = np.zeros((10000))
    med_c = np.zeros((10000))
    for entry in g:
        if entry['Sig']>3.6 and entry['Source_Detected']==True and entry['Spectral_Match']==True and entry['GLAT']>10.0 and entry['Temperature']<18.0:
            med_hist[i] = entry['V_Recovered']
            med_true[i] = entry['V_True']*0.02/entry['Distance']
            med_c[i] = entry['Temperature']/18.0
            i += 1
    i = 0

    for entry in f:
        if entry['Sig']>3.6 and entry['Source_Detected']==True and entry['Spectral_Match']==True and entry['GLAT']>10.0 and entry['Temperature']<28.0:
            low_hist[i] = entry['V_Recovered']
            low_true[i] = entry['V_True']*0.02/entry['Distance']
            low_c[i] = entry['Temperature']
            i += 1
    i = 0

    for entry in h:
        if entry['Sig']>3.6 and entry['Source_Detected']==True and entry['Spectral_Match']==True and entry['GLAT']>10.0 and entry['Temperature']<28.0:
            high_hist[i] = entry['V_Recovered']
            high_true[i] = entry['V_True']*0.02/entry['Distance']
            high_c[i] = entry['Temperature']
            i += 1

    low_hist = low_hist[np.nonzero(low_hist)]
    med_hist = med_hist[np.nonzero(med_hist)]
    high_hist = high_hist[np.nonzero(high_hist)]
    low_true = low_true[np.nonzero(low_true)]
    med_true = med_true[np.nonzero(med_true)]
    high_true = high_true[np.nonzero(high_true)]
    med_c = med_c[np.nonzero(med_c)]
    low_c = low_c[np.nonzero(low_c)]
    high_c = high_c[np.nonzero(high_c)]

    print np.mean(med_hist/med_true)
    plot_v_hist = True
    if plot_v_hist:
        bins = np.linspace(0.0, 500.0, 50)

        #plt.hist(low_hist,bins,histtype='step',color='red',label='Low Reconstruction',linewidth = 2.0)
        #plt.hist(med_hist,bins,histtype='step',color='green',label='Med Reconstruction',linewidth = 2.0)
        #plt.hist(high_hist,bins,histtype='step',color='blue',label='High Reconstruction',linewidth = 2.0)
        #plt.hist(low_true,bins,histtype='step',color='red',label='Low True', linewidth = 0.5)
        #plt.hist(med_true,bins,histtype='step',color='green',label='Med True',linewidth = 0.5)
        #plt.hist(high_true,bins,histtype='step',color='blue',label='High True',linewidth = 0.5)
        plt.scatter(med_true, med_hist,c=med_c)#cmap='Greens', bins=np.linspace(0.0, 500.0, 30))
        #plt.scatter(med_true, med_hist, color='blue', label='Baseline Detectability')
        #plt.scatter(low_true, low_hist, color='red', label='Low Detectability')
        plt.plot(bins, bins, color='black', linewidth=0.5, linestyle='--')
        plt.ylabel('Reconstructed Velocity [km/s]')
        plt.xlabel('True Velocity [km/s]')
        plt.xlim([0, 500.0])
        plt.ylim([0.0, 500.0])
        plt.show()


    detected_map, spectral_map, motion_map, trials = load_data(g)
    print "Total simulated PBHs: " + str(sum(sum(trials)))
    kernel = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0], [1.0, 1.0,1.0]])/9.0
    motion_map = convolve2d(motion_map,kernel)[1:-1, 1:-1]
    calculate_limit(motion_map)
    plot_map(motion_map, 'baseline')

    detected_map, spectral_map, motion_map, trials = load_data(h)
    print "Total simulated PBHs: " + str(sum(sum(trials)))
    kernel = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0], [1.0, 1.0,1.0]])/9.0
    motion_map = convolve2d(motion_map,kernel)[1:-1, 1:-1]
    calculate_limit(motion_map)

    detected_map, spectral_map, motion_map, trials = load_data(f)
    print "Total simulated PBHs: " + str(sum(sum(trials)))
    kernel = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0], [1.0, 1.0,1.0]])/9.0
    motion_map = convolve2d(motion_map,kernel)[1:-1, 1:-1]
    calculate_limit(motion_map)

if __name__=='__main__':
    main()
