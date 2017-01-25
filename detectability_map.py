import numpy as np
import matplotlib.pyplot as plt
import pickle


critical_chi2_value = 11.3
cutoff_temperature = 17.0
MET_start = 239902981.0 #Time limits for 3FGL
MET_end = 365467563.0
elapsed_seconds = MET_end-MET_start
year_length = 86400.0*365.0
distances = np.linspace(0.01, 0.10, 30.0)
temps = np.linspace(5.5, 61.5, 30.0)

def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))


file = open('detectability_results.pk1','rb')
g = pickle.load(file)
file.close()

detected_map = np.zeros((len(temps)-1, len(distances)-1))
spectral_map = np.zeros((len(temps)-1, len(distances)-1))
motion_map = np.zeros((len(temps)-1, len(distances)-1))
trials = np.zeros((len(temps)-1,len(distances)-1))
ts_map = np.zeros((len(temps)-1,len(distances)-1))

q = []
ts = []
for entry in g:
    if 'Source_Detected' in entry  and 'Motion_Detected' in entry and 'Temperature' in entry and 'Distance' in entry and 'Spectral_Match' in entry and 'MIN_CHI2' in entry and 'Convergence' in entry and 'GLON' in entry and 'GLAT' in entry and 'V_Recovered' in entry and 'V_True' in entry and 'TS' in entry:
        row = np.argmax(np.histogram(entry['Temperature'], temps)[0])
        column = np.argmax(np.histogram(entry['Distance'], distances)[0])
        trials[row, column] += 1.0
        entry['V_Recovered'] *= entry['Distance']/0.03
        q.append(entry)
file = open('detectability_results_clean.pk1','wb')
pickle.dump(q,file)
file.close()
for entry in q:
    if 'Source_Detected' in entry and 'Motion_Detected' in entry and 'Temperature' in entry and 'Distance' in entry and 'Spectral_Match' in entry:
        row = np.argmax(np.histogram(entry['Temperature'], temps)[0])
        column = np.argmax(np.histogram(entry['Distance'], distances)[0])
        ts_map[row,column] += entry['TS']/trials[row,column]
        if  entry['Source_Detected']:
            detected_map[row,column] += 1.0/trials[row, column]
        if  entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected']:
            spectral_map[row,column] += 1.0/trials[row, column]
        if  entry['Motion_Detected'] and entry['MIN_CHI2']<critical_chi2_value and entry['Source_Detected']:
            motion_map[row, column] += 1.0/trials[row, column]
print "Total simulated PBHs: " + str(sum(sum(trials)))
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(ts_map, interpolation='none', extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]')
plt.xlabel('Distance [pc]')
plt.title('Point Source Detectability')

ax2 = fig.add_subplot(132)
ax2.imshow(trials, interpolation='none',extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]')
plt.xlabel('Distance [pc]')
plt.title('Spectral Matching')

ax3 = fig.add_subplot(133)
mappable = ax3.imshow(motion_map, interpolation='none',extent=[min(distances),max(distances),max(temps),min(temps)], aspect='auto')
plt.ylabel('Temperature [GeV]')
plt.xlabel('Distance [pc]')
plt.title('Proper Motion Sensitivity')
plt.colorbar(mappable)
fig.suptitle('Primordial Black Hole Detection Efficiency',fontsize=24)
print "spectral efficiency: " + str(100.0*sum(sum(spectral_map))/sum(sum(detected_map)))+"%"
print 'Motion efficiency: ' + str(100.0*sum(sum(motion_map))/sum(sum(spectral_map)))+"%"

x = np.linspace(min(temps),max(temps),100)
cutoff_row = np.argmin(np.abs(x-cutoff_temperature))
fraction = get_integral(x[cutoff_row:],x[cutoff_row:]**-2)/get_integral(x,x**-2)

cutoff_row = np.argmin(np.abs(temps-cutoff_temperature))

def weighted_avg(my_array, weights):
    #weights = np.arange(1, len(my_array)+1, 1.0)**index
    return sum(weights*my_array)/sum(weights)

limit = np.zeros((len(distances)))
for column in range(len(distances)-1):
    limit[column] = weighted_avg(motion_map[:,column], temps[:len(temps)-1]**-2.0)


volume = (4/3)*np.pi*max(distances)**3
efficiency = weighted_avg(limit, distances**2.0)
print "efficiency: " + str(efficiency)
years = (elapsed_seconds/year_length)
the_limit = fraction*3.09/(volume*years*efficiency)
print "limit: " + str(the_limit)

plt.show()
