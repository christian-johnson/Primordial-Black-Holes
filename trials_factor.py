import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, factorial
#Goal: given N trials, what's the new sigma that we care about, if originally it was 2 sigma
def num_over_x(z,x):
    return len(z[np.nonzero(np.clip(z,x,100.0)-x)])

def gamma(x):
    if x%1 == 0:
        return factorial(x-1)
    if x%1 == 0.5:
        return np.sqrt(np.pi)*factorial(2*(x-0.5))/(4.**(x-0.5)*factorial((x-0.5)))
        
def chi_square_pdf(k,x):
    return 1.0/(2**(k/2.)*gamma(k/2.))*x**(k/2.-1)*np.exp(-0.5*x)
    
def chi_square_cdf(k,x):
    return gammainc(k/2.,x/2.)/gamma(k/2.)
    
def chi_square_quantile(k,f):
    #Essentially do a numerical integral, until the value is greater than f
    integral_fraction = 0.0
    x = 0.0
    dx = 0.01
    while chi_square_cdf(k,x)<f:
        x += dx
    return x

print 0.5*chi_square_quantile(2*(1+1), 0.99)
raw_input('wait for key')

Ns = np.linspace(10, 1000,400)
trials = 10000
results = np.zeros((trials,len(Ns)))
mc_results = np.zeros((len(Ns)))
theory_results = np.zeros((len(Ns)))

x = np.linspace(-200, 200, 500000)
c = np.cumsum(1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.))/sum(1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.))

for i in range(len(Ns)):
    #MC: Draw N normal values, find the maximum value drawn
    #Do this for many trials, find the 95% percentile value for maximum value drawn
    #i.e. mc_results[N] is the 95% confidence value, such that if you drew from a normal distribution
    #N times, there's a 95% chance that the highest value you got would be less than
    mc_results[i] = np.sort(np.max(np.random.randn(int(Ns[i]),trials),axis=0))[::-1][int(0.05*trials)]
    theory_results[i] = x[np.argmin(np.abs(c**Ns[i]-0.95))]
    print mc_results[i]
    print theory_results[i]
print theory_results[np.argmin(np.abs(318-Ns))]
#Answer: 3.65 sigma locally corresponds to a 2 sigma global significance with 386 trials

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Ns, mc_results,linewidth=2.0, color='blue',label='Monte Carlo')
ax.plot(Ns, theory_results, color='green', linewidth=2.0, label='Theory')
ax.axvline(565,linestyle='--',linewidth=0.5)
plt.xlabel('Number of Trials')
plt.ylabel('Local significance needed for p<0.05')
plt.legend(loc=2)
plt.show()
