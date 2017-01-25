#This script is similar to the PBH Detectability script, but instead of simulating a PBH, it scans through the 3FGL
#The exact same procedures for spectral analysis and proper motion analysis are used, for consistency
#The only extra cut here is that each candidate source must not be associated with an astrophysical object

#Importing useful packages
print "initializing"
import numpy as np
import random
import os
import gt_apps
import pyfits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
import pickle
from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
import sys
import subprocess
#import matplotlib.pyplot as plt
print "done!"

#Defining constants & functions for later use
G = 6.67*10**(-11)
c = 3.0*10**8
hbar = 1.055*10**(-34)
joulestogev = 6.242*(10**9)
seconds_per_year = float(86400*365)
elapsed_seconds = float(333849586)-float(239557447)
kmtoparsecs = 3.241*(10**-14)
parsec_to_km = 3.086*(10**13)
parsecstometers = 3.086*(10**16)
def T(M):
    G = 6.67*10**(-11)
    c = 3.0*10**8
    #What's the temperature of the PBH in GeV?
    return joulestogev*hbar*c**3/(8*np.pi*G*M)
def M(T):
    G = 6.67*10**(-11)
    c = 3.0*10**8
    return joulestogev*hbar*c**3/(8*np.pi*G*T)
def alpha(M):
    #Alpha factor tells us about the number of particle degrees of freedom. Paramaterized from Halzen review paper
    #interpolated from Halzen et al graph
    alphas = np.array([7.727704,7.7369766,7.7057424,7.562091,7.386425,7.2107096,7.0027847,6.76265,6.618852,6.540108,6.477299,6.3338914,6.1743546,5.934317,5.7263923,5.4542427,5.2143517,4.9904923,4.7504554,4.526401,4.3185735,4.1111364,3.6149325,3.134565,1.8849653,1.0679307,0.8448036,0.83150476,0.8335057])
    temperatures = np.array([8519.18,80.54961,53.073032,38.554485,27.32478,19.846907,15.514125,13.051488,10.205263,5.008235,2.581585,1.6588818,1.1194917,0.8966952,0.7009375,0.5752979,0.42810825,0.31860077,0.2551942,0.20949882,0.15592195,0.09536693,0.06425724,0.04776353,0.030534271,0.022661088,0.011672345,0.0029544712,0.001080562])
    return alphas[np.argmin(np.abs(temperatures-T(M)))]*10**17
def celestial_to_galactic(ra,dec):
    ag = radians(192.85)
    dg = radians(27.128)
    ac = radians(266.4)
    dc = radians(-28.9297)
    b = asin(sin(dec)*sin(dg)+cos(dec)*cos(dg)*cos(ra-ag))
    j = (sin(dec)*cos(dg)-cos(dec)*sin(dg)*cos(ra-ag))/cos(b)
    k = asin(cos(dec)*sin(ra-ag)/cos(b))
    q = acos(sin(dc)/cos(dg))
    if j<0.0:
        l = q+k-180.0
    else:
        l = q-k
    if l<0:
        l += 360.0

    return l,b
def spatial_resolution(E): #In MeV
	energy = np.array([9.760642,16.96123,30.866728,52.83083,94.68691,172.33113,304.18912,545.2184,962.40643,1725.1139,2998.5986,5457.3774,9630.715,16482.188,30451.303,52101.32,94780.69,164671.23,290505.56,528506.0,946869.94,1645084.5,2858157.2])
	psf = np.array([22.738272,18.45705,13.17205,9.101765,6.090297,4.0102134,2.4757671,1.5785031,0.9589487,0.5732728,0.3654938,0.244569,0.18614233,0.13940534,0.12265361,0.11508227,0.11333075,0.10806294,0.10814128,0.10147373,0.09369465,0.089339554,0.08518689])
	closest = np.argmin(np.abs(E-energy))
	if E-energy[closest]>0.:
		frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
		return psf[closest]+frac*(psf[closest+1]-psf[closest])
	else:
		frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
		return psf[closest-1]+frac*(psf[closest]-psf[closest-1])
def psf_array(E):
    energy = np.array([9.91152,17.36871,31.150045,54.59528,96.42895,171.62605,303.14316,539.58026,967.85913,1709.5619,3066.256,5374.1895,9712.058,17151.041,29366.348,52649.074,92947.98,167911.25,298723.0,527422.3,952855.0,1682382.6,2993103.8])
    psf = np.array([22.122343,17.216175,11.960119,8.108732,5.279108,3.5216076,2.2375877,1.3988715,0.8535155,0.53358656,0.347393,0.23173566,0.17039458,0.12837319,0.112826064,0.10581638,0.10334797,0.10426899,0.10101496,0.09097172,0.08671612,0.07683781,0.073241934])
    result = np.zeros(len(E))
    for i in range(len(result)):
        closest = np.argmin(np.abs(E[i]-energy))
        if E[i]-energy[closest]>0.:
            frac = (E[i]-energy[closest])/(energy[closest+1]-energy[closest])
            result[i] = psf[closest]+frac*(psf[closest+1]-psf[closest])
        else:
            frac = (E[i]-energy[closest-1])/(energy[closest]-energy[closest-1])
            result[i] = psf[closest-1]+frac*(psf[closest]-psf[closest-1])
    return result
def make_random(x,g):
    if len(x) != len(g):
        print "Random number generation must be performed with equal-sized arrays!"
    #Make cumulative distribution function from the pdf
    cdf = np.zeros(len(x))
    for i in range(len(x)):
        cdf[i] = get_integral(x[0:i+1],g[0:i+1])
    cdf *= 1.0/get_integral(x,g)
    #Pick a random number between 0 and 1
    numb = np.random.rand(1)[0]
    #Find the closest value in the cdf
    index = np.argmin(np.abs(cdf-numb))
    value = min(np.random.normal(x[index],0.005),x[len(x)-1])

    return value
def gamma(x):
	if x%1 == 0:
		return factorial(x-1)
	if x%1 == 0.5:
		return np.sqrt(np.pi)*factorial(2*(x-0.5))/(4**(x-0.5)*factorial((x-0.5)))
def chi_square_pdf(k,x):
    return 1.0/(2**(k/2)*gamma(k/2))*x**(k/2-1)*np.exp(-0.5*x)
def chi_square_cdf(k,x):
    return gammainc(k/2,x/2)
def chi_square_quantile(k,f):
    #Essentially do a numerical integral, until the value is greater than f
    integral_fraction = 0.0
    x = 0.0
    dx = 0.01
    while chi_square_cdf(k,x)<f:
        x += dx
    return x
def upper_limit_pdg(N,alpha,b):
    dof = 2*(N+1)
    p = 1-alpha*(1-(chi_square_cdf(dof, 2*b)))
    sup = 0.5*chi_square_quantile(dof, p)-b
    return sup
def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))
def sigma_given_p(p):
    x = np.linspace(-200, 200, 500000)
    g = 1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.)
    c = np.cumsum(g)/sum(g)
    value = x[np.argmin(np.abs(c-(1.0-p)))]
    return value
def pvalue_given_chi2(x, k):
    y = np.concatenate([np.concatenate([np.arange(0., 1.0, 0.0001),np.arange(1.0,10,0.1)]),np.arange(10.0,1000.0,1.0)])
    g = (y**(k/2.-1.0)*np.exp(-0.5*y))/(2.**(k/2.0)*gamma(k/2.))
    initial_pos = np.argmin(np.abs(y-x))
    return get_integral(y[initial_pos:], g[initial_pos:])
def multiply_multidimensional_array(vec,cube):
    #Stupid function bc apparently numpy can't do this natively??
    result = np.zeros((cube.shape))
    for i in range(len(vec)):
        result[i,:,:] = vec[i]*cube[i,:,:]
    return result
exposurefile = fits.open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3fgl_all_sky_exposure.fits')
def find_exposure(l, b, e):
    e_index = np.argmin(np.abs(e-10**np.linspace(2.0, np.log10(500000),51)))
    return np.mean(exposurefile[0].data[e_index][np.nonzero(exposurefile[0].data[e_index])])
    """
    print "reconstructed l = " + str(l)
    print "reconstructed b = " + str(b)
    g = wcs.WCS(exposurefile[0].header, naxis=2)
    print "dec index = " + str(int(g.wcs_world2pix(l, b, 0)[1]))
    print "ra index = " + str(int(g.wcs_world2pix(l, b, 0)[0]))
    return exposurefile[0].data[e_index, int(g.wcs_world2pix(l, b, 0)[1]), int(g.wcs_world2pix(l, b, 0)[0])]
    """
def chi_squared(theory,unc, data):
    return sum((theory-data)**2/(unc**2))
def analyze_source_spectrum(flux, flux_err, ra, dec):
    print "spec = " + str(flux)
    print "spec unc = " + str(flux_err)
    #x is the column array of the flux-matrix, i.e. energy
    x = 10**np.linspace(-2.5, 2.5, 200)
    #Here, in GeV because the flux matrix is in GeV (ultimately because MacGibbon's data is in GeV)
    bin_edges = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0, 100000.0],'d')/1000.0
    bin_indices = np.array([0,0,0,0,0,0],'i')
    exposure = np.zeros((5))
    l = SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='icrs').galactic.l.degree
    b = SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='icrs').galactic.b.degree

    for s in range(len(bin_edges)):
        bin_indices[s] = int(np.argmin(np.abs(x-bin_edges[s])))
    for s in range(5):
        exposure[s] = find_exposure(l, b, bin_edges[s]*1000.0)
    #print "exposure = " + str(exposure)

    #Loop through the temperatures in the flux matrix, at each temperature find the appropriate distance
    #Find the chi-squared value between the flux-matrix and the SED of the source
    chi2_values = np.zeros((len(flux_matrix)))
    for i in range(len(flux_matrix)):
        sim_flux = np.zeros((5))
        #Find the flux in each of the 5 bins for that temperature
        sim_flux[0] = get_integral(x[bin_indices[0]:bin_indices[1]],integrated_flux_matrix[i,bin_indices[0]:bin_indices[1]])
        sim_flux[1] = get_integral(x[bin_indices[1]:bin_indices[2]],integrated_flux_matrix[i,bin_indices[1]:bin_indices[2]])
        sim_flux[2] = get_integral(x[bin_indices[2]:bin_indices[3]],integrated_flux_matrix[i,bin_indices[2]:bin_indices[3]])
        sim_flux[3] = get_integral(x[bin_indices[3]:bin_indices[4]],integrated_flux_matrix[i,bin_indices[3]:bin_indices[4]])
        sim_flux[4] = get_integral(x[bin_indices[4]:bin_indices[5]],integrated_flux_matrix[i,bin_indices[4]:bin_indices[5]])
        sim_flux *= sum(flux)/sum(sim_flux)
        data_counts = exposure*flux
        sim_counts = exposure*sim_flux
        data_err = np.sqrt(flux_err)/np.sqrt(exposure)

        chi2_values[i] = chi_squared(flux, np.sqrt(flux_err**2+data_err**2), sim_flux)
    if min(chi2_values)<11.3:
        spectral_match = True
        reconstructed_temp = np.arange(0.3, 100.0, 0.1)[np.argmin(chi2_values)]
    else:
        spectral_match = False
        reconstructed_temp = 0.0
        print "Not spectrally consistent"

    return spectral_match, reconstructed_temp

#Signal fraction is the proportion of the total flux coming from the signal, at that particular energy
def gaussian_with_motion(x, y, e, t, sigma, x0, y0, vx, vy, distance):
    vx *= 360.0/(2*np.pi*distance*parsec_to_km)
    vy *= 360.0/(2*np.pi*distance*parsec_to_km)
    t2 = t-239902981.0
    result = (np.exp(-1.0*(np.sqrt((x-(x0+vx*t2))**2+(y-(y0+vy*t2))**2))**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2))#
    return result#[resid>0.0]

def likelihood(photons, params, distance, npred):
    x0 = params[0]
    y0 = params[1]
    vx = params[2]
    vy = params[3]
    result = gaussian_with_motion(photons[:,0], photons[:,1], photons[:,2], photons[:,3], photons[:,4], x0, y0, vx, vy, distance)
    return sum(np.log(np.sort(result[np.nonzero(result)])[::-1][0:int(npred)]))

#Function which takes the photons, and returns the best estimates of initial position, velocity, and the significance
def likelihood_ratio_test(photons_init, distance, ra, dec, npred):
    photons = np.zeros((len(photons_init),5))
    photons[:,0] = photons_init[:,0]
    photons[:,1] = photons_init[:,1]
    photons[:,2] = photons_init[:,2]
    photons[:,3] = photons_init[:,3]
    photons[:,4] = np.sqrt(2)*psf_array(photons[:,2])
    params = [ra, dec, 0.0, 0.0]
    dx = 2.5
    dv = 350.0
    bestx0 = ra
    besty0 = dec
    bestvx = 0.0
    bestvy = 0.0
    res = 25
    print "Finding f0..."
    test_likelihood = np.zeros((res, res))
    x0_arr = np.linspace(bestx0-dx, bestx0+dx, res)
    y0_arr = np.linspace(besty0-dx, besty0+dx, res)
    for i in range(res):
        print i
        for j in range(res):
            params = [x0_arr[i], y0_arr[j], 0.0, 0.0]
            test_likelihood[i,j] = likelihood(photons, params, distance, npred)
    bestx0 = x0_arr[np.mod(int(np.argmax(test_likelihood)/(res**1)),res)]
    besty0 = y0_arr[np.mod(int(np.argmax(test_likelihood)/(res**0)),res)]
    f0 = np.max(test_likelihood)
    #Next, maximize the likelihood by adjusting the values of the 4 degrees of freedom
    test_likelihood = np.zeros((res, res, res, res))
    x0_arr = np.linspace(bestx0-dx, bestx0+dx, res)
    y0_arr = np.linspace(besty0-dx, besty0+dx, res)
    vx_arr = np.linspace(bestvx-dv, bestvx+dv, res)
    vy_arr = np.linspace(bestvy-dv, bestvy+dv, res)
    for i in range(res):
        print i
        for j in range(res):
            for k in range(res):
                for l in range(res):
                    params = [x0_arr[i], y0_arr[j], vx_arr[k], vy_arr[l]]
                    test_likelihood[i,j,k,l] = likelihood(photons, params, distance, npred)
                    #print "x0 = " + str(x0_arr[i]) + " y0 = "+ str(y0_arr[j]) + " vx = " + str(vx_arr[k]) + " vy = " + str(vy_arr[l]) + " like = " + str(test_likelihood[i,j,k,l])
    bestx0 = x0_arr[np.mod(int(np.argmax(test_likelihood)/(res**3)),res)]
    besty0 = y0_arr[np.mod(int(np.argmax(test_likelihood)/(res**2)),res)]
    bestvx = vx_arr[np.mod(int(np.argmax(test_likelihood)/(res**1)),res)]
    bestvy = vy_arr[np.mod(int(np.argmax(test_likelihood)/(res**0)),res)]
    f1 = np.max(test_likelihood)
    print "f0 = " + str(f0)
    print "f1 = " + str(f1)
    print "v = " + str(np.sqrt(bestvx**2+bestvy**2)) + " km/s  "
    return 2*(f1-f0), np.sqrt(bestvx**2+bestvy**2)

def movement_significance(photons_init, distance, ra, dec, npred):
    photons_init[:,0] = ra+np.cos(2*np.pi*dec/360.0)*(photons_init[:,0]-ra)
    ref_value, v_recovered = likelihood_ratio_test(photons_init, distance, ra, dec, npred)
    scrambled_values = np.zeros((10))
    for i in range(10):
        photons_init[:,3] = photons_init[:,3][np.argsort(np.random.rand(len(photons_init[:,3])))]
        scrambled_values[i], v_recovered_rand = likelihood_ratio_test(photons_init, distance, ra, dec, npred)
    print ref_value
    print np.mean(scrambled_values)
    print np.std(scrambled_values)
    print str((ref_value-np.mean(scrambled_values))/np.std(scrambled_values)) + " sigma"
    return (ref_value-np.mean(scrambled_values))/np.std(scrambled_values), v_recovered

# "Flux Matrix" = lookup table for the spectrum of the PBH at the source. Interpolated from MacGibbon 1990 paper
#Should have temperatures as rows, and energy bins as columns. Gives the total flux emitted by the PBH in ph/s

file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/flux_matrix.pk1', 'rb')
flux_matrix = pickle.load(file)
file.close()
file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/integrated_flux_matrix.pk1', 'rb')
integrated_flux_matrix = pickle.load(file)
file.close()

#Looping over a grid of temperatures and distances
def analyze_a_source(src_name, npred, ra, dec):
    #First, check the source spectrum
    print "analyzing source " + str(src_name)
    #spectral_match, reconstructed_temperature = analyze_source_spectrum(src_spectrum, src_spectrum_unc, ra, dec)
    #if spectral_match:
    #    print "Spectral match!"
    #    print "npred = " + str(npred)
    MET_start = 239902981 #Time limits for 3FGL
    MET_end = 365467563
    #Subselect some data around the location of the candidate source
    from gt_apps import filter
    prefix = '/scratch/johnsarc/'+os.environ['LSB_JOBID']
    filter['infile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3fgl_all_sky-ft1.fits'
    filter['outfile'] = prefix+'/raw_data.fits'
    filter['ra'] = ra
    filter['dec'] = dec
    filter['rad'] = 3
    filter['tmin'] = MET_start
    filter['tmax'] = MET_end
    filter['emin'] = 1000.0
    filter['emax'] = 300000.0
    filter.run()
    print "Loading photons..."
    dist = 0.02
    g = pyfits.open(prefix+'/raw_data.fits')
    photons = np.zeros((len(g[1].data),4))
    for q in range(len(g[1].data)):
        photons[q,0] = g[1].data[q]['RA']
        photons[q,1] = g[1].data[q]['DEC']
        photons[q,2] = g[1].data[q]['ENERGY']
        photons[q,3] = g[1].data[q]['TIME']
    g.close()

    print "done!"
    print "Checking movement..."
    significance, v_recovered = movement_significance(photons, dist, ra, dec, npred)
    print "significance = " + str(significance)
    print "v recovered = " + str(v_recovered)

    if significance>3.65:
        print "Significant proper motion!"
        file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/photons.pk1','wb')
        pickle.dump([src_name, photons, dist, ra, dec, npred])
        file.close()
    return significance

candidate = str(sys.argv[2])
npred = int(float(sys.argv[3]))
ra = float(sys.argv[4])
dec = float(sys.argv[5])

significance = analyze_a_source(candidate,npred,ra,dec)

print "saving results..."
file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/fgl_results.pk1','rb')
g = pickle.load(file)
file.close()


g.append(significance)
file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/fgl_results.pk1','wb')
pickle.dump(g,file)
file.close()
print "done saving!"

"""
#Comb the 3FGL for candidate sources
fgl = pyfits.open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3FGL.fit')
for source in fgl[1].data:
    print "Checking source: " + str(source['SOURCE_NAME'])
    print source['CLASS1']
    print source['ASSOC1']
    if source['CLASS1']=='' and source['ASSOC1'] =='' and np.abs(source['GLAT'])>5.0:
        print "Candidate source! "

        spec1 = source['FLUX100_300']
        if np.abs(source['Unc_FLUX100_300'][0])>0.:
            spec1_unc = (source['Unc_FLUX100_300'][1]-source['Unc_FLUX100_300'][0])/2.0
        else:
            spec1_unc =source['Unc_FLUX100_300'][1]

        spec2 = source['FLUX300_1000']
        if np.abs(source['Unc_FLUX300_1000'][0])>0.:
            spec2_unc = (source['Unc_FLUX300_1000'][1]-source['Unc_FLUX300_1000'][0])/2.0
        else:
            spec2_unc =source['Unc_FLUX300_1000'][1]

        spec3 = source['FLUX1000_3000']
        if np.abs(source['Unc_FLUX1000_3000'][0])>0.:
            spec3_unc = (source['Unc_FLUX1000_3000'][1]-source['Unc_FLUX1000_3000'][0])/2.0
        else:
            spec3_unc =source['Unc_FLUX1000_3000'][1]

        spec4 = source['FLUX3000_10000']
        if np.abs(source['Unc_FLUX3000_10000'][0])>0.:
            spec4_unc = (source['Unc_FLUX3000_10000'][1]-source['Unc_FLUX3000_10000'][0])/2.0
        else:
            spec4_unc =source['Unc_FLUX3000_10000'][1]

        spec5 = source['FLUX10000_100000']
        if np.abs(source['Unc_FLUX10000_100000'][0])>0.:
            spec5_unc = (source['Unc_FLUX10000_100000'][1]-source['Unc_FLUX10000_100000'][0])/2.0
        else:
            spec5_unc =source['Unc_FLUX10000_100000'][1]

        src_spectrum = np.array([spec1, spec2, spec3, spec4, spec5],'d')
        src_spectrum_unc = np.array([spec1_unc, spec2_unc, spec3_unc, spec4_unc, spec5_unc],'d')
        ra = source['RAJ2000']
        dec = source['DEJ2000']
        print "ra = " + str(ra)
        print "dec = "+ str(dec)
        print "l = " + str(source['GLON'])
        print "b = " + str(source['GLAT'])
        src_name = source['SOURCE_NAME']
        npred = 0.0
        energy = [200.0, 650.0, 2000.0, 6500.0, 55000.0]
        for i in range(5):
            npred += src_spectrum[i]*find_exposure(source['GLON'], source['GLAT'], energy[i])

        analyze_a_source(src_name, src_spectrum, src_spectrum_unc, npred, ra, dec)
"""
