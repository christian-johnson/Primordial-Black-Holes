#This script simulates the Fermi response to the gamma rays from PBH at a random point in the sky
#It then analyzes the resulting patch of sky to see if the PBH was detected above the background
#If the PBH was detected, the spectrum is analyzed to see if it is consistent with a PBH spectrum
#Finally, if the spectrum is consistent, the individual gamma rays are analyzed to search for proper motion

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
from gt_apps import evtbin
from gt_apps import gtexpcube2
from gt_apps import srcMaps
from BinnedAnalysis import *
import pyLikelihood as pyLike
print "done!"

if sys.argv[3]=='batch':
    prefix = '/scratch/johnsarc/'+os.environ['LSB_JOBID']
    print "Running on batch cluster"
else:
    prefix = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability'
    print "Running locally"

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
def make_data(distance, mass_init, flux_matrix, filename):
    MET_start = 239902981 #Time limits for 3FGL
    MET_end = 365467563

    elapsed_time = MET_end-MET_start

    #Random start location in the sky & speed
    angle_relative_to_earth = make_random(y,h)
    total_v = np.random.normal(v_disk,dm_dispersion)
    v_tang = total_v*np.sin(angle_relative_to_earth)#in km/s
    v_ra = v_tang
    v_dec = 0.0
    l_0 = 360*np.random.rand(1)[0] #phi
    b_0 = (make_random(y,h)-np.pi*0.5)*360/(2*3.141593) #theta
    ra_init = SkyCoord(l=l_0*u.degree,b=b_0*u.degree,frame='galactic').icrs.ra.degree
    dec_init = SkyCoord(l=l_0*u.degree,b=b_0*u.degree,frame='galactic').icrs.dec.degree
    print "RA_init = " + str(ra_init) + " DEC_init = " + str(dec_init)
    parsec_to_km = 3.086*(10**13)
    dt = 86400.0 #86400 seconds is 1 day
    met = MET_start
    ra_final = ra_init
    dec_final = dec_init
    mass = mass_init
    while mass>0 and met<MET_end:
        Temp = T(mass)
        met += dt
        mass -= dt*alpha(mass)/(mass**2)
        ra_final += v_ra*dt*360/(parsec_to_km*distance*2*np.pi)
        dec_final += v_dec*dt*360/(distance*parsec_to_km*2*np.pi)

    #Subselect some data around the final location of the PBH, to save memory
    from gt_apps import filter
    filter['infile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3fgl_all_sky-ft1.fits'
    filter['outfile'] = prefix+'/bkg.fits'
    filter['ra'] = ra_final
    filter['dec'] = dec_final
    filter['rad'] = 7
    filter['tmin'] = MET_start
    filter['tmax'] = MET_end
    filter.run()

    g = fits.open(prefix+'/bkg.fits')
    a = g[0]
    c = g[2]
    table = Table.read(g[1])
    bkg_photons = []
    for j in range(len(g[1].data)):
        bkg_photons.append(g[1].data[j])
    tot_phots = 0
    #Evolve forward in time & make gamma rays
    w = 10**np.linspace(-2.5, 2.5, 200)
    met = MET_start
    ra = ra_init
    dec = dec_init
    mass = mass_init

    #Energy binning for exposure file
    e_bin_edges_MEV = 10**np.linspace(2.0, np.log10(500000),51)
    e_bin_edges_GEV = 10**np.linspace(2.0, np.log10(500000),51)/1000.0

    folded_flux = np.zeros((200))
    eff_area = np.zeros((len(folded_flux)))
    for i in range(len(folded_flux)):
        e_index = np.argmin(np.abs(e_bin_edges_GEV-w[i]))
        eff_area[i] = find_exposure(l_0, b_0, e_bin_edges_MEV[e_index])/(elapsed_time*10000.)
    print "Eff area = " + str(eff_area)
    #Evolve forward in time, create photons
    tot_ra = 0.0
    tot_dec = 0.0
    while mass>0 and met<MET_end:
        Temp = T(mass)
        row = np.argmin(np.abs(np.arange(0.3, 100.0, 0.1)-Temp))
        folded_flux = eff_area*flux_matrix[row,:]
        emitted_flux = get_integral(w, folded_flux)
        #fermi_capture_rate = avg total number of photons recorded in time range [met, met+dt]
        fermi_capture_rate = emitted_flux*dt/(4.0*np.pi*(distance*parsec_to_km*1000.0)**2)

        for k in range(0, np.random.poisson(fermi_capture_rate)):
            theEnergy = 1000.0*make_random(w,folded_flux)
            tot_phots +=1
            rand_bkg_photon = random.choice(bkg_photons)
            #MC data variables
            theTime = met+np.random.rand(1)[0]*dt
            theRa = np.random.normal(ra, spatial_resolution(theEnergy)/np.sqrt(2))
            theDec = np.random.normal(dec, spatial_resolution(theEnergy)/np.sqrt(2))
            theL = SkyCoord(ra=theRa*u.degree, dec=theDec*u.degree,frame='icrs').galactic.l.degree
            theB = SkyCoord(ra=theRa*u.degree, dec=theDec*u.degree,frame='icrs').galactic.b.degree
            #Other variables: stolen from the gtobssim photons
            theTheta = rand_bkg_photon[5]
            thePhi = rand_bkg_photon[6]
            theZenithAngle = rand_bkg_photon[7]
            theEarthAngle = rand_bkg_photon[8]
            theEventId = rand_bkg_photon[10]
            theRunId = rand_bkg_photon[11]
            theReconVersion = rand_bkg_photon[12]
            theCalibVersion = rand_bkg_photon[13]
            theEventClass = rand_bkg_photon[14]
            #theEventType = rand_bkg_photon[15]#included for Pass 8, not Pass 7
            theConversionType = rand_bkg_photon[15]
            theLivetime =rand_bkg_photon[16]
            theDiff0 = rand_bkg_photon[17]
            theDiff1 = rand_bkg_photon[18]
            theDiff2 = rand_bkg_photon[19]
            theDiff3 = rand_bkg_photon[20]
            theDiff4 = rand_bkg_photon[21]
            #theMC_SRC_ID = rand_bkg_photon[23]#included for Pass 8, not Pass 7
            #Add background photons
            tot_ra += theRa
            tot_dec += theDec
            table.add_row([theEnergy, theRa, theDec, theL, theB, theTheta, thePhi, theZenithAngle, theEarthAngle, theTime, theEventId, theRunId, theReconVersion, theCalibVersion, theEventClass, theConversionType, theLivetime, theDiff0, theDiff1, theDiff2, theDiff3, theDiff4])
        #Update variables for next time step
        met += dt
        mass -= dt*alpha(mass)/(mass**2)
        ra += v_ra*dt*360/(parsec_to_km*distance*2*np.pi)
        dec += v_dec*dt*360/(distance*parsec_to_km*2*np.pi)
    if tot_phots>0:
        weighted_average_ra = tot_ra/(tot_phots)
        weighted_average_dec = tot_dec/(tot_phots)
    else:
        weighted_average_dec = dec
        weighted_average_ra = ra
    print "Total photons added = " + str(tot_phots)
    #Write out FITS file
    table.write("table_data.fits",overwrite=True)
    q = fits.open('table_data.fits')
    b = q[1]

    # the column attribute is the column definitions
    hdulist = fits.HDUList([a,b,c])
    hdulist.writeto(prefix+'/raw_data.fits')
    q.close()
    os.system('rm table_data.fits')
    print "Ra = " + str(ra) + " dec = " + str(dec)+ " v_tang = " + str(v_ra)
    print "weighted avg ra = " + str(weighted_average_ra) + " weighted avg dec = " + str(weighted_average_dec)
    return weighted_average_ra, weighted_average_dec, v_tang
def chi_squared(theory, unc, data):
    return sum((theory-data)**2/(unc**2))
def analyze_source_spectrum(flux, flux_err, ra, dec):
    print "flux = " + str(flux)
    print "flux err = " + str(flux_err)
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
        print "data err = " + str(data_err)
        print "exposure = " + str(exposure)
        print "sim flux = " + str(sim_flux)
        chi2_values[i] = chi_squared(flux, np.sqrt(flux_err**2+data_err**2), sim_flux)
        #uncertainty = np.maximum(1.3*sim_flux,unc)
        print "Temperature = " + str(np.linspace(0.3,100, 1000)[i]) + " GeV, chi2-value = " + str(chi2_values[i])
        print sim_flux
    if min(chi2_values)<11.3:
        spectral_match = True
        reconstructed_temp = np.arange(0.3, 100.0, 0.1)[np.argmin(chi2_values)]
    else:
        spectral_match = False
        reconstructed_temp = 0.0
    return spectral_match, reconstructed_temp, min(chi2_values)
#Signal fraction is the proportion of the total flux coming from the signal, at that particular energy
def gaussian_with_motion(x, y, e, t, sigma, x0, y0, vx, vy, distance):
    vx *= 360.0/(2*np.pi*distance*parsec_to_km)
    vy *= 360.0/(2*np.pi*distance*parsec_to_km)
    t2 = t-239902981.0
    result = np.exp(-1.0*(np.sqrt((x-(x0+vx*t2))**2+(y-(y0+vy*t2))**2))**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)#
    return result
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
    num_trials = 20
    scrambled_values = np.zeros((num_trials))
    for i in range(num_trials):
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

#Parameters of the simulation
y = np.linspace(0, np.pi, 500)
h = np.sin(y)
dm_dispersion = 270.01 #km/s
v_disk = 250.0#km/s

def test_spectral_model(prefix, pbh_ra, pbh_dec):

    subprocess.call('python /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/make3FGLxml.py -G /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/gll_iem_v05.fits -g gll_iem_v05 -GIF True -i iso_source_v05 -r 3.0 -ER 2.0 -I /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/iso_source_v05.txt -e /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/Templates/ -o '+ prefix +'/xmlmodel.xml /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3FGL.fit '+ prefix +'/raw_data.fits',shell=True)
    subprocess.call('python /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/add_pbh_src_logparabola.py '+ str(pbh_ra) + ' ' + str(pbh_dec) + ' ' + prefix, shell=True)

    srcMaps['scfile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/SC.fits'
    srcMaps['expcube'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/ltcube.fits'
    srcMaps['cmap'] = prefix+'/ccube.fits'
    srcMaps['srcmdl'] = prefix+'/xmlmodel.xml'
    srcMaps['bexpmap'] = prefix+'/exposure.fits'
    srcMaps['outfile'] = prefix+'/srcmap.fits'
    srcMaps['rfactor'] = 4
    srcMaps['emapbnds'] = 'no'
    srcMaps.run()

    obs = BinnedObs(srcMaps=prefix+'/srcmap.fits', expCube='/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/ltcube.fits',binnedExpMap=prefix+'/exposure.fits', irfs='P7REP_SOURCE_V15')#
    like1 = BinnedAnalysis(obs, prefix+'/xmlmodel.xml', optimizer='MINUIT')
    like1.tol = 0.1
    like1obj = pyLike.Minuit(like1.logLike)
    like1.fit(verbosity=3,covar=True,optObject=like1obj)
    like1.logLike.writeXml(prefix+'/fit1.xml')
    like2 = BinnedAnalysis(obs, prefix+'/fit1.xml', optimizer='NewMinuit')
    like2.tol = 1e-8
    like2obj = pyLike.NewMinuit(like2.logLike)
    like2.fit(verbosity=3,covar=True,optObject=like2obj)
    like2.logLike.writeXml(prefix+'/fit2.xml')
    convergence = like2obj.getRetCode()

    ts = like2.Ts('PBH_Source')
    if ts>25.0:
        source_flux = np.array([like2.flux('PBH_Source',emin=100.0, emax=300.0), like2.flux('PBH_Source',emin=300.0, emax= 1000.0), like2.flux('PBH_Source',emin=1000.0, emax= 3000.0), like2.flux('PBH_Source',emin=3000.0, emax= 10000.0), like2.flux('PBH_Source',emin=10000.0, emax=100000.0)],'d')
        source_flux_unc = np.array([like2.fluxError('PBH_Source',emin=100.0, emax= 300.0), like2.fluxError('PBH_Source',emin=300.0, emax= 1000.0), like2.fluxError('PBH_Source',emin=1000.0, emax= 3000.0), like2.fluxError('PBH_Source',emin=3000.0, emax= 10000.0), like2.fluxError('PBH_Source',emin=10000.0, emax=100000.0)],'d')
        spectral_match, reconstructed_temperature, chi2value = analyze_source_spectrum(source_flux, source_flux_unc, pbh_ra, pbh_dec)
    else:
        spectral_match = False
        chi2value = 0.0
    npred = like2.NpredValue('PBH_Source')
    npred1000 = np.sum(like1._srcCnts('PBH_Source')[np.argmin(np.abs(like2.energies-1000.0)):])

    return ts, spectral_match, chi2value, npred, npred1000, convergence

def simulate_and_analyze(temp, dist):
    source_detected = False
    spectral_match = False
    motion_detected = False
    #First, simulate the PBH
    filename = prefix+'/raw_data.fits'

    print "Simulating photons..."
    pbh_ra, pbh_dec, v_tang = make_data(dist, M(temp), flux_matrix, filename)
    print "Done!"

    evtbin['evfile'] = prefix+'/raw_data.fits'
    evtbin['scfile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/SC.fits'
    evtbin['outfile'] = prefix+'/ccube.fits'
    evtbin['ebinalg'] = 'LOG'
    evtbin['emin'] = 100.0
    evtbin['emax'] = 100000.0
    evtbin['enumbins'] = 30
    evtbin['algorithm'] = 'ccube'
    evtbin['coordsys'] = 'CEL'
    evtbin['nxpix'] = 100
    evtbin['nypix'] = 100
    evtbin['binsz'] = 0.1
    evtbin['xref'] = pbh_ra
    evtbin['yref'] = pbh_dec
    evtbin['axisrot'] = 0.0
    evtbin['proj'] = 'AIT'
    evtbin.run()

    gtexpcube2['infile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/ltcube.fits'
    gtexpcube2['outfile'] = prefix+'/exposure.fits'
    gtexpcube2['cmap'] = prefix+'/ccube.fits'
    gtexpcube2['irfs'] = 'P7REP_SOURCE_V15'
    gtexpcube2.run()

    chi2value = 0.0

    ts, test_spectral_match, chi2value, npred, npred1000, convergence = test_spectral_model( prefix, pbh_ra, pbh_dec)
    print "TS: " + str(ts)
    print "Spectral match: " + str(test_spectral_match)
    print "npred = " + str(npred)
    print "npred1000 = " + str(npred1000)
    if ts>25.0:
        source_detected = True
    v_recovered = 0.0
    if source_detected:
        MET_start = 239902981 #Time limits for 3FGL
        MET_end = 365467563

        from gt_apps import filter
        filter['infile'] = prefix+'/raw_data.fits'
        filter['outfile'] = prefix+'/ROI.fits'
        filter['ra'] = pbh_ra
        filter['dec'] = pbh_dec
        filter['rad'] = 3
        filter['tmin'] = MET_start
        filter['tmax'] = MET_end
        filter['emin'] = 1000.0
        filter['emax'] = 300000.0
        filter.run()


        g = pyfits.open(prefix+'/raw_data.fits')
        photons = np.zeros((len(g[1].data),4))
        for q in range(len(g[1].data)):
            photons[q,0] = g[1].data[q]['RA']
            photons[q,1] = g[1].data[q]['DEC']
            photons[q,2] = g[1].data[q]['ENERGY']
            photons[q,3] = g[1].data[q]['TIME']
        g.close()

        sig, v_recovered = movement_significance(photons, dist, pbh_ra, pbh_dec, npred1000)


        if sig>3.65:
            print "Significant proper motion!"
            motion_detected = True

    print "source detected: "+ str(source_detected)
    print "spectral match: "+ str(spectral_match)
    print "proper motion: " + str(motion_detected)
    #Save results to a file
    file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/detectability_results.pk1','rb')
    detectability_map = pickle.load(file)
    file.close()
    detectability_map.append({"Temperature":temp, "Distance":dist, "Convergence":convergence, "Source_Detected":source_detected, "Motion_Detected":motion_detected, "GLON":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.l.degree, "GLAT":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.b.degree, "V_Recovered":v_recovered , "Spectral_Match": spectral_match, "MIN_CHI2":chi2value, "V_True": v_tang, "TS":ts})
    file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/detectability_results.pk1','wb')
    pickle.dump(detectability_map, file)
    file.close()

print "Temperature = " + str(float(sys.argv[1])) + " Distance = " + str(float(sys.argv[2]))
simulate_and_analyze(float(sys.argv[1]),float(sys.argv[2]))
