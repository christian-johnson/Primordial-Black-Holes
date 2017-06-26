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

#Defining constants for later use
G = 6.67*10**(-11)
c = 3.0*10**8
hbar = 1.055*10**(-34)
joulestogev = 6.242*(10**9)
seconds_per_year = float(86400*365)
#Time limits of 3FGL catalog
elapsed_seconds = float(333849586)-float(239557447)
kmtoparsecs = 3.241*(10**-14)
parsec_to_km = 3.086*(10**13)
parsecstometers = 3.086*(10**16)
MET_start = 239902981

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
    alphas = np.array([
    7.727704,
    7.7369766,
    7.7057424,
    7.562091,
    7.386425,
    7.2107096,
    7.0027847,
    6.76265,
    6.618852,
    6.540108,
    6.477299,
    6.3338914,
    6.1743546,
    5.934317,
    5.7263923,
    5.4542427,
    5.2143517,
    4.9904923,
    4.7504554,
    4.526401,
    4.3185735,
    4.1111364,
    3.6149325,
    3.134565,
    1.8849653,
    1.0679307,
    0.8448036,
    0.83150476,
    0.8335057])
    temperatures = np.array([
    8519.18,
    80.54961,
    53.073032,
    38.554485,
    27.32478,
    19.846907,
    15.514125,
    13.051488,
    10.205263,
    5.008235,
    2.581585,
    1.6588818,
    1.1194917,
    0.8966952,
    0.7009375,
    0.5752979,
    0.42810825,
    0.31860077,
    0.2551942,
    0.20949882,
    0.15592195,
    0.09536693,
    0.06425724,
    0.04776353,
    0.030534271,
    0.022661088,
    0.011672345,
    0.0029544712,
    0.001080562])
    return alphas[np.argmin(np.abs(temperatures-T(M)))]*10**17

#Pass 7 PSF
def spatial_resolution(E): #In MeV
	energy = np.array([
    9.760642,
    16.96123,
    30.866728,
    52.83083,
    94.68691,
    172.33113,
    304.18912,
    545.2184,
    962.40643,
    1725.1139,
    2998.5986,
    5457.3774,
    9630.715,
    16482.188,
    30451.303,
    52101.32,
    94780.69,
    164671.23,
    290505.56,
    528506.0,
    946869.94,
    1645084.5,
    2858157.2])
	psf = np.array([
    22.738272,
    18.45705,
    13.17205,
    9.101765,
    6.090297,
    4.0102134,
    2.4757671,
    1.5785031,
    0.9589487,
    0.5732728,
    0.3654938,
    0.244569,
    0.18614233,
    0.13940534,
    0.12265361,
    0.11508227,
    0.11333075,
    0.10806294,
    .10814128,
    0.10147373,
    0.09369465,
    0.089339554,
    0.08518689])
	closest = np.argmin(np.abs(E-energy))
	if E-energy[closest]>0.:
		frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
		return psf[closest]+frac*(psf[closest+1]-psf[closest])
	else:
		frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
		return psf[closest-1]+frac*(psf[closest]-psf[closest-1])

#Same PSF as above, but takes as an argument and returns an array
def psf_array(E):
    energy = np.array([
    9.760642,
    16.96123,
    30.866728,
    52.83083,
    94.68691,
    172.33113,
    304.18912,
    545.2184,
    962.40643,
    1725.1139,
    2998.5986,
    5457.3774,
    9630.715,
    16482.188,
    30451.303,
    52101.32,
    94780.69,
    164671.23,
    290505.56,
    528506.0,
    946869.94,
    1645084.5,
    2858157.2])
    psf = np.array([
    22.738272,
    18.45705,
    13.17205,
    9.101765,
    6.090297,
    4.0102134,
    2.4757671,
    1.5785031,
    0.9589487,
    0.5732728,
    0.3654938,
    0.244569,
    0.18614233,
    0.13940534,
    0.12265361,
    0.11508227,
    0.11333075,
    0.10806294,
    0.10814128,
    0.10147373,
    0.09369465,
    0.089339554,
    0.08518689])
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

def chi_square_cdf(k,x):
    return gammainc(k/2,x/2)

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
def find_exposure(e):
    e_index = np.argmin(np.abs(e-10**np.linspace(2.0, np.log10(500000),51)))
    return np.mean(exposurefile[0].data[e_index][np.nonzero(exposurefile[0].data[e_index])])


#Function to generate PBH photons, and place them into a background FITS file
#gamma_fraction is the fraction of PBH rest energy that is converted to the energy of gamma rays
#v_disk is the rotation speed of the Milky Way
#v_dm is the local velocity dispersion of dark matter
#z is an integer between 0 and 100 that tells us where on the sky we are looking
#   We could generate a truly random position on the sky, but this saves a lot of computation expense
def make_data(distance, mass_init, flux_matrix, filename, gamma_fraction, v_disk, v_dm, z):
    MET_start = 239902981 #Time limits for 3FGL
    MET_end = 365467563
    #Random velocity
    y = np.linspace(0, np.pi, 500)
    h = np.sin(y)
    theta = make_random(y,h)
    x_vel = np.sin(theta)*v_disk+np.random.normal(0., v_dm/np.sqrt(3.))
    y_vel = np.random.normal(0., v_dm/np.sqrt(3))
    z_vel = np.random.normal(0., v_dm/np.sqrt(3))
    tot_v = np.sqrt(x_vel**2+y_vel**2+z_vel**2)
    new_theta = make_random(y,h)
    phi = np.random.rand(1)[0]*2.*np.pi
    rx = np.cos(new_theta)
    ry = np.sin(new_theta)*np.sin(phi)
    rz = np.cos(phi)*np.sin(new_theta)
    v_tang = np.sqrt(tot_v**2-(rx*x_vel+ry*y_vel+rz*z_vel)**2)
    angle_on_sky = np.random.rand(1)[0]*np.pi*2.0
    v_ra = v_tang*np.cos(angle_on_sky)
    v_dec = v_tang*np.sin(angle_on_sky)
    print "V_RA = " + str(v_ra) + " km/s    V_DEC = " + str(v_dec) + " km/s     V_TANG = " + str(np.sqrt(v_ra**2+v_dec**2)) + " km/s"
    dt = 0.25*86400.0 #86400 seconds is 1 day
    met = MET_start

    #Work backwards given the final position and velocity to find the starting position
    f = pyfits.open(prefix+'/ROIs/'+str(z)+'_bkg.fits')
    ra_final = float(f[1].header['DSVAL2'].split('(')[1].split(',')[0])
    dec_final = float(f[1].header['DSVAL2'].split('(')[1].split(',')[1])
    f.close()
    print "Center of ROI: RA = " + str(ra_final) + " DEC = " + str(dec_final)
    mass = mass_init
    ra_init = ra_final
    dec_init = dec_final
    while mass>0 and met<MET_end:
        Temp = T(mass)
        met += dt
        mass -= dt*alpha(mass)/(mass**2)
        ra_init -= v_ra*dt*360/(parsec_to_km*distance*2*np.pi)
        dec_init -= v_dec*dt*360/(distance*parsec_to_km*2*np.pi)
    print "Starting RA: " + str(ra_init)
    print "Starting DEC: " + str(dec_init)
    g = fits.open(prefix+'/ROIs/'+str(z)+'_bkg.fits')
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
        eff_area[i] = find_exposure(e_bin_edges_MEV[e_index])/(elapsed_seconds*10000.)
    #Evolve forward in time, create photons
    tot_ra = 0.0
    tot_dec = 0.0
    while mass>0 and met<MET_end:
        Temp = T(mass)
        row = np.argmin(np.abs(np.arange(0.3, 100.0, 0.1)-Temp))
        folded_flux = eff_area*flux_matrix[row,:]
        emitted_flux = gamma_fraction*get_integral(w, folded_flux)
        #fermi_capture_rate = avg total number of photons recorded in time range [met, met+dt]
        fermi_capture_rate = emitted_flux*dt/(4.0*np.pi*(distance*parsecstometers)**2)

        for k in range(0, np.random.poisson(fermi_capture_rate)):
            tot_phots +=1
            #Some of the variables recorded for each photon are not easy to simulate, or important
            #So, just steal them from a real photon
            rand_bkg_photon = random.choice(bkg_photons)

            #MC data variables
            theEnergy = 1000.0*make_random(w,folded_flux)
            theTime = met+np.random.rand(1)[0]*dt
            theRa = np.random.normal(ra, spatial_resolution(theEnergy)/np.sqrt(2))
            theDec = np.random.normal(dec, spatial_resolution(theEnergy)/np.sqrt(2))
            theL = SkyCoord(ra=theRa*u.degree, dec=theDec*u.degree,frame='icrs').galactic.l.degree
            theB = SkyCoord(ra=theRa*u.degree, dec=theDec*u.degree,frame='icrs').galactic.b.degree
            tot_ra += theRa
            tot_dec += theDec

            #Other variables: stolen from the background photons (things like zenith angle, Run ID etc)
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
            table.add_row([
            theEnergy,
            theRa,
            theDec,
            theL,
            theB,
            theTheta,
            thePhi,
            theZenithAngle,
            theEarthAngle,
            theTime,
            theEventId,
            theRunId,
            theReconVersion,
            theCalibVersion,
            theEventClass,
            theConversionType,
            theLivetime,
            theDiff0,
            theDiff1,
            theDiff2,
            theDiff3,
            theDiff4])
        #Update variables for next time step
        met += dt
        mass -= dt*alpha(mass)/(mass**2)
        ra += v_ra*dt*360/(parsec_to_km*distance*2*np.pi) #Velocities are in km/s
        dec += v_dec*dt*360/(distance*parsec_to_km*2*np.pi)

    if tot_phots>0:
        average_ra = tot_ra/(tot_phots)
        average_dec = tot_dec/(tot_phots)
    else:
        average_dec = dec
        average_ra = ra
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
    print "RA_Final = " + str(ra) + " DEC_Final = " + str(dec)
    print "Average RA = " + str(average_ra) + " Average DEC = " + str(average_dec)
    print "V_RA = " + str(v_ra) + " km/s    V_DEC = " + str(v_dec) + " km/s     V_TANG = " + str(np.sqrt(v_ra**2+v_dec**2)) + " km/s"

    return average_ra, average_dec, v_tang, ra_init, dec_init, ra, dec

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
    for s in range(len(bin_edges)):
        bin_indices[s] = int(np.argmin(np.abs(x-bin_edges[s])))
    #Find the chi^2 value between the source flux and the model of PBH flux
    #Loop over every temperature to cover all possibilities (not computationally expensive)
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
        chi2_values[i] = chi_squared(flux, flux_err, sim_flux)
    if min(chi2_values)<11.3:
        spectral_match = True
        reconstructed_temp = np.arange(0.3, 100.0, 0.1)[np.argmin(chi2_values)]
    else:
        spectral_match = False
        reconstructed_temp = 0.0
    return spectral_match, reconstructed_temp, min(chi2_values)

#Code to search for proper motion

#not so much a gaussian, but a chi^2 for the fit. Now, with even more weight!
def gaussian_with_motion(x, y, e, t, weights, sigma, x0, y0, vx, vy, distance):
    vx2 = 360.0*vx/(2*np.pi*distance*parsec_to_km)
    vy2 = 360.0*vy/(2*np.pi*distance*parsec_to_km)
    t2 = t-MET_start
    result = (-1.0*((y-(y0+vy2*t2))**2+(np.sin(2*np.pi*y/360.0)*(x-(x0+vx2*t2)))**2)/(sigma**2))*weights
    return result

def likelihood(phots, params, distance, npred):
    x0 = params[0]
    y0 = params[1]
    vx = params[2]
    vy = params[3]
    result = gaussian_with_motion(phots[:,0], phots[:,1], phots[:,2], phots[:,3], phots[:,4], phots[:,5], x0, y0, vx, vy, distance)
    return sum(np.sort(result[np.nonzero(result)])[::-1][0:int(npred)])

def likelihood_ratio_test(photons, ra, dec, npred):
    distance = 0.02
    phots = np.zeros((len(photons),6))
    phots[:,0:4] = photons[:,0:4]
    phots[:,5] = psf_array(photons[:,2])
    params = [ra, dec, 0.0, 0.0]
    bestvx = 0.0
    bestvy = 0.0
    res = 11
    minx = ra-rad
    maxx = ra+rad
    miny = dec-rad
    maxy = dec+rad
    print "Optimizing with 2 free parameters..."
    for k in range(5):
        test_likelihood = np.zeros((res, res))
        x0_arr = np.linspace(minx, maxx, res)
        y0_arr = np.linspace(miny, maxy, res)

        for i in range(res):
            for j in range(res):
                params = [x0_arr[i], y0_arr[j], 0.0, 0.0]
                test_likelihood[i,j] = likelihood(phots, params, distance, npred)
        bestx0 = x0_arr[np.mod(int(np.argmax(test_likelihood)/(res**1)),res)]
        besty0 = y0_arr[np.mod(int(np.argmax(test_likelihood)/(res**0)),res)]
        new_spacing = 0.5*np.mean(np.diff(x0_arr))
        minx = bestx0 - new_spacing
        maxx = bestx0 + new_spacing
        miny = besty0 - new_spacing
        maxy = besty0 + new_spacing
    print "best x = " +str(bestx0) + " best y = " + str(besty0)

    params = [bestx0, besty0, 0.0, 0.0]
    f0 = likelihood(phots, params, distance, npred)

    print "Optimizing with 4 free parameters..."
    #Next, maximize the likelihood by adjusting the values of the 4 degrees of freedom
    dv = (4.0/elapsed_seconds)*(2*np.pi/360.0)*(distance*parsec_to_km)
    minx = ra-rad
    maxx = ra+rad
    miny = dec-rad
    maxy = dec+rad
    minvx = -1.0*dv
    maxvx = dv
    minvy = -1.0*dv
    maxvy = dv
    for q in range(5):
        test_likelihood = np.zeros((res, res, res, res))
        x0_arr = np.linspace(minx, maxx, res)
        y0_arr = np.linspace(miny, maxy, res)
        vx_arr = np.linspace(minvx, maxvx, res)
        vy_arr = np.linspace(minvy, maxvy, res)

        for i in range(res):
            for j in range(res):
                for k in range(res):
                    for l in range(res):
                        #startingx = bestx0-(vx_arr[i]*elapsed_seconds*(360/(2*np.pi))/(distance*parsec_to_km))
                        #startingy = bestx0-(vy_arr[j]*elapsed_seconds*(360/(2*np.pi))/(distance*parsec_to_km))
                        params = [x0_arr[i], y0_arr[j], vx_arr[k], vy_arr[l]]
                        test_likelihood[i, j, k, l] = likelihood(phots, params, distance, npred)
        bestx0 = x0_arr[np.mod(int(np.argmax(test_likelihood)/(res**3)),res)]
        besty0 = y0_arr[np.mod(int(np.argmax(test_likelihood)/(res**2)),res)]
        bestvx = vx_arr[np.mod(int(np.argmax(test_likelihood)/(res**1)),res)]
        bestvy = vy_arr[np.mod(int(np.argmax(test_likelihood)/(res**0)),res)]
        new_spacing_v = 0.5*np.mean(np.diff(vx_arr))
        new_spacing_x = 0.5*np.mean(np.diff(x0_arr))
        minx = bestx0 - new_spacing_x
        maxx = bestx0 + new_spacing_x
        miny = besty0 - new_spacing_x
        maxy = besty0 + new_spacing_x
        minvx = bestvx-new_spacing_v
        maxvx = bestvx+new_spacing_v
        minvy = bestvy-new_spacing_v
        maxvy = bestvy+new_spacing_v


    f1 = np.max(test_likelihood)
    print "best x = " +str(bestx0) + " best y = " + str(besty0)
    print "bestvx = " + str(bestvx) + " bestvy= " + str(bestvy)
    print "f0 = " + str(f0)
    print "f1 = " + str(f1)
    print " "
    return 2*(f1-f0), np.sqrt(bestvx**2+bestvy**2)


def movement_significance(photons, ra, dec, npred):
    trials = 20

    ref_value, v_recovered = likelihood_ratio_test(photons, ra, dec, npred)
    scrambled_values = np.zeros((trials))
    for i in range(trials):
        photons[:,3] = np.random.rand(len(photons[:,3]))*elapsed_seconds+MET_start
        scrambled_values[i], v_recovered_rand = likelihood_ratio_test(photons, distance, ra, dec, npred, rad)

    print "Data: " + str(ref_value)
    print "Simulation: " + str(np.mean(scrambled_values)) + " +/- " + str(np.std(scrambled_values))
    print str((ref_value-np.mean(scrambled_values))/np.std(scrambled_values)) + " sigma"
    return (ref_value-np.mean(scrambled_values))/np.std(scrambled_values), v_recovered

def perform_likelihood(elow, ehigh, num_ebins, pbh_ra, pbh_dec, prefix, pbh_present=False):
    subprocess.call('python /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/make3FGLxml.py -G /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/gll_iem_v05.fits -g gll_iem_v05 -GIF True -i iso_source_v05 -r 3.0 -ER 2.0 -I /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/iso_source_v05.txt -e /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/Templates/ -o '+ prefix +'/xmlmodel.xml /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3FGL.fit '+ prefix +'/raw_data.fits',shell=True)
    if pbh_present:
        subprocess.call('python /nfs/farm/g/glast/u/johnsarc/PBH_Detectability/add_pbh_src_logparabola.py '+ str(pbh_ra) + ' ' + str(pbh_dec) + ' ' + prefix, shell=True)
        
    #run gtbin
    evtbin['evfile'] = prefix+'/raw_data.fits'
    evtbin['outfile'] = prefix+'/ccube.fits'
    evtbin['ebinalg'] = 'LOG'
    evtbin['emin'] = elow
    evtbin['emax'] = ehigh
    evtbin['enumbins'] = num_ebins
    evtbin['algorithm'] = 'ccube'
    evtbin['coordsys'] = 'CEL'
    evtbin['nxpix'] = 70
    evtbin['nypix'] = 70
    evtbin['binsz'] = 0.1
    evtbin['xref'] = pbh_ra
    evtbin['yref'] = pbh_dec
    evtbin['axisrot'] = 0.0
    evtbin['proj'] = 'AIT'
    evtbin.run()

    #run gtexpcube2
    gtexpcube2['infile'] = prefix+'/ltcube.fits'
    gtexpcube2['outfile'] = prefix+'/exposure.fits'
    gtexpcube2['cmap'] = prefix+'/ccube.fits'
    gtexpcube2['irfs'] = 'P7REP_SOURCE_V15'
    gtexpcube2.run()
    
    #run gtsrcmaps
    srcMaps['expcube'] = prefix+'/ltcube.fits'
    srcMaps['cmap'] = prefix+'/ccube.fits'
    srcMaps['srcmdl'] = prefix+'/xmlmodel.xml'
    srcMaps['bexpmap'] = prefix+'/exposure.fits'
    srcMaps['outfile'] = prefix+'/srcmap.fits'
    srcMaps['rfactor'] = 4
    srcMaps['emapbnds'] = 'no'
    srcMaps.run()
    
    #run gtlike
    obs = BinnedObs(srcMaps=prefix+'/srcmap.fits', expCube='/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/ltcube.fits', binnedExpMap=prefix+'/exposure.fits', irfs='P7REP_SOURCE_V15')
    like = BinnedAnalysis(obs, prefix+'/xmlmodel.xml', optimizer='NEWMINUIT')
    like.tol=1e-8
    like.fit(verbosity=0,covar=True)
    
    return like
    
def make_model_cube(prefix, like):
    f = pyfits.open(prefix+'/srcmap.fits')
    model_map = np.zeros((70,70,10))
    for source in like.sourceNames():
        print "source = " + str(source)
        print like._srcCnts(source)
        for j in range(3,len(f)):
            if source == f[j].header['EXTNAME']:
                the_index = j
            
        num_photons = like._srcCnts(source)
        model_counts = num_photons*f[the_index].data/np.sum(np.sum(f[the_index].data, axis=1), axis=2)
        my_arr += model_counts
    f.close()
    return my_arr


# "Flux Matrix" = lookup table for the spectrum of the PBH at the source. Interpolated from MacGibbon 1990 paper
#Should have temperatures as rows, and energy bins as columns. Gives the total flux emitted by the PBH in ph/s
file = open(prefix+'/flux_matrix.pk1', 'rb')
flux_matrix = pickle.load(file)
file.close()
flux_matrix *= (0.35/0.25)

file = open(prefix+'/integrated_flux_matrix.pk1', 'rb')
integrated_flux_matrix = pickle.load(file)
file.close()

def test_spectral_model(like):
    ts = like.Ts('PBH_Source')
    if ts>25.0:
        source_flux = np.array([like.flux('PBH_Source',emin=100.0, emax=300.0), like.flux('PBH_Source',emin=300.0, emax= 1000.0), like.flux('PBH_Source',emin=1000.0, emax= 3000.0), like.flux('PBH_Source',emin=3000.0, emax= 10000.0), like.flux('PBH_Source',emin=10000.0, emax=100000.0)],'d')
        source_flux_unc = np.array([like.fluxError('PBH_Source',emin=100.0, emax= 300.0), like.fluxError('PBH_Source',emin=300.0, emax= 1000.0), like.fluxError('PBH_Source',emin=1000.0, emax= 3000.0), like.fluxError('PBH_Source',emin=3000.0, emax= 10000.0), like.fluxError('PBH_Source',emin=10000.0, emax=100000.0)],'d')
        spectral_match, reconstructed_temperature, chi2value = analyze_source_spectrum(source_flux, source_flux_unc, pbh_ra, pbh_dec)
    else:
        spectral_match = False
        chi2value = 0.0
    npred = like.NpredValue('PBH_Source')
    npred1000 = np.sum(like._srcCnts('PBH_Source')[np.argmin(np.abs(like.energies-1000.0)):])

    return ts, spectral_match, chi2value, npred, npred1000

def simulate_and_analyze(temp, dist, gamma_fraction, v_disk, v_dm, z, j):
    source_detected = False
    test_spectral_match = False
    motion_detected = False
    #First, simulate the PBH
    filename = prefix+'/raw_data.fits'

    print "Simulating photons..."
    pbh_ra, pbh_dec, v_tang, ra_init, dec_init, ra_final, dec_final = make_data(dist, M(temp), flux_matrix, filename, gamma_fraction, v_disk, v_dm, j)
    print "Done!"

    like = perform_likelihood(100, 100000, 10, pbh_ra, pbh_dec, prefix, pbh_present=True)

    chi2value = 0.0

    ts, spectral_match, chi2value, npred, npred1000 = test_spectral_model(like)
    print "TS: " + str(ts)
    print "Spectral match: " + str(spectral_match)
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
        filter['outfile'] = prefix+'/high_e_ROI.fits'
        filter['ra'] = pbh_ra
        filter['dec'] = pbh_dec
        filter['rad'] = 5.0
        filter['tmin'] = MET_start
        filter['tmax'] = MET_end
        filter['emin'] = 1000.0
        filter['emax'] = 300000.0
        filter.run()


        g = pyfits.open(prefix+'/high_e_ROI.fits')
        photons = np.zeros((len(g[1].data),4))
        for q in range(len(g[1].data)):
            photons[q,0] = g[1].data[q]['RA']
            photons[q,1] = g[1].data[q]['DEC']
            photons[q,2] = g[1].data[q]['ENERGY']
            photons[q,3] = g[1].data[q]['TIME']
        g.close()

        #Load disk model
        file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/disk_component.pk1', 'rb')
        g = pickle.load(file)
        file.close()
        like = perform_likelihood(1000, 100000, 5, pbh_ra, pbh_dec, prefix, pbh_present=False)

        my_arr = make_model_cube(prefix,like)
        #Assign weights to each photon
        prob_array = g[0].data*npred/my_arr
        prob_photons= np.zeros((len(photons), 5))
        prob_photons[:,0:3] = photons
        e_array = 10**np.linspace(3, 5, 5)
        for i in range(len(photons)):
            x_index = int((photons[i,0]-pbh_ra)/0.1+34)
            y_index = int((photons[i,1]-pbh_dec)/0.1+34)
            z_index = np.argmin(np.abs(e_array-photons[i,2]))
            prob_photons[i, 4] = prob_array[z_index,x_index, y_index]
    

        #run the algorithm
        significance, v_recovered = movement_significance(prob_photons, ra, dec, npred)


        if sig>4.0:
            print "Significant proper motion!"
            motion_detected = True

    print "source detected: "+ str(source_detected)
    print "spectral match: "+ str(spectral_match)
    print "proper motion: " + str(motion_detected)
    #Save results to a file
    if z == 0:
        file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/results_low/'+os.environ['LSB_JOBID']+"_"+str(z)+'.pk1','wb')
        pickle.dump({"Temperature":temp, "Distance":dist, "Convergence":convergence, "Source_Detected":source_detected, "Motion_Detected":motion_detected, "GLON":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.l.degree, "GLAT":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.b.degree, "V_Recovered":v_recovered , "Spectral_Match": spectral_match, "MIN_CHI2":chi2value, "V_True": v_tang, "TS":ts, "Sig": sig}, file)
        file.close()

    if z == 1:
        file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/results/'+os.environ['LSB_JOBID']+"_"+str(z)+'.pk1','wb')
        pickle.dump({"Temperature":temp, "Distance":dist, "Convergence":convergence, "Source_Detected":source_detected, "Motion_Detected":motion_detected, "GLON":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.l.degree, "GLAT":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.b.degree, "V_Recovered":v_recovered , "Spectral_Match": spectral_match, "MIN_CHI2":chi2value, "V_True": v_tang, "TS":ts, "Sig": sig}, file)
        file.close()

    if z == 2:
        file = open('/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/results_high/'+os.environ['LSB_JOBID']+"_"+str(z)+'.pk1','wb')
        pickle.dump({"Temperature":temp, "Distance":dist, "Convergence":convergence, "Source_Detected":source_detected, "Motion_Detected":motion_detected, "GLON":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.l.degree, "GLAT":SkyCoord(ra=pbh_ra*u.degree, dec=pbh_dec*u.degree,frame='icrs').galactic.b.degree, "V_Recovered":v_recovered , "Spectral_Match": spectral_match, "MIN_CHI2":chi2value, "V_True": v_tang, "TS":ts, "Sig": sig}, file)
        file.close()



#Command-line arguments:
#1: Temperature in GeV
#2: Distance in pc
#3: batch or local
#4: which model are we using (worst-case, baseline, best-case)
#5: Random integer between 0 and 100 (determines spot in the sky)
print "Temperature = " + str(float(sys.argv[1])) + " Distance = " + str(float(sys.argv[2]))

#To get the upper and lower confidence intervals for the limit
gamma_fraction = np.array([0.25/0.35, 1.0, 0.45/0.35],'d')
v_disk = np.array([300.0, 50.0, 100.0],'d')
v_dm = np.array([350.0, 70.0, 150.0],'d')
z = int(sys.argv[4])
j = int(sys.argv[5])
simulate_and_analyze(float(sys.argv[1]),float(sys.argv[2]), gamma_fraction[z], v_disk[z], v_dm[z], z, j)
