#This script is similar to the PBH Detectability script, but instead of simulating a PBH, it scans through the 3FGL
#The exact same procedures for spectral analysis and proper motion analysis are used, for consistency
#The only extra cut here is that each candidate source must not be associated with an astrophysical object

#Importing useful packages
print "initializing"
import numpy as np
import pyfits
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
print "done!"

#Defining constants & functions for later use
G = 6.67*10**(-11)
c = 3.0*10**8
hbar = 1.055*10**(-34)
joulestogev = 6.242*(10**9)
seconds_per_year = float(86400*365)
MET_start = 239902981 #Time limits for 3FGL
MET_end = 365467563#508966570#
elapsed_seconds = float(MET_end)-float(MET_start)
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

#not so much a gaussian, but a chi^2 for the fit. Now, with even more weight!
def gaussian_with_motion(x, y, e, t, weights, sigma, x0, y0, vx, vy, distance):
    vx2 = 360.0*vx/(2*np.pi*distance*parsec_to_km)
    vy2 = 360.0*vy/(2*np.pi*distance*parsec_to_km)
    t2 = t-MET_start
    result = (-1.0*((y-(y0+vy2*t2))**2+(np.sin(2*np.pi*y/360.0)*(x-(x0+vx2*t2)))**2)/(sigma**2))*(1.0-weights)
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
    rad = 1.0
    phots = np.zeros((len(photons),6))
    phots[:,0:4] = photons[:,0:4]
    phots[:,5] = psf_array(photons[:,2])
    params = [ra, dec, 0.0, 0.0]
    bestvx = 0.0
    bestvy = 0.0
    res = 5
    minx = ra-rad
    maxx = ra+rad
    miny = dec-rad
    maxy = dec+rad
    print "Optimal likelihood = " + str(likelihood(phots, [236.239840427, -35.3952977267, -150.288464424, 128.12239613], 0.02, 671))
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
        scrambled_values[i], v_recovered_rand = likelihood_ratio_test(photons, ra, dec, npred)

    print "Data: " + str(ref_value)
    print "Simulation: " + str(np.mean(scrambled_values)) + " +/- " + str(np.std(scrambled_values))
    print str((ref_value-np.mean(scrambled_values))/np.std(scrambled_values)) + " sigma"
    return (ref_value-np.mean(scrambled_values))/np.std(scrambled_values), v_recovered



#Step 1: gtlike calculation
#Step 1a: Make xml model
#Step 1b: Make sourcemap, exposure map, counts cube

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
    like = BinnedAnalysis(obs_complete, prefix+'/xmlmodel.xml', optimizer='NEWMINUIT')
    like.tol=1e-8
    like.fit(verbosity=0)
    
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

pbh_ra = 236.239840427
pbh_dec = -35.3952977267
npred = 671

#Load photons
file = open('photons.pk1')
prob_photons = pickle.load(file)
file.close()  

file = open('prob_array.pk1', 'rb')
g = pickle.load(file)
file.close()

plt.imshow(g[0], interpolation='none')
plt.show()  
    
significance, v_recovered = movement_significance(prob_photons, pbh_ra, pbh_dec, 671)
    


