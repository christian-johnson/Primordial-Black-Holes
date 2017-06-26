import numpy as np
from gt_apps import filter
from astropy import units as u
from astropy.coordinates import SkyCoord
MET_start = 239902981 #Time limits for 3FGL
MET_end = 365467563
y = np.linspace(0, np.pi, 500)
h = np.sin(y)
def get_integral(x,g):
    if len(x) != len(g):
        print "Integral must be performed with equal-sized arrays!"
        print "Length of x is " + str(len(x)) + " Length of g is " + str(len(g))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))
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
    
for i in range(100):
    print "ROI " + str(i+1)
    l_0 = 360*np.random.rand(1)[0] #phi
    b_0 = (make_random(y,h)-np.pi*0.5)*360/(2*3.141593) #theta
    ra = SkyCoord(l=l_0*u.degree,b=b_0*u.degree,frame='galactic').icrs.ra.degree
    dec = SkyCoord(l=l_0*u.degree,b=b_0*u.degree,frame='galactic').icrs.dec.degree
    print "RA = " + str(ra) + " DEC = " + str(dec)

    prefix = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/ROIs/'+str(i)+'_'
    filter['infile'] = '/nfs/farm/g/glast/u/johnsarc/PBH_Detectability/3fgl_all_sky-ft1.fits'
    filter['outfile'] = prefix+'bkg.fits'
    filter['ra'] = ra
    filter['dec'] = dec
    filter['rad'] = 7
    filter['emin'] = 100.0
    filter['emax'] = 100000.0
    filter['tmin'] = MET_start
    filter['tmax'] = MET_end
    print "Selecting photons..."
    filter.run()
    print "Done!"
    print " "
