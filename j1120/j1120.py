#Code to make cool plot of 3FGL data vs 9 years of Pass 8 data
#Should show that proper motion algorithm is confused by nearby point sources

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyfits
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as colors
import pickle

MET_start = 239902981. #Time limits for 3FGL
MET_end = 365467563.#508966570#
elapsed_seconds = float(MET_end)-float(MET_start)
kmtoparsecs = 3.241*(10**-14)
parsec_to_km = 3.086*(10**13)


def sigma(E): #In MeV
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

def make_4fgl_plot():

    roi_ra = 170.172
    roi_dec =  7.2235
    fgl_4 = pyfits.open('4fgl_prelim.fits')
    image_data = fits.getdata('9year_image.fits')
    filename = get_pkg_data_filename('9year_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[15,12])
    plt.subplot(projection=wcs)
    ax=plt.gca()
    #ax.grid(color='white',ls='dotted')
    n_sources = 0
    for entry in fgl_4[1].data:
        dist = np.sqrt((entry['RAJ2000']-roi_ra)**2+(entry['DEJ2000']-roi_dec)**2)

        if dist<5.0 and n_sources == 0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'),label='4FGL Preliminary Sources')
            n_sources+=1
        if dist<5.0 and n_sources>0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'))
            
    #c = Wedge((roi_ra, roi_dec), 15.0, theta1=0.0, theta2=360.0, width=10.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('fk5'))
    #ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=50.0)
    cb = plt.colorbar(mappable,label='Counts per pixel')

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')


    plt.xlabel('RA J2000')
    plt.ylabel('DEC J2000')
    plt.savefig('4fgl_overlay.pdf',bbox_inches='tight')
    #plt.show()

def make_3fgl_plot():
    roi_ra = 170.172
    roi_dec =  7.2235
    fgl_4 = pyfits.open('3FGL.fits')
    image_data = fits.getdata('4year_image.fits')
    filename = get_pkg_data_filename('4year_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[10,7])
    plt.subplot(projection=wcs)
    ax=plt.gca()
    #ax.grid(color='white',ls='dotted')
    n_sources = 0
    for entry in fgl_4[1].data:
        dist = np.sqrt((entry['RAJ2000']-roi_ra)**2+(entry['DEJ2000']-roi_dec)**2)

        if dist<5.0 and n_sources == 0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'),label='3FGL Point Sources')
            n_sources+=1
        if dist<5.0 and n_sources>0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'))
    
    ax.scatter(170.253018519, 7.20652469136, marker='x',color='white',s=150, transform=ax.get_transform('world'), label='Best Fit 2 parameters')
    ax.scatter(170.063975309, 7.08435390947, marker='x', color='red', s=150, transform=ax.get_transform('world'), label='Starting position')
    endx = 170.063975309+37.6875356105*elapsed_seconds*360.0/(2.*np.pi*0.02*parsec_to_km)
    endy = 7.08435390947+21.8005182572*elapsed_seconds*360.0/(2.*np.pi*0.02*parsec_to_km)
    ax.scatter(endx, endy, marker='x', color='Blue', s=150, transform=ax.get_transform('world'), label='Ending position')
    print endx
    print endy
    #c = Wedge((roi_ra, roi_dec), 15.0, theta1=0.0, theta2=360.0, width=10.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('fk5'))
    #ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=50.0)
    cb = plt.colorbar(mappable,label='Counts per pixel')

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')


    plt.xlabel('RA J2000')
    plt.ylabel('DEC J2000')
    plt.savefig('3fgl_overlay.pdf',bbox_inches='tight')
    plt.show()

def make_halftime_plot(half):
    roi_ra = 170.172
    roi_dec =  7.2235
    image_data = fits.getdata('half'+half+'_image.fits')
    filename = get_pkg_data_filename('half'+half+'_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[15,12])
    plt.subplot(projection=wcs)
    ax=plt.gca()
    #ax.grid(color='white',ls='dotted')
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=15.0)
    cb = plt.colorbar(mappable,label='Counts per pixel')


    plt.xlabel('RA J2000')
    plt.ylabel('DEC J2000')
    plt.savefig('half'+half+'.pdf',bbox_inches='tight')
    #plt.show()

print np.logspace(3.0, 5.0, 3.0)
g = pyfits.open('J1120.6+0713_raw.fits')
phots = np.zeros((len(g[1].data), 4))
i = 0
for entry in g[1].data:
    phots[i] = np.array([entry['RA'], entry['DEC'], entry['ENERGY'], entry['TIME']])
    i += 1
print phots

file = open('photons.pk1')
[f, a, b, c] = pickle.load(file)
file.close()

file = open('prob_array.pk1','rb')
h = pickle.load(file)
file.close()


ax = plt.gca()

my_colors = cm.RdYlGn((f[:,3]-np.min(f[:,3]))/(np.max(f[:,3])-np.min(f[:,3])))
for entry in f:
    if entry[2]>1000.0 and entry[0]<171.0 and entry[0]>169.0 and entry[1]<8.0 and entry[1]>6.0:
        c = Circle((entry[0], entry[1]), radius=0.025, edgecolor='black',facecolor=cm.plasma((entry[3]-np.min(f[:,3]))/(np.max(f[:,3])-np.min(f[:,3]))),alpha = 0.5) 
        ax.add_patch(c)

ax.scatter(170.253018519, 7.20652469136, marker='x',color='white',s=150, label='Best Fit 2 parameters')
ax.scatter(170.063975309, 7.08435390947, marker='x', color='red', s=150, label='Starting position')
endx = 170.063975309+37.6875356105*elapsed_seconds*360.0/(2.*np.pi*0.02*parsec_to_km)
endy = 7.08435390947+21.8005182572*elapsed_seconds*360.0/(2.*np.pi*0.02*parsec_to_km)
ax.scatter(endx, endy, marker='x', color='Blue', s=150, label='Ending position')
plt.legend()
ax.set_axis_bgcolor('black')
plt.show()
#make_3fgl_plot()
#make_4fgl_plot()
"""
from gt_apps import filter
from gt_apps import evtbin

for i in range(10):
    max_met = 521483035
    min_met = 239557417
    filter['infile'] = 'j1120.fits'
    filter['outfile'] = 'half'+str(i)+'.fits'
    filter['ra'] = 170.172
    filter['dec'] = 7.2235
    filter['rad'] = 5.0
    filter['tmin'] = min_met+i*(max_met-min_met)/10.0
    filter['tmax'] = min_met+(i+1)*(max_met-min_met)/10.0
    filter['emin'] = 1000.0
    filter['emax'] = 800000.0
    filter.run()
    
    evtbin['evfile'] = 'half'+str(i)+'.fits'
    evtbin['outfile'] = 'half'+str(i)+'_image.fits'
    evtbin['ebinalg'] = 'LOG'
    evtbin['algorithm'] = 'cmap'
    evtbin['coordsys'] = 'CEL'
    evtbin['nxpix'] = 100
    evtbin['nypix'] = 100
    evtbin['binsz'] = 0.1
    evtbin['xref'] = 170.172
    evtbin['yref'] = 7.2235
    evtbin['axisrot'] = 0.0
    evtbin['proj'] = 'AIT'
    evtbin.run()

    make_halftime_plot(str(i))
    
"""