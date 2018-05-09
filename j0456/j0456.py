#Code to make cool plot of 3FGL data vs 9 years of Pass 8 data
#Should show that proper motion algorithm is confused by nearby point sources

import numpy as np
import matplotlib.pyplot as plt
import pyfits
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as colors


def make_4fgl_plot():

    roi_ra = 74.0667
    roi_dec =  -69.4077
    
    roi_l = 280.7476
    roi_b = -35.28291
    
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
        dist = np.sqrt((entry['GLON']-roi_l)**2+(entry['GLAT']-roi_b)**2)

        if dist<5.0 and n_sources == 0:
            ax.scatter([float(entry['GLON'])], [float(entry['GLAT'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'),label='4FGL Preliminary Sources')
            n_sources+=1
        if dist<5.0 and n_sources>0:
            ax.scatter([float(entry['GLON'])], [float(entry['GLAT'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'))
            
    #c = Wedge((roi_l, roi_b), 15.0, theta1=0.0, theta2=360.0, width=10.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
    #ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6), vmin=0.0, vmax=100.0)
    cb = plt.colorbar(mappable,label='Counts per pixel')

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')


    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.savefig('4fgl_overlay.pdf',bbox_inches='tight')
    #plt.show()

def make_3fgl_plot():
    roi_ra = 74.0667
    roi_dec =  -69.4077
    
    roi_l = 280.7476
    roi_b = -35.28291
    
    fgl_4 = pyfits.open('3FGL.fits')
    image_data = fits.getdata('4year_image.fits')
    filename = get_pkg_data_filename('4year_image.fits')
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[15,12])
    plt.subplot(projection=wcs)
    ax=plt.gca()
    #ax.grid(color='white',ls='dotted')
    n_sources = 0
    for entry in fgl_4[1].data:
        dist = np.sqrt((entry['GLON']-roi_l)**2+(entry['GLAT']-roi_b)**2)

        if dist<5.0 and n_sources == 0:
            ax.scatter([float(entry['GLON'])], [float(entry['GLAT'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'),label='3FGL Point Sources')
            n_sources+=1
        if dist<5.0 and n_sources>0:
            ax.scatter([float(entry['GLON'])], [float(entry['GLAT'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'))
            
    #c = Wedge((roi_ra, roi_dec), 15.0, theta1=0.0, theta2=360.0, width=10.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('fk5'))
    #ax.add_patch(c)
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=100.0)
    cb = plt.colorbar(mappable,label='Counts per pixel')

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')


    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.savefig('3fgl_overlay.pdf',bbox_inches='tight')
    #plt.show()

make_3fgl_plot()
make_4fgl_plot()