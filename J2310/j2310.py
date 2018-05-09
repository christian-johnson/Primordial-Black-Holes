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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib.pyplot import rc
import os

rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['axes.labelcolor'] = '#000000'

rcParams['xtick.labelsize'] =14
rcParams['ytick.labelsize'] =14
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['xtick.color'] = '#000000'
rcParams['ytick.color'] = '#000000'

rcParams['text.color'] = '#000000'
rcParams['font.size'] = 18


def make_4fgl_plot(fits_file, image_path, time_display=False, time_start = 0.0, time_end = 1.0):
    met_start4 = 239557417
    met_end4 = 521403273

    roi_ra = 347.507419753
    roi_dec =  -5.8267
    fgl_4 = pyfits.open('4fgl_prelim.fits')
    image_data = fits.getdata(fits_file)
    filename = get_pkg_data_filename(fits_file)
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[10,10])
    if time_display:
        ax1 = plt.axes([0.0, 0.1, 1.0, 1.0],projection=wcs)
    else:
        ax1 = plt.axes([0.0, 0.0, 1.0, 1.0],projection=wcs)
    
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
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=5.0)
    cb = plt.colorbar(mappable,label='Counts per pixel',fraction=0.046, pad=0.04)

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')


    plt.xlabel('RA J2000')
    plt.ylabel('DEC J2000')
    
    if time_display:
        ax2 = plt.axes([0.0, 0.0, 1.0, 0.1])
        ax2.set_position([0.0,0.0, 1.0, 0.05])
    
        plt.xlim([met_start4, met_end4])
        plt.ylim([0.0, 0.5])
        plt.axvline(352700000,linestyle='--',linewidth=2.0, color='red')
        plt.axvspan(time_start, time_end, alpha=0.5, color='green')
        plt.gca().axes.get_yaxis().set_visible(False)
        ax2.set_xlabel('MET')
    
    plt.savefig(image_path,bbox_inches='tight')
    #plt.show()
    
def make_3fgl_plot(fits_file, image_path, time_display=False, time_start = 0.0, time_end = 1.0):
    roi_ra = 347.507419753
    roi_dec =  -5.8267
    
    met_start3 = 239902981
    met_end3 = 365467563
    
    fgl_4 = pyfits.open('3FGL.fits')
    image_data = fits.getdata(fits_file)
    filename = get_pkg_data_filename(fits_file)
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=[10,10])
    if time_display:
        ax1 = plt.axes([0.0, 0.1, 1.0, 1.0],projection=wcs)
    else:
        ax1 = plt.axes([0.0, 0.0, 1.0, 1.0],projection=wcs)
        
    ax=plt.gca()
    n_sources = 0
    for entry in fgl_4[1].data:
        dist = np.sqrt((entry['RAJ2000']-roi_ra)**2+(entry['DEJ2000']-roi_dec)**2)

        if dist<5.0 and n_sources == 0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'),label='3FGL Point Sources')
            n_sources+=1
        if dist<5.0 and n_sources>0:
            ax.scatter([float(entry['RAJ2000'])], [float(entry['DEJ2000'])], color='#2dff7b',marker='x',s=100.0,transform=ax.get_transform('world'))
            
    mappable = plt.imshow(image_data,cmap='magma',interpolation='nearest',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0.0, vmax=5.0)

    leg = plt.legend(loc=1,frameon=True)
    leg.get_frame().set_alpha(0.25)
    leg.get_frame().set_edgecolor('white')

    text1 = leg.get_texts()
    text1[0].set_color('white')

    cb = plt.colorbar(mappable,label='Counts per pixel',fraction=0.046, pad=0.04)

    plt.xlabel('RA J2000')
    plt.ylabel('DEC J2000')
    
    if time_display:
        ax2 = plt.axes([0.0, 0.0, 1.0, 0.1])
        ax2.set_position([0.0,0.0, 1.0, 0.05])
    
        plt.xlim([met_start3, met_end3])
        plt.ylim([0.0, 0.5])
        plt.axvline(352700000,linestyle='--',linewidth=2.0, color='red')
        plt.axvspan(time_start, time_end, alpha=0.5, color='green')
        plt.gca().axes.get_yaxis().set_visible(False)
        ax2.set_xlabel('MET')
    
    plt.savefig(image_path,bbox_inches='tight')
    #plt.show()


def make_movie():
    #Run gtselect on sequential time bins, get images
    from gt_apps import filter
    from gt_apps import evtbin

    fgl3_raw_filename = 'J2310.1-0557_raw.fits'
    fgl4_raw_filename = 'j2310.fits'

    met_start4 = 239557417
    met_end4 = 521403273

    met_start3 = 239902981
    met_end3 = 365467563

    num_bins = 5
    for i in range(num_bins):
        tmin = met_start3+i*(met_end3-met_start3)/num_bins
        tmax = met_start3+(i+1)*(met_end3-met_start3)/num_bins
        filter['infile'] = fgl3_raw_filename
        filter['outfile'] = '3fgl_fits_files/'+str(i)+'_raw.fits'
        filter['ra'] = 347.507419753
        filter['dec'] = -5.8267
        filter['rad'] = 5.0
        filter['tmin'] = tmin
        filter['tmax'] = tmax
        filter['emin'] = 1000.0
        filter['emax'] = 800000.0
        filter.run()
    
        evtbin['evfile'] = '3fgl_fits_files/'+str(i)+'_raw.fits'
        evtbin['outfile'] = '3fgl_fits_files/'+str(i)+'_image.fits'
        evtbin['algorithm'] = 'cmap'
        evtbin['coordsys'] = 'CEL'
        evtbin['nxpix'] = 100
        evtbin['nypix'] = 100
        evtbin['binsz'] = 0.1
        evtbin['xref'] = 347.507419753
        evtbin['yref'] = -5.8267
        evtbin['axisrot'] = 0.0
        evtbin['proj'] = 'AIT'
        evtbin.run()
        if i<10:
            image_path = '3fgl_images/0'+str(i)+'_image.png'
        else:
            image_path = '3fgl_images/'+str(i)+'_image.png'
            
        make_3fgl_plot(fits_file = '3fgl_fits_files/'+str(i)+'_image.fits', image_path = image_path, time_display=True, time_start = tmin, time_end=tmax)
    
    num_bins = 12
    for i in range(num_bins):
        tmin = met_start4+i*(met_end4-met_start4)/num_bins
        tmax = met_start4+(i+1)*(met_end4-met_start4)/num_bins
        filter['infile'] = fgl4_raw_filename
        filter['outfile'] = '4fgl_fits_files/'+str(i)+'_raw.fits'
        filter['ra'] = 347.507419753
        filter['dec'] = -5.8267
        filter['rad'] = 5.0
        filter['tmin'] = tmin
        filter['tmax'] = tmax
        filter['emin'] = 1000.0
        filter['emax'] = 800000.0
        filter.run()
    
        evtbin['evfile'] = '4fgl_fits_files/'+str(i)+'_raw.fits'
        evtbin['outfile'] = '4fgl_fits_files/'+str(i)+'_image.fits'
        evtbin['algorithm'] = 'cmap'
        evtbin['coordsys'] = 'CEL'
        evtbin['nxpix'] = 100
        evtbin['nypix'] = 100
        evtbin['binsz'] = 0.1
        evtbin['xref'] = 347.507419753
        evtbin['yref'] = -5.8267
        evtbin['axisrot'] = 0.0
        evtbin['proj'] = 'AIT'
        evtbin.run()
        if i<10:
            image_path = '4fgl_images/0'+str(i)+'_image.png'
        else:
            image_path = '4fgl_images/'+str(i)+'_image.png'
    
        make_4fgl_plot(fits_file = '4fgl_fits_files/'+str(i)+'_image.fits', image_path = image_path, time_display=True, time_start = tmin, time_end=tmax)
        
    os.system('convert -antialias -delay 50 3fgl_images/*.png 3fgl_images/j2310_3fgl.gif')
    os.system('convert -antialias -delay 50 4fgl_images/*.png 4fgl_images/j2310_4fgl.gif')


make_movie()
#make_3fgl_plot(fits_file = '4year_image.fits',image_path = '3fgl_overlay.pdf', time_display=False)
#make_4fgl_plot(fits_file = '9year_image.fits',image_path = '4fgl_overlay.pdf', time_display=False)

