import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import netCDF4 as nc

# laptop test

# scitools
import cartopy as cp
import cartopy.io.shapereader as shpreader

 # homemade extras
import cartopyXtras as cpx
import palettable as palet

def make_mesh(c,x_offset=0.,y_offset=0.,multiplier=1.,dx=None,dy=None, transpose=False, unstacked=False, keep_axes=False):
    """ 
    This function converts a pandas Series with lon/lat index to plottable grids.
    Also adds an extra row and column for correct pcolormesh plotting, checks
    for incorrect grid spacing, and masks the array for nans.
    
    usage:
    
    make_mesh(c)
    
    c = pandas Series with lon/lat MultiIndex.
    
    returns: 
    
    x, y, c
    
    x = meshgrid of x coordinates
    y = meshgrid of y coordinates
    c = masked array of data values
    
    """
    # unstack the series to a DataFrame if needed
    if not unstacked:
        if transpose:
            c = c.unstack(level=1).sort_index(ascending=0)
        else:
            c = c.unstack(level=0).sort_index(ascending=0)
    
    if keep_axes:
        x, y = np.meshgrid(c.columns, c.index) # take x and y from columns
    else:
        # if not set, get grid spacing
        if dx == None: dx = grid_spacing(c.columns)
        if dy == None: dy = grid_spacing(c.index)

        # make x and y with regular grid spacing, add an extra entry for pcolormesh plotting
        x = np.arange(c.columns[0], c.columns[-1] + 2*dx, dx)
        y = np.arange(c.index[0], c.index[-1] + 2*dy, dy)
    
        # reindex the DataFrame according to x and y. fill empty values with nans
        c = c.reindex(index=y,columns=x,fill_value=np.nan)
    
        # make meshgrids
        x, y = np.meshgrid(x*multiplier + x_offset,y*multiplier + y_offset)
    
    return x,y,sp.ma.masked_invalid(c.values)

def grid_spacing(x):
    """
    finds out smallest spacing, assume it's dx. Also assumes array is increasing in one direction.
    
    usage: grid_spacing(x)
    returns: smallest spacing in between values
    
    where x is a pandas Series or an array
    """
    
    diff = np.asarray(x[1:],dtype=float)-np.asarray(x[:-1],dtype=float)
    try: 
        step = np.min(diff[diff>0])
    except:
        step = np.max(diff[diff<0])
        
    return step
    
# close all previously open windows
plt.close('all')

# define the extent
lat_bot, lat_top    = (60, 90.)  # southern and northern limits of the domain
lon_left, lon_right = (-180., 180.) # western and eastern limits of the domain
central_lon = -45                   # central longitude (-45 deg is a projection centered on Greenland)
model_cmap = palet.scientific.sequential.LaJolla_20.get_mpl_colormap()  # color map to use for the model, see https://jiffyclub.github.io/palettable

# define some projections
npstere = cp.crs.NorthPolarStereo(central_longitude=central_lon)
carree = cp.crs.PlateCarree()
ortho = cp.crs.Orthographic(central_longitude=central_lon,central_latitude=90)
geostat = cp.crs.Geostationary(central_longitude=central_lon,false_northing=80)
lambert = cp.crs.LambertConformal(central_latitude=90)

map_prj = npstere # projection of the map is north polar stereographic
model_prj = carree # projection of the model is plate carree (normal lat/lon grid)

# load the model output
dynNEE = pd.read_table('Dnee.txt', index_col=[0,1], sep='\s+')
dynNEE_winter = dynNEE[['Dec', 'Jan', 'Feb']].sum(axis=1)*1000 # select only the winter months and convert to g/m2

statNEE = pd.read_table('Snee.txt', index_col=[0,1], sep='\s+')
statNEE_winter = statNEE[['Dec', 'Jan', 'Feb']].sum(axis=1)*1000 # select only the winter months and convert to g/m2

# difference of dynamic and static
diffNEE_winter = dynNEE_winter-statNEE_winter

# minimum and maximum values of both model outputs
model_minmax = (np.min([statNEE_winter.min(), dynNEE_winter.min()]), 
                np.max([statNEE_winter.max(), dynNEE_winter.max()]))

# loop through the three types of plots
for plot_type, model_output in [('static', statNEE_winter), ('dynamic', dynNEE_winter), ('difference', diffNEE_winter)]:

    # make new figure
    fig = plt.figure(figsize=(6, 6), dpi=150)

    # define figure and extent
    ax = plt.axes(projection=map_prj)
    ax.set_extent([lon_left,lon_right-1,lat_bot-1,lat_top], crs=carree) # define the extent of the map

    # turn model output into a plottable grid and apply offsets (Python uses top-left corner as coordinates, not center of grid cell)
    plot_data = make_mesh(model_output, x_offset=-0.25, y_offset=0.25)

    # plot the model data
    p1 = plt.pcolormesh(*plot_data, cmap=model_cmap, vmin=model_minmax[0], vmax=model_minmax[-1], transform=model_prj)

    # Add some natural earth features to the map, but only those north from the southern limit of the domain (optional)
    cpx.add_natural_earth_features(ax, features=['coastline', 'lakes', 'glaciated'], scale='50m', poly=cpx.LonLatPolygon(lat_bot))
    # cpx.add_natural_earth_features(ax, features=['rivers'], scale='110m', poly=cpx.LonLatPolygon(lat_bot))

    # clip the rest of the graph to a circle defined by lat_bot, and add bounding circle 
    cpx.clip_to_circle(ax, lat_bot, transform=map_prj, central_lon=central_lon)

    # add colorbar
    cbar = cpx.add_colorbar(ax, p1, title=r'NEE (g m$^{-2}$)', fontsize=12, location='bottom', shrink=0.7)
    # cbar.ax.invert_yaxis() # invert the y-axis of the colorbar if you need to

    # save the figure
    print('saving figure..')
    plt.savefig('NEE_{0}.png'.format(plot_type), dpi=150)
