import os, sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import cartopy as cp
import custom as cs
# import colormaps as cmaps

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from descartes import PolygonPatch
from matplotlib.colorbar import make_axes
from matplotlib.text import Text as txtType
from matplotlib.colors import LogNorm, SymLogNorm

import cartopy.crs as ccrs

# # register colormaps
# cmaps.brew2('RdBu','Diverging', 11, name='div_trend_r')
# cmaps.brew2('RdBu','Diverging', 11, reverse=True, name='div_trend')
# cmaps.brew2('RdYlBu','Diverging', 11, reverse=True, name='div_BuYlRd')
# cmaps.brew2('RdYlBu','Diverging', 11, reverse=False, name='div_BuYlRd_r')
# cmaps.brew2('PuBu','Sequential', 5, reverse=True, name='seq_BuPu')
# cmaps.brew2('Oranges','Sequential', 9, name='seq_oranges')
# cmaps.brew2('GnBu','Sequential', 9, name='seq_GnBu')
# cmaps.brew2('YlOrBr','Sequential', 9, name='seq_YlOrBr')
# cmaps.brew2('YlOrRd','Sequential', 9, name='seq_YlOrRd')
# cmaps.brew2('YlGnBu','Sequential', 9, name='seq_YlGnBu')
# cmaps.brew2('OrRd','Sequential', 9, name='seq_OrRd')

def plot_map(land_data, ocean_data=False, projection=cp.crs.NorthPolarStereo(central_longitude=-45), 
            fig_ax=None, fig_dpi=150, fig_title='', fontsize=8,
            lat_bot=49.75, lat_top=90., lon_left=-180., lon_right=180., central_lon=-45.,
            feat_props='none', feat_scale='110m', linewidth=0.5, land_projection = cp.crs.PlateCarree(), ocean_projection=cp.crs.PlateCarree(),
            land_title = r'',  land_cmap = 'div_trend', land_minmax=[np.nan,np.nan], land_scale='Linear', land_cb_extend ='neither', land_cb_bounds=None,
            ocean_title = r'', ocean_cmap = 'bone',     ocean_minmax=[-30.,0.],     cb_location='bottom', ocean_cb_extend='neither', ocean_cb_bounds=None):

    """
    Function that plots a map with data on land and/or ocean

    Usage:
    
    plot_map(land_data, ocean_data, projection, 
                fig_ax, fig_dpi, fig_title, fontszie,
                lat_bot, lat_top, lon_left, lon_right, central_lon,
                feat_props, feat_scale, land_projection, ocean_projection,
                land_title, land_cmap,  land_minmax,  land_scale,  land_cb_extend, 
                ocean_title, ocean_cmap, ocean_minmax, cb_location, ocean_cb_extend)
                
    Default values:
    
    land_data        !should be provided!
    ocean_data       : False
    projection       : cp.crs.NorthPolarStereo(central_longitude=-45), 
    fig_ax           : None
    fig_dpi          : 150
    fig_title        : ''
    fontsize         : 10
    feat_props       : 'none'
    feat_scale       : '110m'
    lat_bot          : 49.75
    lat_top          : 90.
    lon_left         : 180.
    lon_right        : 180.
    central_lon      : -45.
    land_projection  : cp.crs.PlateCarree() 
    ocean_projection : cp.crs.PlateCarree()
    land_title       : ''
    land_cmap        : 'trend' (alternatives: corr, ice_mean, flux_mean)
    land_minmax      : [np.nan,np.nan]
    land_scale       : 'Linear'
    land_cb_extend   : 'neither'
    land_cb_bound    : None
    ocean_title      : ''
    ocean_cmap       : 'bone'
    ocean_minmax     : [-30.,0.] 
    ocean_cb_extend  : 'neither'
    ocean_cb_bound   : None
    cb_location      : 'bottom'
    linewidth        : 1
    """
    if fig_ax == None:
        # make new figure
        fig = plt.figure(figsize=(3.74,5.0), dpi=int(fig_dpi))
        fig.figurePatch.set_alpha(0.0) 

        # define figure and extent
        ax = plt.axes([0.05,0.05,0.9,0.9], projection=projection)
    else:
        fig_dpi=fig_ax.figure.dpi
        ax = fig_ax
        fig = None

    ax.set_extent([lon_left,lon_right,lat_bot-1,lat_top],cp.crs.PlateCarree()) # extent in a circle around the north pole
    
    # unpack Series if it is one, otherwise it should be a tuple like this: (lon, lat, values)
    if type(land_data) == pd.Series: 
        land_data = cs.make_mesh(land_data, transpose=True)
    else:
        land_data = (land_data[0].values, land_data[1].values, sp.ma.masked_invalid(land_data[2].values))

    # get the ocean data into the right format
    plot_ocean = True
    if ocean_data == False:
        plot_ocean = False
    elif type(ocean_data) == pd.Series: 
        ocean_data = cs.make_mesh(ocean_data, multiplier=1e3)
    elif len(ocean_data) == 3:
        ocean_data = (ocean_data[0].values, ocean_data[1].values, sp.ma.masked_invalid(ocean_data[2].values))

    # plot the ocean data
    if plot_ocean: p2 = plt.pcolormesh(*ocean_data, cmap=ocean_cmap, vmin=ocean_minmax[0], vmax=ocean_minmax[-1], transform=ocean_projection,zorder=1)

    # Add some natural earth features to the map, but only where they intersect the defined polygon (optional)
    add_natural_earth_features(ax, features=['ocean','land', 'coastline', 'glaciated'], poly=LonLatPolygon(lat_bot), linewidth=linewidth, props=feat_props, scale=feat_scale)
    
    # plot the land data
    if np.any(np.isnan(land_minmax)): land_minmax = [-abs(land_data[2]).max(), abs(land_data[2]).max()]
    
    if str.lower(land_scale) == 'linear':
        land_scale = 'Linear'
        cb_norm = None
    elif str.lower(land_scale) == 'log': 
        land_scale = 'Log'
        cb_norm = LogNorm(vmin=land_minmax[0], vmax=land_minmax[-1])
    elif str.lower(land_scale) == 'symlog':
        land_scale = 'SymLog'
        lin_thresh = abs(np.asarray([land_minmax[0], land_minmax[1]])).max()/500.
        cb_norm = SymLogNorm(vmin=land_minmax[0], vmax=land_minmax[-1], linthresh=lin_thresh)
    else:
        sys.exit("wrong value {0}' for land_scale. Choose from 'Linear', 'Log' or 'SymLog'.".format(land_scale))
        
    p1 = plt.pcolormesh(*land_data, cmap=land_cmap, vmin=land_minmax[0], vmax=land_minmax[-1], norm=cb_norm, transform=land_projection, zorder=10)

    # clip the rest of the graph to the circle, and add circle patch
    clip_to_circle(ax, lat_bot, transform=projection, linewidth=linewidth)

    if fig_title != '': # if set, plot title
        ax.set_title(fig_title, fontsize=fontsize+2, zorder=200)
    
    if (cb_location != 'none') | (fig != None): # add colorbars, if cb_location is not set to 'none'
        cb_land  = add_colorbar(ax, p1, minmax=land_minmax,  title=land_title, extend=land_cb_extend, scale=land_scale, location=cb_location, fontsize=fontsize, linewidth=linewidth, boundaries=land_cb_bounds)
        if plot_ocean: cb_ocean = add_colorbar(ax, p2, minmax=ocean_minmax, title=ocean_title, extend=ocean_cb_extend, scale='Linear',  location=cb_location, fontsize=fontsize, linewidth=linewidth, boundaries=ocean_cb_bounds)    
            
    if fig_ax == None:
        return ax, cb_land
    elif not plot_ocean:
        return ax, p1
    else:
        return ax, p1, p2

def add_colorbar(ax, patch, minmax=[0,0], title='none', extend='neither', fontsize=8, shrink=0.8, scale='Linear', location=None, pad=0.08, boundaries=None, cb_ticks=None, linewidth=0.5):

    # sort out the orientation and location
    if (location == 'bottom') | (location == 'top'):
        orientation = 'horizontal'
    elif (location == 'left') | (location == 'right'):
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
        location = 'bottom'
    
    # try:
    #     fig_dpi=fig.dpi
    # except:
    #     fig_dpi=ax.figure.dpi
    
    # get bounds of current axes
    left, bottom, width, height = ax.get_position().bounds
    
    # Add axes to the figure, to place the color bar
    colorbar_axes, niks = make_axes(ax, fraction=.05, pad=pad, shrink=shrink, aspect=25, location=location)

    # Add the color bar
    cbar = plt.colorbar(patch, colorbar_axes, extend=extend, orientation=orientation, boundaries=boundaries)
    
    # set minmax if not supplied. If minmax is given, adjust clim
    if minmax[0] == minmax[1]: 
        minmax = patch.get_clim()

    # define cbticks along linear, logarithmic or semilogarithmic scale if not already supplied
    if cb_ticks == None:
        if str.lower(scale) == 'linear':
            cb_ticks = np.linspace(minmax[0], minmax[1],5)
        elif str.lower(scale) == 'log': 
            log10_max = int(np.log10(abs(np.asarray([minmax[0], minmax[1]])).max())) # find highest log10 within range minmax
            cb_ticks = [10**n for n in range(log10_max-4,log10_max+1)]  # get 5 ticks, exponentially spaced
        elif str.lower(scale) == 'symlog':
            log10_max = int(np.log10(abs(np.asarray([minmax[0], minmax[1]])).max())) # find highest log10 within range minmax
        
            cb_ticks = [10**n for n in range(log10_max-1,log10_max+1)]      # get 3 ticks, exponentially spaced
            cb_ticks = np.asarray([-nr for nr in cb_ticks[::-1]]+cb_ticks)  # mirror the axis, and add zero in the middle
            cb_ticks = cb_ticks[(np.abs(cb_ticks)>patch.norm.linthresh)]    # remove abs(values) > linthresh
            cb_ticks = np.concatenate((cb_ticks,[-patch.norm.linthresh, 0., patch.norm.linthresh])) # add indices for linthresh and 0
            cb_ticks.sort()
        
    # Label the color bar and add ticks
    if title != 'none': 
        if orientation == 'vertical':
            colorbar_axes.set_title(title, position=(-0.75, 0.5), fontsize=fontsize, rotation='vertical', verticalalignment='center', loc='center')
        else:
            # x_position = np.abs(cbar.ax.get_xlim()[1]-cbar.ax.get_xlim()[0])/2 + np.min(cbar.ax.get_xlim())
            colorbar_axes.set_title(title, position=(0.5,1.1), fontsize=fontsize, loc='center')

    cbar.ax.tick_params(length=0, pad=3, labelsize=fontsize)
    cbar.ax.get_children()[2].set_linewidth(linewidth) # adjust the linewidth of the outside line of the colorbar
    cbar.set_ticks(cb_ticks)
    cbar.update_ticks()
    
    return cbar

# add North Polar Lambert Azimuthal Equal Area projection class
class EASE_North(cp.crs.Projection):

    def __init__(self):

        # see: http://www.spatialreference.org/ref/epsg/3408/
        proj4_params = {'proj': 'laea',
            'lat_0': 90.,
            'lon_0': 0,
            'x_0': 0,
            'y_0': 0,
            'a': 6371228,
            'b': 6371228,
            'units': 'm',
            'no_defs': ''}

        super(EASE_North, self).__init__(proj4_params)

    @property
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),(self.x_limits[1], self.y_limits[0]),
                  (self.x_limits[1], self.y_limits[1]),(self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[0], self.y_limits[0]))

        return cp.crs.sgeom.Polygon(coords).exterior

    @property
    def threshold(self):
        return 1e5

    @property
    def x_limits(self):
        return (-9000000, 9000000)

    @property
    def y_limits(self):
        return (-9000000, 9000000)

def LonLatPolygon(lat_bot, lat_top=90., lon_left=-180., lon_right=180.):
    """
    Convenience function that makes a polygon according to lon/lat boundaries:
    
    Usage:

    LonLatPolygon(lat_bot, lat_top=90., lon_left=-180., lon_right=180.)
    
    Where:
    
    lat_bot: bottom latitude of polygon in deg N
    lat_top: top latitude of polygon in deg N (default=90)
    lon_left: left longitude in deg E (default=-180)
    lon_right: right longitude in deg E (default=180)
    """
    
    # define borders of polygon
    x = np.linspace(lon_left, lon_right, lon_right-lon_left) # longitudes of polygon
    y = np.linspace(lat_bot, lat_top, lat_top-lat_bot) # latitudes of polygon

    # get x, y points and make the polygon
    points = list(zip(np.concatenate((x, np.ones(len(y))*lon_right, x[::-1], np.ones(len(y))*lon_left)), np.concatenate((np.ones(len(x))*lat_bot, y, np.ones(len(x))*lat_top, y[::-1]))))
    LonLatPoly = cp.feature.sgeom.Polygon(points)
    
    return LonLatPoly

def clip_to_circle(ax, lat_bot, transform=None, central_lon=-45., linewidth=0.5):

    """
    Function that clips the figure to a Circle around the North Pole.
    Only to be used with North Polar projections
    
    Usage:

    clip_to_circle(ax, lat_edge, transform=prj, central_lon=-45.)
    
    Where:
    
    lat_bot: latitude of circle edge (i.e. the radius)
    prj: the projection used by the current figure
    central_lon: the central longitude of the current polar projection
    """
    
    # add the circle to the plot as a patch
    cirkel_patch = ax.add_patch(Circle((0,0), transform.transform_point(180+central_lon,lat_bot,cp.crs.PlateCarree())[1], edgecolor='k', facecolor='none', linewidth=linewidth, transform=transform, zorder=100))

    # cut out the raster data to the circle patch
    for obj in ax.findobj():
        if type(obj) != txtType: # we don't want to clip any text
            obj.set_clip_on(True)
            obj.set_clip_path(cirkel_patch)

    # remove background and border frame
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    
    return cirkel_patch


def poly_clip(geoms, poly): 
    """ 
    Function that returns a list of geometries that intersect with the provided polygon
    
    Usage: poly_clip(geoms, poly)
    
    geoms: geometry collection
    poly: polygon to clip to
    
    returns: list of geometries
    """
     
    return [geom.intersection(poly) for geom in geoms.geometries() if geom.intersects(poly)]

def add_natural_earth_feature(ax, feature, categorie='physical', poly='none', scale='50m', props='none', linewidth=0.5, hairline=False):
    """ 
    Function that plots a natural earth feature, optionally clipping it with poly_clip
    
    Usage: add_clipped_feature(feature, ax, poly, scale, props, proj)
    
    feature: name of the feature (NaturalEarthFeature)
    ax: axis to plot the feature
    poly: polygon to clip to. if not set, no clipping will occur (default=none)
    scale: scale of the feature, choose between 10m, 50m and 110m (default=50m)
    props: list with properties of the geometry (facecolor, edgecolor, linewidth and alpha. In that order)
    
    returns: axis object
    
    This function only plots physical data from http://www.naturalearthdata.com/downloads/ when available
    
    Currently this function has predefined colors for ocean, land, coastline, rivers, lakes, and glaciated
    """
    
    # lines to be able to use shorter names for some features
    if feature == 'rivers': feature = 'rivers_lake_centerlines'
    if feature == 'glaciated': feature = 'glaciated_areas'
    if hairline: linewidth = linewidth/2.

    # default layout for some features, in this order: facecolor, edgecolor, linewidth and alpha
    std_props = {'ocean' :                   ['#FFFFFF', 'none',     linewidth, 1.],
                 'land' :                    ['#E0E0E0', 'none',     linewidth, 1.],
                 'coastline' :               ['none',    '#2F4F4F',  linewidth, 1.],
                 'rivers_lake_centerlines' : ['none',    '#4682B4',  linewidth, .5],
                 'lakes' :                   ['#4682B4', 'none',     linewidth, 1.],
                 'glaciated_areas':          ['#EFF8FF', 'none',     linewidth, 1.],
                 }
    
    if props != 'none':
        if feature in props.keys():     
            props = props[feature]
        elif feature in std_props.keys():     
            props = std_props[feature]
    else:
        try:
            props = std_props[feature]
        except:
            if categorie=='raster':
                props = ['none', 'none', 0., 1.]
            else:
                props = ['#D0D0D0', 'none', 1., 1.]
        
    # get the feature geometries
    geoms = cp.feature.NaturalEarthFeature(category=categorie, name=feature, scale=scale)
    
    # define zorder of this feature
    if feature == 'ocean': 
        zorder=0
    else:
        zorder=2
    
    # add it to the specified axis
    if poly == 'none':
        ax.add_geometries(geoms.geometries(), cp.crs.PlateCarree(), facecolor=props[0], edgecolor=props[1], linewidth=props[2], alpha=props[3], zorder=zorder)
    else:
        ax.add_geometries(poly_clip(geoms, poly), cp.crs.PlateCarree(), facecolor=props[0], edgecolor=props[1], linewidth=props[2], alpha=props[3],zorder=zorder)

    return ax

def add_natural_earth_features(ax, features, categorie='physical', poly='none', scale='110m', props='none', gridlines=True, linewidth=0.5):
    
    """ 
    Simple wrapper around add_natural_earth_feature.
    
    Usage: add_clipped_feature(feature, ax, poly, scale, gridlines, props, proj)
    
    features: name of the feature (NaturalEarthFeature)
    ax: axis to plot the feature
    poly: polygon to clip to. if not set, no clipping will occur (default=none)
    scale: scale of the feature, choose between 10m, 50m and 110m (default=110m)
    props: list with properties of the geometry (facecolor, edgecolor, linewidth and alpha. In that order)
    gridlines: whether to add gridlines (True or False, default=True)
    
    returns: axis object
    
    """
    [add_natural_earth_feature(ax, feature, poly=poly, scale=scale, categorie=categorie, props=props, linewidth=linewidth) for feature in features]
    if gridlines: ax = add_gridlines(ax, poly, linewidth=linewidth)
    
    return ax

def add_gridlines(ax, poly=None, deg_sep=30, N80=True, linewidth=0.5):
    """
    Convenience function to plot not only gridlines, but also the Arctic Circle. 
    Also adds an extra 80N line if deg_sep == 30 (looks nicer for north polar plots)
    
    Usage: add_gridlines(ax, cirkel, deg_sep=30)
    
    Where:
    
    ax : axes to plot the gridlines in (default plt.gca())
    poly : polygon to clip the gridlines to
    deg_sep: horizontal seperation of gridlines (default=30)
    N80: whether to plot 80 deg N line (True/False, default=True)
    """
    
    # get grid lines at 30 deg, and define LineStrings for the Arctic Circle and 80 deg N
    arctic_circle = cp.feature.sgeom.LineString(list(zip(np.linspace(-180,180,360), 66.5622*np.ones(360))))
    lines = cp.feature.NaturalEarthFeature(category='physical', name='graticules_{0}'.format(deg_sep), scale='50m')

    # add the grid lines
    if poly == None:
        ax.add_geometries(lines.geometries(), cp.crs.PlateCarree(), facecolor='none', edgecolor='#303030', alpha=0.5, linestyle=':', linewidth=linewidth)
    else:
        ax.add_geometries(poly_clip(lines,poly), cp.crs.PlateCarree(), facecolor='none', edgecolor='#303030', alpha=0.5, linestyle=':', linewidth=linewidth)
        
    ax.add_geometries([arctic_circle], cp.crs.PlateCarree(), facecolor='none', edgecolor='#303030', alpha=0.5, linestyle='--', linewidth=linewidth)

    if (deg_sep == 30) & (N80):
        lat80N = cp.feature.sgeom.LineString(list(zip(np.linspace(-180,180,360), 80*np.ones(360))))
        ax.add_geometries([lat80N], cp.crs.PlateCarree(), facecolor='none', edgecolor='#303030', alpha=0.5, linestyle=':', linewidth=linewidth) # add an extra dashed line for 80 deg latitude

    return ax

def add_scalebar(ax, ax_crs, scale=100, scale_unit='m', max_stripes=5, x0y0=(0.565, 0.065), y_height=0.02, utm_zone=33, font_color='#000000'):
    """
    Add a scalebar to a GeoAxes

    Args:
    * ax: axis to plot the scale bar in
    * ax_crs: cartopy coordinate system of ax
    * scale: width of the scale bar
    * scale_unit: base unit of the scale bar (m or km)
    * max_strips: typical/maximum number of black+white regions
    * x0x0: bottom left corner of scale bar in native axes coordinates
    * y_height: height of the scale bar in native axes coordinates
    * utm_zone: UTM zone nr, to project the scale bar in meters
    * font_color: color of the unit and ticks
    """
    
    if scale_unit == 'km': scale = scale* 1000.
    
    #Projection in metres, need to change this to suit your own figure
    utm = ccrs.UTM(utm_zone)
    
    # fetch axes coordinate mins+maxes
    x_left, x_right, y_bottom, y_top = ax.get_extent(ax_crs)
    
    # determine lower left corner
    xll = x_left + x0y0[0] * (x_right - x_left)
    yll = y_bottom + x0y0[1] * (y_top - y_bottom)
    ylr, xul = yll, xll
    
    # get range to find indices on
    x_range = np.linspace(xll, x_right, 1000)
    y_range = np.ones(len(x_range))*yll
    
    # calculate utm coordinates for this coordinate system
    utm_range = utm.transform_points(ax_crs,x_range,y_range)[:,:2]
    
    # find nr of max_stripes points along utm x-axis at interval of size scale in m (improve with pythagoras)
    x_inds = np.asarray([np.where(utm_range[:,0] - utm_range[0,0] >= step*scale)[0][0] for step in range(max_stripes)])
    x_vals = x_range[x_inds]
    
    # transform xy coordinates back to ax_crs coordinate system
    xy_ticks = ax_crs.transform_points(utm, x_vals, utm_range[:max_stripes,1])[:,:2]
     
    # calculate Axes Y coordinates of box top+bottom
    xlr = xur = x_vals[-1]
    yul = yur = yll + y_height * (y_top - y_bottom)
    
    # calculate Axes Y distance of ticks + label margins
    y_margin = (yul-yll)*0.25

    # fill black/white 'stripes' and draw their boundaries
    fill_colors = ['black', 'white']
    i_color = 0
    for xi0, xi1 in zip(x_vals[:-1],x_vals[1:]):
        # fill region
        ax.fill((xi0, xi0, xi1, xi1, xi0), (yll, yul, yur, ylr, yll), fill_colors[i_color], transform=ax_crs, zorder=1000)
        # draw boundary
        ax.plot((xi0, xi0, xi1, xi1, xi0), (yll, yul, yur, ylr, yll), color='#000000', marker='', linestyle='-', linewidth=.5, transform=ax_crs, zorder=1000)
        i_color = 1 - i_color

    # add short tick lines
    for x in x_vals: ax.plot((x, x), (yll, yll-y_margin), color='#000000', marker='', linestyle='-', linewidth=.5, transform=ax_crs, zorder=1000)

    # add a scale legend 'Km'
    ax.text(0.5 * (xul + xur), yul + y_margin, scale_unit, verticalalignment='bottom', horizontalalignment='center', fontsize=8, transform=ax_crs, zorder=1000, color=font_color)

    # add numeric labels
    for x in range(max_stripes): ax.text(x_vals[x], yll - 2 * y_margin, '{:g}'.format(x*scale), verticalalignment='top', horizontalalignment='center', fontsize=8, transform=ax_crs, color=font_color, zorder=1000)

def scale_bar(ax, length, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    """
    #Projection in metres, need to change this to suit your own figure
    utm = ccrs.UTM(36)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    #Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * 500, sbcx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbcx, sbcy, str(length) + ' km', transform=utm,
            horizontalalignment='center', verticalalignment='bottom')