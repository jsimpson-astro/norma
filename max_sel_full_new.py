#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:34:29 2019
19.04.19
@author: michael cretignier & jémémie francfort

# =====================================================================================
# Rolling Alpha Shape for a Spectral Improved Normalisation Estimator (RASSINE)
# =====================================================================================

	   ^				  .-=-.		  .-==-.
	  {}	  __		.' O o '.	   /   ^  )
	 { }	.' O'.	 / o .-. O \	 /  .--`\
	 { }   / .-. o\   /O  /   \  o\   /O /	^  (RASSSSSSINE)
	  \ `-` /   \ O`-'o  /	 \  O`-`o /
  jgs  `-.-`	 '.____.'	   `.____.'

"""
from __future__ import print_function

import matplotlib
import pickle
import sys

import argparse

#matplotlib.use('Qt5Agg',force=True)
#matplotlib.use('TkAgg',force=True)
import getopt
import os
import sys
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erf

from rassine_.rassine.rassine import plot_output
from tqdm import tqdm

def ras_troncated(array, spectre, treshold=5):
	maxi = np.percentile(spectre,99.9)
	mini = np.percentile(spectre,0.1)
	tresh = (maxi-mini)/treshold
	array[array<mini-tresh] = mini
	array[array>maxi+tresh] = maxi
	return array

def ras_find_nearest(array,value,dist_abs=True):
	"""
	Find the closest value in an array
	
	Parameters
	----------
	
	dist_abs : provide the distance output in absolute
	
	Return
	------
	
	index of th closest element, value and distance
	
	"""
	if type(value)!=np.ndarray:
		value = np.array([value])
	idx = np.argmin((np.abs(array-value[:,np.newaxis])),axis=1)
	distance = abs(array[idx]-value) 
	if dist_abs==False:
		distance = array[idx]-value
	return idx, array[idx], distance

def ras_check_none_negative_values(array):
	"""
	Interpolate over negative values
	
	"""
	neg = np.where(array<=0)[0]
	if len(neg)==0:
		pass
	elif len(neg)==1:
		array[neg] = 0.5*(array[neg+1]+array[neg-1])
	else:
		where = np.where(np.diff(neg)!=1)[0]
		if (len(where)==0)&(np.mean(neg)<len(array)/2):
			array[neg] = array[neg[-1]+1]
		elif (len(where)==0)&(np.mean(neg)>=len(array)/2):
			array[neg] = array[neg[0]-1]			
		else:
			where = np.hstack([where,np.array([0,len(neg)-1])])
			where = where[where.argsort()]
			for j in range(len(where)-1):
				if np.mean(array[neg[where[j]]:neg[where[j+1]]+1])<len(array)/2:
					array[neg[where[j]]:neg[where[j+1]]+1] = array[neg[where[j+1]]+1]
				else:
					array[neg[where[j]]:neg[where[j+1]]+1] = array[neg[where[j]]-1]
	return array

def ras_make_continuum(wave, flux, grid, spectrei):
	
	continuum1 = np.zeros(len(grid))
	continuum3 = np.zeros(len(grid))
	  

	Interpol1 = interp1d(wave, flux, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
	continuum1 = Interpol1(grid)
	continuum1 = ras_troncated(continuum1,spectrei)
	continuum1 = ras_check_none_negative_values(continuum1)

	Interpol3 = interp1d(wave, flux, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')
	continuum3 = Interpol3(grid)
	continuum3 = ras_troncated(continuum3,spectrei)
	continuum3 = ras_check_none_negative_values(continuum3)

	return continuum1, continuum3

np.warnings.filterwarnings('ignore', category=RuntimeWarning)

######## NEW FUNCTIONS  ########


def ras_troncated(array, spectre, treshold=5):
	maxi = np.percentile(spectre,99.9)
	mini = np.percentile(spectre,0.1)
	tresh = (maxi-mini)/treshold
	array[array<mini-tresh] = mini
	array[array>maxi+tresh] = maxi
	return array

def get_data(file) -> dict:
	"""
	Reads pickle, takes important data and stores in a dict
	Returns dict
	"""
	# nb, normalisation flipped
	spec_dict = {}

	with open(file, 'rb') as p:
		in_p = pickle.load(p)
		spec_dict['spectrei'] = in_p['flux'] # spectre is (seemingly) the unmodified flux (may need to apply par_stretching)
		spec_dict['normalisation'] = in_p['parameters']['normalisation']
		
		#spec_dict['spectre'] = in_p['flux'] / spec_dict['normalisation']
		spec_dict['spectre'] = in_p['flux'] 

		spec_dict['locmaxx'] = in_p['output']['anchor_wave_original'] # local maxima, i.e. 'wave', from earlier on in the code, before penalty computation
		#spec_dict['locmaxy'] = in_p['output']['anchor_flux_original'] # * in_p['parameters']['normalisation'] # same as above, for flux
		spec_dict['locmaxy'] = in_p['output']['anchor_flux_original'] * spec_dict['normalisation']
		
		spec_dict['locmaxz'] = in_p['output']['anchor_index_original'] # 
		spec_dict['wave'] = in_p['output']['anchor_wave']
		
		#spec_dict['flux'] = in_p['output']['anchor_flux'] / spec_dict['normalisation']
		spec_dict['flux'] = in_p['output']['anchor_flux']
		
		spec_dict['index'] = in_p['output']['anchor_index']
		spec_dict['interpol'] = 'cubic' # or linear, for interp1d
		
		spec_dict['wavelengths'] = in_p['wave']
		spec_dict['grid'] = in_p['grid']
		
		# anchors are computed on the grid
		spec_dict['locmaxx_fixed'] = spec_dict['wavelengths'][spec_dict['locmaxz']]
		spec_dict['wave_fixed'] = spec_dict['wavelengths'][spec_dict['index']]
		
		# manual points read/verify
		spec_dict['wave_man'] = in_p['output']['wave_man'] if 'wave_man' in in_p['output'] else np.array([])
		#spec_dict['flux_man'] = in_p['output']['flux_man']  / spec_dict['normalisation'] if 'flux_man' in in_p['output'] else np.array([])
		spec_dict['flux_man'] = in_p['output']['flux_man'] if 'flux_man' in in_p['output'] else np.array([])

	return spec_dict

def write_data(file, spec_dict):
	"""
	Takes spec_dict, writes to pickle 'file'
	"""
	# nb, normalisation flipped

	with open(file, 'rb') as pi:
		out_p = pickle.load(pi)

		out_p['output']['anchor_wave'] = spec_dict['wave']
		out_p['output']['anchor_flux'] = spec_dict['flux']
		out_p['output']['anchor_index'] = spec_dict['index']
		
		out_p['output']['wave_man'] = spec_dict['wave_man']
		out_p['output']['flux_man'] = spec_dict['flux_man']

		out_p['parameters']['filename'] = file
		out_p['parameters']['number_anchors'] = len(spec_dict['wave'])

		# combine with wave and flux vectors
		combx = np.append(spec_dict['wave_fixed'], spec_dict['wave_man'])
		comby = np.append(spec_dict['flux'], spec_dict['flux_man'])

		# sort correctly
		combx_as = np.argsort(combx)
		comby = comby[combx_as]
		combx = combx[combx_as]

		continuum1, continuum3 = ras_make_continuum(
			 combx,
			 comby,
			 spec_dict['wavelengths'],
			 spec_dict['spectrei'],
		)

		out_p['output']['continuum_linear'] = continuum1
		out_p['output']['continuum_cubic'] = continuum3

	with open(file, 'wb') as pf:
		pickle.dump(out_p, pf)




def plot_data(ax, spec_dict, spec_kwargs=None, conti_kwargs=None, anchor_kwargs=None, manual_kwargs=None, locmax_kwargs=None) -> dict:
	"""
	Takes axis, spec_dict, returns a dictionary of plot objects and adds them to axis
	"""

	# defaults
	if not spec_kwargs:
		spec_kwargs = {'color': '#074517', 'alpha': 0.3}
	if not conti_kwargs:
		conti_kwargs = {'color': '#0d7a2a'}
	if not anchor_kwargs:
		anchor_kwargs = {'color': '#09521c', 'marker': 'o', 'linestyle': 'none', 'markersize': 6}
	if not manual_kwargs:
		manual_kwargs = {'color': '#12cc43', 'marker': 'o', 'linestyle': 'none', 'markersize': 7}
	if not locmax_kwargs:
		locmax_kwargs = {'color': 'k', 's': 5}
	
	spec, = ax.plot(spec_dict['wavelengths'], spec_dict['spectre'], **spec_kwargs)
	locmax = ax.scatter(spec_dict['locmaxx_fixed'],spec_dict['locmaxy'], zorder = 3, **locmax_kwargs)
		
	# combine with wave and flux vectors
	combx = np.append(spec_dict['wave_fixed'], spec_dict['wave_man'])
	comby = np.append(spec_dict['flux'], spec_dict['flux_man'])
	#comby = np.append(spec_dict['flux']  / spec_dict['normalisation'], spec_dict['flux_man'] / spec_dict['normalisation'])
	
	# sort correctly
	combx_as = np.argsort(combx)
	comby = comby[combx_as]
	combx = combx[combx_as]
	
	Interpol3 = interp1d(combx, comby, kind = spec_dict['interpol'], bounds_error = False, fill_value = 'extrapolate')
	continuum3 = Interpol3(spec_dict['wavelengths'])
	# continuum3 = ras_troncated(continuum3,spec_dict['spectre'])
	continuum3 = ras_troncated(continuum3,spec_dict['spectre']) #* spec_dict['normalisation']
		
	l1, = ax.plot(spec_dict['wavelengths'], continuum3, zorder=5, **conti_kwargs)
	l2, = ax.plot(spec_dict['wave_fixed'], spec_dict['flux'], zorder=6, **anchor_kwargs)
	lm, = ax.plot(spec_dict['wave_man'], spec_dict['flux_man'], zorder=7, **manual_kwargs)

	return {'spec': spec, 'conti': l1, 'anchors': l2, 'manual': lm, 'locmax': locmax}


def change_data(plot_dict, spec_dict) -> dict:
	"""
	Updates plot_dict objects with new data from spec_dict, returns new plot_dict
	"""
	#spec, = ax.plot(spec_dict['wavelengths'], spec_dict['spectre'], **spec_kwargs)
	plot_dict['spec'].set_data(spec_dict['wavelengths'], spec_dict['spectre'])
	
	#locmax = ax.scatter(spec_dict['locmaxx_fixed'],spec_dict['locmaxy'], zorder = 3, **locmax_kwargs)
	plot_dict['locmax'].set_offsets(np.c_[spec_dict['locmaxx_fixed'], spec_dict['locmaxy']])

	# combine with wave and flux vectors
	combx = np.append(spec_dict['wave_fixed'], spec_dict['wave_man'])
	comby = np.append(spec_dict['flux'], spec_dict['flux_man'])
	#comby = np.append(spec_dict['flux']  / spec_dict['normalisation'], spec_dict['flux_man'] / spec_dict['normalisation'])
	
	# sort correctly
	combx_as = np.argsort(combx)
	comby = comby[combx_as]
	combx = combx[combx_as]
	
	Interpol3 = interp1d(combx, comby, kind = spec_dict['interpol'], bounds_error = False, fill_value = 'extrapolate')
	continuum3 = Interpol3(spec_dict['wavelengths'])
	#continuum3 = ras_troncated(continuum3,spec_dict['spectre'])
	continuum3 = ras_troncated(continuum3,spec_dict['spectre']) #* spec_dict['normalisation']

	#l1, = ax.plot(spec_dict['wavelengths'], continuum3, zorder=5, **conti_kwargs)
	plot_dict['conti'].set_data(spec_dict['wavelengths'], continuum3)
	#l2, = ax.plot(spec_dict['wave_fixed'], spec_dict['flux'], zorder=6, **anchor_kwargs)
	plot_dict['anchors'].set_data(spec_dict['wave_fixed'], spec_dict['flux'])
	#lm, = ax.plot(spec_dict['wave_man'], spec_dict['flux_man'], zorder=7, **manual_kwargs)
	plot_dict['manual'].set_data(spec_dict['wave_man'], spec_dict['flux_man'])

	return plot_dict

def hide_plot(plot_dict) -> dict:
	"""
	Hide all objects in the given plot dict
	Returns a copy with the modifications
	"""
	plot_dict['spec'].set_visible(False)
	plot_dict['locmax'].set_visible(False)
	plot_dict['conti'].set_visible(False)
	plot_dict['anchors'].set_visible(False)
	plot_dict['manual'].set_visible(False)
	return plot_dict

def unhide_plot(plot_dict) -> dict:
	"""
	Hide all objects in the given plot dict
	Returns a copy with the modifications
	"""
	plot_dict['spec'].set_visible(True)
	plot_dict['locmax'].set_visible(True)
	plot_dict['conti'].set_visible(True)
	plot_dict['anchors'].set_visible(True)
	plot_dict['manual'].set_visible(True)
	return plot_dict

def hide_points(plot_dict) -> dict:
	"""
	Hide all points in the given plot dict
	Returns a copy with the modifications
	"""
	plot_dict['locmax'].set_visible(False)
	return plot_dict

def unhide_points(plot_dict) -> dict:
	"""
	Hide all points in the given plot dict
	Returns a copy with the modifications
	"""
	plot_dict['locmax'].set_visible(True)
	return plot_dict


def hide_all(wd, current_index, extra) -> dict:
	"""
	Hide all that can be hidden
	"""

	above = below = 0

	if current_index < extra:
		# at lower edge, so hide some
		below = extra - current_index
	elif current_index > len(df) - 1 - extra:
		# at upper edge, hide some
		above = current_index + extra + 1 - len(df)
	else:
		# nothing to hide/unhide, no issues
		pass

	for i in range(extra-above):
		wd['above'][i]['plot'] = hide_plot(wd['above'][i]['plot'])

	for i in range(extra-below):
		wd['below'][i]['plot'] = hide_plot(wd['below'][i]['plot'])

	fig.canvas.draw_idle()

	return wd

def unhide_all(wd, current_index, extra) -> dict:
	"""
	Unhide all that should be unhidden
	"""

	above = below = 0

	if current_index < extra:
		# at lower edge, so hide some
		below = extra - current_index
	elif current_index > len(df) - 1 - extra:
		# at upper edge, hide some
		above = current_index + extra + 1 - len(df)
	else:
		# nothing to hide/unhide, no issues
		pass

	for i in range(extra-above):
		wd['above'][i]['plot'] = unhide_plot(wd['above'][i]['plot'])

	for i in range(extra-below):
		wd['below'][i]['plot'] = unhide_plot(wd['below'][i]['plot'])

	fig.canvas.draw_idle()

	return wd

def hide_all_points(wd, current_index, extra) -> dict:
	"""
	Hide all that can be hidden
	"""

	above = below = 0

	if current_index < extra:
		# at lower edge, so hide some
		below = extra - current_index
	elif current_index > len(df) - 1 - extra:
		# at upper edge, hide some
		above = current_index + extra + 1 - len(df)
	else:
		# nothing to hide/unhide, no issues
		pass

	for i in range(extra-above):
		wd['above'][i]['plot'] = hide_points(wd['above'][i]['plot'])

	for i in range(extra-below):
		wd['below'][i]['plot'] = hide_points(wd['below'][i]['plot'])

	fig.canvas.draw_idle()

	return wd

def unhide_all_points(wd, current_index, extra) -> dict:
	"""
	Unhide all that should be unhidden
	"""

	above = below = 0

	if current_index < extra:
		# at lower edge, so hide some
		below = extra - current_index
	elif current_index > len(df) - 1 - extra:
		# at upper edge, hide some
		above = current_index + extra + 1 - len(df)
	else:
		# nothing to hide/unhide, no issues
		pass

	for i in range(extra-above):
		wd['above'][i]['plot'] = unhide_points(wd['above'][i]['plot'])

	for i in range(extra-below):
		wd['below'][i]['plot'] = unhide_points(wd['below'][i]['plot'])

	fig.canvas.draw_idle()

	return wd

def set_thinning(wd, thinning=10) -> dict:
	"""
	Enables a given thinning factor to improve performance
	(can try thinning continuum too)
	"""
	spec_dict_list = [i['data'] for i in wd['above'] + [wd['current']] + wd['below']]
	plot_dict_list = [i['plot'] for i in wd['above'] + [wd['current']] + wd['below']]

	for spec_dict, plot_dict in zip(spec_dict_list, plot_dict_list):
		plot_dict['spec'].set_data(spec_dict['wavelengths'][::thinning], spec_dict['spectre'][::thinning])
	
		combx = np.append(spec_dict['wave_fixed'], spec_dict['wave_man'])
		comby = np.append(spec_dict['flux']  / spec_dict['normalisation'], spec_dict['flux_man'] / spec_dict['normalisation'])
		
		combx_as = np.argsort(combx)
		comby = comby[combx_as]
		combx = combx[combx_as]
		
		Interpol3 = interp1d(combx, comby, kind = spec_dict['interpol'], bounds_error = False, fill_value = 'extrapolate')
		continuum3 = Interpol3(spec_dict['wavelengths'])
		continuum3 = ras_troncated(continuum3,spec_dict['spectre']) * spec_dict['normalisation']

		plot_dict['conti'].set_data(spec_dict['wavelengths'][::thinning], continuum3[::thinning])

	fig.canvas.draw_idle()
	return wd


def reset_home(wd, border=0.05):
	"""
	Reset the matplotlib 'home' button to proper limits
	"""

	new_xmin, new_xmax = wd['current']['data']['wavelengths'][0], wd['current']['data']['wavelengths'][-1]
	new_ymin, new_ymax = 0, wd['above'][-1]['data']['spectrei'].max()
	new_xmin = new_xmin - border *(new_xmax - new_xmin)
	new_xmax = new_xmax + border *(new_xmax - new_xmin)
	new_ymin = new_ymin - border *(new_ymax - new_ymin)
	new_ymax = new_ymax + border *(new_ymax - new_ymin)

	if len(fig.canvas.toolbar._nav_stack._elements) > 0:
		# Get the first key in the navigation stack
		key = list(fig.canvas.toolbar._nav_stack._elements[0].keys())[0]
		# Construct a new tuple for replacement
		alist = []
		for x in fig.canvas.toolbar._nav_stack._elements[0][key]:
			alist.append(x)
		alist[0] = (new_xmin, new_xmax, new_ymin, new_ymax)
		# Replace in the stack
		fig.canvas.toolbar._nav_stack._elements[0][key] = tuple(alist)


def move_up(wd, df, current_index, extra):
	"""
	Move up one slot, updating wd as needed
	
	"""
	new_index = current_index + 1

	to_hide = to_unhide = None
	if new_index == len(df):
		# can't move up, already at max
		return wd, current_index
	elif new_index > len(df) - 1 - extra:
		to_hide = len(df) - 1 - new_index
	elif current_index < extra:
		# something is hidden, unhide it
		to_unhide = current_index
	else:
		# nothing to hide/unhide, no issues
		to_hide = to_unhide = None
	
	# first, write current spec_data out
	write_data(df['out file'][current_index], wd['current']['data'])	

	below_index = [new_index - i - 1 if new_index - i > 0 else 0 for i in range(extra)]
	above_index = [new_index + i + 1 if new_index + i + 1 < len(df) else len(df) - 1 for i in range(extra)]
	
	below_list = df['out file'][below_index].to_list()
	above_list = df['out file'][above_index].to_list()
	current = df['out file'][new_index]

	new_wd = {'above': [{} for i in above_list], 'current': {}, 'below': [{} for i in below_list]}

	# change below plot to current data
	new_wd['below'][0]['plot'] = change_data(wd['below'][0]['plot'], wd['current']['data'])
	new_wd['below'][0]['data'] = wd['current']['data']

	# change current plot to 1 above
	new_wd['current']['plot'] = change_data(wd['current']['plot'], wd['above'][0]['data'])
	new_wd['current']['data'] = wd['above'][0]['data']

	# now loop and move all above up one, all below up one
	for i in range(len(wd['below']) - 1):
		new_wd['below'][i+1]['plot'] = change_data(wd['below'][i+1]['plot'], wd['below'][i]['data'])
		new_wd['below'][i+1]['data'] = wd['below'][i]['data']

	for i in range(len(wd['above'])):
		if i+1 == len(wd['above']):
			if len(above_list) > 1:
				new_wd['above'][-1]['data'] = get_data(above_list[-1]) if above_list[-1] != above_list[-2] else wd['above'][-1]['data']
			else:
				new_wd['above'][-1]['data'] = get_data(above_list[-1])
			new_wd['above'][-1]['plot'] = change_data(wd['above'][-1]['plot'], new_wd['above'][-1]['data'])
		else:
			new_wd['above'][i]['plot'] = change_data(wd['above'][i]['plot'], wd['above'][i+1]['data'])
			new_wd['above'][i]['data'] = wd['above'][i+1]['data']

	# now hide/unhide
	if to_hide is not None:
		new_wd['above'][to_hide]['plot'] = hide_plot(new_wd['above'][to_hide]['plot'])

	if to_unhide is not None:
		new_wd['below'][to_unhide]['plot'] = unhide_plot(new_wd['below'][to_unhide]['plot'])
	
	fig.canvas.draw_idle()
	ax.set_title(f"Spectrum {current_index}: {os.path.split(df.loc[df['out file'] == current, 'in file'].values[0])[-1]}")
	reset_home(new_wd)

	return new_wd, new_index


def move_down(wd, df, current_index, extra):
	"""
	Move down one slot, updating wd as needed

	"""
	new_index = current_index - 1
	
	to_hide = to_unhide = None
	if current_index == 0:
		# can't move down, already at min
		return wd, current_index
	elif new_index < extra:
		# something to hide
		to_hide = new_index
	elif current_index > len(df) - 1 - extra:
		# something to unhide
		to_unhide = len(df) - 1 - current_index
	else:
		# nothing to hide/unhide, no issues
		to_hide = to_unhide = None
		
	# first, write current spec_data out
	write_data(df['out file'][current_index], wd['current']['data'])

	below_index = [new_index - i - 1 if new_index - i > 0 else 0 for i in range(extra)]
	above_index = [new_index + i + 1 if new_index + i + 1 < len(df) else len(df) - 1 for i in range(extra)]
	
	below_list = df['out file'][below_index].to_list()
	above_list = df['out file'][above_index].to_list()
	current = df['out file'][new_index]

	new_wd = {'above': [{} for i in above_list], 'current': {}, 'below': [{} for i in below_list]}

	# change above plot to current data
	new_wd['above'][0]['plot'] = change_data(wd['above'][0]['plot'], wd['current']['data'])
	new_wd['above'][0]['data'] = wd['current']['data']

	# change current plot to 1 below
	new_wd['current']['plot'] = change_data(wd['current']['plot'], wd['below'][0]['data'])
	new_wd['current']['data'] = wd['below'][0]['data']

	# now loop and move all above up one, all below up one
	for i in range(len(wd['above']) - 1):
		new_wd['above'][i+1]['plot'] = change_data(wd['above'][i+1]['plot'], wd['above'][i]['data'])
		new_wd['above'][i+1]['data'] = wd['above'][i]['data']

	for i in range(len(wd['below'])):
		if i+1 == len(wd['below']):
			if len(below_list) > 1:
				new_wd['below'][-1]['data'] = get_data(below_list[-1]) if below_list[-1] != below_list[-2] else wd['below'][-1]['data']
			else:
				new_wd['below'][-1]['data'] = get_data(below_list[-1])
			new_wd['below'][-1]['plot'] = change_data(wd['below'][-1]['plot'], new_wd['below'][-1]['data'])
		else:
			new_wd['below'][i]['plot'] = change_data(wd['below'][i]['plot'], wd['below'][i+1]['data'])
			new_wd['below'][i]['data'] = wd['below'][i+1]['data']

	# now hide/unhide
	if to_hide is not None:
		new_wd['below'][to_hide]['plot'] = hide_plot(new_wd['below'][to_hide]['plot'])

	if to_unhide is not None:
		new_wd['above'][to_unhide]['plot'] = unhide_plot(new_wd['above'][to_unhide]['plot'])

	fig.canvas.draw_idle()
	ax.set_title(f"Spectrum {current_index}: {os.path.split(df.loc[df['out file'] == current, 'in file'].values[0])[-1]}")
	reset_home(new_wd)

	return new_wd, new_index


########  TWEAKED CODE  ########

def_thin = 10

parser = argparse.ArgumentParser(prog="maxima_select_full.py",
								 description="Interactively edit local maxima from (modified) rassine outputs",
								 epilog="Rassine outputs must contain anchor_wave_original, anchor_flux_original, and anchor_index_original along with normalisation.")

parser.add_argument('filename',
					help="filename of paths to the spectra to work with")
parser.add_argument('-n', '--number',
					default=1,
					help="number of spectra above and below to plot, must be > 1 (default: 2)")
parser.add_argument('-i', '--start',
					default=0,
					help="starting point for spectra, in order of input list (default: 0)")
parser.add_argument('-u', '--update',
					action='store_true',
					default=False,
					help="update input files in-place rather than make new (default: False)")
parser.add_argument('-t', '--thinning',
					default=def_thin,
					help="thinning factor to use, enable with \'t\' (default: 10)")
parser.add_argument('-d', '--dark-mode',
					action='store_true',
					default=False,
					help="plot in dark mode (default: False)")
parser.add_argument('-p', '--plot-on-exit',
					 action='store_true',
					 default=False,
					 help="plot output of all on exit (warning - can be slow) (default: False)")
parser.add_argument('-px1',
					action='store',
					default=None,
					help="lower plot output wavelength limit (default: None)")
parser.add_argument('-px2',
					action='store',
					default=None,
					help="upper plot output wavelength limit (default: None)")

# parser.add_argument('-hc', '--hard-copy',
#					 action='store_true',
#					 default=False,
#					 help="save output plot, name is output name with _plot.png appended (default: False)")
# parser.add_argument('-j', '--balmer-jump',
#					 action='store_true',
#					 default=True,
#					 help="add line at 3645AA to mark the Balmer jump (default: True)")

args = parser.parse_args()

list_file = args.filename
extra = int(args.number)
current_index = int(args.start)
update = args.update
thinning_fact = int(args.thinning)
dark = args.dark_mode

plot_on_exit = args.plot_on_exit
px1 = args.px1
px2 = args.px2

interpol = 'cubic'

out_suf = '_mod'
plot_suf = '_plot'

if update:
	out_list_filename = list_file
else:
	out_list_filename = os.path.splitext(list_file)[0] + out_suf + os.path.splitext(list_file)[-1]

########  INIT + CHECKS  ########

print("Initialising...")

df = pd.read_csv(list_file, names=['in file'])

out_list = []
for f in df['in file']:

	# file checks
	with open(f, 'rb') as p:
		in_p = pickle.load(p)
	
	extra_data_kws = ['anchor_wave_original', 'anchor_flux_original', 'anchor_index_original']
	extra_data_exists = [True if k in in_p['output'] else False for k in extra_data_kws]

	if not all(extra_data_exists):
		raise KeyError("File does not include required output from modified rassine - please ensure save_maxima=True")
	
	if 'normalisation' not in in_p['parameters']:
		raise KeyError("Normalisation missing from input pickle")
	
	# if update, we don't change the output file
	if update:
		out_list.append(f)
	else:
		head, tail = os.path.splitext(f)
		out = head + out_suf + tail
		out_list.append(out)

df['out file'] = out_list

# now we write a copy of the files as output if not update
# and also make a list of the files for easy rerunning
if update:
	pass
else:
	for in_f, out_f in zip(df['in file'], df['out file']):

		with open(in_f, 'rb') as pi:
			in_p = pickle.load(pi)

		with open(out_f, 'wb') as pf:
			pickle.dump(in_p, pf)

	with open(out_list_filename, 'w') as f:
		f.writelines([i+'\n' for i in df['out file']])

above_to_hide = below_to_hide = None

if current_index < extra:
	# at lower edge, so hide some
	below_to_hide = extra - current_index
elif current_index > len(df) - 1 - extra:
	# at upper edge, hide some
	above_to_hide = current_index + extra + 1 - len(df)
else:
	# nothing to hide/unhide, no issues
	pass

below_index = [current_index - i - 1 if current_index - i > 0 else 0 for i in range(extra)]
above_index = [current_index + i + 1 if current_index + i + 1 < len(df) else len(df) - 1 for i in range(extra)]

below_list = df['out file'][below_index].to_list()
above_list = df['out file'][above_index].to_list()

current = df['out file'][current_index]


########  FIGURE/DICT INIT  ########

if not dark:
	current_plot_params = dict(spec_kwargs = {'color': '#074517', 'alpha': 0.3},
		 					   conti_kwargs = {'color': '#0d7a2a'},
		 					   anchor_kwargs = {'color': '#09521c', 'marker': 'o', 'linestyle': 'none', 'markersize': 6},
		 					   manual_kwargs = {'color': '#12cc43', 'marker': 'D', 'linestyle': 'none', 'markersize': 7},
		 					   locmax_kwargs = {'color': 'k', 's': 5}
							  )

	below_plot_params = dict(spec_kwargs={'color': '#726bd6', 'alpha': 0.2},
							 conti_kwargs={'color': '#14a3fc', 'alpha': 0.3}, 
							 anchor_kwargs={'color': '#3054e3', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
							 manual_kwargs={'color': '#38b8eb', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
							 locmax_kwargs={'color': '#14257a', 'alpha': 0.3, 's': 3}
							)

	above_plot_params = dict(spec_kwargs={'color': '#f55668', 'alpha': 0.2},
							 conti_kwargs={'color': '#ff6176', 'alpha': 0.3}, 
							 anchor_kwargs={'color': '#e32228', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
							 manual_kwargs={'color': '#ff1925', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
							 locmax_kwargs={'color': '#66090e', 'alpha': 0.3, 's': 3}
							)
else:
	current_plot_params = dict(spec_kwargs = {'color': '#6bff93', 'alpha': 0.8, 'linewidth': 1.5},
		 					   conti_kwargs = {'color': '#43fa74', 'linewidth': 1.5},
		 					   anchor_kwargs = {'color': '#ccffd9', 'marker': 'o', 'linestyle': 'none', 'markersize': 6},
		 					   manual_kwargs = {'color': '#11f54d', 'marker': 'D', 'linestyle': 'none', 'markersize': 7},
		 					   locmax_kwargs = {'color': 'white', 's': 5}
							  )

	below_plot_params = dict(spec_kwargs={'color': '#30f1ff', 'alpha': 0.5, 'linewidth': 0.7},
							 conti_kwargs={'color': '#73f1fa', 'alpha': 0.6, 'linewidth': 0.7}, 
							 anchor_kwargs={'color': '#30f1ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
							 manual_kwargs={'color': '#6df3fc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
							 locmax_kwargs={'color': '#bffbff', 'alpha': 0.6, 's': 3}
							)

	above_plot_params = dict(spec_kwargs={'color': '#f14efc', 'alpha': 0.5, 'linewidth': 0.7},
							 conti_kwargs={'color': '#fd80ff', 'alpha': 0.6, 'linewidth': 0.7}, 
							 anchor_kwargs={'color': '#cf30ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
							 manual_kwargs={'color': '#f76dfc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
							 locmax_kwargs={'color': '#fde0ff', 'alpha': 0.6, 's': 3}
							)

# initialise

fig, ax = plt.subplots(figsize=(16,6))
fig.subplots_adjust(left=0.07,right=0.96,hspace=0,top=0.95)

wd = {'above': [{} for i in above_list], 'current': {}, 'below': [{} for i in below_list]}

for i, file in enumerate(below_list):
	below_plot_params_i = below_plot_params.copy()
	for kwargs in below_plot_params_i.values():
		kwargs['alpha'] = kwargs['alpha']/(i+1)

	wd['below'][i]['data'] = get_data(file)
	wd['below'][i]['plot'] = plot_data(ax, wd['below'][i]['data'], **below_plot_params_i)

for i, file in enumerate(above_list):
	above_plot_params_i = above_plot_params.copy()
	for kwargs in above_plot_params_i.values():
		kwargs['alpha'] = kwargs['alpha']/(i+1)

	wd['above'][i]['data'] = get_data(file)
	wd['above'][i]['plot'] = plot_data(ax, wd['above'][i]['data'], **above_plot_params_i)
	
wd['current']['data'] = get_data(current)
wd['current']['plot'] = plot_data(ax, wd['current']['data'], **current_plot_params)

if above_to_hide:
	for i in range(above_to_hide):
		wd['above'][-(i+1)]['plot'] = hide_plot(wd['above'][-(i+1)]['plot'])

if below_to_hide:
	for i in range(below_to_hide):
		wd['below'][-(i+1)]['plot'] = hide_plot(wd['below'][-(i+1)]['plot'])

if dark:
	f_c = '#111111'
	a_c = '#222222'
	text_c = '#ffffff'
	fig.set_facecolor('#111111')
	ax.set_facecolor('#444444')
	ax.set_xlabel(r'Wavelength [$\AA$]',fontsize=14, color=text_c)
	ax.set_ylabel('Flux [arb. unit]',fontsize=14, color=text_c)
	ax.set_title(f"Spectrum {current_index}: {os.path.split(df.loc[df['out file'] == current, 'in file'].values[0])[-1]}", color=text_c)
	ax.xaxis.label.set_color(text_c)
	ax.xaxis.label.set_color(text_c)
	ax.tick_params(axis='both', colors=text_c)
else:
	ax.set_xlabel(r'Wavelength [$\AA$]',fontsize=14)
	ax.set_ylabel('Flux [arb. unit]',fontsize=14)
	ax.set_title(f"Spectrum {current_index}: {os.path.split(df.loc[df['out file'] == current, 'in file'].values[0])[-1]}")
#ax.legend(loc='upper right')

# performance improvements?
plt.style.use('fast')
plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams['path.simplify'] = True
#plt.rcParams['path.simplify_threshold'] = 0.1

hidden = False
thinned = False
points = True
cur_points = True
# adjust
plt.show(block=False)


class Index():

	spec_dict = wd['current']['data']
	plot_dict = wd['current']['plot']

	def change_current(self, spec_dict, plot_dict):

		self.spec_dict = wd['current']['data']
		self.plot_dict = wd['current']['plot']

	def update_data(self, newx, newy):

		newy = newy / self.spec_dict['normalisation']

		# compute distances from event to all locmax
		dist1 = (newx-self.spec_dict['locmaxx_fixed'])**2+(newy-self.spec_dict['locmaxy']/self.spec_dict['normalisation'])**2
		# compute distances from event to all selected max
		dist2 = (newx-self.spec_dict['wave_fixed'])**2+(newy-self.spec_dict['flux']/self.spec_dict['normalisation'])**2

		# list of [closest dist1, its index, second closest dist1, its index]
		#dist_min1 = [np.sort(dist1)[0], np.argsort(dist1)[0], np.sort(dist1)[1], np.argsort(dist1)[1]]
		# faster calculation
		dist1_as = np.argsort(dist1)
		dist_min1 = [dist1[dist1_as[0]], dist1_as[0], dist1[dist1_as[1]], dist1_as[1]]

		# list of [closest dist2, its index]
		dist_min2 = [np.min(dist2), np.argmin(dist2)]

		# if second closest locmax is closer than closest max
		# add the first closest locmax to max? weird logic
		# else, find the closest point in max and remove it
		if dist_min1[0]<dist_min2[0]:
			self.spec_dict['wave_fixed'] = np.append(self.spec_dict['wave_fixed'], self.spec_dict['locmaxx_fixed'][dist_min1[1]])
			self.spec_dict['flux'] = np.append(self.spec_dict['flux'],  self.spec_dict['locmaxy'][dist_min1[1]])
			self.spec_dict['index'] = np.append(self.spec_dict['index'], self.spec_dict['locmaxz'][dist_min1[1]])
		else:
			# what is the point in this function? we already have the index of the closest element
			#where = ras_find_nearest(self.spec_dict['wave_fixed'],self.spec_dict['wave_fixed'][dist_min2[1]])[0]
			self.spec_dict['wave_fixed'] = np.delete(self.spec_dict['wave_fixed'], dist_min2[1])
			self.spec_dict['flux'] = np.delete(self.spec_dict['flux'], dist_min2[1])
			self.spec_dict['index'] = np.delete(self.spec_dict['index'], dist_min2[1])

		self.spec_dict['flux'] = self.spec_dict['flux'][np.argsort(self.spec_dict['wave_fixed'])]
		self.spec_dict['index'] = self.spec_dict['index'][np.argsort(self.spec_dict['wave_fixed'])]
		self.spec_dict['wave_fixed'] = np.sort(self.spec_dict['wave_fixed'])

		# combine with wave and flux vectors
		combx = np.append(self.spec_dict['wave_fixed'], self.spec_dict['wave_man'])
		comby = np.append(self.spec_dict['flux'], self.spec_dict['flux_man'])

		# sort correctly
		combx_as = np.argsort(combx)
		comby = comby[combx_as]
		combx = combx[combx_as]

		Interpol3 = interp1d(combx, comby, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
		continuum3 = Interpol3(self.spec_dict['wavelengths'])
		continuum3 = ras_troncated(continuum3,self.spec_dict['spectre'])

		self.plot_dict['conti'].set_ydata(continuum3)
		self.plot_dict['anchors'].set_xdata(self.spec_dict['wave_fixed'])
		self.plot_dict['anchors'].set_ydata(self.spec_dict['flux'])
		fig.canvas.draw_idle()

	def add_data(self, newx, newy):

		# add data
		self.spec_dict['wave_man'] = np.append(self.spec_dict['wave_man'], newx)
		self.spec_dict['flux_man'] = np.append(self.spec_dict['flux_man'], newy)

		# sort data correctly
		manx_as = np.argsort(self.spec_dict['wave_man'])
		self.spec_dict['flux_man'] = self.spec_dict['flux_man'][manx_as]
		self.spec_dict['wave_man'] = self.spec_dict['wave_man'][manx_as]

		# combine with wave and flux vectors
		combx = np.append(self.spec_dict['wave_fixed'], self.spec_dict['wave_man'])
		comby = np.append(self.spec_dict['flux'], self.spec_dict['flux_man'])

		# sort correctly
		combx_as = np.argsort(combx)
		comby = comby[combx_as]
		combx = combx[combx_as]

		# interpolation
		Interpol3 = interp1d(combx, comby, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
		continuum3 = Interpol3(self.spec_dict['wavelengths'])
		continuum3 = ras_troncated(continuum3, self.spec_dict['spectre'])

		self.plot_dict['conti'].set_ydata(continuum3)
		self.plot_dict['manual'].set_xdata(self.spec_dict['wave_man'])
		self.plot_dict['manual'].set_ydata(self.spec_dict['flux_man'])
		fig.canvas.draw_idle()

	def remove_data(self, delx, dely):

		# need to separate into deleting manual points 
		# vs. deleting existing points
		# only need to update lines of those modified
		dely = dely / self.spec_dict['normalisation']

		# compute distances from event to all manual max
		distman = (delx-self.spec_dict['wave_man'])**2+(dely-self.spec_dict['flux_man']/self.spec_dict['normalisation'])**2
		# compute distances from event to all selected max
		dist1 = (delx-self.spec_dict['wave_fixed'])**2+(dely-self.spec_dict['flux']/self.spec_dict['normalisation'])**2

		# list of [closest dist1, its index, second closest dist1, its index]
		#dist_min1 = [np.sort(dist1)[0], np.argsort(dist1)[0], np.sort(dist1)[1], np.argsort(dist1)[1]]
		# faster calculation
		if len(self.spec_dict['wave_man']) == 0:
			dist_minman = [1e99, -1]
		else:
			dist_minman = [np.min(distman), np.argmin(distman)]


		# list of [closest dist2, its index]
		dist_min1 = [np.min(dist1), np.argmin(dist1)]

		if len(self.spec_dict['wave_man']) == 0 or dist_min1[0] < dist_minman[0]:
			# remove from vecx
			self.spec_dict['wave_fixed'] = np.delete(self.spec_dict['wave_fixed'], dist_min1[1])
			self.spec_dict['flux'] = np.delete(self.spec_dict['flux'], dist_min1[1])
			self.spec_dict['index'] = np.delete(self.spec_dict['index'], dist_min1[1])

			self.spec_dict['flux'] = self.spec_dict['flux'][np.argsort(self.spec_dict['wave_fixed'])]
			self.spec_dict['index'] = self.spec_dict['index'][np.argsort(self.spec_dict['wave_fixed'])]
			self.spec_dict['wave_fixed'] = np.sort(self.spec_dict['wave_fixed'])

			self.plot_dict['anchors'].set_xdata(self.spec_dict['wave_fixed'])
			self.plot_dict['anchors'].set_ydata(self.spec_dict['flux'])

		elif len(self.spec_dict['wave_man']) > 0 and dist_min1[0] > dist_minman[0]:
			# remove from manx
			self.spec_dict['wave_man'] = np.delete(self.spec_dict['wave_man'], dist_minman[1])
			self.spec_dict['flux_man'] = np.delete(self.spec_dict['flux_man'], dist_minman[1])

			manx_as = np.argsort(self.spec_dict['wave_man'])
			self.spec_dict['flux_man'] = self.spec_dict['flux_man'][manx_as]
			self.spec_dict['wave_man'] = self.spec_dict['wave_man'][manx_as]

			self.plot_dict['manual'].set_xdata(self.spec_dict['wave_man'])
			self.plot_dict['manual'].set_ydata(self.spec_dict['flux_man'])

		# combine with wave and flux vectors
		combx = np.append(self.spec_dict['wave_fixed'], self.spec_dict['wave_man'])
		comby = np.append(self.spec_dict['flux'], self.spec_dict['flux_man'])

		# sort correctly
		combx_as = np.argsort(combx)
		comby = comby[combx_as]
		combx = combx[combx_as]

		Interpol3 = interp1d(combx, comby, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
		continuum3 = Interpol3(self.spec_dict['wavelengths'])
		continuum3 = ras_troncated(continuum3,self.spec_dict['spectre'])

		self.plot_dict['conti'].set_ydata(continuum3)
		fig.canvas.draw_idle()


t = Index()


def onclick(event):
	newx = event.xdata
	newy = event.ydata
	if event.dblclick:
		t.update_data(newx,newy)  
		
def onpress(event):
	global wd, df, current_index, extra
	global hidden, thinned, points, cur_points
	if event.key in ['up', 'down', 'x', 't', 'z', 'w']:
		if event.key == 'up':
			wd['current']['data'] = t.spec_dict
			wd['current']['plot'] = t.plot_dict
			wd, current_index = move_up(wd, df, current_index, extra)
			t.change_current(wd['current']['data'], wd['current']['plot'])
			if thinned:
				wd = set_thinning(wd, thinning_fact)
			if points is False:
				wd = hide_all_points(wd, current_index, extra)

		elif event.key == 'down':
			wd['current']['data'] = t.spec_dict
			wd['current']['plot'] = t.plot_dict
			wd, current_index = move_down(wd, df, current_index, extra)
			t.change_current(wd['current']['data'], wd['current']['plot'])
			if thinned:
				wd = set_thinning(wd, thinning_fact)
			if points is False:
				wd = hide_all_points(wd, current_index, extra)

		elif event.key == 'x':
			if hidden is True:
				wd = unhide_all(wd, current_index, extra)
				if points is False:
					wd = hide_all_points(wd, current_index, extra)
				hidden = False
			elif hidden is False:
				wd = hide_all(wd, current_index, extra)
				hidden = True
			else:
				print("Hidden is ", hidden)

		elif event.key == 't':
			if thinned is True:
				wd = set_thinning(wd, 1)
				thinned = False
			elif thinned is False:
				wd = set_thinning(wd, thinning_fact)
				thinned = True
			else:
				print("Thinned is ", thinned)

		elif event.key == 'z':
			if points is True:
				wd = hide_all_points(wd, current_index, extra)
				points = False
			elif points is False:
				wd = unhide_all_points(wd, current_index, extra)
				points = True
			else:
				print("Points is ", points)

		elif event.key == 'w':
			if cur_points is True:
				wd['current']['plot'] = hide_points(wd['current']['plot'])
				fig.canvas.draw_idle()
				cur_points = False
			elif cur_points is False:
				wd['current']['plot'] = unhide_points(wd['current']['plot'])
				fig.canvas.draw_idle()
				cur_points = True
			else:
				print("Cur_points is ", cur_points)


		else:
			print('How did you get here?')
	else:
		newx = event.xdata
		newy = event.ydata
		if event.key == 'a':
			t.add_data(newx, newy)

		elif event.key == 'd':
			t.remove_data(newx, newy)

		else:
			pass
		

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onpress)


help_str = """Maxima modification:
Current spectrum shown in green
Double click to enable/disable points
up/down - scroll through spectra
a - add manual points\td - remove points
x - spectra ghosts\tt - thinning
z - ghost points\tw - current points
"""
print(help_str)
prompt = 'none'
while prompt.strip() != '':
	prompt = input("Press enter to quit: ")

plt.close()

# write current out
write_data(df['out file'][current_index], wd['current']['data'])

if plot_on_exit:
	for f in tqdm(df['out file']):
	
		head, tail = os.path.splitext(f)
		plot_path = head + plot_suf + '.png'

		with open(f, 'rb') as p:
			plot_p = pickle.load(p)

		fig, ax = plot_output(plot_p)

		# xlim logic
		cur_xlims = ax[0].get_xlim()
		in_xlims = (px1, px2)

		new_xlims = [float(in_xlims[i]) if in_xlims[i] is not None else cur_xlims[i] for i in range(2)]

		dx = new_xlims[1] - new_xlims[0]

		xmin, xmax = new_xlims

		ax[0].set_xlim(xmin - 0.01 * dx, xmax + 0.01 * dx)
		ax[1].set_xlim(xmin - 0.01 * dx, xmax + 0.01 * dx)

		ymin = -0.1
		ymax = 1.1
		ax[1].set_ylim(ymin, ymax)

		fig.savefig(plot_path)


print("Done!")

print("To resume, run the following:")
print(f"./maxima_select_full.py {out_list_filename} -u -i {current_index}" + (f"-t {thinning_fact}" if thinning_fact != def_thin else ""))



