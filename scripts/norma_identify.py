#!/usr/bin/env python3
"""
"""
import os
import argparse

import numpy as np
from tqdm import tqdm

from norma import find_max
from norma.io import write_index_file, assemble_index_array


fig_files = [os.path.splitext(f)[0] + plot_suf for f in spec_files]

		# setup figure
		artist_dict = {}
		fig, axes = plt.subplots(figsize=(15, 7), nrows=2, sharex=True)
		fig.subplots_adjust(left=0.07, right=0.96, hspace=0, top=0.95)

		# set labels, title
		axes[-1].set_xlabel("Wavelength ($\mathrm{\AA}$)", fontsize=14)
		axes[0].set_ylabel("Flux (arb. unit)", fontsize=14)
		axes[-1].set_ylabel("Normalised flux", fontsize=14)
		axes[0].set_title(f"Spectrum {'placeholder_num'} - {'placeholder_file'}")

		# setup colours to match 
		base_plot_params = inorma.base_plot_params

		fig.set_facecolor(base_plot_params['fig_facecolor'])

		axes[0].title.set_color(base_plot_params['text_color'])

		for ax in axes:
			ax.set_facecolor(base_plot_params['axes_facecolor'])
			ax.xaxis.label.set_color(base_plot_params['text_color'])
			ax.yaxis.label.set_color(base_plot_params['text_color'])

			ax.tick_params(color=base_plot_params['axes_edgecolor'], labelcolor=base_plot_params['text_color'])
			for spine in ax.spines.values():
				spine.set_edgecolor(base_plot_params['axes_edgecolor'])

		# now plot everything with placeholder values
		plot_params = inorma.current_plot_params

		ones = np.ones(1000)

		artist_dict['spec'], = axes[0].plot(ones, ones, **plot_params['spec_kws'])
		artist_dict['index'], = axes[0].plot(ones, ones, zorder=3, **plot_params['index_kws'])
		artist_dict['sel'], = axes[0].plot(ones, ones, zorder=5,  **plot_params['sel_kws'])
		artist_dict['man'], = axes[0].plot(ones[::10], ones[::10], zorder=6, **plot_params['man_kws'])
		artist_dict['cont'], = axes[0].plot(ones, ones, zorder=4, **plot_params['cont_kws'])

		artist_dict['norm_spec'], = axes[1].plot(ones, ones, **plot_params['spec_kws'])
		artist_dict['norm_index'], = axes[1].plot(ones, ones, zorder=3, **plot_params['index_kws'])
		artist_dict['norm_sel'], = axes[1].plot(ones, ones, zorder=5,  **plot_params['sel_kws'])
		artist_dict['norm_man'], = axes[1].plot(ones[::10], ones[::10], zorder=6, **plot_params['man_kws'])
		artist_dict['norm_cont'], = axes[1].plot(ones, ones, zorder=4, **plot_params['cont_kws'])

	output_on_exit = True

	if output_on_exit:
		for i, out_f in tqdm(enumerate(spec_files_out)):
			spec_file = spec_files[i]
			index_file = index_files[i]
			spec_data = np.loadtxt(spec_file)
			index_data = read_index_file(index_file)

			sel = index_data['sel']

			# normalise with cubic interpolation
			interp = interp1d(index_data['wvs'][sel], index_data['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
			cont = interp(spec_data[:, 0])
			norm_spec_data = spec_data.copy()
			norm_spec_data[:, 1] = spec_data[:, 1] / cont

			np.savetxt(out_f, spec_data)

			if plot_on_exit:
				fig_f = fig_files[i]
				wv_min, wv_max = spec_data[:, 0].min(), spec_data[:, 0].max()
				flux_min, flux_max = spec_data[:, 1].min(), spec_data[:, 1].max()
				# save time, just update data on fig
				data_dict = {
					'spec': spec_data[:, :2],
					'index': spec_data[:, :2][index_data['index']],
					'sel': spec_data[:, :2][index_data['index'][sel]],
					'man': spec_data[:, :2][index_data['index'][index_data['index'] == -1]],
					'cont': np.c_[spec_data[:, 0], cont]
					}

				data_dict = data_dict | {
					'norm_spec': norm_spec_data[:, :2],
					'norm_index': np.c_[data_dict['index'][:, 0], data_dict['index'][:, 1] / cont[index_data['index']]],
					'norm_sel': np.c_[data_dict['sel'][:, 0], data_dict['sel'][:, 1] / cont[index_data['index'][sel]]],
					'norm_man': np.c_[data_dict['man'][:, 0], data_dict['man'][:, 1] / cont[index_data['index'][index_data['index'] == -1]]],
					'norm_cont': np.array([(wv_min, 1.), (wv_max, 1.)])
					}

				_ = [artist_dict[k].set_data(data.T) for k, data in data_dict.items()]

				axes[0].set_title(f"Spectrum {i+1} - {spec_file}")

				#axes[-1].hlines(1, wv_min, wv_max, color='r', linewidth=0.5)

				# tweak xlims
				if not (px1 is None and px2 is None):
					cur_xlims = (wv_min, wv_max)
					in_xlims = (px1, px2)
					xmin, xmax = [float(in_xlims[i]) if in_xlims[i] is not None else cur_xlims[i] for i in range(2)]
					#dx = xmax - xmin
					axes[-1].set_xlim(xmin, xmax)
				else:
					axes[-1].set_xlim(wv_min, wv_max)

				axes[0].set_ylim(0, flux_max + 0.1 * (flux_max - flux_min))
				axes[-1].set_ylim(0, 1.2)

				fig.savefig(fig_f)


####

## need to convert this to a function #
import matplotlib.pyplot as plt

def plot_result(
	wvs: np.ndarray, 
	flux: np.ndarray, 
	index_array: np.ndarray,
	_plot_param_dict: dict | None = None,
	_return_artists: bool = False
	) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, plt.Axes, dict]:
	"""
	Plot spectra continuum fit and normalised spectrum using an `index_array` from find_max.
	`_plot_param_dict` and `_return_artists` are intended for advanced internal use.
	Returns the created figure and axes.
	Also returns a dictionary of artists if `_return_artists` is True.
	"""

	# setup figure
	artist_dict = {}
	fig, axes = plt.subplots(figsize=(15, 7), nrows=2, sharex=True)
	fig.subplots_adjust(left=0.07, right=0.96, hspace=0, top=0.95)

	# set labels, title
	axes[-1].set_xlabel("Wavelength ($\mathrm{\AA}$)", fontsize=14)
	axes[0].set_ylabel("Flux (arb. unit)", fontsize=14)
	axes[-1].set_ylabel("Normalised flux", fontsize=14)
	#axes[0].set_title(f"Spectrum {'placeholder_num'} - {'placeholder_file'}")

	# # setup colours to match 
	# base_plot_params = inorma.base_plot_params

	# fig.set_facecolor(base_plot_params['fig_facecolor'])

	# axes[0].title.set_color(base_plot_params['text_color'])

	# for ax in axes:
	# 	ax.set_facecolor(base_plot_params['axes_facecolor'])
	# 	ax.xaxis.label.set_color(base_plot_params['text_color'])
	# 	ax.yaxis.label.set_color(base_plot_params['text_color'])

	# 	ax.tick_params(color=base_plot_params['axes_edgecolor'], labelcolor=base_plot_params['text_color'])
	# 	for spine in ax.spines.values():
	# 		spine.set_edgecolor(base_plot_params['axes_edgecolor'])

	# # now plot everything with placeholder values
	# plot_params = inorma.current_plot_params

	man_mask = index_array['index'] == -1
	locmax = index_array['index']
	sel = index_array['sel']
	man_wvs, man_flux = index_array['wvs'][man_mask], index_array['flux'][man_mask]

	# normalise with cubic interpolation
	interp = interp1d(index_array['wvs'][sel], index_array['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
	cont = interp(spec_data[:, 0])
	norm_spec_data = spec_data.copy()
	norm_spec_data[:, 1] = spec_data[:, 1] / cont

	artist_dict['spec'], = axes[0].plot(wvs, flux, **plot_param_dict['spec'])
	artist_dict['index'], = axes[0].plot(wvs[locmax], flux[locmax], zorder=3, **plot_param_dict['index'])
	artist_dict['sel'], = axes[0].plot(wvs[locmax[sel]], flux[locmax[sel]], zorder=5, **plot_param_dict['sel'])
	artist_dict['man'], = axes[0].plot(man_wvs, man_flux, zorder=6, **plot_param_dict['man'])
	artist_dict['cont'], = axes[0].plot(ones, ones, zorder=4, **plot_param_dict['cont'])

	artist_dict['norm_spec'], = axes[1].plot(ones, ones, **plot_param_dict['norm_spec'])
	artist_dict['norm_index'], = axes[1].plot(ones, ones, zorder=3, **plot_param_dict['norm_index'])
	artist_dict['norm_sel'], = axes[1].plot(ones, ones, zorder=5,  **plot_param_dict['norm_sel'])
	artist_dict['norm_man'], = axes[1].plot(ones[::10], ones[::10], zorder=6, **plot_param_dict['norm_man'])
	artist_dict['norm_cont'], = axes[1].plot(ones, ones, zorder=4, **plot_param_dict['norm_cont'])

		
	spec_file = spec_files[i]
	index_file = index_files[i]
	spec_data = np.loadtxt(spec_file)
	index_data = read_index_file(index_file)

	sel = index_data['sel']

	# normalise with cubic interpolation
	interp = interp1d(index_data['wvs'][sel], index_data['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
	cont = interp(spec_data[:, 0])
	norm_spec_data = spec_data.copy()
	norm_spec_data[:, 1] = spec_data[:, 1] / cont

	wv_min, wv_max = spec_data[:, 0].min(), spec_data[:, 0].max()
	flux_min, flux_max = spec_data[:, 1].min(), spec_data[:, 1].max()
	# save time, just update data on fig
	data_dict = {
		'spec': spec_data[:, :2],
		'index': spec_data[:, :2][index_data['index']],
		'sel': spec_data[:, :2][index_data['index'][sel]],
		'man': spec_data[:, :2][index_data['index'][index_data['index'] == -1]],
		'cont': np.c_[spec_data[:, 0], cont]
		}

	data_dict = data_dict | {
		'norm_spec': norm_spec_data[:, :2],
		'norm_index': np.c_[data_dict['index'][:, 0], data_dict['index'][:, 1] / cont[index_data['index']]],
		'norm_sel': np.c_[data_dict['sel'][:, 0], data_dict['sel'][:, 1] / cont[index_data['index'][sel]]],
		'norm_man': np.c_[data_dict['man'][:, 0], data_dict['man'][:, 1] / cont[index_data['index'][index_data['index'] == -1]]],
		'norm_cont': np.array([(wv_min, 1.), (wv_max, 1.)])
		}

def main():
	pass