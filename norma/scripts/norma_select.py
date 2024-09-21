#!/usr/bin/env python3
"""
`norma_select` is a command-line script to interface with norma.InteractiveNorma,
to allow tweaking of selected indices output by norma.find_max
"""
import os
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from norma import InteractiveNorma
from norma.io import read_index_file
from norma.plotting import plot_output, plot_styles

def main():
	#### parse args ####

	def_thin = 10
	index_suf = '.index'
	out_suf = '.out'
	plot_suf = '.png'

	desc = f"""Interactively edit local maxima from spectra using outputs from `norma-identify` (or norma.find_max).

Index files should have the same names as their corresponding spectrum, but with the extension `{index_suf}`.
Index files will be updated in-place during editing.
Normalised spectra will be output to the same filenames with the extension `{out_suf}`.

"""

	parser = argparse.ArgumentParser(prog="norma-select",
									 description=desc,
									 epilog=" ",
									 formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('spec_files',
	                    nargs='+',
						help=f"Paths to spectra files. Index files should have the same paths with the extension `{index_suf}`")
	parser.add_argument('-n', '--number',
						default=3,
						help="number of spectra to plot at once, must be >= 1 (default: 3)")
	parser.add_argument('-i', '--start',
						default=0,
						help="starting point for spectra, in order of input list (default: 0)")
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

	args = parser.parse_args()

	spec_files = args.spec_files
	index_files = [os.path.splitext(f)[0] + index_suf for f in spec_files]

	number = int(args.number)
	start_index = int(args.start)
	thinning_fact = int(args.thinning)
	dark = args.dark_mode

	plot_on_exit = args.plot_on_exit
	px1 = args.px1
	px2 = args.px2

	# class will verify all these params
	inorma = InteractiveNorma(spec_files, 
							  index_files, 
							  start_index=start_index, 
							  n_plot=number, 
							  thinning_factor=thinning_fact)

	# set dark mode by changing params
	if dark:
		plot_params = plot_styles['dark']
	else:
		plot_params = plot_styles['light']
		
	inorma.base_plot_params = plot_params['base']
	inorma.current_plot_params = {k+'_kws': v for k, v in plot_params['current'].items()}
	inorma.above_plot_params = {k+'_kws': v for k, v in plot_params['above'].items()}
	inorma.below_plot_params = {k+'_kws': v for k, v in plot_params['below'].items()}

	inorma.start()

	# finished - now need to write out normalised spectra
	# and plot if requested
	spec_files_out = [os.path.splitext(f)[0] + out_suf for f in spec_files]

	if plot_on_exit:
		fig_files = [os.path.splitext(f)[0] + plot_suf for f in spec_files]

		spec_data = np.loadtxt(spec_files[0])
		index_data = read_index_file(index_files[0])

		out_plot_params = plot_params['current'] | plot_params['out']
		fig, axes, artist_dict = plot_output(spec_data[:, 0], spec_data[:, 1], index_data, 
											 _plot_param_dict=out_plot_params, _return_artists=True)

		axes[0].set_title(f"Spectrum {1} - {spec_files[0]}")

		# setup colours to match 
		base_plot_params = plot_params['base']

		handles = [artist_dict[k] for k in out_plot_params.keys()]
		legend = axes[-1].legend(handles=handles, labelcolor=base_plot_params['text_color'], loc='lower right')

		fig.set_facecolor(base_plot_params['fig_facecolor'])
		axes[0].title.set_color(base_plot_params['text_color'])

		legend.get_frame().set_facecolor(base_plot_params['axes_facecolor'])

		for ax in axes:
			ax.set_facecolor(base_plot_params['axes_facecolor'])
			ax.xaxis.label.set_color(base_plot_params['text_color'])
			ax.yaxis.label.set_color(base_plot_params['text_color'])

			ax.tick_params(color=base_plot_params['axes_edgecolor'], labelcolor=base_plot_params['text_color'])
			for spine in ax.spines.values():
				spine.set_edgecolor(base_plot_params['axes_edgecolor'])

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

			np.savetxt(out_f, norm_spec_data)

			if plot_on_exit:
				fig_f = fig_files[i]
				wv_min, wv_max = spec_data[:, 0].min(), spec_data[:, 0].max()
				flux_min, flux_max = spec_data[:, 1].min(), spec_data[:, 1].max()

				man_mask = index_data['index'] == -1
				man_wvs, man_flux = index_data['wvs'][man_mask], index_data['flux'][man_mask]

				# save time, just update data on fig
				data_dict = {
					'spec': spec_data[:, :2],
					'index': spec_data[:, :2][index_data['index']],
					'sel': spec_data[:, :2][index_data['index'][sel]],
					'man': np.c_[man_wvs, man_flux],
					'cont': np.c_[spec_data[:, 0], cont]
					}

				data_dict = data_dict | {
					'norm_spec': norm_spec_data[:, :2],
					'norm_index': np.c_[data_dict['index'][:, 0], data_dict['index'][:, 1] / cont[index_data['index']]],
					'norm_sel': np.c_[data_dict['sel'][:, 0], data_dict['sel'][:, 1] / cont[index_data['index'][sel]]],
					'norm_man': np.c_[data_dict['man'][:, 0], data_dict['man'][:, 1] / interp(data_dict['man'][:, 0])],
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
			

	print("Done!")



