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

def main():
	#### parse args ####

	def_thin = 10
	index_suf = '.index'
	out_suf = '.out'
	plot_suf = '.png'

	desc = "Interactively edit local maxima from spectra using outputs from fitter.find_max"
	epilog = (
		f"Index files should have the same names as their corresponding spectrum, but with the suffix `{index_suf}`."
		"Index files will be updated in-place during editing."
		f"Normalised spectra will be output to the same filenames with `{out_suf}` appended."
		)

	parser = argparse.ArgumentParser(prog="norma_select",
									 description=desc,
									 epilog=epilog)

	parser.add_argument('spec_files',
	                    nargs='+',
						help=f"Paths to spectra files. Index files should have the same paths with the suffix `{index_suf}`")
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
		inorma.current_plot_params = dict(
			spec_kws = {'color': '#6bff93', 'alpha': 0.8, 'linewidth': 1.5},
			cont_kws = {'color': '#43fa74', 'linewidth': 1.5},
			index_kws = {'color': 'white', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
			sel_kws = {'color': '#ccffd9', 'marker': 'o', 'linestyle': 'none', 'markersize': 6},
			man_kws = {'color': '#11f54d', 'marker': 'D', 'linestyle': 'none', 'markersize': 7}
			)

		inorma.above_plot_params = dict(
			spec_kws={'color': '#30f1ff', 'alpha': 0.5, 'linewidth': 0.7},
			cont_kws={'color': '#73f1fa', 'alpha': 0.6, 'linewidth': 0.7}, 
			index_kws={'color': '#bffbff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
			sel_kws={'color': '#30f1ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
			man_kws={'color': '#6df3fc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4},
			)

		inorma.below_plot_params = dict(
			spec_kws={'color': '#f14efc', 'alpha': 0.5, 'linewidth': 0.7},
			cont_kws={'color': '#fd80ff', 'alpha': 0.6, 'linewidth': 0.7}, 
			index_kws={'color': '#fde0ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
			sel_kws={'color': '#cf30ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
			man_kws={'color': '#f76dfc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
			)

		inorma.base_plot_params = dict(
			fig_facecolor='#111111',
			axes_facecolor='#222222',
			text_color='#ffffff',
			axes_edgecolor='#ffffff'
			)

	inorma.start()

	# finished - now need to write out normalised spectra
	# and plot if requested
	spec_files_out = [os.path.splitext(f)[0] + out_suf for f in spec_files]

	if plot_on_exit:
		fig_files = [os.path.splitext(f)[0] + plot_suf for f in spec_files]

		spec_data = np.loadtxt(spec_files[0])
		index_data = read_index_file(index_files[0])
		fig, ax, artist_dict = plot_output(spec_data[:, 0], spec_data[:, 1], _plot_param_dict=plot_params, _return_artists=True)

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

				man_mask = index_array['index'] == -1
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
					'norm_man': np.c_[data_dict['man'][:, 0], data_dict['man'][:, 1] / cont[index_data['index'][man_mask]]],
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



