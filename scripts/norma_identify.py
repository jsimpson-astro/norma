#!/usr/bin/env python3
"""
`norma_identify` is a command-line script to interact with norma.find_max,
to output identified maxima points as files which can be read by `norma_select`.
"""
import os
import argparse

import numpy as np
from tqdm import tqdm

from norma import find_max
from norma.io import write_index_file, assemble_index_array
from norma.plotting import plot_output, plot_styles

def main():

	#### parse args ####

	index_suf = '.index'
	out_suf = '.out'
	plot_suf = '.png'

	desc = "Identify local maxima from spectra and automatically select a continuum for normalisation."
	epilog = (
		f"Index files will be created for each spectrum with the extension `{index_suf}`."
		f"Normalised spectra will be output to the same filenames with the extension `{out_suf}`."
		)

	parser = argparse.ArgumentParser(prog="norma-identify",
									 description=desc,
									 epilog=epilog)

	parser.add_argument('spec_files',
	                    nargs='+',
						help=f"Paths to spectra files. Index files will be created with the same names with the extension `{index_suf}`")

	parser.add_argument('--vfwhm',
						type=float,
						help="FWHM of spectrum in km/s. Automatically determined if not provided.")

	parser.add_argument('-r0', '--min-radius',
						type=float,
						help="Minimum radius of `rolling pin` in angstroms, in blue end of spectrum. Automatically determined if not provided from vfwhm.")
	parser.add_argument('-r1', '--max-radius',
						type=float,
						help="Maximum radius of `rolling pin` in angstroms, in red end of spectrum. Automatically determined if not provided.")

	parser.add_argument('-pr', '--pr-func',
						default='power',
						help="Functional form to use for scaling the radius of the `rolling pin` in large gaps - either `power` or `sigmoid`. (default: `power`).")
	parser.add_argument('-prp1', '--pr-func-param1',
						type=float,
						help="First parameter for the penalty-radius function. Exponent of if pr-func is `power` (default: 1.0), centre of sigmoid if pr-func is `sigmoid` (default: 0.5).")
	parser.add_argument('-prp2', '--pr-func-param2',
						type=float,
						help="Second parameter for the penalty-radius function. Steepness of sigmoid if pr-func is `sigmoid` (default: 0.5). Unused for pr-func `power`.")

	parser.add_argument('-s', '--stretching',
						type=float,
						help="Stretch to apply to wavelength axis - affects tightness of fit. Automatically determined if not provided.")
	parser.add_argument('-t', '--auto-stretching-tightness',
						type=float,
						default=0.5,
						help="Scaling to apply when determining stretching automatically, between 1 and 0. Ignored if stretching is set (default: 0.5).")
	
	parser.add_argument('-win', '--maxima-window',
						type=int,
						default=7,
						help="Half-window size to search for local maxima (int, default: 7).")
	parser.add_argument('-k', '--smoothing-kernel',
						help="Kernel to use for smoothing of spectrum, `rectangular`, `gaussian`, or `savgol` (default: `rectangular`).")
	parser.add_argument('-w', '--smoothing-width',
						type=int,
						help="Half-width of smoothing kernel. Must be > 2 if kernel is `savgol` (int, default: 6)")

	parser.add_argument('-syn', '--synthetic',
						action='store_true',
						help="inject a small (1e-5) amount of noise into the spectrum for numerical stability (default: False).")

	# parser.add_argument('-ef', '--edge-fix',
	# 					action='store-false',
	# 					help="")
	parser.add_argument('-cc', '--clip-cosmics',
						action='store_true',
						help="Enable cosmic-ray cleaning step using sigma-clipping (default: False).")
	parser.add_argument('-ncc', '--clip-iters',
						type=int,
						help="Number of iterations of cosmic sigma-clipping cleaning to perform. Ignored if clip_cosmics is False (int, default: 5).")
	parser.add_argument('-ccp', '--clean-cosmic-peaks',
						action='store_true',
						help="Enable removal of cosmic-ray peaks (default: False).")

	parser.add_argument('-nt', '--disable-telluric-mask',
						action='store_true',
						help="Disable the default telluric line mask (valid < 10000 A) when calculating `vfwhm`. No effect if `vfwhm` set (default: False).")

	parser.add_argument('-seed', '--random-seed',
						type=int,
						help="Random seed to use for noisy processes.")

	parser.add_argument('-p', '--plot-on-exit',
						 action='store_true',
						 help="plot output of all on exit (warning - can be slow) (default: False)")
	parser.add_argument('-pd', '--plot-diagnostics',
						 action='store_true',
						 help="plot additional algorithm diagnostics on output. no effect without `-p` (default: False)")
	parser.add_argument('-px1',
						default=None,
						help="lower plot output wavelength limit (default: None)")
	parser.add_argument('-px2',
						default=None,
						help="upper plot output wavelength limit (default: None)")

	parser.add_argument('-d', '--dark-mode',
						action='store_true',
						help="plot in dark mode (default: False)")

	args = parser.parse_args()

	spec_files = args.spec_files
	# check index files exist/don't, add overwrite arg clobber
	index_files = [os.path.splitext(f)[0] + index_suf for f in spec_files]
	out_files = [os.path.splitext(f)[0] + out_suf for f in spec_files]

	kwargs = {}
	kwargs['vfwhm'] = args.vfwhm
	kwargs['min_radius'] = args.min_radius
	kwargs['max_radius'] = args.max_radius

	# deal with pr_func_params correctly
	kwargs['pr_func'] = args.pr_func
	pr_func_p1 = args.pr_func_param1
	pr_func_p2 = args.pr_func_param2
	if kwargs['pr_func'] == 'power':
		kwargs['pr_func_params'] = 1.0 if pr_func_p1 is None else pr_func_p1
	elif kwargs['pr_func'] == 'sigmoid':
		p1 = 0.5 if pr_func_p1 is None else pr_func_p1
		p2 = 0.5 if pr_func_p1 is None else pr_func_p1
		kwargs['pr_func_params'] = (p1, p2)
	else:
		raise ValueError(f"Invalid pr-func provided: {kwargs['pr_func']}")

	kwargs['stretching'] = args.stretching
	kwargs['auto_stretching_tightness'] = args.auto_stretching_tightness

	kwargs['maxima_window'] = args.maxima_window
	kwargs['smoothing_kernel'] = args.smoothing_kernel
	kwargs['smoothing_width'] = args.smoothing_width

	kwargs['telluric_mask'] = [] if args.disable_telluric_mask else None

	kwargs['synthetic'] = args.synthetic
	kwargs['edge_fix'] = True

	kwargs['clip_cosmics'] = args.clip_cosmics
	kwargs['clip_iters'] = args.clip_iters
	kwargs['clean_cosmic_peaks'] = args.clean_cosmic_peaks

	kwargs['random_seed'] = args.random_seed

	dark = args.dark_mode

	plot_on_exit = args.plot_on_exit
	plot_diagnostics = args.plot_diagnostics
	px1 = args.px1
	px2 = args.px2

	# remove unspecified kwargs, find_max will fill in defaults
	kwargs = {k: v for k, v in kwargs.items() if v is not None}

	print("Input parameters:"
		  "-----------------")
	for k, v in kwargs.items():
		print(f"  {k}: {v}")
	print('\n')

	# similar to norma_select plot on exit, but also adds diagnostics if requested
	if plot_on_exit:
		if dark:
			plot_params = plot_styles['dark']
		else:
			plot_params = plot_styles['light']

		fig_files = [os.path.splitext(f)[0] + plot_suf for f in spec_files]

		spec_data = np.loadtxt(spec_files[0])

		# create fake index data
		index_data = assemble_index_array(spec_data[::100, 0], 
										  spec_data[::100, 1], 
										  np.arange(spec_data[::100, 0].size), 
										  np.ones(spec_data[::100, 0].size, dtype=bool))

		out_plot_params = plot_params['current'] | plot_params['out']
		fig, axes, artist_dict = plot_output(spec_data[:, 0], spec_data[:, 1], index_data, 
											 _plot_param_dict=out_plot_params, _return_artists=True)

		axes[0].set_title(f"Spectrum {1} - {spec_files[0]}")

		# setup colours to match 
		base_plot_params = plot_params['base']

		# add diagnostics if needed, use continuum params
		if plot_diagnostics:
			diag_plot_params = out_plot_params['cont']
			penalty_params = {'linestyle': ':', 'linewidth': 0.5}
			radius_params = {'linestyle': '-.', 'linewidth': 0.5}

			artist_dict['pen'], = axes[0].plot(spec_data[:, 0], spec_data[:, 1], **diag_plot_params | penalty_params, label='Penalty')
			rax = axes[0].twinx()
			artist_dict['rad'], = rax.plot(spec_data[:, 0], spec_data[:, 1], **diag_plot_params | radius_params, label='Radius')

			handles = [artist_dict[k] for k in out_plot_params.keys() if k != 'man'] + [artist_dict['pen'], artist_dict['rad']]
			legend = axes[-1].legend(handles=handles, labelcolor=base_plot_params['text_color'], loc='lower right', ncols=2)
		else:
			handles = [artist_dict[k] for k in out_plot_params.keys() if k != 'man']
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

		if plot_diagnostics:
			rax.tick_params(color=base_plot_params['axes_edgecolor'], labelcolor=base_plot_params['text_color'])
			for spine in rax.spines.values():
				spine.set_edgecolor(base_plot_params['axes_edgecolor'])

	output_on_exit = True

	for i, (in_f, out_f) in tqdm(enumerate(zip(spec_files, index_files))):

		spec_data = np.loadtxt(in_f)

		if plot_diagnostics:
			index_data, penalty, radius = find_max(spec_data[:, 0], spec_data[:, 1], **kwargs, return_penalty=True)	
		else:
			index_data = find_max(spec_data[:, 0], spec_data[:, 1], **kwargs)
		
		write_index_file(out_f, index_data)

		sel = index_data['sel']

		# normalise with cubic interpolation
		interp = interp1d(index_data['wvs'][sel], index_data['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
		cont = interp(spec_data[:, 0])
		norm_spec_data = spec_data.copy()
		norm_spec_data[:, 1] = spec_data[:, 1] / cont

		np.savetxt(out_files[i], norm_spec_data)

		if plot_on_exit:
			
			wv_min, wv_max = spec_data[:, 0].min(), spec_data[:, 0].max()
			flux_min, flux_max = spec_data[:, 1].min(), spec_data[:, 1].max()

			#man_mask = index_data['index'] == -1
			#man_wvs, man_flux = index_data['wvs'][man_mask], index_data['flux'][man_mask]

			# save time, just update data on fig
			data_dict = {
				'spec': spec_data[:, :2],
				'index': spec_data[:, :2][index_data['index']],
				'sel': spec_data[:, :2][index_data['index'][sel]],
				#'man': np.c_[man_wvs, man_flux],
				'cont': np.c_[spec_data[:, 0], cont]
				}

			if plot_diagnostics:
				data_dict = data_dict | {
					'pen': np.c_[spec_data[:, 0], penalty],
					'rad': np.c_[spec_data[:, 0][index_data['index']], radius],
					}

			data_dict = data_dict | {
				'norm_spec': norm_spec_data[:, :2],
				'norm_index': np.c_[data_dict['index'][:, 0], data_dict['index'][:, 1] / cont[index_data['index']]],
				'norm_sel': np.c_[data_dict['sel'][:, 0], data_dict['sel'][:, 1] / cont[index_data['index'][sel]]],
				#'norm_man': np.c_[data_dict['man'][:, 0], data_dict['man'][:, 1] / interp(data_dict['man'][:, 0])],
				'norm_cont': np.array([(wv_min, 1.), (wv_max, 1.)])
				}

			_ = [artist_dict[k].set_data(data.T) for k, data in data_dict.items()]

			axes[0].set_title(f"Spectrum {i+1} - {in_f}")

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

			if plot_diagnostics:
				rax.set_ylim(0, radius.max() + 0.1 * (radius.max() - radius.min()))

			axes[-1].set_ylim(0, 1.2)

			fig.savefig(fig_files[i])
			

	print("Done!")