import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

_default_plot_param_dict = {
	'spec': {'color': '#074517', 'alpha': 0.3},
	'cont': {'color': '#0d7a2a'},
	'index': {'color': 'k', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
	'sel': {'color': '#09521c', 'marker': 'o', 'linestyle': 'None', 'markersize': 6},
	'man':  {'color': '#12cc43', 'marker': 'D', 'linestyle': 'None', 'markersize': 7}
	}

_plot_keys = set(_default_plot_param_dict.keys())
_norm_plot_keys = {'norm_' + k for k in _plot_keys}

# # duplicate for norm plots
# _default_plot_param_dict = _default_plot_param_dict | {'norm_' + k: v for k, v in _default_plot_param_dict.items()}


def plot_output(
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
	plot_params = _default_plot_param_dict if _plot_param_dict is None else _plot_param_dict

	# check all plot params keys are present
	missing_keys = [k for k in _plot_keys if k not in plot_params.keys()]
	if len(missing_keys) != 0:
		raise KeyError(f"Missing keys in _plot_param_dict: {missing_keys}.")

	# check `norm_` keys - duplicate them from the regular keys if they are missing
	missing_norm_keys = [k for k in _norm_plot_keys if k not in plot_params.keys()]
	if len(missing_norm_keys) == 0:
		pass
	elif len(missing_norm_keys) == len(_norm_plot_keys):
		# none provided, duplicate across
		plot_params = plot_params | {'norm_' + k: v for k, v in plot_params.items()}
	else:
		current_norm_keys = [k for k in _norm_plot_keys if k in plot_params.keys()]
		raise KeyError(f"Missings keys in _plot_param_dict: {missing_norm_keys}."
			f"Either provide missing keys or omit keys {current_norm_keys}, and {_plot_keys} will be used instead.")

	# setup figure
	artist_dict = {}
	fig, axes = plt.subplots(figsize=(15, 7), nrows=2, sharex=True)
	fig.subplots_adjust(left=0.07, right=0.96, hspace=0, top=0.95)

	# set labels, title
	axes[-1].set_xlabel("Wavelength ($\mathrm{\AA}$)", fontsize=14)
	axes[0].set_ylabel("Flux (arb. unit)", fontsize=14)
	axes[-1].set_ylabel("Normalised flux", fontsize=14)

	man_mask = index_array['index'] == -1
	locmax = index_array['index']
	sel = index_array['sel']
	man_wvs, man_flux = index_array['wvs'][man_mask], index_array['flux'][man_mask]

	# normalise with cubic interpolation
	interp = interp1d(index_array['wvs'][sel], index_array['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
	cont = interp(wvs)
	norm_flux = flux / cont

	artist_dict['spec'], = axes[0].plot(wvs, flux, **plot_params['spec'])
	artist_dict['index'], = axes[0].plot(wvs[locmax], flux[locmax], zorder=3, **plot_params['index'])
	artist_dict['sel'], = axes[0].plot(wvs[locmax[sel]], flux[locmax[sel]], zorder=5, **plot_params['sel'])
	artist_dict['man'], = axes[0].plot(man_wvs, man_flux, zorder=6, **plot_params['man'])
	artist_dict['cont'], = axes[0].plot(wvs, cont, zorder=4, **plot_params['cont'])

	artist_dict['norm_spec'], = axes[1].plot(wvs, norm_flux, **plot_params['norm_spec'])
	artist_dict['norm_index'], = axes[1].plot(wvs[locmax], flux[locmax] / cont[locmax], zorder=3, **plot_params['norm_index'])
	artist_dict['norm_sel'], = axes[1].plot(wvs[locmax[sel]], flux[locmax[sel]] / cont[locmax[sel]], zorder=5,  **plot_params['norm_sel'])
	artist_dict['norm_man'], = axes[1].plot(man_wvs, man_flux / cont[locmax[man_mask]], zorder=6, **plot_params['norm_man'])
	artist_dict['norm_cont'], = axes[1].plot([wvs.min(), wvs.max()], [1, 1], zorder=4, **plot_params['norm_cont'])

	if _return_artists:
		return fig, axes, artist_dict
	else:
		return fig, axes

#### plotting parameters for InteractiveNorma and norma_identify ####

# plot parameters for light mode
_light_plot_params = {
	# base properties of fig/axes
    'base': dict(
        fig_facecolor='white',
        axes_facecolor='white',
        text_color='black',
        axes_edgecolor='black'
        ),
    # for currently selected spectrum
    'current': dict(
        spec = {'color': '#074517', 'alpha': 0.3},
        cont = {'color': '#0d7a2a'},
        index = {'color': 'k', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
        sel = {'color': '#09521c', 'marker': 'o', 'linestyle': 'None', 'markersize': 6},
        man = {'color': '#12cc43', 'marker': 'D', 'linestyle': 'None', 'markersize': 7},
        ),
    # for spectra above - alpha is reduced by 1/i for i above
    'above': dict(
        spec={'color': '#726bd6', 'alpha': 0.2},
        cont={'color': '#14a3fc', 'alpha': 0.3}, 
        index={'color': '#14257a', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3},
        sel={'color': '#3054e3', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3}, 
        man={'color': '#38b8eb', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'None', 'markersize': 4}
        ),
    # for spectra below - alpha is reduced by 1/i for i below
    'below': dict(
        spec={'color': '#f55668', 'alpha': 0.2},
        cont={'color': '#ff6176', 'alpha': 0.3}, 
        index={'color': '#66090e', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3},
        sel={'color': '#e32228', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3}, 
        man={'color': '#ff1925', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'None', 'markersize': 4}
        ),
    # for output - updates `current` parameters
    'out': dict(
        spec = {'color': '#074517', 'alpha': 0.3},
        cont = {'color': '#0d7a2a'},
        index = {'color': 'k', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
        sel = {'color': '#09521c', 'marker': 'o', 'linestyle': 'None', 'markersize': 6},
        man = {'color': '#12cc43', 'marker': 'D', 'linestyle': 'None', 'markersize': 7},
        ),
	}


# plot parameters for dark mode
_dark_plot_params = {
	# base properties of fig/axes
	'base': dict(
		fig_facecolor='#111111',
		axes_facecolor='#222222',
		text_color='#ffffff',
		axes_edgecolor='#ffffff'
		),
    # for currently selected spectrum
	'current': dict(
		spec={'color': '#6bff93', 'alpha': 0.8, 'linewidth': 1.5},
		cont={'color': '#43fa74', 'linewidth': 1.5},
		index={'color': 'white', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
		sel={'color': '#ccffd9', 'marker': 'o', 'linestyle': 'none', 'markersize': 6},
		man={'color': '#11f54d', 'marker': 'D', 'linestyle': 'none', 'markersize': 7}
		),
    # for spectra above - alpha is reduced by 1/i for i above
	'above': dict(
		spec={'color': '#30f1ff', 'alpha': 0.5, 'linewidth': 0.7},
		cont={'color': '#73f1fa', 'alpha': 0.6, 'linewidth': 0.7}, 
		index={'color': '#bffbff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
		sel={'color': '#30f1ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
		man={'color': '#6df3fc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4},
		),
	# for spectra below - alpha is reduced by 1/i for i below
	'below': dict(
		spec={'color': '#f14efc', 'alpha': 0.5, 'linewidth': 0.7},
		cont={'color': '#fd80ff', 'alpha': 0.6, 'linewidth': 0.7}, 
		index={'color': '#fde0ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
		sel={'color': '#cf30ff', 'alpha': 0.6, 'marker': 'o', 'linestyle': 'none', 'markersize': 3}, 
		man={'color': '#f76dfc', 'alpha': 0.6, 'marker': 'D', 'linestyle': 'none', 'markersize': 4}, 
		),
    # for output - updates `current` parameters
	'out': dict(
		spec={'color': '#6bff93', 'alpha': 0.8, 'linewidth': 1.5},
		cont={'color': '#43fa74', 'linewidth': 1.5},
		index={'color': 'white', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
		sel={'color': '#ccffd9', 'marker': 'o', 'linestyle': 'none', 'markersize': 6},
		man={'color': '#11f54d', 'marker': 'D', 'linestyle': 'none', 'markersize': 7}
		)
	}

# simple dict to access these params
plot_styles = {'light': _light_plot_params, 'dark': _dark_plot_params}