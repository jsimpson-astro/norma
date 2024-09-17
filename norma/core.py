import numpy as np
from scipy.interpolate import interp1d
import logging

from norma.utils import *
from norma.io import assemble_index_array

def find_max(
    wvs: np.ndarray,
    flux: np.ndarray,
    **kwargs
    ) -> tuple[np.ndarray[int], np.ndarray[bool]]:
    """
    Identify local maxima along a spectrum and select continuum points.
    Based on the RASSINE program.

    Parameters
    ----------
    wvs: np.ndarray
        Wavelengths of spectrum
    flux: np.ndarray
        Fluxes of spectrum, same length as `wvs`

    Returns
    -------
    index_array: structured np.ndarray of float, float, int, bool
        All identified maxima points along spectrum, as indexed from the original spectrum.
        The structured array has 4 columns: `wvs` and `flux` for the wavelength and
        flux of each point in the original spectrum, `index` for its index, and `sel`
        which is a boolean mask indicating the selected points.

    To normalise, use norma.utils.normalise with the input spectrum and output index_array:
        norm_flux = norma.utils.normalise(wvs, flux, index_array)

    The selected points in `index_array` will be interpolated over with a cubic spline,
    and used to divide the spectrum's flux.

    Other parameters
    ----------------
    vfwhm: float, optional
        FWHM of spectrum in km/s. Automatically determined if not provided.

    min_radius: float, optional
        Minimum radius of `rolling pin` in angstroms, in blue end of spectrum.
        If not given, it is automatically determined using `vfwhm`.

    max_radius: float, optional
        Maximum radius of `rolling pin` in angstroms, in red end of spectrum.
        If not given, it is automatically determined from the spectrum.

    pr_func: {'power', 'sigmoid'}, default: `power`
        Functional form to use for scaling the radius of the `rolling pin` in large gaps.
        `poly` for polynomial, `sigmoid` for a sigmoid function.

    pr_func_params: float | tuple[float, float], optional
        Parameters for the penalty-radius function. 
        If penalty_radius_func is `power`, should be a float for the exponent.
        If penalty_radius_func is `sigmoid`, should be a tuple of 2 floats, where:
            - first float is the centre of the signoid
            - second float is the steepness of the sigmoid
        Defaults to 1.0 for `power`, (0.5, 0.5) for `sigmoid`
    
    stretching: float, optional
        Stretch to apply to wavelength axis before normalisation.
        Determines the `tightness` of the continuum fit - lower values = tighter fit.
        If not given, automatically determined using `vfwhm`.
        
    auto_stretching_tightness: float, default: 0.5
        Scaling to apply when determining stretching automatically, between 1 and 0.
        Ignored if stretching is set.

    maxima_window: int, default: 7
        Half-window size to search for local maxima - lower values = less anchor points.

    smoothing_kernel: {`rectangular`, `gaussian`, `savgol`, `erf`, `hat_exp`}, default: `rectangular`
        Kernel to use for smoothing of spectrum.
        
    smoothing_width: float, optional
        Half-width of smoothing kernel. 
        Rounded to nearest integer and used a window if smoothing_kernel is `rectangular`, `gaussian`, or `savgol`.
        For these kernels, the default value is 6. Must be larger than 2 for the `savgol` kernel.
        
        Used as half-width half-maximum (= FWHM / 2) if smoothing_kernel is `erf` or `hat_exp`. 
        For these kernels, the default value is the value of `vfwhm`.
        
    telluric_mask: list of tuple[float, float], default: see below
        Telluric mask to apply to spectrum during calculation of `vfwhm`.
        Default mask covers the optical and near-IR O2 and H20 bands up to 10000 AA:
            [(6275., 6330.), (6470., 6577.), (6866., 6970.), (7190., 7400.), (7580., 7750.), (8120., 8350.), (8900., 10000.)]

    synthetic: bool, default: False
        If True, inject a small (0.01%) amount of noise into the spectrum for numerical stability.

    edge_fix: bool, default: True
        Always select the first and last local maxima.

    clip_cosmics: bool, default: False
        Enable cosmic-ray cleaning step using sigma-clipping.

    clip_iters: int, default: 5
        Number of iterations of cosmic sigma-clipping cleaning to perform.
        Ignored if clip_cosmics is False.

    clean_cosmic_peaks: bool, default: True
        Enable removal of cosmic-ray peaks.
    
    random_seed: int
        Random seed to use for noisy processes.

    CCF_mask: currently unused
        Mask to apply to spectrum during calculation of `vfwhm`.
        Defaults to 'master' in rassine - need to evaluate what this is and if it is useful.
        Temporarily ignoring in the meantime.
        
    """
    # old arg, currently unused:
    #    denoising_width: int, default: 5
    #       Half-window size to average points around local maxima to form the continuuum.
    
    # get params and check
    vfwhm = kwargs.get('vfwhm', None)
    min_radius = kwargs.get('min_radius', None)
    max_radius = kwargs.get('max_radius', None)

    # check pr_func and params are valid
    pr_funcs = {'power', 'sigmoid'}
    default_pr_func_params = {'power': 1.0, 'sigmoid': (0.5, 0.5)}
    
    pr_func = kwargs.get('pr_func', 'power')
    if pr_func not in pr_funcs:
        raise ValueError(f"Invalid pr_func given: {pr_func}. Valid choices are: {pr_funcs}.")
        
    pr_func_params = kwargs.get('pr_func_params', default_pr_func_params[pr_func])
    if pr_func == 'power' and not isinstance(pr_func_params, float | int):
        raise ValueError(f"Invalid pr_func_params for pr_func `{pr_func}`: {pr_func_params}.")
    elif (pr_func == 'sigmoid' and 
          not isinstance(pr_func_params, tuple) and 
          not len(pr_func_params > 1) and
          not all(isinstance(i, float | int) for i in pr_func_params)):
        raise ValueError(f"Invalid pr_func_params for pr_func `{pr_func}`: {pr_func_params}.")

    stretching = kwargs.get('stretching', None)
    auto_stretching_tightness = kwargs.get('auto_stretching_tightness', 0.5)
    maxima_window = kwargs.get('maxima_window', 7)

    # get and check smoothing kernels
    smoothing_kernels = {'rectangular', 'gaussian', 'savgol', 'erf', 'hat_exp'}
    default_smoothing_width = {'rectangular': 6., 'gaussian': 6., 'savgol': 6., 
                               'erf': vfwhm, 'hat_exp': vfwhm}
    
    smoothing_kernel = kwargs.get('smoothing_kernel', 'rectangular')
    if smoothing_kernel not in smoothing_kernels:
        raise ValueError(f"Invalid smoothing_kernel given: {smoothing_kernel}. Valid choices are {smoothing_kernels}.")

    # round to int if a window half-width is needed
    smoothing_width = kwargs.get('smoothing_width', default_smoothing_width[smoothing_kernel])
    smoothing_width = int(smoothing_width) if smoothing_kernel in {'rectangular', 'gaussian', 'savgol'} else smoothing_width

    # get and check mask
    # this will fail if it isn't iterable and can't be unpacked to two elements anyways
    default_telluric_mask = [(6275., 6330.), (6470., 6577.), (6866., 6970.), (7190., 7400.), (7580., 7750.), (8120., 8350.), (8900., 10000.)]
    telluric_mask = kwargs.get('telluric_mask', default_telluric_mask)
    if telluric_mask != default_telluric_mask:
        for lb, ub in telluric_mask:
            if not (isinstance(lb, float | int) and isinstance(ub, float | int)):
                raise TypeError(f"Provided telluric_mask contains invalid bounds of type ({type(lb)}, {type(ub)}): ({lb}, {ub}).")
            if lb > ub:
                raise ValueError(f"Provided telluric_mask contains invalid bounds: ({lb}, {ub}).")

    synthetic = kwargs.get('synthetic', False)

    #denoising_width = kwargs.get('denoising_width', 5)
    edge_fix = kwargs.get('edge_fix', True)
    clip_cosmics = kwargs.get('clip_cosmics', False)
    clip_iters = kwargs.get('clip_iters', 5)
    clean_cosmic_peaks = kwargs.get('clean_cosmic_peaks', False)
    random_seed = kwargs.get('random_seed')
                          
    # check values are positive, and integers where needed
    check_positive = {
        'vfwhm': vfwhm,
        'min_radius': min_radius,
        'max_radius': max_radius,
        'stretching': stretching,
        'auto_stretching_tightness': auto_stretching_tightness,
        'maxima_window': maxima_window,
        'smoothing_width': smoothing_width,
        #'denoising_width': denoising_width,
        'clip_iters': clip_iters
        }
    check_int_names = [
        'maxima_window',
        #'denoising_width',
        'clip_iters'
        ]
    
    for arg_name, arg in check_positive.items():
        if arg and arg <= 0:
            raise ValueError(f"{arg_name} must be positive: {arg}")
        if arg_name in check_int_names and not isinstance(arg, int):
            raise TypeError(f"{arg_name} must be an integer, not {type(arg)}.")

    #### make grid ####
    
    # debating the use of an equidistant grid
    dgrid = dwv = (wvs.max() - wvs.min()) / wvs.size
    if kwargs.get('use_grid'):
        grid = np.linspace(wvs.min(), wvs.min() + (wvs.size - 1) * dwv,
                           wvs.size, dtype=np.float64)
    else:
        grid = wvs.copy()
    fluxes = flux.copy()

    # Preprocess spectrum
    assert np.all(np.isfinite(grid)), "Grid points must be finite"
    assert np.all(np.isfinite(fluxes)), "Flux values must be finite"

    # clamp value to non-negative
    fluxes[fluxes < 0] = 0

    #### find SNR ####
    
    wave_min = grid[0]
    wave_max = grid[-1]

    SNR_at = kwargs.get('wv_for_SNR')
    SNR_width = kwargs.get('width_for_SNR', 50)

    if SNR_at:
        if not grid[0] < SNR_at < grid[-1]:
            raise ValueError(f"wv_for_SNR of {SNR_at} out of range for spectrum from {wave_min} to {wave_max}.")
    else:
        wv_range = grid[-1] - grid[0]
        midpoint = (grid[-1] + grid[0]) / 2

        # find a good midpoint
        SNR_index0 = np.abs(grid - (midpoint - 0.1*wv_range)).argmin()
        SNR_index1 = np.abs(grid - (midpoint + 0.1*wv_range)).argmin()
        
        SNR_at = grid[SNR_index0:SNR_index1][flux[SNR_index0:SNR_index1].argmax()]
        
    SNR_index0 = np.abs(grid - (SNR_at - SNR_width / 2)).argmin()
    SNR_index1 = np.abs(grid - (SNR_at + SNR_width / 2)).argmin()
    SNR = np.quantile(fluxes[SNR_index0:SNR_index1], 0.95)**0.5

    #### preprocessing ####
    
    # "normalise" i.e. scale
    norm = (fluxes.max() - fluxes.min()) / (grid[-1] - grid[0])
    flux_norm = fluxes / norm

    # inject 1e-5 of noise if synthetic
    # no clue what the np.min(np.diff) line is, I think its incorrect
    if synthetic:
        if random_seed:
            np.random.seed(random_seed)
        np.diff(flux) != 0
        flux_norm += np.random.randn(flux_norm.size) * 1e-5 

    # cosmic cleaning in rolling windows
    # uses maxima_window (par_vicinity)
    if clip_cosmics:
        for i in range(clip_iters):
            maxi_roll_fast = rolling_quantile(flux_norm, int(100/dgrid), q=0.99, min_periods=1, center=True)
            Q3_fast = rolling_quantile(flux_norm, int(5/dgrid), q=0.75, min_periods=1, center=True)
            Q2_fast = rolling_quantile(flux_norm, int(5/dgrid), q=0.50, min_periods=1, center=True)
            
            IQ_fast = 2 * (Q3_fast - Q2_fast)
            sup_fast = Q3_fast + 1.5 * IQ_fast
            mask = (flux_norm > sup_fast) & (flux_norm > maxi_roll_fast)

            # suppress cosmic peaks over the range of maxima_window
            for j in range(int(maxima_window / 2)):
                mask = mask | np.roll(mask, -j) | np.roll(mask, j)
            if sum(mask) == 0:
                break
            flux_norm[mask] = Q2_fast[mask]

    #### apply smoothing ####

    # compute vfwhm if needed
    out_of_calibration = False
    if vfwhm is None:
        vfwhm, vfwhm_err = calculate_fwhm(wvs, flux_norm, telluric_mask, plot_ccf=kwargs.get('plot_ccf', False))
        if wave_min * (vfwhm / 2.997e5) > 15:
            vfwhm = 2.997e5 * 15 / wave_min
            logging.warning("Star out of the FWHM calibration range - FWHM capped to 15 A.")
            out_of_calibration = True
        
    #return fwhm, fwhm_err
    fwhm2sig = 2**1.5 * np.log(2)**0.5
    #print('vfwhm', vfwhm)
    # convert vfwhm to fwhm using min wavelength (gives largest fwhm when using equidistant grid)
    fwhm = wave_min * (vfwhm / 2.997e5) 
    #print('converted fwhm:', fwhm)

    # determine stretching parameter if not given
    if stretching is None:
        if out_of_calibration:
            stretching = 7.0
        else:
            calib_low = np.polyval([-0.08769286, 5.90699857], fwhm)
            calib_high = np.polyval([-0.38532535, 20.17699949], fwhm)
            stretching = calib_low + (calib_high - calib_low) * auto_stretching_tightness
            #print('stretching:', stretching)

    if smoothing_kernel in {'erf', 'hat_exp'}:
        raise NotImplementedError
        # auto smoothing
        def perform_auto_smoothing() -> np.ndarray:
            # grille en vitesse radiale (unités km/s)
            grid_vrad = (grid - wave_min) / grid * c_lum / 1000
            # new grid equidistant
            grid_vrad_equi = np.linspace(grid_vrad.min(), grid_vrad.max(), len(grid))
            # delta velocity
            dv = np.diff(grid_vrad_equi)[0]
            spectrum_vrad = interp1d(
                grid_vrad, spectre, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )(grid_vrad_equi)

            sp = np.fft.fft(spectrum_vrad)
            # List of frequencies
            freq: NDArray[np.float64] = np.fft.fftfreq(grid_vrad_equi.shape[-1]) / dv  # type: ignore
            sig1 = par_fwhm / 2.35  # fwhm-sigma conversion

            if par_smoothing_kernel == "erf":
                # using the calibration curve calibration
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.00210819, -0.04581559, 0.49444111, -1.78135102]), np.log(SNR_0)
                    )
                )
                alpha2 = np.polyval(np.array([-0.04532947, -0.42650657, 0.59564026]), SNR_0)
            elif par_smoothing_kernel == "hat_exp":
                # using the calibration curve calibration
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.01155214, -0.20085361, 1.34901688, -3.63863408]), np.log(SNR_0)
                    )
                )
                alpha2 = np.polyval(np.array([-0.06031564, -0.45155956, 0.67704286]), SNR_0)
            else:
                raise NotImplementedError
            fourier_center = alpha1 / sig1
            fourier_delta = alpha2 / sig1
            cond = abs(freq) < fourier_center

            if par_smoothing_kernel == "erf":
                # erf function
                fourier_filter = 0.5 * (erf((fourier_center - abs(freq)) / fourier_delta) + 1)
            elif par_smoothing_kernel == "hat_exp":
                # Top hat with an exp
                fourier_filter = cond + (1 - cond) * np.exp(
                    -(abs(freq) - fourier_center) / fourier_delta
                )
            else:
                raise NotImplementedError

            fourier_filter = fourier_filter / fourier_filter.max()

            spectrei_ifft = np.fft.ifft(fourier_filter * (sp.real + 1j * sp.imag))
            spectrei_ifft = np.abs(spectrei_ifft)
            spectre_back = interp1d(
                grid_vrad_equi,
                spectrei_ifft,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(grid_vrad)
            median = np.median(abs(spectre_back - spectre))
            IQ = np.percentile(abs(spectre_back - spectre), 75) - median
            mask_out_fourier = np.where(abs(spectre_back - spectre) > (median + 20 * IQ))[0]
            length_oversmooth = int(1 / fourier_center / dv)
            mask_fourier = np.unique(
                mask_out_fourier
                + np.arange(-length_oversmooth, length_oversmooth + 1, 1)[:, np.newaxis]
            )
            mask_fourier = mask_fourier[(mask_fourier >= 0) & (mask_fourier < len(grid))]
            # supress the smoothing of peak to sharp which create sinc-like wiggle
            spectre_back[mask_fourier] = spectre[mask_fourier]
            # suppression of the border which are at high frequencies
            spectre_back[0 : length_oversmooth + 1] = spectre[0 : length_oversmooth + 1]
            spectre_back[-length_oversmooth:] = spectre[-length_oversmooth:]
            return spectre_back

        spectre = perform_auto_smoothing()
    else:
        
        flux_norm_smooth = apply_smooth(flux_norm, 2*smoothing_width, smoothing_kernel)
        smooth_dif = np.abs(flux_norm - flux_norm_smooth)
        median = np.median(smooth_dif)
        IQR = np.quantile(smooth_dif, 0.75) - median

        # mask everything greater than a threshold (= median + 20 * IQR from 0.5 to 0.75
        mask = np.asarray(smooth_dif > (median + 20 * IQR)).nonzero()
        # select a smoothing width around masked elements
        mask = np.unique(mask + np.arange(-smoothing_width, smoothing_width + 1)[:, None])
        # clip these expanded indices to valid ones within the spectrum
        mask = mask[(mask >= 0) & (mask < grid.size)]

        # replaced masked values with original
        flux_norm_smooth[mask] = flux_norm[mask]
        flux_norm = flux_norm_smooth

    # find maxima
    maxima_idxs, maxima_fluxes = local_max(flux_norm, maxima_window)
    maxima_wvs = grid[maxima_idxs]

    # add points to ends if they are below the edge fluxes
    if maxima_fluxes[0] < flux_norm[0]:
        maxima_wvs = np.insert(maxima_wvs, 0, grid[0])
        maxima_fluxes = np.insert(maxima_fluxes, 0, flux_norm[0])
        maxima_idxs = np.insert(maxima_idxs, 0, 0)

    if maxima_fluxes[-1] < flux_norm[-1]:
        maxima_wvs = np.hstack([maxima_wvs, grid[-1]])
        maxima_fluxes = np.hstack([fluxes, flux_norm[-1]])
        maxima_idxs = np.hstack([maxima_idxs, flux_norm.size - 1])

    # removal of cosmic peaks     
    if clean_cosmic_peaks:
        median = rolling_quantile(maxima_fluxes, 10, 0.5, center=True)
        IQ = rolling_quantile(maxima_fluxes, 10, 0.75, center=True) - median
        #median = np.ravel(pd.DataFrame(maxima_fluxes).rolling(10, center=True).quantile(0.50))
        #IQ = np.ravel(pd.DataFrame(maxima_fluxes).rolling(10, center=True).quantile(0.75)) - median
        IQ[np.isnan(IQ)] = flux_norm.max()
        median[np.isnan(median)] = flux_norm.max()
        mask = flux > median + 20 * IQ
        if np.sum(mask) > 0:
            print(f" Cosmic peaks removed : {np.sum(mask):.0f}")
        
        maxima_wvs = maxima_wvs[~mask]
        maxima_fluxes = maxima_fluxes[~mask]
        maxima_idxs = maxima_idxs[~mask]

    flux_norm = flux_norm / stretching
    maxima_fluxes = maxima_fluxes / stretching
    norm = norm * stretching

    # save these for later - these are all the anchor points
    _maxima_fluxes = maxima_fluxes.copy()
    _maxima_wvs = maxima_wvs.copy()
    _maxima_idxs = maxima_idxs.copy()

    #### calculate penalty ####

    min_radius = min(5., 10 * fwhm) if min_radius is None else min_radius
    
    if out_of_calibration:
        narrow_win = 2.0  # 2 typical line width scale (small window for the first continuum)
        broad_win = 20.0  # 20 typical line width scale (large window for the second continuum)
    else:
        narrow_win = 10.0  # 10 typical line width scale (small window for the first continuum)
        broad_win = 100.0  # 100 typical line width scale (large window for the second continuum)
        
    iterations = 5

    chromatic_law = maxima_wvs / wave_min
    radius = min_radius * np.ones_like(maxima_wvs) * chromatic_law

    # removed this from if, as the conditions should always be True unless max_radius == min_radius
    # note - changed broad_win to 10 * narrow_win in q1/q3 calcs - unsure if correct

    # start the huge penalty map computation
    # basically taking several IQR ranges in a broad and narrow window and comparing the two
    # this gives a statistic that increases in large absorption features
    # which is used to scale the radius of the rolling pin
    dx = fwhm / np.median(np.diff(grid))

    # ensure dx*window is always more than one
    dx = 1 / narrow_win if dx * narrow_win < 1 else dx

    #print(f"converted fwhm: {fwhm:.4f} A. dx = {dx:.4f}, windows = {narrow_win}, {broad_win}")
    
    if max_radius is None:
        penalty, max_radius = compute_penalty(grid, 
                                              flux_norm, 
                                              dx, 
                                              narrow_win=narrow_win, 
                                              broad_win=broad_win, 
                                              get_max_radius=True
                                             )
        
        # failed to calculate max_radius
        # this logic seems a bit strange
        if max_radius is None:
            #print('Failed to calculate max_radius')
            #max_radius = 5 * min_radius if out_of_calibration else min_radius
            #max_radius = 5 * min_radius
            max_radius = max(150, 5 * min_radius)
    else:
        penalty = compute_penalty(grid, flux_norm, dx, narrow_win=narrow_win, broad_win=broad_win)
        
    penalty_adj = penalty.copy()

    #print(f"Minimum radius: {min_radius:.3f} A. Maximum radius: {max_radius:.3f} A.")
    
    # this broadens the size of each penalty peak slightly
    for i in range(iterations):
        continuum_right = rolling_quantile(penalty_adj, int(narrow_win * dx), 1)
        continuum_left = rolling_quantile(penalty_adj[::-1], int(narrow_win * dx), 1)[::-1]

        # define for the left border all nan value to the first non nan value
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]

        # define for the right border all nan value to the first non nan value
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
        
        #both = np.array([continuum_right, continuum_left])
        #penalite = np.max(both, axis=0)
        penalty_adj = np.c_[continuum_right, continuum_left].max(axis=1)

    # normalise penalty
    penalty_norm = (penalty_adj - penalty_adj.min()) / (penalty_adj.max() - penalty_adj.min())

    # get the value of the penalty at the local maxima
    maxima_penalty = penalty_norm[maxima_idxs]
    
    # apply penalty-radius law
    dr = max_radius - min_radius
    if pr_func == 'power':
        exponent = pr_func_params
        maxima_radii = min_radius + dr * maxima_penalty ** exponent
        maxima_radii = chromatic_law * maxima_radii
    elif pr_func == 'sigmoid':
        centre, steep = pr_func_params
        maxima_radii = min_radius + dr * (1 + np.exp(-steep * (maxima_penalty - centre) )) ** -1
        maxima_radii = chromatic_law * maxima_radii
    else:
        raise ValueError(f"Invalid pr_func: {pr_func}.")

    #### rolling pin selection ####
    
    # apply one threshold scaling before
    maxima_radii[0] = maxima_radii[0] / 1.5
    keep = [0]
    j = 0
    prev_radius = min_radius

    # removed the distance array that is useless
    # instead it is computed at each j
    numero = np.arange(maxima_wvs.size, dtype='int')

    while maxima_wvs.size - j > 3:
        # take the radius from the penalty law
        # par_R -> radius
        # R_old -> prev_radius
        radius = maxima_radii[j]
        
        # recompute the points closer than the diameter if Radius changed with the penality
        distances = np.sign(maxima_wvs - maxima_wvs[j]) * np.sqrt((maxima_wvs - maxima_wvs[j])**2 + (maxima_fluxes - maxima_fluxes[j])**2)
        mask = (distances > 0) & (distances < 2.0 * radius)
        while np.sum(mask) == 0:
            radius = 1.5 * radius
            # recompute the points closer than the diameter if Radius changed with the penality
            mask = (distances > 0) & (distances < 2.0 * radius)
        
        # local maximum
        p1 = np.array((maxima_wvs[j], maxima_fluxes[j]))
        
        # vector of all the maxima in the diameter zone
        p2 = np.c_[maxima_wvs[mask], maxima_fluxes[mask]]
        
        delta = p2 - p1  # delta x delta y
        
        c = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)  # euclidian distance
        h = np.sqrt(radius**2 - 0.25 * c**2)
        
        cx = p1[0] + 0.5 * delta[:, 0] - h / c * delta[:, 1]  # x coordinate of the circles center
        cy = p1[1] + 0.5 * delta[:, 1] + h / c * delta[:, 0]  # y coordinates of the circles center

        cond1 = (cy - p1[1]) >= 0
        thetas = cond1 * (-1 * np.arccos((cx - p1[0]) / radius) + np.pi) + (1 - 1 * cond1) * (
            -1 * np.arcsin((cy - p1[1]) / radius) + np.pi
        )
        j2 = thetas.argmin()
        j = numero[mask][j2]  # take the numero of the local maxima falling in the diameter zone
        keep.append(j)
    
    # local maxima selected by rolling pin
    #maxima_fluxes_sel = maxima_fluxes[keep]
    #maxima_wvs_sel = maxima_wvs[keep]
    #maxima_idxs_sel = maxima_idxs[keep]

    if edge_fix:
        keep = [0] + keep
        keep = keep + [maxima_idxs.size - 1]
    
    # # CAII MASKING
    # mask_caii = ((wave > 3929) & (wave < 3937)) | ((wave > 3964) & (wave < 3972))
    # wave = wave[~mask_caii]
    # flux = flux[~mask_caii]
    # index = index[~mask_caii]
    
    # assemble and return a structured index array
    selected = np.isin(np.arange(maxima_idxs.size), keep)
    index_array = assemble_index_array(wvs, flux, maxima_idxs, selected)

    if kwargs.get('return_penalty'):
        return index_array, penalty_norm, maxima_radii
    else:
        return index_array
