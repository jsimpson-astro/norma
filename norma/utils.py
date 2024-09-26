import logging

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, correlate
from scipy.ndimage import gaussian_filter1d
from scipy.stats.distributions import norm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


__all__ = [
    'rolling_quantile',
    'apply_smooth',
    'calculate_fwhm',
    'compute_penalty',
    'local_max',
    'grouping',
    'produce_line',
    'clip'
    ]

def normalise(
    wvs: np.ndarray,
    flux: np.ndarray,
    index_array: np.ndarray
    ) -> np.ndarray:
    """
    Normalise a spectrum using an index structured array.
    See also norma.io.assemble_index_array to convert norma.find_max outputs.
    Returns the flux divided by a continuum fit through the selected points
    """
    sel = index_array['sel']
    # normalise with cubic interpolation
    interp = interp1d(index_array['wvs'][sel], index_array['flux'][sel], 
                      kind='cubic', bounds_error=False, fill_value='extrapolate')
    cont = interp(wvs)
    flux_norm = flux / cont

    return flux_norm

def rolling_quantile(
    x: np.ndarray, 
    window: int, 
    q: float, 
    min_periods: int | None = None,
    center: bool = False
    ) -> np.ndarray:
    
    df = pd.DataFrame(x)
    out = df.rolling(window, min_periods=min_periods, center=center).quantile(q)
    return np.ravel(out)

def rectangular_smooth(
    flux: np.ndarray, 
    width: int
    ) -> np.ndarray:
    """
    Apply rectangular smoothing
    """
    if width <= 0:
        raise ValueError("width must be greater than 0")
    
    half_width = width // 2 + 1
    window = np.ones(width) / width
    flux_smooth = np.convolve(flux, window, mode='same')

    # remove end effects
    flux_smooth[:half_width] = flux[:half_width]
    flux_smooth[-half_width:] = flux[-half_width:]
    
    return flux_smooth

def gaussian_smooth(
    flux: np.ndarray, 
    width: int
    ) -> np.ndarray:
    """
    Apply gaussian smoothing
    """
    if width <= 0:
        raise ValueError("width must be greater than 0")
    
    # 6 sigma contained in the width (3-sigma window)
    sigma = width / 6
    
    flux_smooth = gaussian_filter1d(flux, sigma, radius=width)

    # remove end effects
    flux_smooth[:width] = flux[:width]
    flux_smooth[-width:] = flux[-width:]
    
    return flux_smooth

def savgol_smooth(
    flux: np.ndarray, 
    width: int,
    polyorder: int = 3,
    ) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing
    """
    if width <= 0:
        raise ValueError("width must be greater than 0")
    
    flux_smooth = savgol_filter(flux, width, polyorder)
    
    return flux_smooth

_smooth_funcs = {
    'rectangular': rectangular_smooth, 
    'gaussian': gaussian_smooth, 
    'savgol': savgol_smooth
    }

def apply_smooth(
    flux: np.ndarray, 
    width: int, 
    kernel: str = 'rectangular'
    ) -> np.ndarray:
    """
    Apply smoothing kernel `kernel` to `flux` in window `width`.
    Available kernels are `rectangular`, `gaussian`, and `savgol`.
    
    """
    if kernel not in _smooth_funcs:
        raise ValueError(f"Invalid kernel provided: {kernel}.")
        
    func = _smooth_funcs[kernel]
    # convert width to fwhm, taking width as a 3-sigma window
    #width_ = (width / 6) * (2**1.5 * np.log(2) ** 0.5) if kernel == 'gaussian' else width
    # double width for gaussian for a similar scale
    width_ = 2 * width if kernel == 'gaussian' else width
    flux_smooth = func(flux, width_)
    
    return flux_smooth

def calculate_fwhm(
    wvs: np.ndarray, 
    flux: np.ndarray,
    telluric_mask: list[tuple[float, float]] | None = None,
    **kwargs
    ) -> float:
    """
    Calculate the velocity resolution of a spectrum using scipy.signal.correlate.
    Returns fwhm in velocity units (km/s).
    
    """
    
    dwv = (wvs.max() - wvs.min()) / wvs.size
    mask = np.zeros_like(flux)

    # by default rolling maxima in a 30 angstrom window
    continuum_right = rolling_quantile(flux, int(30/dwv), 1)
    continuum_left = rolling_quantile(flux[::-1], int(30/dwv), 1)[::-1]
    
    continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
    continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
    continuum = np.c_[continuum_right, continuum_left].min(axis=1)
    
    # rectangular smoothing of 15 AA provides more accurate weighting
    continuum = apply_smooth(continuum, int(15 / dwv), 'rectangular')

    flux_norm = flux / continuum

    if telluric_mask is not None:
        for lb, ub in telluric_mask:
            tmask = (wvs > lb) & (wvs < ub)
            flux_norm[tmask] = 1
    
    # new ccf
    extend = 30
    
    # perform correlation on mask, padded with zeros
    ccf = correlate(flux_norm-1, np.r_[np.zeros(extend), flux_norm, np.zeros(extend)]-1, mode='valid')

    max_idx = ccf.argmax()
    ccf_max = ccf[max_idx]

    # clip edges
    max_idx = 1 if max_idx == 0 else max_idx
    max_idx = max_idx - 1 if max_idx == ccf.size - 1 else max_idx
    
    # quadratic approx to peak
    ccf_max_a, ccf_max_b = ccf[max_idx-1], ccf[max_idx+1]
    z1, z2 = ccf_max_b - ccf_max_a, ccf_max_a + ccf_max_b - 2 * ccf_max
    max_loc = int(max_idx  - 0.5 * (z1 / z2))
    error = 1 / (-0.5 * z2)**0.5

    # convert error to fwhm, convert fwhm to velocity units
    fwhm = ((np.diff(wvs).mean() * error * 2**1.5 * np.log(2)**0.5) / wvs[wvs.size // 2 - extend + max_loc]) * 2.998e5

    if kwargs.get('plot_ccf'):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xcor_wvs, ccf, color='r', lw=1)
        ax.plot(xcor_wvs, offset_gaus(xcor_wvs, *popt), color='r', lw=1, ls='--')
        ax.set_title(f"vfwhm = {fwhm:.3f} km/s, offset = {popt[0]:.3f} km/s")

        plt.show(block=False)
    
    return fwhm

def compute_penalty(
    wvs: np.ndarray, 
    flux: np.ndarray, 
    dx: float,
    narrow_win: float = 10., 
    broad_win: float = 100.
    ) -> np.ndarray:
    """
    Compute a penalty map of a spectrum, which indicates large absorption features.
    """
    
    wv_min, wv_max = wvs.min(), wvs.max()

    # rolling maximum small, large from right, large from left
    continuum_small_win = rolling_quantile(flux, int(narrow_win * dx), 1, center=True)
    continuum_right = rolling_quantile(flux, int(broad_win * dx), 1, center=True)
    continuum_left = rolling_quantile(flux[::-1], int(broad_win * dx), 1, center=True)[::-1]
    
    continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
    continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
    #both = np.array([continuum_right, continuum_left])
    both = np.c_[continuum_right, continuum_left].min(axis=1)
    
    continuum_small_win[np.isnan(continuum_small_win) & (2 * wvs < (wv_min + wv_max))] = continuum_small_win[~np.isnan(continuum_small_win)][0]
    continuum_small_win[np.isnan(continuum_small_win) & (2 * wvs > (wv_min + wv_max))] = continuum_small_win[~np.isnan(continuum_small_win)][-1]
    #continuum_large_win = np.min(both, axis=0)  
    continuum_large_win = both
    
    # when taking a large window, the rolling maximum depends on the direction - make both directions and take the minimum
    # take median and q1, q3 in large and small windows, on continuum_large_win
    median_large = rolling_quantile(continuum_large_win, int(10 * broad_win * dx), 0.5, min_periods=1, center=True)
    Q1_large = rolling_quantile(continuum_large_win, int(10 * broad_win * dx), 0.25, min_periods=1, center=True)
    Q3_large = rolling_quantile(continuum_large_win, int(10 * broad_win * dx), 0.75, min_periods=1, center=True)
    q1_large = rolling_quantile(continuum_large_win, int(10 * narrow_win * dx), 0.25, min_periods=1, center=True)
    q3_large = rolling_quantile(continuum_large_win, int(10 * narrow_win * dx), 0.75, min_periods=1, center=True)
    
    IQ1_large = Q3_large - Q1_large
    IQ2_large = q3_large - q1_large
    sup_large = np.min([Q3_large + 1.5 * IQ1_large, q3_large + 1.5 * IQ2_large], axis=0)
    
    mask = continuum_large_win > sup_large
    continuum_large_win[mask] = median_large[mask]

    # take median and q1, q3 in large and small windows, on continuum_small_win
    median_small = rolling_quantile(continuum_small_win, int(10 * broad_win * dx), 0.5, min_periods=1, center=True)
    Q1_small = rolling_quantile(continuum_small_win, int(10 * broad_win * dx), 0.25, min_periods=1, center=True)
    Q3_small = rolling_quantile(continuum_small_win, int(10 * broad_win * dx), 0.75, min_periods=1, center=True)
    q1_small = rolling_quantile(continuum_small_win, int(10 * narrow_win * dx), 0.25, min_periods=1, center=True)
    q3_small = rolling_quantile(continuum_small_win, int(10 * narrow_win * dx), 0.75, min_periods=1, center=True)
    
    IQ1_small = Q3_small - Q1_small
    IQ2_small = q3_small - q1_small
    sup_small = np.min([Q3_small + 1.5 * IQ1_small, q3_small + 1.5 * IQ2_small], axis=0)
    
    mask = continuum_small_win > sup_small
    continuum_small_win[mask] = median_small[mask]
    
    loc_out = local_max(continuum_large_win, 2)[0]
    for k in loc_out.astype("int"):
        continuum_large_win[k] = np.min([continuum_large_win[k - 1], continuum_large_win[k + 1]])
        
    loc_out = local_max(continuum_small_win, 2)[0]
    for k in loc_out.astype("int"):
        continuum_small_win[k] = np.min([continuum_small_win[k - 1], continuum_small_win[k + 1]])
    
    # replace null values
    #continuum_large_win = np.where(continuum_large_win == 0, 1.0, continuum_large_win)
    continuum_large_win[continuum_large_win == 0] = 1
    
    penalty = (continuum_large_win - continuum_small_win) / continuum_large_win
    
    penalty[penalty < 0] = 0

    return penalty

###############################################################################

# Here be dragons

def produce_line(
    grid: np.ndarray,
    spectre: np.ndarray,
    box: int = 5,
    shape: str = "savgol",
    vic: int = 7,
    ):
    """
    No idea what this is cause they didn't comment it
    """
    index, line_flux = local_max(-apply_smooth(spectre, box, shape), vic)
    line_flux = -line_flux
    line_index = index.astype("int")
    line_wave = grid[line_index]
    
    index2, line_flux2 = local_max(apply_smooth(spectre, box, shape), vic)
    line_index2 = index2.astype("int")
    line_wave2 = grid[line_index2]

    if line_wave[0] < line_wave2[0]:
        line_wave2 = np.insert(line_wave2, 0, grid[0])
        line_flux2 = np.insert(line_flux2, 0, spectre[0])
        line_index2 = np.insert(line_index2, 0, 0)

    if line_wave[-1] > line_wave2[-1]:
        line_wave2 = np.insert(line_wave2, -1, grid[-1])
        line_flux2 = np.insert(line_flux2, -1, spectre[-1])
        line_index2 = np.insert(line_index2, -1, len(grid) - 1)

    memory = np.hstack([-1 * np.ones(len(line_wave)), np.ones(len(line_wave2))])
    stack_wave = np.hstack([line_wave, line_wave2])
    stack_flux = np.hstack([line_flux, line_flux2])
    stack_index = np.hstack([line_index, line_index2])

    memory = memory[stack_wave.argsort()]
    stack_flux = stack_flux[stack_wave.argsort()]
    stack_wave = stack_wave[stack_wave.argsort()]
    stack_index = stack_index[stack_index.argsort()]

    trash, matrix = grouping(memory, 0.01, 0)

    delete_liste = []
    for j in range(len(matrix)):
        number = np.arange(matrix[j, 0], matrix[j, 1] + 2)
        fluxes = stack_flux[number].argsort()
        if trash[j][0] == 1:
            delete_liste.append(number[fluxes[0:-1]])
        else:
            delete_liste.append(number[fluxes[1:]])
    delete_liste = np.hstack(delete_liste)

    memory = np.delete(memory, delete_liste)
    stack_flux = np.delete(stack_flux, delete_liste)
    stack_wave = np.delete(stack_wave, delete_liste)
    stack_index = np.delete(stack_index, delete_liste)

    minima = np.where(memory == -1)[0]
    maxima = np.where(memory == 1)[0]

    index = stack_index[minima]
    index2 = stack_index[maxima]
    flux = stack_flux[minima]
    flux2 = stack_flux[maxima]
    wave = stack_wave[minima]
    wave2 = stack_wave[maxima]

    index = np.hstack([index[:, np.newaxis], index2[0:-1, np.newaxis], index2[1:, np.newaxis]])
    flux = np.hstack([flux[:, np.newaxis], flux2[0:-1, np.newaxis], flux2[1:, np.newaxis]])
    wave = np.hstack([wave[:, np.newaxis], wave2[0:-1, np.newaxis], flux2[1:, np.newaxis]])

    return index, wave, flux

def local_max(
    spectre: np.ndarray, vicinity: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a local maxima algorithm of a vector.

    Args:
        spectre: The vector to investigate.
        vicinity: The half window in which a local maxima is searched.

    Returns:
        A tuple containing the index and vector values of the detected local maxima.
    """

    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1, vicinity):
        maxima *= (
            0.5
            * (1 + np.sign(vec_base - spectre[vicinity - k : -vicinity - k]))
            * 0.5
            * (1 + np.sign(vec_base - spectre[vicinity + k : -vicinity + k]))
        )

    index = np.where(maxima == 1)[0] + vicinity
    flux = spectre[index]
    return (index, flux)

def grouping(
    array: np.ndarray, 
    threshold: float, 
    num: int
    ) -> tuple[list[np.ndarray], np.ndarray[int]]:
    """
    No idea what this is
    """
    
    difference = abs(np.diff(array))
    cluster = difference < threshold
    indices = np.arange(len(cluster))[cluster]

    j = 0
    border_left = [indices[0]]
    border_right = []
    while j < len(indices) - 1:
        if indices[j] == indices[j + 1] - 1:
            j += 1
        else:
            border_right.append(indices[j])
            border_left.append(indices[j + 1])
            j += 1
    border_right.append(indices[-1])
    border = np.array([border_left, border_right]).T
    border = np.hstack([border, (1 + border[:, 1] - border[:, 0])[:, np.newaxis]])

    kept: List[NDArray[np.float64]] = []
    for j in range(len(border)):
        if border[j, -1] >= num:
            kept.append(array[border[j, 0] : border[j, 1] + 2])
    return kept, border

def clip(interpolated_flux, original_flux):
    # version of 'ras_troncated'
    maxi, mini = np.percentile(original_flux, 99.9), np.percentile(original_flux, 0.1)
    t = (maxi-mini) / 5
    interpolated_flux[interpolated_flux < mini - t] = mini
    interpolated_flux[interpolated_flux > maxi + t] = maxi
    return interpolated_flux