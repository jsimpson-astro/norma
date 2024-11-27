# norma

**norma** is a normalisation tool for astronomical spectra primarily based on [RASSINE](https://github.com/MichaelCretignier/Rassine_public).

norma contains two important command-line tools: `norma-identify` and `norma-select`.

`norma-identify` identifies and selects local maxima using the same 'rolling pin' approach as RASSINE. 
`norma-identify` greatly simplifies this process, taking only a wavelength and flux array as required inputs (rather than an arbitrary pickle with several mandatory keys) and outputting .index files containing only the identified local maxima, and those which the algorithm has selected for the continuum.
It offers several parameters for tweaking the algorithm, which can help in achieving a good fit to the continuum.

`norma-select` takes the .index files output by `norma-select` and allows the selection to be manually tweaked, including by adding manual points if desired to better pin down the interpolated continuum. This works over a list of spectra and index files at once, allowing easy normalisation of many spectra simultaneously.

Both commands are well-documented and make good use of `argparse`, so help can be found for both commands by running `norma-identify -h` and `norma-select -h`.

## Installation

To install norma, you can simply download and install in one line with `pip`:

```
pip install https://github.com/jsimpson-astro/norma/archive/main.zip
```

## Acknowledgement

If you find norma useful in your work, please reference this repository (https://github.com/jsimpson-astro/norma) in a footnote, and be sure to cite the original authors of [RASSINE](https://github.com/MichaelCretignier/Rassine_public) too.

## Advanced usage

While norma is intended to be used as a command-line tool, norma also provides easy access to its methods natively, within Python.

The main Python methods of norma are:

- The `norma.find_max` function, which is the underlying function wrapped by `norma-identify`
- The `norma.InteractiveNorma` class, which handles the interactive elements used by `norma-select`
- The `plot_output` function from `norma.plotting`, which offers convenient plotting of results from `find_max`
- The `norma.normalise` function, which simply normalises the flux of a spectrum using an index array
- The `read_index_file` and `write_index_file` functions from `norma.io`, which handle reading and writing of the .index files used by norma

An example usage to normalise a spectrum using norma is as follows:

```py
import numpy as np
import norma

# load a spectrum, with column 1 being wavelengths (in angstroms) and column 2 being fluxes
spec = np.loadtxt('spectrum.dat')
wvs, flux = spec[:, 0], spec[:, 1]

index_array = norma.find_max(wvs, flux)
norm_flux = norma.normalise(wvs, flux, index_array)

# save normalised spectrum
norm_spec = spec.copy()
norm_spec[:, 1] = norm_flux
np.savetxt('norm_spectrum.dat', norm_spec)

```

It is important to note that while `find_max` (and by extent, `norma-identify`) can produce good results without adjustment, 
it is strongly recommended to tweak the input parameters to achieve a good fit. 
The min/max radii, stretching, and pr_func parameters in particular can have a large effect on the results.

Continuing from the above example, we can plot and inspect the fit like so:

```py
import matplotlib.pyplot as plt
from norma.plotting import plot_output

fig, ax = plot_output(wvs, flux, index_array)

plt.show()

```

This will display a figure with your fit spectrum in the top panel, and the normalised spectrum in the lower panel.

If you decide your continuum fit with these parameters is almost satisfactory, but requires a little tweaking, you could then use the `InteractiveNorma` class to make some adjustments:

```py
from norma.io import write_index_file

# the name does not have to match the spectrum as index files are specified explicitly to InteractiveNorma
# however, it does when using norma-select, so it is useful to keep matching names
write_index_file('spectrum.index', index_array)

# note InteractiveNorma takes lists of files
inorma = norma.InteractiveNorma(['spectrum.dat'], ['spectrum.index'])

# begin the interactive programme
inorma.start()

```
`InteractiveNorma` will print out a short help string explaining its usage, and an interactive figure will appear.

Note that `InteractiveNorma` works on saved files. This keeps it from hogging lots of memory when working on many large spectra at once.
The .index files are updated in place, either on exiting `InteractiveNorma` or while scrolling through multiple spectra.

If we want to see our normalised result after our tweaking, we can simply do the following:

```py
from norma.io import read_index_file

# load the new, edited index file
index_array = read_index_file('spectrum.index')

fig, ax = plot_output(wvs, flux, index_array)

plt.show()

```
Our new, tweaked spectrum should appear now, in the same format as before but with the changes made in `InteractiveNorma` applied.

This is the extent of the main usage of norma as of the 0.1.0 version.
Should you encounter any issues, or need any help with using norma, please feel free to contact me or raise an issue here.






