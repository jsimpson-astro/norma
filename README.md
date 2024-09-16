## norma

**norma** is a normalisation tool for astronomical spectra primarily based on [RASSINE](https://github.com/MichaelCretignier/Rassine_public).

norma contain two important tools: the `find_max` function, and the `InteractiveNorma` class.

`norma.find_max` identifies and selects local maxima using the same 'rolling pin' approach as RASSINE. 
`find_max` greatly simplifies this process, taking only a wavelength and flux array as required inputs (rather than an arbitrary pickle with several mandatory keys) and outputting only the identified local maxima, and those which the algorithm has selected for the continuum.

`InteractiveNorma` takes files containing these local maxima and allows the selection to be manually tweaked, including by adding manual points if desired to better pin down the interpolated continuum. This works over a list of spectra and index files at once, allowing easy normalisation of many spectra simultaneously.

