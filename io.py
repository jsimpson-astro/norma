import numpy as np

# numpy setup for read/write of index files
_index_dtypes = [('wvs', 'float'), ('flux', 'float'), ('index', 'int'), ('sel', 'bool')]
_index_fmts = ['%3.9f', '%.18e', '%i', '%i']

def write_index_file(file: str, arr: np.ndarray):
	"""
	Write an index structured array `arr` to file `file`.
	"""
	sorted_arr = arr[arr['wvs'].argsort()]    
	np.savetxt(file, sorted_arr, fmt=_index_fmts)

def read_index_file(file: str) -> np.ndarray:
	"""
	Read an index structured array from file `file`.
	"""
	return np.loadtxt(file, dtype=_index_dtypes)

def assemble_index_array(
	wvs: np.ndarray[float], 
	flux: np.ndarray[float], 
	index: np.ndarray[int], 
	sel: np.ndarray[bool]
	) -> np.ndarray:
	"""
	Assemble an index structured array from a spectrum and the output of find_max.
	`wvs` and `flux` are the wavelengths and flux of the spectrum.
	`index` and `sel` are the first and second outputs of find_max,
	i.e. the local maxima and the selection mask.
	"""
	if index.size != sel.size:
		raise IndexError("index and sel must have equal sizes.")
	index_data = np.zeros(index.size, dtype=_index_dtypes)
	index_data['wvs'] = wvs[index]
	index_data['flux'] = flux[index]
	index_data['index'] = index
	index_data['sel'] = sel
	return index_data