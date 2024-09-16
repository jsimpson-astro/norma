import os
import numpy as np
import matplotlib.pyplot as plt

class InteractiveNormalisation:
    """
    """
    _help_str = (
        "Maxima modification:\n"
        "Current spectrum shown in green\n"
        "Double click to enable/disable points\n"
        "up/down - scroll through spectra\n"
        "a - add manual points\td - remove points\n"
        "x - spectra ghosts\tt - thinning\n"
        "z - ghost points\tw - current points\n"
        )
    
    # numpy setup for read/write of index files
    _index_dtypes = [('wvs', 'float'), ('flux', 'float'), ('index', 'int'), ('sel', 'bool')]
    _index_fmts = ['%3.9f', '%.18e', '%i', '%i']
    
    # matplotlib axes.plot arguments for current spectrum
    current_plot_params = dict(spec_kws = {'color': '#074517', 'alpha': 0.3},
                               cont_kws = {'color': '#0d7a2a'},
                               index_kws = {'color': 'k', 'marker': 'o', 'linestyle': 'None', 'markersize': 5},
                               sel_kws = {'color': '#09521c', 'marker': 'o', 'linestyle': 'None', 'markersize': 6},
                               man_kws = {'color': '#12cc43', 'marker': 'D', 'linestyle': 'None', 'markersize': 7},
                              )
    # matplotlib axes.plot arguments for spectra show above the current spectrum
    # note that alpha is scaled to decrease the further away from `current` the spectrum is, starting at this value
    above_plot_params = dict(spec_kws={'color': '#726bd6', 'alpha': 0.2},
                             cont_kws={'color': '#14a3fc', 'alpha': 0.3}, 
                             index_kws={'color': '#14257a', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3},
                             sel_kws={'color': '#3054e3', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3}, 
                             man_kws={'color': '#38b8eb', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'None', 'markersize': 4}
                            )
    # matplotlib axes.plot arguments for spectra show below the current spectrum
    # note that alpha is scaled to decrease the further away from `current` the spectrum is, starting at this value
    below_plot_params = dict(spec_kws={'color': '#f55668', 'alpha': 0.2},
                             cont_kws={'color': '#ff6176', 'alpha': 0.3}, 
                             index_kws={'color': '#66090e', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3},
                             sel_kws={'color': '#e32228', 'alpha': 0.3, 'marker': 'o', 'linestyle': 'None', 'markersize': 3}, 
                             man_kws={'color': '#ff1925', 'alpha': 0.3, 'marker': 'D', 'linestyle': 'None', 'markersize': 4}
                            )

    def __init__(
        self, 
        spec_files: list[str], 
        index_files: list[str], 
        start_index: int = 0,
        n_plot: int = 3
        ):
        
        self._on_press_dict = {
            'up': self.move_up_event,
            'down': self.move_down_event,
            'a': self.add_manual_point_event,
            'v': self.toggle_point_event,
            'd': self.remove_point_event,
            'x': self.hide_event,
            't': self.thin_event,
            'z': self.hide_points_event,
            'w': self.hide_current_points_event
            }

        self._started = False
        
        # check all files
        if not len(spec_files) == len(index_files):
            raise IndexError("Lengths of spec_files and index_files do not match.")
        
        missing_files = [f for f in spec_files + index_files if not os.path.isfile(f)]
        if len(missing_files) != 0:
            raise FileNotFoundError(f"Missing files: {missing_files}")

        self.n_plot = n_plot
        
        # set attributes
        self._spec_files = spec_files
        self._index_files = index_files
        self._max_index = len(spec_files) - 1

        spec_file, index_file = self._spec_files[start_index], self._index_files[start_index]
        self.current_index = start_index
        
        self._enable_wrapping = False

    
    def start(self, fig=None, ax=None):
        """
        Start the interactive plot
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(16, 6))
            fig.subplots_adjust(left=0.07,right=0.96,hspace=0,top=0.95)
        else:
            ax.clear()
        
        fig.canvas.mpl_connect('button_press_event', self._on_click_event)
        fig.canvas.mpl_connect('key_press_event', self._on_press_event)

        self._fig, self._axes = fig, ax
        
        self._initialise_plots()
        self._started = True
        # wait in loop
        plt.show(block=False)
        print(self._help_str)
        prompt = 'none'
        while prompt.strip() != '':
        	prompt = input("Press enter to quit: ")
        
        plt.close()

    
    def _initialise_plots(self):
        """
        Populate the axis with the desired range of spectra.
        """
        fig, ax = self._fig, self._axes
        
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
        ax.set_ylabel('Flux (arb. unit)', fontsize=14)
        ax.set_title(f"Spectrum {self.current_index+1}")

        # performance improvements?
        plt.style.use('fast')
        plt.rcParams['agg.path.chunksize'] = 1000
        #plt.rcParams['path.simplify'] = True
        #plt.rcParams['path.simplify_threshold'] = 0.1
        
        current_spec_data, current_index_data = self._read_single(self.current_index)
        current_artist_dict = self._plot_single(current_spec_data, current_index_data, **self.current_plot_params)

        spec_data_list = [current_spec_data]
        index_data_list = [current_index_data]
        artist_list = [current_artist_dict]
        
        # get indices for below and above spectra
        # clip at 0 and max_index, and note the ones we have clipped for hiding
        above_idxs = range(self.current_index + 1, self.current_index + self.n_ghosts + 1)
        above_to_hide = [i+1 for i, idx in enumerate(above_idxs) if idx > self.max_index]
        above_idxs = [min(idx, self.max_index) for idx in above_idxs]

        above_plot_params = self.above_plot_params.copy()
        for i, idx in enumerate(above_idxs):
            spec_data, index_data = self._read_single(idx)
            
            above_plot_params = {kw: {k: v/(i+1) if k == 'alpha' else v for k, v in pars.items()} 
                                 for kw, pars in above_plot_params.items()}
            
            artist_dict = self._plot_single(spec_data, index_data, **above_plot_params)
            
            spec_data_list.append(spec_data)
            index_data_list.append(index_data)
            artist_list.append(artist_dict)

        # use negative indices for below, for relative indexing
        below_idxs = range(self.current_index - 1, self.current_index - self.n_ghosts - 1, -1)
        below_to_hide = [-(i+1) for i, idx in enumerate(below_idxs) if idx < 0]
        below_idxs = [max(idx, 0) for idx in below_idxs]

        # hold these temporarily, we want to put them in in reverse
        below_spec_data_list = []
        below_index_data_list = []
        below_artist_list = []
        
        below_plot_params = self.below_plot_params.copy()
        for i, idx in enumerate(below_idxs):
            spec_data, index_data = self._read_single(idx)
            
            below_plot_params = {kw: {k: v/(i+1) if k == 'alpha' else v for k, v in pars.items()} 
                                 for kw, pars in below_plot_params.items()}
            
            artist_dict = self._plot_single(spec_data, index_data, **below_plot_params)

            below_spec_data_list.append(spec_data)
            below_index_data_list.append(index_data)
            below_artist_list.append(artist_dict)

        # setup data lists (use get_rel to pull from them)
        self._spec_data_list = spec_data_list + below_spec_data_list[::-1]
        self._index_data_list = index_data_list + below_index_data_list[::-1]
        self._artist_list = artist_list + below_artist_list[::-1]

        # setup hidden
        self._hidden = [False] * (2 * self.n_ghosts + 1)
        for rel_idx in above_to_hide + below_to_hide:
            self._hide_single(rel_idx)

        self._reset_home()

    def _plot_single(
        self, 
        spec_data: np.ndarray, 
        index_data: np.ndarray, 
        spec_kws: dict = {}, 
        cont_kws: dict = {}, 
        index_kws: dict = {}, 
        sel_kws: dict = {}, 
        man_kws: dict = {}
        ) -> dict:
        """
        Plot a single set of data and return a dict of artists
        """
        ax = self._axes
        artist_dict = {}
        artist_dict['spec'], = ax.plot(spec_data[:, 0], spec_data[:, 1], **spec_kws)

        # index data should be wvs | flux | index | sel
        # index = -1 indicates manual
        # sel indicates selected (including manual)
        idxs = index_data['index']
        sel = index_data['sel']
        man_mask = ~(idxs != -1)
        man = index_data[man_mask]

        # may need to ensure this is sorted somewhere
        locmax = idxs[~man_mask]
        sel_filt = sel[~man_mask]
        locmax_sel = locmax[sel_filt]
        
        artist_dict['index'], = ax.plot(spec_data[:, 0][idxs], spec_data[:, 1][idxs], zorder=3, **index_kws)
        artist_dict['sel'], = ax.plot(spec_data[:, 0][locmax_sel], spec_data[:, 1][locmax_sel], zorder=5,  **sel_kws)
        artist_dict['man'], = ax.plot(man['wvs'], man['flux'], zorder=6, **man_kws)

        interp = interp1d(index_data['wvs'][sel], index_data['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        artist_dict['cont'], = ax.plot(spec_data[:, 0], interp(spec_data[:, 0]), zorder=4, **cont_kws)

        return artist_dict

    def _update_single(
        self, 
        spec_data: np.ndarray, 
        index_data: np.ndarray, 
        artist_dict: dict,
        adjust: bool = False
        ) -> None:
        """
        Updates data in artist_dict with given spec data and index data.
        Set `adjust` to True to only alter continuum and selected points.
        
        """
        # index data should be wvs | flux | index | sel
        # index = -1 indicates manual
        # sel indicates selected (including manual)
        idxs = index_data['index']
        sel = index_data['sel']
        man_mask = ~(idxs != -1)
        man = index_data[man_mask]

        # may need to ensure this is sorted somewhere
        locmax = idxs[~man_mask]
        sel_filt = sel[~man_mask]
        locmax_sel = locmax[sel_filt]

        if not adjust:
            artist_dict['spec'].set_data(spec_data[:, 0], spec_data[:, 1])
            artist_dict['index'].set_data(spec_data[:, 0][idxs], spec_data[:, 1][idxs])
        artist_dict['sel'].set_data(spec_data[:, 0][locmax_sel], spec_data[:, 1][locmax_sel])
        artist_dict['man'].set_data(man['wvs'], man['flux'])

        interp = interp1d(index_data['wvs'][sel], index_data['flux'][sel], kind='cubic', bounds_error=False, fill_value='extrapolate')

        artist_dict['cont'].set_data(spec_data[:, 0], interp(spec_data[:, 0]))

    def _update_all(
        self,
        spec_data_list: list[np.ndarray], 
        index_data_list: list[np.ndarray]
        ) -> None:
        """
        Updates all artists in self.artist_list with the provided data
        
        """
        rel_idxs = range(-self.n_ghosts, self.n_ghosts + 1)
        
        for rel_idx in rel_idxs:
            spec_data, index_data = spec_data_list[rel_idx], index_data_list[rel_idx]
            artist_dict = self._artist_list[rel_idx]
            self._update_single(spec_data, index_data, artist_dict)

    def _update_title(self):
        """
        Updates the title to match the current index
        """
        title = f"Spectrum {self.current_index+1}"
        self._axes.set_title(title)

    def _get_rel(self, rel_idx: int):
        """
        Get the correct spec data, index data, and artists dict relative to the current index.
        Positive indices get successively further above the current index.
        Negative indices get successively further below the current index.
        0 gets the current index.
        
        """
        
        if abs(rel_idx) > self.n_ghosts:
            print("Incorrect index")

        # when we initialised the list we put current first, then above, then below in reverse
        # so the index will be correct because of this
        spec_data = self._spec_data_list[rel_idx]
        index_data = self._index_data_list[rel_idx]
        artist_dict = self._artist_list[rel_idx]
        
        return spec_data, index_data, artist_dict

    def _write_single(self, rel_idx: int):
        """
        Write given relative result to file
        """
        # translate rel_idx to absolute idx to get filename
        index = min(max(self.current_index + rel_idx, 0), self.max_index)
        index_file = self._index_files[index]

        print(f"Writing index {index}, rel_idx {rel_idx}, file {index_file}")
        
        index_data = self._index_data_list[rel_idx]
        
        index_data = index_data[index_data['wvs'].argsort()]
        
        np.savetxt(index_file, index_data, fmt=self._index_fmts)

    def _read_single(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a given index from file
        """
        spec_file = self._spec_files[index]
        index_file = self._index_files[index]
        print(f"Reading index: {index}, file: {index_file}")

        spec_data = np.loadtxt(spec_file)
        index_data = np.loadtxt(index_file, dtype=self._index_dtypes)
        
        return spec_data, index_data

    def change_index(self, new_index: int) -> int:
        """
        Change the current index
        """
        # get currently loaded indices
        index, n_ghosts, max_index = self.current_index, self.n_ghosts, self.max_index
        delta_i = new_index - index
        
        current_indices = [min(max(0, idx), max_index) for idx in range(index - n_ghosts, index + n_ghosts + 1)]

        # new indices
        new_indices = [min(max(0, idx), max_index) for idx in range(new_index - n_ghosts, new_index + n_ghosts + 1)]

        # need to map currently loaded data to new data, so we know what to read
        index_map = {idx: new_idx for idx, new_idx in zip(current_indices, new_indices)}

        rel_idxs = range(-n_ghosts, n_ghosts + 1)
        rel_idx_map = {i: idx for i, idx in zip(rel_idxs, current_indices)}
        inv_rel_idx_map = {idx: i for i, idx in rel_idx_map.items()}
        new_rel_idx_map = {i: idx for i, idx in zip(rel_idxs, new_indices)}
        
        new_spec_data_list = [None] * len(rel_idxs)
        new_index_data_list = [None] * len(rel_idxs)

        #_index_list = ['test'] * len(rel_idxs)
        #_rel_map = {}

        #print(rel_idxs)
        #print(inv_rel_idx_map)
        #print(new_rel_idx_map)
        #print([inv_rel_idx_map.get(new_rel_idx_map[idx]) for idx in rel_idxs])
        
        for idx in rel_idxs:
            # new absolute index from map
            new_idx = new_rel_idx_map[idx]
            # this gets the rel_idx of the new data in the current data
            cur_rel_idx = inv_rel_idx_map.get(new_idx)
            # load data if required, else get it
            spec_data, index_data =  self._read_single(new_idx) if cur_rel_idx is None else (self._spec_data_list[cur_rel_idx], self._index_data_list[cur_rel_idx])
            #spec_data, index_data = cur_rel_idx) if cur_rel_idx else (*self.read_single(new_idx), None)
            #_index_list[idx] = f"current: {cur_rel_idx}, new: {new_idx}"

            #_rel_map[idx] = cur_rel_idx
            
            new_spec_data_list[idx] = spec_data
            new_index_data_list[idx] = index_data

        #print('debug', _index_list)
        #print('debug', _rel_map)
        
        # write out
        to_write = [rel_idx for rel_idx, idx in rel_idx_map.items() if idx not in new_rel_idx_map.values()]
        _ = [self._write_single(rel_idx) for rel_idx in to_write]

        # update plots with data
        self._update_all(new_spec_data_list, new_index_data_list)

        # update lists
        self._spec_data_list = new_spec_data_list
        self._index_data_list = new_index_data_list
        
        # hide setting for spectra in new (hide_single will only update if needed)
        hide = [True if idx < 0 or idx > max_index else False for idx in range(new_index - n_ghosts, new_index + n_ghosts + 1)]
        
        for rel_idx, h in zip(rel_idxs, hide):
            self._hide_single(rel_idx, h)

        self._current_index = new_index
        self._update_title()
        

    def _hide_single(self, rel_idx: int, hide=True):
        """
        Change visibility of a spectrum's artists using relative indexing
        """
        spec_data, index_data, artist_dict = self._get_rel(rel_idx)

        # only update if needed
        if hide != self._hidden[rel_idx]:
            [a.set_visible(not hide) for a in artist_dict.values()]
            self._hidden[rel_idx] = hide
        else:
            pass

    def _hide_points(self, rel_idx: int, hide_points=True):
        """
        Change the visibility of a spectrum's points using relative indexing
        """
        spec_data, index_data, artist_dict = self._get_rel(rel_idx)
        raise NotImplementedError

        # # only update if needed
        # if hide != self.hidden[rel_idx]:
        #     [a.set_visible(hide) for a in artist_dict.values()]
        #     self.hidden[rel_idx] = hide
        # else:
        #     pass

    def _get_closest(
        self, 
        x: float, 
        y: float, 
        only_selected: bool = False, 
        include_manual: bool = False
        ) -> int:
        """
        Find the closest data point to a given point.
        Works in display coordinates, not axes coordinates.
        `only_selected` to pick only selected indices instead of all indices (default: False).
        `include_manual` to also consider manual points if `selected` is also  True (default: False).
        
        """
        index_data = self._index_data_list[0]
        
        if only_selected:
            mask = index_data['sel'] if include_manual else (index_data['sel'] & index_data['index'] != -1)
            wvs, flux = index_data['wvs'][mask], index_data['flux'][mask]
        else:
            mask = np.ones_like(index_data['sel'])
            wvs, flux = index_data['wvs'], index_data['flux']

        # transform wvs and flux to display coordinates
        xy_data = self._axes.transData.transform(np.c_[wvs, flux])
        xs, ys = xy_data[:, 0], xy_data[:, 1]
        
        distances = (x - xs)**2 + (y - ys)**2
        
        # find index in unmasked array
        return np.arange(index_data.size)[mask][distances.argmin()]
        

    def _reset_home(self, border=0.05):
        """
        Reset the matplotlib home button to the correct limits
        """
        wvs = self._spec_data_list[0][:, 0]
        above_flux = self._get_rel(self.n_ghosts)[0][:, 1]
        
        new_xmin, new_xmax = wvs.min(), wvs.max()
        new_ymin, new_ymax = 0, above_flux.max()
        
        new_xmin = new_xmin - border *(new_xmax - new_xmin)
        new_xmax = new_xmax + border *(new_xmax - new_xmin)
        new_ymin = new_ymin - border *(new_ymax - new_ymin)
        new_ymax = new_ymax + border *(new_ymax - new_ymin)

        if self._fig.canvas.toolbar and len(self._fig.canvas.toolbar._nav_stack._elements) > 0:
            key = list(self._fig.canvas.toolbar._nav_stack._elements[0].keys())[0]
            alist = [x for x in self._fig.canvas.toolbar._nav_stack._elements[0][key]]
            alist[0] = (new_xmin, new_xmax, new_ymin, new_ymax)
            # Replace in the stack
            self._fig.canvas.toolbar._nav_stack._elements[0][key] = tuple(alist)

    #### event callback functions ####

    def _on_click_event(self, event):
        """
        Routes click events appropriately
        """
        if event.dblclick:
            #print("click event:", event.xdata, event.ydata)
            self.toggle_point_event(event)
            self._fig.canvas.draw_idle()

    def _on_press_event(self, event):
        """
        Routes key press events to the appropriate method.
        """
        method = self._on_press_dict.get(event.key)
        if method is None:
            self.invalid_key_event(event)
        else:
            method(event)
            self._fig.canvas.draw_idle()

    def move_up_event(self, event):
        """
        Increase current_index by one, wrapping if enabled
        """
        if self.current_index == self.max_index:
            if self._enable_wrapping:
                self.change_index(0)
        else:
            self.change_index(self.current_index + 1)

    def move_down_event(self, event):
        """
        Decrease current_index by one, wrapping if enabled
        """
        if self.current_index == 0:
            if self._enable_wrapping:
                self.change_index(self.max_index)
        else:
            self.change_index(self.current_index - 1)

    def toggle_point_event(self, event):
        """
        Gets the closest point to the event and selects/deselects it.
        """
        idx = self._get_closest(event.x, event.y)
        print(f"{idx}: {self._index_data_list[0]['sel'][idx]}")
        self._index_data_list[0]['sel'][idx] = not self._index_data_list[0]['sel'][idx]
        print(f"{idx}: {self._index_data_list[0]['sel'][idx]}")
        self._update_single(*self._get_rel(0), adjust=True)
    
    def add_manual_point_event(self, event):
        """
        Adds a manual point at the event location
        """
        pass

    def remove_point_event(self, event):
        """
        Removes the nearest point to the event location.
        """
        pass

    def hide_event(self, event):
        pass

    def hide_points_event(self, event):
        pass

    def hide_current_points_event(self, event):
        pass

    def thin_event(self, event):
        pass

    def invalid_key_event(self, event):
        print(f"Invalid key pressed: {event.key}")

    #### properties ####

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, new_index):
        if not isinstance(new_index, int):
            raise TypeError("New index must be an integer.")
        if 0 <= new_index <= self._max_index:
            if self._started:
                self.change_index(new_index)
            else:
                self._current_index = new_index
        else:
            raise IndexError(f"Invalid index for spec_files and index_files of length {len(spec_files)}.")

    @property
    def n_plot(self):
        return self._n_plot
    
    @n_plot.setter
    def n_plot(self, new_n_plot):
        # make sure n_plot is an odd integer, more than 1
        if new_n_plot < 1 or new_n_plot % 2 != 1 or not isinstance(new_n_plot, int):
            raise ValueError("n_plot must be an odd integer greater than 1")
        self._n_plots = new_n_plot
        self._n_ghosts = (new_n_plot - 1) // 2

    @property
    def n_ghosts(self):
        return self._n_ghosts

    @property
    def max_index(self):
        return self._max_index

    @property
    def spec_files(self):
        return self._spec_files

    @property
    def index_files(self):
        return self._index_files