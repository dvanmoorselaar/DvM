import mne
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from statsmodels.stats.multitest import fdrcorrection
from analysis.ERP import *
from stats.stats_utils import bootstrap_SE, perform_stats
from typing import Optional, Generic, Union, Tuple, Any, List, Dict
from support.preprocessing_utils import get_time_slice, get_diff_pairs
from visualization.plot_utils import shifted_color_map

# set general plotting parameters
# inspired by http://nipunbatra.github.io/2014/08/latexify/
params = {
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'arial',
}
matplotlib.rcParams.update(params)

meanlineprops = dict(linestyle='--', linewidth=1, color='black')
medianlineprops = dict(linestyle='-', linewidth=1, color='black')

def plot_timecourse(x: np.ndarray, y: np.ndarray,
                    show_SE: bool = False, smooth: bool = False,
                    **kwargs):
    """
    Plot a timecourse line with optional error bands and smoothing.

    Plots neural timecourse data on the current matplotlib axis. If data 
    contains multiple subjects, can either average across subjects or 
    compute bootstrap standard errors for visualization of variability.

    Parameters
    ----------
    x : np.ndarray
        X-axis values (typically time points in seconds). Should have 
        shape (n_timepoints,).
    y : np.ndarray
        Y-axis values. Can be 1D with shape (n_timepoints,) for 
        pre-averaged data, or 2D with shape (n_subjects, n_timepoints) 
        for subject-level data. If 2D, averaging or error computation 
        is applied based on show_SE parameter.
    show_SE : bool, default=False
        If True and y is 2D, compute bootstrap standard errors and shade 
        the area around the mean line. If False, simply average across 
        subjects. Ignored if y is 1D. Default is False.
    smooth : bool, default=False
        If True, apply Savitzky-Golay smoothing (window=9, order=1) to 
        the data before plotting. Provides mild smoothing suitable for 
        neural timecourse data. Default is False.
    **kwargs
        Additional keyword arguments passed to plt.plot() for styling 
        the line, such as label, color, linewidth, linestyle, etc.

    Returns
    -------
    None
        Modifies the current matplotlib figure by adding the plotted 
        timecourse line and optional error band.

    Notes
    -----
    When show_SE=True, the error band uses transparency (alpha=0.2) for 
    clear visualization. The Savitzky-Golay filter uses a window length 
    of 9 and polynomial order of 1, which provides mild smoothing 
    without over-filtering neural data.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-0.5, 1, 300)
    >>> y = np.random.randn(20, 300)  # 20 subjects, 300 timepoints
    >>> plt.figure()
    >>> plot_timecourse(x, y, show_SE=True, label='Mean ± SE', 
    ...                 color='blue')
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Amplitude (µV)')
    """

    if y.ndim > 1:
        if show_SE:
            err, y = bootstrap_SE(y)
        else:
            y = y.mean(axis=0)

    if smooth:
        y = savgol_filter(y, 9, 1)

    plt.plot(x, y, **kwargs)

    if show_SE:
        kwargs.pop('label', None)
        plt.fill_between(x, y + err, y - err, alpha=0.2, **kwargs)

def plot_2d(Z:np.array,x_val:np.array=None,
            y_val:np.array=None,colorbar:bool=True,
            nr_ticks_x:np.array=None,nr_ticks_y:np.array=5, 
            set_y_ticks:bool=True,interpolation:str='bilinear',
            cbar_label:str=None,mask:Union[np.ndarray,list]=None,
            mask_value:float=0,p_vals:np.ndarray=None,
            p_thresh:float=0.05,diverging_cmap:bool=False,
            cmap:str=None,center:float=0,
            **kwargs):
    """Plot 2D heatmap with optional masking and diverging colormap 
    support.

    Displays 2D neural data as a heatmap with support for masking 
    non-significant voxels, applying diverging colormaps centered at 
    arbitrary points (useful for AUC/accuracy visualization), and 
    overlay of statistical significance markers.

    Parameters
    ----------
    Z : ndarray
        2D or 3D array to plot. If 3D with shape 
        (n_subjects, n_rows, n_cols), averaged over subjects 
        (first dimension). If 2D, plotted as-is.
    x_val : ndarray, optional
        X-axis values (e.g., test times in ms). If None, uses array 
        indices. Default: None.
    y_val : ndarray, optional
        Y-axis values (e.g., train times, frequencies). If None, uses 
        array indices. Default: None.
    colorbar : bool, default=True
        If True, display colorbar showing data range. When using 
        diverging colormaps, tick labels are automatically adjusted to 
        show original (unshifted) values.
    nr_ticks_x : int, optional
        Number of x-axis ticks. If None, uses matplotlib defaults. 
        Default: None.
    nr_ticks_y : int, default=5
        Number of y-axis ticks. Adjusted if y_val has fewer than 5 
        unique values.
    set_y_ticks : bool, default=True
        If True, explicitly set y-axis ticks based on y_val. Useful for 
        non-linear scales (detects and applies log scaling if needed).
    interpolation : str, default='bilinear'
        Interpolation method for imshow. Use 'nearest' for sharp pixels, 
        'bilinear' for smooth transitions. Automatically switches to 
        'nearest' for masked arrays to prevent interpolation artifacts.
    cbar_label : str, optional
        Label for colorbar. Default: None.
    mask : ndarray or list, optional
        Mask indicating which values to display. Can be boolean ndarray 
        (True = keep, False = hide) or list of cluster indices 
        (for permutation test results).
        Default: None (displays all values).
    mask_value : float, default=0
        Value to set masked pixels to. Only used when mask is a boolean 
        array. Default: 0.
    p_vals : ndarray, optional
        P-values for each cluster (one per cluster in mask). Only used
        when mask is a list. Clusters with p_vals[i] > p_thresh are 
        excluded. Default: None.
    p_thresh : float, default=0.05
        P-value threshold for including clusters. Only used when mask is 
        a list.
    diverging_cmap : bool, default=False
        If True, use diverging colormap centered at the point specified 
        by `center`. Useful for data where a neutral point 
        (e.g., chance level, zero) should map to the middle of the 
        colormap. When True and cmap=None, uses 'RdBu_r'.
    cmap : str, optional
        Colormap name. If None and diverging_cmap=True, uses 'RdBu_r'. 
        Default: None.
    center : float, default=0
        Point to center the diverging colormap. Only used when 
        diverging_cmap=True. Data is internally shifted by this amount. 
        For AUC/accuracy, set to 0.5.Colorbar tick labels automatically 
        adjusted to show original values.
    **kwargs
        Additional arguments passed to plt.imshow().

    Returns
    -------
    None
        Modifies the current matplotlib figure by displaying the
        heatmap.

    Notes
    -----
    When diverging_cmap=True with center != 0, data is shifted 
    internally (Z - center), colormap applied to shifted values, but 
    colorbar labels show unshifted values.

    See Also
    --------
    plot_significance : Overlay statistical significance contours on 
    2D plots
    plot_bdm_timecourse : High-level wrapper for BDM/decoding 2D 
    visualization
    """

    if Z.ndim > 2:
        Z = Z.mean(axis=0)
    
    # Apply mask to set non-significant values to mask_value
    # This is for visualization only - does not affect statistical testing
    if mask is not None:
        Z = Z.copy()  # Don't modify the original array
        
        # Handle different mask formats
        if isinstance(mask, list):
            # Convert cluster list to boolean mask (for perm test results)
            bool_mask = np.zeros(Z.shape, dtype=bool)
            for i, cluster in enumerate(mask):
                # Only include significant clusters
                if p_vals is None or p_vals[i] <= p_thresh:
                    bool_mask[cluster] = True
            mask = bool_mask
        
        # Store unmasked data range BEFORE applying mask (for colorbar limits)
        Z_unmasked_values = Z[mask]
        
        # Use masked array to properly handle non-significant values
        # This prevents interpolation artifacts - 
        # masked pixels won't be displayed
        Z = np.ma.masked_where(~mask, Z)
    else:
        Z_unmasked_values = None

    # set extent
    x_lim = [0,Z.shape[-1]] if x_val is None else [x_val[0],x_val[-1]]
    y_lim = [0,Z.shape[-2]] if y_val is None else [y_val[0],y_val[-1]]
    extent = [x_lim[0],x_lim[1],y_lim[0],y_lim[1]]

    # Set up colormap and normalization for diverging data
    #TODO: fix or remove
    if diverging_cmap:
        if cmap is None:
            # Red-Blue colormap, reversed (red=positive, blue=negative)
            cmap = 'RdBu_r'  
        
        # Calculate data range for colorbar limits
        if Z_unmasked_values is not None and len(Z_unmasked_values) > 0:
            data_min = Z_unmasked_values.min()
            data_max = Z_unmasked_values.max()
        else:
            if isinstance(Z, np.ma.MaskedArray):
                data_min = Z.compressed().min()
                data_max = Z.compressed().max()
            else:
                data_min = Z.min()
                data_max = Z.max()

        # Shift data relative to center point, handling masked arrays
        if isinstance(Z, np.ma.MaskedArray):
            Z_shifted = Z.data - center
        else:
            Z_shifted = Z - center
        data_min_shifted = data_min - center
        data_max_shifted = data_max - center
        
        # Apply shifted colormap for asymmetric data
        cmap_obj = plt.cm.get_cmap(cmap)
        shifted_cmap = shifted_color_map(cmap_obj, data_min_shifted, 
                                 data_max_shifted, name='shifted_colormap')
        kwargs.setdefault('cmap', shifted_cmap)
        
        # Display shifted data on the plot
        if isinstance(Z, np.ma.MaskedArray):
            Z = np.ma.masked_array(Z_shifted, mask=Z.mask)
        else:
            Z = Z_shifted

        # Prevent interpolation artifacts for masked arrays
        if isinstance(Z, np.ma.MaskedArray) and np.ma.is_masked(Z):
            interpolation = 'nearest'
    else:
        # Apply colormap when diverging_cmap=False
        if cmap is not None:
            kwargs.setdefault('cmap', cmap)
    
    # do actuall plotting
    plt.imshow(Z,interpolation=interpolation,aspect='auto',origin='lower',
            extent=extent, **kwargs)
    
    # set ticks
    if nr_ticks_x is not None:
        plt.xticks(np.linspace(x_lim[0],x_lim[1],nr_ticks_x))

    if set_y_ticks:
        if isinstance(y_val, list):
            y_val = np.array(y_val)
        if nr_ticks_y is None:
            nr_ticks_y = 5 if len(y_val) > 5 else len(y_val)	
        idx = np.linspace(0, len(y_val)-1, nr_ticks_y).astype(int)
        ticks = y_val[idx]
        if np.allclose(np.diff(y_val),np.diff(y_val)[0],rtol=1e-2, atol=1e-8):
            plt.yscale('linear')
        else:
            plt.yscale('log')
        if np.issubdtype(y_val.dtype, np.floating):
            tick_labels = np.round(ticks).astype(int)
        else:
            tick_labels = ticks
        plt.yticks(ticks, tick_labels)
        plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    
    # add colorbar
    if colorbar:
        cbar = plt.colorbar()
        if cbar_label:
            cbar.set_label(cbar_label)
        
        # If using diverging colormap with shifted data, 
        # adjust tick labels to show original values
        if diverging_cmap and center != 0:
            # Get current tick positions and labels
            ticks = cbar.get_ticks()
            # Add center back to show original values
            tick_labels = [f'{tick + center:.3f}' for tick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)

def _get_continuous_segments(mask: np.ndarray) -> List[np.ndarray]:
    """Convert boolean mask into list of continuous segments.

    Identifies contiguous regions of True values in a boolean array and 
    returns the indices for each continuous segment. Useful for finding 
    time windows where statistical significance occurs.

    Parameters
    ----------
    mask : np.ndarray
        1D boolean array indicating significant timepoints/pixels. 
        True values are grouped into continuous segments, False values 
        serve as boundaries.

    Returns
    -------
    segments : list of np.ndarray
        List of 1D arrays, each containing the integer indices of a 
        continuous segment of True values. If no True values exist in 
        mask, returns empty list. If entire array is True, returns 
        single array [0, 1, ..., len(mask)-1].

    Notes
    -----
    This function is commonly used with permutation test results to 
    identify which time windows show significant differences, or with 
    other boolean masking operations.

    Handles edge cases where significant segments start at index 0 or 
    extend to the end of the array.

    Examples
    --------
    >>> mask = np.array([False, True, True, False, True, False])
    >>> segments = _get_continuous_segments(mask)
    >>> len(segments)
    2
    >>> segments[0]
    array([1, 2])
    >>> segments[1]
    array([4])
    
    >>> # No significant values
    >>> mask_empty = np.array([False, False, False])
    >>> _get_continuous_segments(mask_empty)
    []
    
    >>> # Entire array significant
    >>> mask_all = np.array([True, True, True])
    >>> _get_continuous_segments(mask_all)
    [array([0, 1, 2])]

    See Also
    --------
    plot_significance : Visualizes significance using this function
    """
    # Find boundaries of continuous segments
    diff = np.diff(mask.astype(int))
    segment_starts = np.where(diff == 1)[0] + 1
    segment_ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        segment_starts = np.r_[0, segment_starts]
    if mask[-1]:
        segment_ends = np.r_[segment_ends, len(mask)]

    # Return list of index arrays for each segment
    segments = [np.arange(start, end) for 	
                             start, end in zip(segment_starts, segment_ends)]
    return segments

def plot_significance(x:np.array,y:np.array,chance:float=0,p_thresh:float=0.05,
                    color:str=None,stats:str='perm',
                    smooth:bool=False,line_width:float = 4,
                    y_val:np.array=None,
                    sig_mask:Union[np.ndarray,list]=None,
                    p_cluster:Optional[float]=None,
                    threshold:Optional[float]=None,
                    **kwargs):
    """Overlay statistical significance markers on 1D or 2D neural data 
    plots.

    Automatically detects plot dimensionality and visualizes significant 
    regions using either horizontal lines (1D timecourse) or contours 
    (2D matrices). Can use pre-computed significance masks or compute 
    significance on-the-fly.

    Parameters
    ----------
    x : ndarray
        X-axis values (e.g., timepoints in ms). Shape: (n_timepoints,).
    y : ndarray
        Data array. For significance computation (if sig_mask not 
        provided):
        - 1D: Shape (n_timepoints,)
        - 2D: Shape (n_channels, n_timepoints)
        - 3D: Shape (n_subjects, n_channels, n_timepoints) for group 
        analysis. Only used if sig_mask is None.
    chance : float, default=0
        Chance/baseline level for statistical testing. Typically 0 for 
        amplitude, 0.5 for accuracy/AUC.
    p_thresh : float, default=0.05
        P-value threshold. Regions with p-value <= p_thresh marked as 
        significant.
    color : str, optional
        Color for significance markers. If None, auto-selected based on 
        data. Default: None.
    stats : str, default='perm'
        Statistical test type if computing significance on-the-fly:
        - 'perm': Cluster-based permutation test
        - 'ttest': Independent samples t-test
        - 'fdr': False discovery rate correction
    smooth : bool, default=False
        If True, smooth data before significance computation using
        Gaussian filter.
    line_width : float, default=4
        Width of horizontal lines marking significance in 1D plots.
    y_val : ndarray, optional
        Y-axis values for 2D plots (e.g., train times, frequencies). 
        Required for 2D significance visualization. Default: None.
    sig_mask : ndarray or list, optional
        Pre-computed significance mask. If provided, skips significance 
        computation:
        - ndarray: Boolean array (True = significant)
        - list: Cluster indices (each non-zero value = cluster)
        If None, significance computed from y using stats method. 
        Default: None.
    p_cluster : float, optional
        P-value threshold for cluster-based permutation test. If None, 
        uses p_thresh. Default: None.
    threshold : float, optional
        Threshold for initial thresholding in permutation test 
        (e.g., t-statistic threshold). Default: None.
    **kwargs
        Additional arguments passed to plt.axhline() (1D) or 
        plt.contour() (2D).

    Returns
    -------
    None
        Modifies the current matplotlib figure by overlaying 
        significance markers.

    Notes
    -----
    Automatic plot type detection: Inspects current axis to determine 
    dimensionality.
    - 1D (timecourse): Significant segments marked with horizontal lines 
      at top
    - 2D (matrix): Significant regions marked with contour outlines
    
    For 2D plots, contour lines are drawn around the boundaries of 
    significant clusters, making them visually distinct from the 
    background.

    Examples
    --------
    Mark significant timepoints in a timecourse with permutation test:

        >>> time = np.linspace(0, 1000, 1000)
        >>> amp = np.random.randn(20, 1000)  # 20 subjects
        >>> plt.plot(time, amp.mean(axis=0))
        >>> plot_significance(time, amp, chance=0, stats='perm')

    Overlay pre-computed significance mask on timecourse:

        >>> sig_mask = np.zeros(1000, dtype=bool)
        >>> sig_mask[100:150] = True
        >>> plot_significance(time, amp, sig_mask=sig_mask)

    Mark significant regions in a 2D GAT matrix:

        >>> freq = np.arange(8, 13)
        >>> time = np.arange(-1000, 2000, 10)
        >>> GAT = np.random.randn(20, len(freq), len(time))
        >>> plt.imshow(GAT.mean(axis=0), aspect='auto')
        >>> plot_significance(time, GAT, y_val=freq, stats='perm')

    See Also
    --------
    plot_2d : Plot 2D heatmap with integrated masking
    perform_stats : Underlying statistical computation function
    """
    
    # Infer plot type from data dimensions
    plot_type = '2d' if y.ndim == 3 else '1d'

    # Only compute stats if not provided
    if sig_mask is None:
        _, sig_mask, _ = perform_stats(y, chance, stats, p_thresh,
                                        p_cluster=p_cluster,
                                        threshold=threshold)

    if plot_type == '2d':
        # Require y_val for 2D plots
        if y_val is None:
            raise ValueError("y_val (e.g., freq values) required for 2D plots")
        
        # Handle 2D significance plotting
        x_lim = [x[0], x[-1]]
        y_lim = [y_val[0], y_val[-1]]
        extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]

        if color is None:
            color = 'white'

        # Plot significance contours based on test type
        if stats == 'perm':
            # Handle cluster-based results
            for cluster in sig_mask:
                cluster_mask = np.zeros(y.shape[1:])  # (n_freqs, n_times)
                cluster_mask[cluster] = 1

                # Smooth the cluster mask
                if smooth:
                    cluster_mask_smooth = gaussian_filter(
                                    cluster_mask.astype(float), sigma=1.0)
                else:
                    cluster_mask_smooth = cluster_mask.astype(float)

                # Remove p_vals from kwargs as contour doesn't accept it
                contour_kwargs = {k: v for k, v in kwargs.items() 
                                                      if k != 'p_vals'}
                plt.contour(cluster_mask_smooth, levels=[0.5], colors=color,
                            linestyles='dashed', linewidths=1,
                            extent=extent, **contour_kwargs)
        else:
            # Handle boolean mask results (ttest, fdr)
            # Apply smoothing if requested
            if smooth:
                sig_mask_smooth = gaussian_filter(sig_mask.astype(float), 
                                                      sigma=1.0)
            else:
                sig_mask_smooth = sig_mask.astype(float)

            if sig_mask_smooth.max() > 0.5 and sig_mask_smooth.min() < 0.5:
                plt.contour(sig_mask_smooth, levels=[0.5], colors=color,
                            linestyles='dashed', linewidths=1,
                            extent=extent, **kwargs)
            elif sig_mask.any():  # If there are any significant points
                # Use alternative visualization - highlight significant regions
                plt.contourf(sig_mask.astype(float), levels=[0.5, 1.0], 
                            colors=[color], alpha=0.3, extent=extent, **kwargs)
    
    else:
        # Get current line properties
        if color is None:
            current_line = plt.gca().get_lines()[-1]
            color = current_line.get_color()
            y_data = current_line.get_ydata()
        else:
            y_data = np.mean(y, axis=0)

        if smooth:
            y_data = savgol_filter(y_data, 9, 1)

        if stats == 'perm':
            for cl in sig_mask:
                plt.plot(x[cl], y_data[cl], linewidth=line_width, 
                    color=color, **kwargs)
        else:
            for segment in _get_continuous_segments(sig_mask):
                plt.plot(x[segment], y_data[segment], linewidth=line_width,
                        color=color, **kwargs)

def plot_erp_timecourse(
    erps: Union[list, dict], 
    times: np.array, 
    elec_oi: list, 
    lateralized: bool = False, 
    cnds: list = None, 
    colors: list = None, 
    show_SE: bool = False, 
    smooth: bool = False, 
    stats: Union[str, bool] = False,
    p_thresh: float = 0.05,
    p_cluster: Optional[float] = None,
    threshold: Optional[float] = None,
    window_oi: Tuple = None, 
    offset_axes: int = 10, 
    onset_times: Union[list, bool] = [0], 
    show_legend: bool = True, 
    **kwargs
):
    """Visualize event-related potential (ERP) timecourses.

    Plots averaged ERP waveforms for specified conditions and 
    electrodes. Supports single electrode, multiple electrode groups 
    (averaged within group), and lateralized comparisons 
    (contralateral - ipsilateral differences). Automatically handles 
    unit conversion and time scaling.

    Parameters
    ----------
    erps : list or dict
        ERP data. If list, treated as single condition. If dict, keys 
        are condition names, values are lists of mne.Evoked objects 
        (one per subject).
    times : ndarray
        Time array (seconds or milliseconds) corresponding to ERP 
        timepoints. Automatically converted to milliseconds if detected 
        in seconds (average difference < 0.1).
    elec_oi : list
        Electrode(s) of interest. Can be list of strings for single 
        group (e.g., ['Cz', 'CPz']), or list of lists for multiple 
        groups (e.g., [['C3', 'C5'], ['C4', 'C6']]). 
        Data averaged within groups.
    lateralized : bool, optional
        If True, plots contra-ipsilateral difference. Requires elec_oi 
        to have exactly 2 groups. Default: False.
    cnds : list, optional
        Condition names to include. If None, plots all conditions.
        Default: None.
    colors : list, optional
        Colors for waveforms. If fewer colors than waveforms, uses
        default tableau colors. Default: None.
    show_SE : bool, optional
        If True, displays shaded standard error band around waveform.
        Default: False.
    smooth : bool, optional
        If True, applies Savitzky-Golay smoothing to waveform.
        Default: False.
    stats : {'perm', 'ttest', 'fdr'} or False, optional
        Statistical test type. 'perm': permutation cluster test; 
        'ttest': t-test; 'fdr': false discovery rate correction; 
        False: no statistics. Default: False.
    p_thresh : float, optional
        P-value threshold for significance. Clusters/timepoints with 
        p <= p_thresh considered significant. Default: 0.05.
    p_cluster : float or None, optional
        Cluster-forming p-value threshold (permutation test only). 
        Automatically converts to threshold value as t-statistic. 
        If both p_cluster and threshold are None, MNE uses automatic.
        Default: None.
    threshold : float or None, optional
        Cluster-forming threshold as test statistic value (permutation 
        test only). Overrides p_cluster if specified. Default: None.
    window_oi : tuple, optional
        Time window (start_ms, end_ms) or 
        (start_ms, end_ms, 'pos'/'neg') to highlight with rectangle. 
        Third element specifies polarity to restrict y-axis 
        (e.g., show only positive half). Default: None.
    offset_axes : int, optional
        Pixel offset for despine. Default: 10.
    onset_times : list or False, optional
        Time points (ms) to mark with vertical dashed lines 
        (e.g., stimulus onset, response). If False, no lines drawn. 
        Default: [0].
    show_legend : bool, optional
        If True, displays legend for waveforms. Default: True.
    **kwargs
        Additional keyword arguments passed to plot_timecourse() and 
        plot_significance() (e.g., linewidth, alpha).

    Returns
    -------
    None
        Modifies matplotlib figure directly.

    Notes
    -----
    1. **Data preparation**: Data averaged across subjects and within 
       electrode groups using ERP.group_erp(). Automatically converts 
       volts to microvolts if detected (data range 1 nV - 1 mV).

    2. **Electrode grouping**: Single electrode names auto-wrapped as 
       single-item lists. Multiple groups (e.g., contra/ipsi pairs) 
       plotted as separate waveforms unless lateralized=True.
    
    3. **Lateralization**: When lateralized=True, requires exactly 2 
       electrode groups. Computes difference (group1 - group2), 
       typically contra-ipsi. Result shown as single waveform per 
       condition.

    4. **Time unit handling**: Automatically detects and converts 
       seconds to milliseconds. If window_oi specified, converted 
       accordingly.

    5. **Window highlighting**: Rectangle drawn with dashed outline. 
       Polarity filtering (pos/neg) limits display to upper/lower half 
       of y-axis, useful for highlighting amplitude without obscuring 
       data.

    6. **Color management**: Colors assigned per waveform 
       (condition x group combinations). Removed colors not re-used; 
       auto-cycles through tableau colors if insufficient provided.

    7. **Statistical testing**: When stats enabled, tests each waveform 
       against zero baseline across subjects. For permutation tests, 
       control cluster formation via p_cluster (intuitive p-value) or 
       threshold (direct test statistic). Returns only significant 
       clusters for 'perm' or significant timepoints for 'ttest'/'fdr'.

    Examples
    --------
    1. Plot single electrode ERP for all conditions:

        >>> import numpy as np
        >>> # Simulate ERP data: 30 subjects, 1 condition, 512 samples
        >>> erps_data = {}\n
        >>> for cnd in ['target', 'nontarget']:
        ...     erps_data[cnd] = []\n
        ...     for _ in range(30):  # 30 subjects
        ...         evoked = type('obj', (), {
        ...             'ch_names': ['Pz', 'Cz', 'Fz'],
        ...             'times': np.linspace(-0.2, 0.8, 512),
        ...             'data': np.random.randn(3, 512) * 1e-6,  # volts
        ...             'pick': lambda x: type('obj', (), {
        ...                 'data': np.random.randn(1, 512) * 1e-6
        ...             })()
        ...         })()\n
        ...         erps_data[cnd].append(evoked)

        >>> plot_erp_timecourse(
        ...     erps=erps_data,
        ...     times=np.linspace(-200, 800, 512),
        ...     elec_oi=['Pz'],
        ...     cnds=['target', 'nontarget'],
        ...     colors=['red', 'blue'],
        ...     show_SE=True,
        ...     onset_times=[0],
        ...     show_legend=True
        ... )

    2. Plot contra/ipsi waveforms with statistical significance:

        >>> plot_erp_timecourse(
        ...     erps=erps_data,
        ...     times=np.linspace(-200, 800, 512),
        ...     elec_oi=[['P3', 'P5'], ['P4', 'P6']],
        ...     cnds=['target'],
        ...     colors=['red', 'blue'],
        ...     window_oi=(300, 500, 'pos'),  # Highlight P300 window
        ...     smooth=True,
        ...     show_SE=True,
        ...     stats='ttest',
        ...     p_thresh=0.05
        ... )

    3. Plot lateralized difference with permutation test:

        >>> plot_erp_timecourse(
        ...     erps=erps_data,
        ...     times=np.linspace(-200, 800, 512),
        ...     elec_oi=[['P3', 'P5'], ['P4', 'P6']],
        ...     lateralized=True,
        ...     cnds=['target', 'nontarget'],
        ...     colors=['purple', 'orange'],
        ...     window_oi=(300, 500),
        ...     smooth=True,
        ...     stats='perm',
        ...     p_cluster=0.01,  # More stringent clustering
        ...     show_legend=True
        ... )

    See Also
    --------
    plot_timecourse : Plot generic 1D timecourse data
    plot_significance : Overlay statistical significance on plots
    plot_erp_topography : Plot topographic scalp maps of ERP amplitudes
    """

    # Convert times from seconds to milliseconds if needed
    time_diff = np.diff(times).mean()
    if time_diff < 0.1:  # If average difference < 0.1, assume seconds
        times = times * 1000
        print(f"Times converted from seconds to milliseconds")
        if window_oi is not None:
            window_oi = [window_oi[0]*1000, window_oi[1]*1000] + window_oi[2:]	

    if isinstance(erps, list):
        erps = {'temp':erps}

    if cnds is not None:
        erps = {key:value for (key,value) in erps.items() if key in cnds}

    if isinstance(elec_oi[0],str): 
        elec_oi = [elec_oi]

    # Calculate the actual number of waveforms that will be plotted
    n_waveforms_per_condition = 1 if lateralized else len(elec_oi)
    total_waveforms = len(erps) * n_waveforms_per_condition

    if colors is None or len(colors) < total_waveforms:
        print('not enough colors specified. Using default colors')
        colors = list(mcolors.TABLEAU_COLORS.values())
    
    for cnd in erps.keys():
        # extract all time courses for the current condition
        y = []
        for c, elec in enumerate(elec_oi):
            y_,_ = ERP.group_erp(erps[cnd],elec_oi = elec)
            y.append(y_)
        
        # Auto-detect and convert volts to microvolts for all conditions
        for i, y_ in enumerate(y):
            typical_value = np.abs(y_).mean()
            # Between 1 nV and 1 mV
            if typical_value < 1e-3 and typical_value > 1e-9:  
                y[i] = y_ * 1e6  # Convert volts to microvolts
                if i == 0:  # Only print once per condition
                    print(f"Data for condition '{cnd}' converted from volts to" 
                       " microvolts" )		
                
        # set up timecourses to plot	
        if lateralized:
            y = [y[0] - y[1]]
        
        # Create appropriate labels based on lateralization and 
        # number of conditions
        if lateralized:
            labels = [f'{cnd} (contra-ipsi)']
        elif len(y) == 1:
            labels = [cnd]
        else:
            labels = [f'{cnd} contra', f'{cnd} ipsi']

        #do actual plotting
        for i, y_ in enumerate(y):
            color = colors.pop(0) 
            plot_timecourse(times,y_,show_SE,smooth,
                            label=labels[i],color=color,**kwargs)
            if stats:
                plot_significance(times, y_, 0, color=color, stats=stats,
                             p_thresh=p_thresh, p_cluster=p_cluster,
                             threshold=threshold, smooth=smooth, **kwargs)

    # clarify plot
    if window_oi is not None:
        _, _, ymin, ymax = plt.axis()
        if len(window_oi) == 3:
            ymin, ymax = (0, ymax) if window_oi[-1] == 'pos' else (ymin, 0)

        # Add a small margin to ymin and ymax to ensure visibility
        # Adjust the margin as needed (5% of height)
        margin = 0.05 * (ymax - ymin)  
        ymin += margin
        ymax -= margin
        # Add dashed grey outline for the time window of interest
        rect = plt.Rectangle(
            (window_oi[0], ymin),  
            window_oi[1] - window_oi[0],  
            ymax - ymin, 
            edgecolor='black',  
            facecolor='none', 
            linestyle='--')  
        plt.gca().add_patch(rect)

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc = 'best',
        prop={'size': 7},frameon=False)
    plt.xlabel('Time (ms)')
    plt.ylabel('\u03BC' + 'V')
    plt.axhline(0, color = 'black', ls = '--', lw=1)
    if onset_times:
        for t in onset_times:
            plt.axvline(t,color = 'black',ls='--',lw=1)

    sns.despine(offset = offset_axes)

def plot_tfr_timecourse(tfr:Union[dict,mne.time_frequency.AverageTFR], 
    elec_oi: list, 
    freq_oi: Union[int,Tuple] = None,
    lateralized: bool = False, 
    cnds: list = None, 
    colors: list = None, 
    timecourse: str = '2d',
    stats:Union[str,bool]=False,
    p_thresh:float=0.05,
    p_cluster:Optional[float]=None,
    threshold:Optional[float]=None,
    mask_nonsig:bool=False,
    divergence_cmap:bool=False,
    show_SE: bool = False, 
    smooth: bool = False, 
    window_oi: Tuple = None, 
    offset_axes: int = 10, 
    onset_times: Union[list, bool] = [0], 
    show_legend: bool = True, 
    ls: str = '-', 
    **kwargs
):	
    """Plot time-frequency representation (TFR) timecourse with multiple 
    visualization modes.

    High-level wrapper for visualizing time-frequency power across time 
    with support for 1D timecourse plots (averaged over frequencies) and 
    2D time-frequency matrices. Integrates statistical testing and 
    significance visualization with optional lateralized comparisons 
    (contralateral - ipsilateral differences).

    Parameters
    ----------
    tfr : dict or mne.time_frequency.AverageTFR
        Time-frequency representation data structure(s). For dict 
        format:
        - Keys are condition names
        - Values are lists of TFR objects (one per subject)
        - Each TFR object has attributes: data (n_channels, n_freqs, 
          n_times), freqs, times, ch_names
    elec_oi : list of str or list of list of str
        Electrode(s) of interest. Can be:
        - Single group: ['C3', 'C4'] (averaged together)
        - Multiple groups: [['C3', 'C5'], ['C4', 'C6']] (separate 
        waveforms). Useful for contra/ipsilateral electrode pairs.
    freq_oi : int or tuple, optional
        Frequency range of interest for 1D plotting or selection:
        - int: Single frequency (Hz) - closest frequency selected
        - tuple: (freq_min, freq_max) - frequency band selected
        For 2D plots, ignored. Default: None (uses all frequencies).
    lateralized : bool, default=False
        If True, plots contra-ipsilateral difference. Requires elec_oi 
        to have exactly 2 groups. Difference computed as 
        group1 - group2. Default: False.
    cnds : list of str, optional
        Condition names to include. If None, plots all conditions.
        Default: None.
    colors : list of str, optional
        Colors for waveforms (1D) or colormap (2D). If fewer colors than 
        conditions, cycles through tableau colors. Default: None.
    timecourse : str, default='2d'
        Visualization type:
        - '1d': 1D timecourse (time x power, averaged over frequencies)
        - '2d': 2D time-frequency matrix (time x frequency)
        For 2D mode, only first condition is plotted with warning if 
        multiple conditions specified.
    stats : {'perm', 'ttest', 'fdr'} or False, optional
        Statistical test type for significance overlay:
        - 'perm': Cluster-based permutation test
        - 'ttest': Independent samples t-test
        - 'fdr': False discovery rate correction
        - False: No significance testing
        Default: False.
    p_thresh : float, default=0.05
        P-value threshold for significance marking.
    p_cluster : float, optional
        Cluster-forming p-value threshold (only for stat_test='perm'). 
        Automatically converts to t-statistic threshold. If None, MNE 
        uses automatic threshold. Default: None.
    threshold : float, optional
        Cluster-forming threshold as test statistic value (permutation 
        test only). Overrides p_cluster if specified. Default: None.
    mask_nonsig : bool, default=False
        If True, non-significant voxels displayed in greyscale (2D plots 
        only). Requires significant clusters to be computed. When True, 
        shows dual-visualization: greyscale background + color overlay 
        of significant data. Default: False.
    divergence_cmap : bool, default=False
        If True, use diverging colormap for 2D plots. Useful for 
        lateralization indices or other data with meaningful zero point 
        (e.g., contra-ipsi differences). When True, uses 'RdBu_r' 
        colormap centered at zero. Default: False.
    show_SE : bool, default=False
        If True, overlay standard error bands around mean (1D plots 
        only).Default: False.
    smooth : bool, default=False
        If True, apply Gaussian smoothing to data (both visualization
        and statistical testing). Default: False.
    window_oi : tuple, optional
        Time window (start_ms, end_ms) to highlight with rectangle. 
        Default: None.
    offset_axes : int, default=10
        Pixel offset for despine. Default: 10.
    onset_times : list of float or False, default=[0]
        Stimulus/event onset times to mark with vertical lines. Set to 
        False for no onset markers. Default: [0].
    show_legend : bool, default=True
        If True, display legend with condition names (1D plots only). 
        Default: True.
    ls : str, default='-'
        Line style for 1D timecourse ('-', '--', '-.', ':'). 
        Default: '-'.
    **kwargs
        Additional arguments passed to plot_significance() and 
        plot_2d() functions.

    Returns
    -------
    None
        Modifies current matplotlib figure(s) by plotting TFR 
        timecourse(s).

    Notes
    -----
    Time unit conversion: If times have mean difference < 0.1, assumes 
    input is in seconds and automatically converts to milliseconds.
    
    2D modes and conditions: Each 2D plot can only show one condition. 
    If multiple conditions specified for 2D plot, only first is plotted 
    with warning.
    
    Frequency selection: For 1D plots with freq_oi specified, power is 
    either extracted at single frequency or averaged over frequency 
    band.For 2D plots, freq_oi is ignored and full frequency range 
    shown.
    
    Lateralization: When lateralized=True, contra-ipsilateral difference 
    is computed before visualization. divergence_cmap recommended for 
    proper visualization of difference data (shows positive/negative 
    symmetrically).
    
    Statistical testing: Applied to raw data before smoothing. Smoothing 
    affects visualization but not significance computation.

    Examples
    --------
    Plot 1D timecourse for single frequency:

        >>> tfr_data = {'cond_A': [tfr_obj1, tfr_obj2, ...],
        ...             'cond_B': [tfr_obj1, tfr_obj2, ...]}
        >>> plot_tfr_timecourse(tfr_data, elec_oi=['C3', 'C4'], 
        ...                     freq_oi=10, cnds=['cond_A', 'cond_B'])

    Plot 1D timecourse averaged over frequency band with statistics:

        >>> plot_tfr_timecourse(tfr_data, elec_oi=['C3', 'C4'], 
        ...                     freq_oi=(8, 12), cnds=['cond_A'],
        ...                     stats='perm', p_thresh=0.05)

    Plot 2D time-frequency matrix with statistical masking:

        >>> plot_tfr_timecourse(tfr_data, elec_oi=['C3', 'C4'], 
        ...                     timecourse='2d', stats='perm',
        ...                     mask_nonsig=True, divergence_cmap=False)

    Plot lateralized contra-ipsi difference with diverging colormap:

        >>> plot_tfr_timecourse(tfr_data, 
        ...                     elec_oi=[['C3', 'C5'], ['C4', 'C6']],
        ...                     lateralized=True, timecourse='2d',
        ...                     divergence_cmap=True, stats='perm')

    See Also
    --------
    plot_bdm_timecourse : Plot decoding timecourse with similar 
    interface
    plot_2d : Core 2D heatmap visualization with masking
    plot_timecourse : 1D timecourse plotting
    plot_significance : Statistical significance overlay
    """
    
    if cnds is not None:
        tfr = {key:value for (key,value) in tfr.items() if key in cnds}
    else:
        print('No conditions specified. Using first condition in tfr')
        cnds = list(tfr.keys())

    if timecourse == '2d' and len(cnds) > 1:
        print(f'2d timecourse only supports one condition. '
        f'will show first condition only: {cnds[0]}')
        cnds = [cnds[0]]

    if colors is None or len(colors) < len(cnds):
        print('not enough colors specified. Using default colors')
        colors = list(mcolors.TABLEAU_COLORS.values())

    if isinstance(elec_oi[0],str): 
        elec_oi = [elec_oi]

    times = tfr[cnds[0]][0].times
    # Convert times from seconds to milliseconds if needed
    time_diff = np.diff(times).mean()
    if time_diff < 0.1:  # If average difference < 0.1, assume seconds
        times = times * 1000
        print(f"Times converted from seconds to milliseconds")


    for cnd in cnds:
        # get indices of frequencies of interest
        freqs = tfr[cnd][0].freqs
        if freq_oi is not None:
            if isinstance(freq_oi, tuple):
                freq_idx = (np.abs(freqs - freq_oi[0]).argmin(),
                            np.abs(freqs - freq_oi[1]).argmin())
                freq_idx = slice(freq_idx[0],freq_idx[1]+1)
            else:
                freq_idx = np.abs(freqs - freq_oi).argmin()

            freqs = freqs[freq_idx]

        # extract all time courses for the current condition
        y = []
        for c, elec in enumerate(elec_oi):
            # Stack individual TFR data for the specified electrodes
            idx = [tfr[cnd][0].ch_names.index(e) for e in elec]
            y_ = np.stack([tfr_.data[idx] for tfr_ in tfr[cnd]]).mean(axis = 1)
            if freq_oi is not None:
                if isinstance(freq_idx, int):
                    y_ = np.expand_dims(y_[:, freq_idx], axis=-1)
                else:
                    y_ = y_[:, freq_idx]
            y.append(y_)

        # Create appropriate labels based on lateralization and number of conditions
        if lateralized:
            labels = [f'{cnd} (contra-ipsi)']
        elif len(y) == 1:
            labels = [cnd]
        else:
            labels = [f'{cnd} contra', f'{cnd} ipsi']
                
        # set up timecourses to plot	
        if lateralized:
            y = [y[0] - y[1]]
            
        #do actual plotting
        for i, y_ in enumerate(y):
            if timecourse == '2d':
                # Calculate stats once if needed
                sig_mask = None
                p_vals = None
                if stats:
                    _, sig_mask, p_vals = perform_stats(y_, 0, stats,
                                                       p_thresh,
                                                       p_cluster=p_cluster,
                                                       threshold=threshold)
                
                # Plot with optional masking
                if mask_nonsig and sig_mask is not None:
                    # Step 1: Plot full data in greyscale as background
                    plot_2d(y_, x_val=times, y_val=freqs, colorbar=False,
                                    cbar_label=None,
                                    mask=None, diverging_cmap=False, 
                                    cmap='gray', **kwargs)

                    # Step 2: Overlay only significant data in color
                    plot_2d(y_, x_val=times, y_val=freqs, colorbar=True,
                            cbar_label='Power (au)',
                            mask=sig_mask,  
                            mask_value=np.nan,  # Hide non-significant 
                            p_vals=p_vals, p_thresh=p_thresh,
                            diverging_cmap=divergence_cmap, **kwargs)
                else:
                    plot_2d(y_, x_val=times, y_val=freqs, 
                            cbar_label='Power (au)',colorbar=True,
                            mask=None,mask_value=0, p_vals=p_vals, 
                            p_thresh=p_thresh,
                            diverging_cmap=divergence_cmap,**kwargs)
            else:
                color = colors.pop(0) 
                plot_timecourse(times,y_.mean(axis = 1),show_SE,smooth,
                        label=labels[i],color=color,ls=ls,**kwargs)
                if stats:		
                    plot_significance(times,y_.mean(axis = 1),0,
                                    color=color,stats=stats,
                                    p_thresh=p_thresh,
                                    p_cluster=p_cluster,
                                    threshold=threshold,
                                    smooth=smooth,**kwargs)

    if show_legend and timecourse == '1d':
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc = 'best',
        prop={'size': 7},frameon=False)


    plt.xlabel('Time (ms)')
    if timecourse == '2d':
        plt.ylabel('Frequency (Hz)')
    else:
        plt.ylabel('Power (au)')
    
    sns.despine(offset = offset_axes)
                
def plot_bdm_timecourse(bdms:Union[list,dict],cnds:list=None,timecourse:str='1d',
                    colors:list=None,
                    show_SE:bool=False,smooth:bool=False,method:str='auc',
                    chance_level:float=0.5,stats:Union[str,bool]='perm',
                    p_thresh:float=0.05,mask_nonsig:bool=False,
                    diverging_cmap:bool=False,
                    p_cluster:Optional[float]=None,
                    threshold:Optional[float]=None,
                    freq_oi: Union[int,Tuple] = None,
                    onset_times:Union[list,bool]=[0],offset_axes:int=10,
                    show_legend:bool=True,ls = '-',**kwargs):
    """Plot backward decoding model (BDM) timecourse with multiple 
    visualization modes.

    High-level wrapper for visualizing decoding performance across time 
    with support for 1D timecourse plots, 2D generalization-across-time 
    (GAT) matrices, and 2D time-frequency representations. Integrates 
    statistical testing and significance visualization.

    Parameters
    ----------
    bdms : dict or list of dict
        Decoding model structure(s) containing 'info' and condition 
        keys. Each condition contains 'dec_scores' (decoding scores 
        across time/time-frequency).
        - Single BDM: dict with keys like {'cond1': {'dec_scores': ...}, 
          'cond2': {...}, 'info': {...}}
        - Multiple subjects: list of such dicts (averaged across list)
        
        Required 'info' keys:
        - 'times': Timepoints in seconds or milliseconds
        - 'test_times': Test times (for GAT, same length as train times)
        - 'freqs': Frequency values (for TFR)
    cnds : list of str, optional
        Condition names to plot. If None, plots all conditions except 
        'info'. Default: None.
    timecourse : str, default='1d'
        Visualization type:
        - '1d': Diagonal generalization timecourse (extracts diagonal 
           from 2D)
        - '2d_GAT': Full generalization-across-time matrix
        - '2d_tfr': Time-frequency representation matrix
        For 2D modes, only first condition (cnds[0]) is plotted.
    colors : list of str, optional
        Colors for each condition line (1D) or colormap (2D). If None or 
        insufficient for conditions, uses matplotlib tableau colors. 
        Default: None.
    show_SE : bool, default=False
        If True, overlay standard error bands around mean (1D plots 
        only).
    smooth : bool, default=False
        If True, apply Gaussian smoothing to data (both for 
        visualization and statistical testing).
    method : str, default='auc'
        Decoding metric type. Used for axis labeling. Common values: 
        'auc', 'accuracy'. Default: 'auc'.
    chance_level : float, default=0.5
        Chance/baseline performance level. Plotted as horizontal 
        reference line. For AUC, typically 0.5; for accuracy with 2 
        classes, 0.5; for n classes, 1/n.
    stats : str or bool, default='perm'
        Statistical test type for significance overlay:
        - 'perm': Cluster-based permutation test
        - 'ttest': Independent samples t-test
        - 'fdr': False discovery rate correction
        - False: No significance testing
    p_thresh : float, default=0.05
        P-value threshold for significance marking.
    mask_nonsig : bool, default=False
        If True, non-significant voxels displayed in greyscale 
        (2D plots only). Requires significant clusters to be computed.
    diverging_cmap : bool, default=False
        If True, use diverging colormap centered at chance_level 
        (2D plots only). Useful for symmetric visualization around 
        baseline performance.
    p_cluster : float, optional
        Cluster-forming p-value threshold (only for stat_test='perm'). 
        Automatically converts to t-statistic threshold. If None,
        MNE uses automatic threshold. Default: None.
    threshold : float, optional
        Initial threshold for cluster formation (e.g., 
        t-statistic threshold). Default: None.
    onset_times : list of float or bool, default=[0]
        Stimulus/event onset times to mark with vertical lines. Set to 
        False for no onset markers. Default: [0].
    offset_axes : int, default=10
        Y-axis limit extension (percentage) beyond data range. 
        Default: 10.
    show_legend : bool, default=True
        If True, display legend with condition names (1D plots). 
        Default: True.
    ls : str, default='-'
        Line style for 1D timecourse ('-', '--', '-.', ':'). 
        Default: '-'.
    **kwargs
        Additional arguments passed to plot_significance() and plotting 
        functions.

    Returns
    -------
    None
        Modifies current matplotlib figure(s) by plotting timecourse(s).

    Notes
    -----
    Time unit conversion: If times have mean difference < 0.1, assumes 
    input is in seconds and automatically converts to milliseconds.
    
    2D modes and conditions: Each 2D plot (GAT, TFR) can only show one 
    condition. If multiple conditions specified for 2D plot, only first
    is plotted with warning.
    
    Diagonal extraction: For '1d' timecourse with 2D input data 
    (e.g., from GAT), automatically extracts diagonal elements 
    (same training and test time).
    
    Statistical testing: Applied to raw data before smoothing. 
    Smoothing affects visualization but not significance computation.

    Examples
    --------
    Plot 1D timecourse for multiple conditions:

        >>> bdm = {'cond_A': {'dec_scores': scores_A}, 
        ...        'cond_B': {'dec_scores': scores_B},
        ...        'info': {'times': times}}
        >>> plot_bdm_timecourse(bdm, cnds=['cond_A', 'cond_B'])

    Plot 2D GAT matrix with statistical masking:

        >>> bdm = {'cond_A': {'dec_scores': gat_matrix},
        ...        'info': {'times': train_times, 
        ...		   'test_times': test_times}}
        >>> plot_bdm_timecourse(bdm, timecourse='2d_GAT', 
        ...                     mask_nonsig=True, diverging_cmap=True)

    Plot 2D TFR with frequency information:

        >>> bdm = {'cond_A': {'dec_scores': tfr_matrix},
        ...        'info': {'times': times, 'freqs': freqs}}
        >>> plot_bdm_timecourse(bdm, timecourse='2d_tfr', 
        ...                     diverging_cmap=True)

    See Also
    --------
    plot_2d : Core 2D heatmap visualization with masking
    plot_timecourse : 1D timecourse plotting
    plot_significance : Statistical significance overlay
    """

    if isinstance(bdms, dict):
        bdms = [bdms]	
    times = bdms[0]['info']['times']
    # Convert times from seconds to milliseconds if needed
    time_diff = np.diff(times).mean()
    if time_diff < 0.1:  # If average difference < 0.1, assume seconds
        times = times * 1000
        print(f"Times converted from seconds to milliseconds")
    
    if cnds is None:
        cnds = [key for key in bdms[0] if 'info' not in key]

    if timecourse != '1d' and len(cnds) > 1:
        print('2d timecourse only supports one condition. Plotting first ' \
        f'condition only {cnds[0]}')
        cnds = [cnds[0]]

    if colors is None or len(colors) < len(cnds) and timecourse == '1d':
        print('not enough colors specified. Using default colors')
        colors = list(mcolors.TABLEAU_COLORS.values())

    for c, cnd in enumerate(cnds):
        # extract data
        y = np.stack([bdm[cnd]['dec_scores'] for bdm in bdms])
        color = colors[c] 

        # Initialize y_label with default (updated in 2D branches)
        y_label = method

        if timecourse == '1d':
            # Extract diagonal if input is 2D (e.g., from GAT/TFR)
            if y.ndim > 2:
                # Get the minimum dimension to extract diagonal safely
                original_shape = y.shape
                min_dim = min(y.shape[1], y.shape[2])
                y = y[:, np.arange(min_dim), np.arange(min_dim)]
                print(f"Extracted diagonal from 2D GAT data "
                    f"{original_shape[1:]} → {y.shape[1:]} "
                    f"for condition '{cnd}'")

            # do actual plotting
            plot_timecourse(times,y,show_SE,smooth,
                            label=cnd,color=color,ls=ls)
            if stats:		
                plot_significance(times,y,chance_level,
                                 color=color,stats=stats,
                                p_thresh=p_thresh,
                                p_cluster=p_cluster,
                                threshold=threshold,
                                smooth=smooth,**kwargs)
        else: 
            if timecourse == '2d_tfr':
                y_range = bdms[0]['info']['freqs']	
                y_label = 'Frequency (Hz)'
                y_ticks = True
            elif timecourse == '2d_GAT':
                test_times_x = bdms[0]['info']['test_times']
                # Convert test_times to milliseconds if needed (same as times)
                time_diff_test = np.diff(test_times_x).mean()
                if time_diff_test < 0.1:
                    test_times_x = test_times_x * 1000
                # Convert y_range (train times) to milliseconds for consistency
                y_range = bdms[0]['info']['times']
                time_diff_train = np.diff(y_range).mean()
                if time_diff_train < 0.1:
                    y_range = y_range * 1000
                y_label = 'Train time (ms)'
                y_ticks = False
            
            # Calculate stats once if needed
            sig_mask = None
            p_vals = None
            if stats:
                _, sig_mask, p_vals = perform_stats(y, chance_level, stats,
                                                    p_thresh, 
                                                    p_cluster=p_cluster,
                                                    threshold=threshold)
            
            # Plot with optional masking
            # For GAT: use test_times_x for x-axis (test time), 
            # train times for y-axis
            x_vals = test_times_x if timecourse == '2d_GAT' else times
            if mask_nonsig and sig_mask is not None:
                # Step 1: Plot full data in greyscale as background
                plot_2d(y, x_val=x_vals, y_val=y_range, colorbar=False,
                                set_y_ticks=y_ticks, cbar_label=None,
                                mask=None, diverging_cmap=False, 
                                cmap='gray', **kwargs)

                # Step 2: Overlay only significant data in color
                # Use default colormap handling (RdBu_r if center_zero=True)
                plot_2d(y, x_val=x_vals, y_val=y_range, colorbar=True,
                        set_y_ticks=y_ticks, cbar_label=method,
                        mask=sig_mask,  
                        mask_value=np.nan,  # Hide non-significant 
                        p_vals=p_vals, p_thresh=p_thresh,
                        diverging_cmap=diverging_cmap, center=chance_level, 
                        **kwargs)
            else:
                plot_2d(y, x_val=x_vals, y_val=y_range, colorbar=True,
                    set_y_ticks=y_ticks, cbar_label=method,
                    mask=None, diverging_cmap=diverging_cmap, 
                    center=chance_level, **kwargs)
            
            # Add contours for significance
            if stats:
                plot_significance(x_vals, y, chance_level, color='white', 
                                  stats=stats,y_val=y_range, 
                                sig_mask=sig_mask, p_vals=p_vals,
                                **kwargs)
        
    # fine tune plot	
    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc = 'best',
        prop={'size': 7},frameon=False)

    if onset_times and timecourse == '1d':
        for t in onset_times:
            plt.axvline(t,color = 'black',ls='--',lw=1)
    
    if timecourse == '2d_GAT':
        plt.xlabel('Test time (ms)')
    else:
        plt.xlabel('Time (ms)')
    plt.ylabel(y_label)
    if timecourse == '1d':
        plt.axhline(chance_level, color = 'black', ls = '--', lw=1)

    sns.despine(offset = offset_axes)

def plot_ctf_timecourse(ctfs:Union[list,dict],cnds:list=None,colors:list=None,
                        show_SE:bool=False,smooth:bool=False,timecourse:str='1d',
                        output:str='raw_slopes',band_oi: str=None,
                        stats:Union[str,bool]='perm',p_thresh:float=0.05,
                        p_cluster:Optional[float]=None,
                        threshold:Optional[float]=None,
                        chance_level:float=0,
                        mask_nonsig:bool=False,diverging_cmap:bool=False,
                        onset_times:Union[list,bool]=[0],offset_axes:int=10,
                        show_legend:bool=True,avg_bins:bool=False,**kwargs):
    """Plot spatial channel tuning function (CTF) timecourse with 
    multiple visualization modes.

    High-level wrapper for visualizing CTF analysis results across time
    with support for 1D timecourse plots, 2D generalization-across-time
    matrices, and 2D time-frequency representations. Integrates 
    statistical testing and significance visualization.

    Parameters
    ----------
    ctfs : dict or list of dict
        CTF analysis structure(s) containing 'info' and condition keys.
        Each condition contains output keys (e.g., 'raw_slopes',
        'param_slopes') with CTF data arrays.
        - Single CTF: dict with keys like {'cond1': {'raw_slopes': ...},
          'cond2': {...}, 'info': {...}}
        - Multiple subjects: list of such dicts (averaged across list)
        
        Required 'info' keys:
        - 'times': Timepoints in seconds or milliseconds
        - 'freqs': Frequency bands (for TFR data)
        - 'bands': Band names (for multiple frequency bands)
    cnds : list of str, optional
        Condition names to plot. If None, plots all conditions except
        'info'. Default: None.
    colors : list of str, optional
        Colors for each condition line (1D) or colormap (2D). If None or
        insufficient for conditions, uses matplotlib tableau colors.
        Default: None.
    show_SE : bool, default=False
        If True, overlay standard error bands around mean (1D plots
        only).
    smooth : bool, default=False
        If True, apply Gaussian smoothing to data (both for
        visualization and statistical testing).
    timecourse : str, default='1d'
        Visualization type:
        - '1d': Timecourse with optional frequency bins
        - '2d_gat': Generalization-across-time matrix (train vs 
          test time)
        - '2d_tfr': Time-frequency representation matrix
        - '2d_ctf': 2D channel tuning function (spatial x time)
        For 2D modes, only first condition is plotted.
    output : str or list, default='raw_slopes'
        Output types to plot (e.g., 'raw_slopes', 'param_slopes').
        Can be single string or list for multiple overlaid outputs.
    band_oi : str, optional
        Frequency band of interest to plot. If None and multiple bands
        exist, averages across all bands. Default: None.
    stats : str or bool, default='perm'
        Statistical test type for significance overlay:
        - 'perm': Cluster-based permutation test
        - 'ttest': Independent samples t-test
        - 'fdr': False discovery rate correction
        - False: No significance testing
    p_thresh : float, default=0.05
        P-value threshold for significance marking.
    p_cluster : float, optional
        Cluster-forming p-value threshold (only for stat_test='perm').
        Automatically converts to t-statistic threshold. If None,
        MNE uses automatic threshold. Default: None.
    threshold : float, optional
        Initial threshold for cluster formation (e.g., t-statistic
        threshold). Default: None.
    chance_level : float, default=0
        Baseline/reference level to compare against. Plotted as 
        horizontal reference line (1D plots). Default: 0 (testing for 
        deviation from zero).
    mask_nonsig : bool, default=False
        If True, non-significant voxels displayed in greyscale
        (2D plots only). Requires significant clusters to be computed.
    diverging_cmap : bool, default=False
        If True, use diverging colormap centered at chance_level
        (2D plots only). Useful for symmetric visualization around
        baseline.
    onset_times : list of float or bool, default=[0]
        Stimulus/event onset times to mark with vertical lines 
        (1D plots).
        Set to False for no onset markers. Default: [0].
    offset_axes : int, default=10
        Y-axis limit extension (percentage) beyond data range.
        Default: 10.
    show_legend : bool, default=True
        If True, display legend with condition/output names (1D plots).
        Default: True.
    avg_bins : bool, default=False
        If True and data contains spatial bins, average across bins
        (removes zero-value bins). Used for 1D plotting. Default: False.
    **kwargs
        Additional arguments passed to plot_2d(), plot_significance(),
        and plotting functions.

    Returns
    -------
    None
        Modifies current matplotlib figure(s) by plotting CTF 
        timecourse(s).

    Notes
    -----
    Time unit conversion: If times have mean difference < 0.1, assumes
    input is in seconds and automatically converts to milliseconds.
    
    2D modes and conditions: Each 2D plot (GAT, TFR, CTF) can only show
    one condition. If multiple conditions specified for 2D plot, only
    first is plotted with warning.
    
    Multiple outputs: When plotting multiple output types (e.g., both
    'raw_slopes' and 'param_slopes'), they are overlaid with different
    line styles and labeled in legend.
    
    Frequency band handling: For time-frequency CTF data with multiple
    frequency bands:
    - If band_oi specified: plots only that band
    - If band_oi=None: averages across all bands (single dimension)
    - For 2D_TFR: Y-axis shows mean frequency per band
    
    Statistical testing: Applied to raw data before smoothing.
    Smoothing affects visualization but not significance computation.

    Examples
    --------
    Plot 1D timecourse for multiple conditions:

        >>> ctf = {'cond_A': {'raw_slopes': slopes_A},
        ...        'cond_B': {'raw_slopes': slopes_B},
        ...        'info': {'times': times}}
        >>> plot_ctf_timecourse(ctf, cnds=['cond_A', 'cond_B'])

    Plot with multiple outputs overlaid:

        >>> ctf = {'cond_A': {'raw_slopes': slopes_A,
        ...                    'param_slopes': param_slopes_A},
        ...        'info': {'times': times}}
        >>> plot_ctf_timecourse(ctf, output=['raw_slopes', 
        ...                   'param_slopes'])

    Plot 2D GAT matrix with statistical masking:

        >>> ctf = {'cond_A': {'raw_slopes': gat_matrix},
        ...        'info': {'times': train_times}}
        >>> plot_ctf_timecourse(ctf, timecourse='2d_gat',
        ...                     mask_nonsig=True, diverging_cmap=True)

    Plot 2D TFR with frequency information:

        >>> ctf = {'cond_A': {'raw_slopes': tfr_matrix},
        ...        'info': {'times': times, 'freqs': freqs}}
        >>> plot_ctf_timecourse(ctf, timecourse='2d_tfr',
        ...                     diverging_cmap=True)

    See Also
    --------
    plot_2d : Core 2D heatmap visualization with masking
    plot_timecourse : 1D timecourse plotting
    plot_significance : Statistical significance overlay
    """
    
    if isinstance(ctfs, dict):
        ctfs = [ctfs]
    times = ctfs[0]['info']['times']

    # Convert times from seconds to milliseconds if needed
    time_diff = np.diff(times).mean()
    if time_diff < 0.1:  # If average difference < 0.1, assume seconds
        times = times * 1000
        print(f"Times converted from seconds to milliseconds")

    if cnds is None:
        cnds = [key for key in ctfs[0] if 'info' not in key]

    if timecourse != '1d' and len(cnds) > 1:
        print('2d timecourse only supports one condition. Plotting first ' \
        f'condition only {cnds[0]}')
        cnds = [cnds[0]]

    if isinstance(output, str):
        output = [output]

    if colors is None or len(colors) < len(cnds) and timecourse == '1d':
        print('not enough colors specified. Using default colors')
        colors = list(mcolors.TABLEAU_COLORS.values())

    if band_oi is not None:
        band_idx = ctfs[0]['info']['bands'].index(band_oi)

    ylabel = f'CTF slope (au) - {band_oi}' if band_oi is not None \
                                            else 'CTF slope (au)'
    for c, cnd in enumerate(cnds):
        color = colors[c] 
        for o, out in enumerate(output):
            # extract data
            y = np.stack([ctf[cnd][out] for ctf in ctfs])
            if len(output) > 1:
                label = f'{cnd} - {out}'
            else:
                label = cnd

            # Select frequency band if needed (applies to both 1d and 2d_gat)
            if timecourse in ['1d', '2d_gat']:
                if y.shape[1]> 1:
                    if band_oi is not None:
                        y = y[:,band_idx,:]
                    else:
                        Warning('Multiple frequency bands detected but no ' \
                        'band_oi specified. Averaging across all frequency ' \
                        'bands.')
                        y = y.mean(axis=1)
                else:
                    y = np.squeeze(y,axis=1)

            # do actual plotting
            if timecourse == '1d':
                if y.ndim > 2 and avg_bins:
                    y = y[:,:,~np.all(y == 0, axis=(0,1))].mean(axis=-1)
                if y.ndim > 2:
                    for b in range(y.shape[-1]):
                        y_ = y[:,:,b]
                        if not np.all(y_ == 0):
                            bin_label = f'{label} - bin_{b}'
                            plot_timecourse(times,y_,show_SE,smooth,
                                        label=bin_label,color=colors[b],
                                        ls=['-','--'][o])
                else:
                    plot_timecourse(times,y,show_SE,smooth,
                                    label=label,color=color,ls=['-','--'][o])

                if stats:
                    #TODO: make also work for individual bins	
                    plot_significance(times,y,chance_level,
                                    color=color,stats=stats,
                                    smooth=smooth,p_cluster=p_cluster,
                                    threshold=threshold,**kwargs)
            elif timecourse == '2d_tfr' or timecourse == '2d_gat':
                if timecourse == '2d_tfr':
                    freqs = ctfs[0]['info']['freqs']	
                    y_range = [np.mean(band) for band in freqs]
                    ylabel = 'Frequency (Hz)'
                    y_ticks = True
                else:  # 2d_gat
                    y_range = times
                    ylabel = 'Train time (ms)'
                    y_ticks = False
                
                # Calculate stats once if needed
                sig_mask = None
                p_vals = None
                if stats:
                    _, sig_mask, p_vals = perform_stats(y, chance_level, stats, 
                                                         p_thresh, p_cluster=p_cluster,
                                                         threshold=threshold)
                
                # Plot with optional masking
                plot_2d(y, x_val=times, y_val=y_range, colorbar=True,
                           set_y_ticks=y_ticks, cbar_label='CTF slope',
                        mask=sig_mask if mask_nonsig else None,
                        mask_value=0, p_vals=p_vals, p_thresh=p_thresh,
                        diverging_cmap=diverging_cmap, **kwargs)
            
                # Add contours only if not masking (when masking, zeros show significance)
                if stats and not mask_nonsig:
                    plot_significance(times, y, 0, color='white', stats=stats,
                                    y_val=y_range, sig_mask=sig_mask, 
                                    p_vals=p_vals, **kwargs)	
                    
            elif timecourse == '2d_ctf':
                if y.shape[1]> 1:
                    Warning('2d CTF timecourse only supports single output.' \
                    ' Plotting first output only.')
                    y = y[:,0]
                else:
                    y = np.squeeze(y,axis=1)
                if y.ndim > 3:
                    Warning('2d CTF timecourse only supports single channel.' \
                    'Individual channels will be averaged.')
                    y = y.mean(axis=-2)
                if y.shape[-1]%2 == 0:
                    y = np.concatenate([y, y[:, :, 0:1] ], axis=2)
                y = np.swapaxes(y,1,2)
                y_range = np.linspace(-180,180,y.shape[1])
                ylabel = 'Channel offset (deg)'
                plot_2d(y,x_val=times,y_val=y_range,colorbar=True,
                           set_y_ticks=False,
                           cbar_label='Channel response',**kwargs)

    # fine tune plot	
    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc = 'best',
        prop={'size': 7},frameon=False)
    
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    if timecourse == '1d':
        if onset_times:
            for t in onset_times:
                plt.axvline(t,color = 'black',ls='--',lw=1)
        plt.axhline(chance_level, color = 'black', ls = '--', lw=1)

    sns.despine(offset = offset_axes)
    
def plot_erp_topography(erps:Union[list,dict],times:np.array,
                        window_oi:tuple=None,cnds:list=None,
                        topo:str='raw',montage:str='biosemi64',**kwargs):

    if isinstance(erps, list):
        erps = {'temp':erps}

    if cnds is not None:
        erps = {key:value for (key,value) in erps.items() if key in cnds}

    if window_oi is None:
        window_oi = (times[0],times[-1])
    idx = get_time_slice(times, window_oi[0],window_oi[1])

    for c, cnd in enumerate(erps.keys()):
        ax = plt.subplot(1,len(erps), c+1, title = cnd)
        _, evoked = ERP.group_erp(erps[cnd],set_mean=True)
        data = evoked._data[:,idx].mean(axis = 1)

        if topo == 'diff':
            preflip = np.copy(data)
            # visualize contra vs. ipsi
            ch_names = evoked.ch_names
            pairs = get_diff_pairs(montage, ch_names)
            # flip data
            for el, pair in pairs.items():
                data[ch_names.index(el)] = preflip[pair[0]] - preflip[pair[1]]

        # do actual plotting
        plot_topography(data,montage=montage,axes=ax,**kwargs)

def plot_topography(X:np.array,ch_types:str='eeg',montage:str='biosemi64',
                    sfreq:int=512.0,**kwargs):

    # create montage 
    ch_names = mne.channels.make_standard_montage(montage).ch_names
    info = mne.create_info(ch_names, ch_types=ch_types,sfreq=sfreq)
    info.set_montage(montage)

    # do actuall plotting
    mne.viz.plot_topomap(X, info,**kwargs)