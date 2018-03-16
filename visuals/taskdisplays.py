import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt

from IPython import embed 
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import PatchCollection

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
	"""
	Make a scatter of circles plot of x vs y, where x and y are sequence 
	like objects of the same lengths. The size of circles are in data scale.

	Parameters
	----------
	x,y : scalar or array_like, shape (n, )
	    Input data
	s : scalar or array_like, shape (n, ) 
	    Radius of circle in data unit.
	c : color or sequence of color, optional, default : 'b'
	    `c` can be a single color format string, or a sequence of color
	    specifications of length `N`, or a sequence of `N` numbers to be
	    mapped to colors using the `cmap` and `norm` specified via kwargs.
	    Note that `c` should not be a single numeric RGB or RGBA sequence 
	    because that is indistinguishable from an array of values
	    to be colormapped. (If you insist, use `color` instead.)  
	    `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
	vmin, vmax : scalar, optional, default: None
	    `vmin` and `vmax` are used in conjunction with `norm` to normalize
	    luminance data.  If either are `None`, the min and max of the
	    color array is used.
	kwargs : `~matplotlib.collections.Collection` properties
	    Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
	    norm, cmap, transform, etc.

	Returns
	-------
	paths : `~matplotlib.collections.PathCollection`

	Examples
	--------
	a = np.arange(11)
	circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
	plt.colorbar()

	License
	--------
	This code is under [The BSD 3-Clause License]
	(http://opensource.org/licenses/BSD-3-Clause)
	"""

	try:
		basestring
	except NameError:
		basestring = str

	if np.isscalar(c):
		kwargs.setdefault('color', c)
		c = None
	if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
	if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
	if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
	if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

	patches = [Circle((x_, y_), s_, fill = False) for x_, y_, s_ in np.broadcast(x, y, s)]
	collection = PatchCollection(patches, **kwargs)
	if c is not None:
		collection.set_array(np.asarray(c))
		collection.set_clim(vmin, vmax)

	ax = plt.gca()
	ax.add_collection(collection)
	ax.autoscale_view()
	if c is not None:
		plt.sci(collection)

	return collection

def searchDisplayEEG(ax, fix = True, stimulus = None, erp_type = 'target_loc'):
	'''

	'''

	if erp_type == 'target_loc':
		letter ='T'
	elif erp_type == 'dist_loc':
		letter = 'D'	

	# draw fixation circle
	display = circles(1920/2, 1080/2, 5, color = 'black', alpha = 1)

	if not fix:
		# draw place holders
		x_list = np.array([960.,765.2,765.2,956.,1154.8,1154.8])
		y_list = np.array([765.,652.5,427.5,315.,427.5,652.5])

		display = circles(x_list, y_list, np.array([75,75,75,75,75,75]), color = 'black', alpha = 0.5)

	if stimulus != None:
		plt.text(x_list[stimulus],y_list[stimulus],letter, ha = 'center',va = 'center', size = 20) 

	ax.set_yticklabels([])	
	ax.set_xticklabels([])

	plt.xlim((480,1440))
	plt.ylim((270,810))

