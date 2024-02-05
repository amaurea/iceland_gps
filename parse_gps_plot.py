import argparse, os, errno
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
args = parser.parse_args()
import numpy as np, glob, numba, pytesseract, datetime, re
from scipy import ndimage
from PIL import Image

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def mkdir(path):
	if path == "": return
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
def replace(istr, ipat, repl):
	ostr = istr.replace(ipat, repl)
	if ostr == istr: raise KeyError("Pattern not found")
	return ostr

def ocr(data):
	res = pytesseract.image_to_data(data, output_type="data.frame")
	res = res[res["conf"]>=0]
	return res

def ocr_column(data):
	res = pytesseract.image_to_data(data, output_type="data.frame", config=r"--psm 6")
	res = res[res["conf"]>=0]
	return res

reader = None
def ocr2(data):
	import easyocr
	global reader
	if reader is None:
		reader = easyocr.Reader(["en"])
	pass

@numba.njit
def _longest_runs(mask, out):
	ny, nx = mask.shape
	for y in range(ny):
		nbest = 0
		nrun  = 0
		for x in range(1,nx):
			if mask[y,x]:
				nrun += 1
			else:
				if nrun > nbest:
					nbest = nrun
				nrun = 0
		out[y] = nbest

def longest_runs(mask, axis=0):
	axis  = axis % mask.ndim
	mflat = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis])
	out   = np.zeros(mflat.shape[:-1], int)
	_longest_runs(mflat, out)
	return out.reshape(mask.shape[:axis]+mask.shape[axis+1:])

@numba.njit
def find_runs(mask1d):
	out   = []
	start = -1
	for i in range(len(mask1d)):
		if mask1d[i]:
			if start < 0:
				start = i
				count = 0
			count += 1
		else:
			if start >= 0:
				out.append((start,count))
				start = -1
	if start >= 0:
		out.append((start,count))
	return np.array(out)

def find_first(mask, axis=0):
	return np.argmax(mask!=0, axis)

def find_last(mask, axis=0):
	axis = axis % mask.ndim
	return (mask.shape[axis]-1)-find_first(mask[(slice(None),)*axis+(slice(None,None,-1),)], axis)

@numba.njit
def _find_first_after(mask, inds, n, step):
	out = np.full(len(inds),-1,int)
	for y in range(mask.shape[0]):
		for x in range(inds[y], n, step):
			if mask[y,x]:
				out[y] = x
				break
	return out

def find_first_after(mask, inds, axis=0):
	fmask = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis])
	finds = inds.reshape(-1)
	fres  = _find_first_after(fmask, finds, fmask.shape[-1], 1)
	return fres.reshape(inds.shape)

def find_first_before(mask, inds, axis=0):
	fmask = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis])
	finds = inds.reshape(-1)
	fres  = _find_first_after(fmask, finds, -1, -1)
	return fres.reshape(inds.shape)

def linear_fit(points):
	x, y = points
	B = np.array([x*0+1,x])
	return np.linalg.solve(B.dot(B.T), B.dot(y))

def robust_fit(points, tol=5):
	if len(points) < 2: raise ValueError("Too few points for linear fit")
	# We can recover missing signs by using our knowledge that the y values
	# should be in descending order
	yabs  = np.abs(points[1])
	diffs = yabs[1:]-yabs[:-1]
	# If all values are > 0, then diffs will all be negative
	# If all values are < 0 then diffs will all be positive
	# If we cross zero somewhere, then diffs will change sign there.
	# We can find the start of the negative region by finding the first
	# non-negative diff
	ineg = np.where(diffs>=0)[0]
	if len(ineg) == 0: ineg = 0
	else: ineg = ineg[0]
	yfix   = yabs.copy()
	yfix[ineg:] *= -1
	# Now that we have recovered the sign, do a standard fit
	fit = linear_fit([points[0],yfix])
	# Check our chisquare in pixel-space
	model = (yfix-fit[0])/fit[1]
	rms   = np.mean((points[0]-model)**2)**0.5
	if rms > tol: raise ValueError("Bad fit")
	return fit

def find_lines(mask, minlen=500, maxwidth=8, axis=0):
	runs = longest_runs(mask, axis)
	# Disqualify too short runs
	good = runs > minlen
	assert good.ndim == 1
	lines = find_runs(good)
	# Disqualify too thick lines
	lines = lines[lines[:,1]<=maxwidth]
	return lines

def find_plots(data):
	black = np.all(data==0,2)
	# Horizontal graph edges
	xlines = find_lines(black, axis=1, minlen=data.shape[1]*3//4)
	ylines = find_lines(black, axis=0, minlen=data.shape[0]//4)
	assert len(ylines) == 2, "Expected one plot horizontally"
	assert len(xlines) & 1 == 0, "Expected a whole number of plots"
	return np.array([[xlines[2*i:2*i+2], ylines] for i in range(len(xlines)//2)])

def find_first_between(edges, poss):
	for i in range(len(edges)-1):
		for j, x in enumerate(poss):
			if edges[i] < x and x < edges[i+1]:
				return i, j

def find_year(data, ymax):
	res = ocr(data[:ymax])
	for text in res["text"]:
		m = re.match(r"\d+/\d+/(\d\d\d\d)", text)
		if m: return int(m.group(1))
	raise ValueError("Could not infer year. ocr was\n" + str(res))

def year_month_ctime(year, month):
	d = datetime.datetime(year+month//12, month%12+1, 1, 0, 0, 0, tzinfo=datetime.UTC)
	return d.timestamp()

def measure_xticks(data, plot, year, axwidth=100):
	# Start by finding the major x ticks
	(top,bottom),(left,right) = plot
	y1 = bottom[0]+bottom[1]
	y2 = y1+axwidth
	x1 = left[0]
	x2 = right[0]+right[1]
	axdata = data[y1:y2,x1:x2]
	dark   = np.mean(axdata,2) < 128
	# Find the separation between the axis ticks and the label text
	ay1, ay2 = np.where(np.all(~dark,1))[0][[0,-1]]
	# Only the major x ticks will extend this far. Use a margin of 2
	# to allow for antialiasing
	ticks = find_runs(dark[ay1-2])
	ticks = ticks[:,0]+(ticks[:,1]-1)/2
	# We now have the relative x positions of the major x ticks.
	# Next we want to try to measure the axis labels.
	res   = ocr(axdata[ay1:ay2])
	# They must be a month name
	good  = [v in months for v in res["text"]]
	res   = res[good]
	month_i = [months.index(v) for v in res["text"]]
	month_x = res.left + (res.width-1)/2
	# We just need to identify a single month with an x tick to
	# know how to interpret them
	ref_tick, ref_label = find_first_between(ticks, month_x)
	tick_months  = np.arange(len(ticks))+month_i[ref_label]-ref_tick
	# Normalize to the last one being in [0:11]
	tick_months -= tick_months[-1]//12*12
	# Need to translate months to dates, and for this we need the year.
	# We can parse this from the header in a separate function.
	tick_times = [year_month_ctime(year, m) for m in tick_months]
	# Translate to absolute x
	ticks += x1
	# return as [{x,ctime},:]
	return np.array([ticks, tick_times])

def fixnum(val):
	try: return float(val)
	except ValueError: return float(val.replace("~","-"))

def isnum(val):
	try:
		fixnum(val)
		return True
	except ValueError:
		return False

def find_pairs(a1, a2, tol=1):
	pairs = []
	for i1, v1 in enumerate(a1):
		besti, bestv = 0, np.inf
		for i2, v2 in enumerate(a2):
			dist = np.abs(v1-v2)
			if dist < bestv and dist <= tol:
				besti, bestv = i2, dist
		if bestv <= tol:
			pairs.append((i1,besti))
	return pairs

def measure_yticks(data, plot, margin=25):
	(top,bottom),(left,right) = plot
	y1 = max(top[0] - margin, 0)
	y2 = min(bottom[0]+bottom[1] + margin, data.shape[0])
	x1 = 0
	x2 = left[0]-1
	axdata = data[y1:y2,x1:x2]
	# Split into the plot label, tick labels and ticks
	dark   = np.mean(axdata,2) < 128
	# First non-blank column
	ax0 = np.where(np.any(dark,0))[0][0]
	# First and last blank column after this
	ax1, ax2 = np.where(np.all(~dark[:,ax0:],0))[0][[0,-1]]+ax0
	# Major x ticks will be just after ax2
	ticks = find_runs(dark[:,ax2+2])
	ticks = ticks[:,0]+(ticks[:,1]-1)/2
	# Now try to measure the axis labels
	res   = ocr_column(axdata[:,ax1:ax2])
	#Image.fromarray(axdata[:,ax1:ax2]).save("test_img.png")
	good  = [isnum(v) for v in res["text"]]
	res   = res[good]
	vals  = [fixnum(v) for v in res["text"]]
	pos   = np.array(res.top + (res.height-1)/2)
	pairs = find_pairs(ticks, pos, 10)
	# Restore global coordinates
	ticks += y1
	# Output [{y,val},:]
	out   = np.array([(ticks[i1],vals[i2]) for i1,i2 in pairs]).T
	return out

def parse_color(color):
	return [int(color[2*i:2*i+2],16) for i in range(3)]

def measure_points(data, plot, colors=["ff0000","00ff00","0000ff","00ffff"], atol=0.75):
	# Cut out our content area
	(top,bottom),(left,right) = plot
	y1 = sum(top)
	y2 = bottom[0]-1
	x1 = sum(left)
	x2 = right[0]-1
	pdata = data[y1:y2,x1:x2]
	# Find the plot point color. This is probably the most common
	# non-black/white color in the plot, but things like the university
	# logo complicate this, so let's just look through an explicit list of colors
	cvals  = [parse_color(color) for color in colors]
	cnums  = np.array([np.sum(np.all(pdata==cval,2)) for cval in cvals])
	cind   = np.argmax(cnums)
	# Find each point. They're surrounded by a different-colored contour,
	# so this should be safe
	labels, nlabel = ndimage.label(np.all(pdata==cvals[cind],2))
	allofthem = np.arange(1,nlabel+1)
	# Find their bounding boxes. Sadly these come as slices, so we
	# must loop in python to get them in a mathable format
	slices = ndimage.find_objects(labels)
	boxes  = np.array([[[a.start,a.stop] for a in s] for s in slices])
	boxes  = np.moveaxis(boxes, 1, 2) # [nobj,{from,to},{y,x}]
	# Boxes will be cut off at the right side, but we know that they
	# should be squares, to restore them. This is only accurate to
	# around one pixel
	height = np.median(boxes[:,1,0]-boxes[:,0,0])
	boxes[:,1,1] = boxes[:,0,1] + height
	# The coordinates will be the center of each box
	poss   = (boxes[:,0]+boxes[:,1]-1)/2
	# Reject points with too low area. This can happen due to the university
	# of iceland logo
	areas  = ndimage.sum_labels(labels*0+1,labels,allofthem)
	good   = areas > np.median(areas)*atol
	poss, boxes = poss[good], boxes[good]
	npoint = len(poss)
	# Try to measure error bars. Ideally this would be as simple as finding
	# the last and first black pixel in the point's column, but due to close
	# points the error bar end-bars can (and usually do overlap). We will still
	# use to to check if we even have any error bars though, by just counting
	# black pixels in these columns.
	iy, ix  = np.round(poss.T).astype(int)
	subblack= np.all(pdata[:,ix]==0,2)
	nblack  = np.sum(subblack,0) 
	if np.mean(nblack>2) < atol:
		# No visible error bars. Just use point radius
		yerr = np.full(npoint,height/2)
	else:
		# Ok, we probably do have error bars. Instead of lookng for the endbars
		# we will look for the solid black columns of the error bars themselves.
		# We need some x tolerance for this because we could have gotten the point
		# position wrong by one pixel
		ptol  = 3
		ylow  = iy.copy()
		yhigh = iy.copy()
		order = np.argsort(ix)
		for xoff in [-1,0,1]:
			ix2      = ix + xoff
			subblack = np.all(pdata[:,ix2]==0,2)
			# We want a continuous black line from height/2+tol above the point
			# and up, and similarly below
			ylow1  = find_first_before(~subblack, iy-height//2-ptol, axis=0)
			yhigh1 = find_first_after (~subblack, iy+height//2+ptol, axis=0)
			# These will be -1 if nothing was found
			ylow [ ylow1>=0] = np.minimum(ylow [ ylow1>=0], ylow1 [ ylow1>=0])
			yhigh[yhigh1>=0] = np.maximum(yhigh[yhigh1>=0], yhigh1[yhigh1>=0])
		# Positive and negative error bars should have same size
		ypos   = yhigh-poss[:,0]
		yneg   = poss[:,0]-ylow
		ysym   = 0.5*(ypos+yneg)
		# Positive and negative error bars should agree
		ratio  = ypos/yneg
		good   = (ratio > atol)&(ratio<1/atol)
		yerr   = np.full(npoint, np.median(ysym[good]))
		yerr[good] = ysym[good]
	# Expand to full coordinates
	poss += [y1,x1]
	# Return as [{y,x,dy},:]
	return np.array([poss[:,0],poss[:,1],yerr])

ifiles = sum([sorted(glob.glob(fname)) for fname in args.ifiles],[])
mkdir(args.odir)
for fi, ifile in enumerate(ifiles):
	print(os.path.basename(ifile))
	data  = np.array(Image.open(ifile).convert("RGB"))
	# Find the plot borders. These will be fully black horizontal and vertical
	# lines. Horizontally we will have have lines > 75% of the image. Vertically
	# it will be 1/3 of that since we have three plots stacked
	plots   = find_plots(data)
	plotsep = plots[1][0][0][0]-sum(plots[0][0][1]) if len(plots) > 1 else sum(plots[0][0][1])-data.shape[0]
	year    = find_year(data, plots[0][0][0][0])
	for pi, plot in enumerate(plots):
		xticks = measure_xticks(data, plot, year, plotsep)
		xscale = linear_fit(xticks)
		yticks = measure_yticks(data, plot)
		yscale = robust_fit(yticks)
		points = measure_points(data, plot)
		# Scale points to physical
		v  = yscale[0]+yscale[1]*points[0]
		dv = yscale[1]*points[2]
		t  = xscale[0]+xscale[1]*points[1]
		# And output
		ofile = args.odir + "/" + ".".join(os.path.basename(ifile).split(".")[:-1]) + "_%d.txt" % pi
		np.savetxt(ofile, np.array([t, v, dv]).T, fmt="%10.0f %8.3f %7.3f")
