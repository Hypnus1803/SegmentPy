import numpy as np
from skimage.util import invert, img_as_float32
from skimage.morphology import reconstruction, square, disk
from skimage.filters.rank import equalize
from skimage.segmentation import watershed
from skimage.measure import label
from shared_state import SharedState


def imhmin(arr, h):
    """
    This function is used to suppress the local minima in the array (arr) using the H-minima transform. 
    The H-minima transform decreases the depth of all regional minima by an amount up to h. As a result, 
    the transform fully suppresses regional minima whose depth is less than h. 
    Function similar to imhmin in matlab.
    :param arr: input array
    :param h: h minima value
    :return: array with suppressed local minima
    Example:
    >>> import numpy as np
    >>> from segmentpy.utilities import imhmin
    >>> a = 10*np.ones([10,10])
    >>> a[1:4,1:4] = 7  
    >>> a[5:8,5:8] = 2
    >>> a[0:3,6:9] = 13
    >>> a[1,7] = 10
    >>> print(a)
        [[10. 10. 10. 10. 10. 10. 13. 13. 13. 10.]
        [10.  7.  7.  7. 10. 10. 13. 10. 13. 10.]
        [10.  7.  7.  7. 10. 10. 13. 13. 13. 10.]
        [10.  7.  7.  7. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10.  2.  2.  2. 10. 10.]
        [10. 10. 10. 10. 10.  2.  2.  2. 10. 10.]
        [10. 10. 10. 10. 10.  2.  2.  2. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]]
    
    >>> h = 4
    >>> output = imhmin(a, h)
    >>> print(output)
        [[10. 10. 10. 10. 10. 10. 13. 13. 13. 10.]
        [10. 10. 10. 10. 10. 10. 13. 13. 13. 10.]
        [10. 10. 10. 10. 10. 10. 13. 13. 13. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10.  6.  6.  6. 10. 10.]
        [10. 10. 10. 10. 10.  6.  6.  6. 10. 10.]
        [10. 10. 10. 10. 10.  6.  6.  6. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
        [10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]]

    """
    im = invert(arr)
    seed = im - h
    suppress = reconstruction(seed, im, footprint=square(3))
    return invert(suppress)

def labels_hmin(image,radius,percentile):
    """
    This function is used to segment the image using the watershed algorithm.
    The function first equalizes the image using the disk structuring element of radius 'radius'.
    Then, it suppresses the local minima in the image using the H-minima transform.
    The function then applies the watershed algorithm to segment the image.
    :param image: input image
    :param radius: radius of the disk structuring element. This radius depends strongly of the scale resolution of the image.
    :param percentile: percentile value for the H-minima transform
    :return: segmented image
    """
    eq_im = equalize(invert(image), disk(radius))
    markers = imhmin(eq_im, np.percentile(eq_im, percentile))
    ws2 = watershed(markers, connectivity=2)
    labels = label(ws2)
    return labels

def normalize_labels(image,labels):
    """
    This function is used to normalize the labels in the image.
    The function first creates a mask of the image using the labels.
    Then, it normalizes the labels using the minimum and maximum values of the image.
    :param image: input image
    :param labels: input labels
    :return: normalized labels
    """
    n_labels = len(np.unique(labels))
    inos = np.zeros_like(image, dtype='int32')

    for l in range(1, n_labels):
        # wn = np.where(labels==l)
        ii = np.float32(image[labels == l])
        i0, i1 = np.nanmin(ii), np.nanmax(ii)
        if i0 < i1:
            inos[labels == l] = np.int32((0.5 + 100 * (ii - i0) / (i1 - i0)))
        else:
            inos[labels == l] = 100

    return inos

def merge_labels(image, normed_labels, labels,ref_level, top_thresh, bright, px_thresh, ref_thresh):
    """
    Merge adjacent segmented features in an image based on normalized reference levels
    and intensity-based criteria.

    Parameters
    ----------
    image : ndarray
        The original grayscale image used to compute intensity-based merging conditions.
    
    normed_labels : ndarray
        Segmented label image normalized to a specific reference level for merging (imer).
        Used to identify features at the normalized reference intensity level.
    
    labels : ndarray
        Original labeled segmentation map (prior to normalization). Tags are used to define
        distinct features and track merging updates.
    
    ref_level : float
        Normalized reference level (in percent) used as the target for shrinking features before merging.
        Corresponds to (imer); converted internally as lmer = 100 * ref_level.

    top_thresh : float
        Threshold intensity level (ltop in %) for detecting bright pixels within a feature.
        Used to exclude features from merging if their top pixel count exceeds px_thresh.

    bright : int
        Pixel count threshold (ntop) for bright regions.
        Features exceeding this number of high-intensity pixels (above top_thresh) will be excluded from merging.

    px_thresh : int
        Minimum number of adjacent contour pixels required between features to be eligible for merging (nmer).

    ref_thresh : float
        Threshold for the scaled mean intensity of features (imex). Features with mean intensities
        exceeding this value will be excluded from merging.

    Returns
    -------
    merged_labels : ndarray
        Updated label image with merged regions based on contour connectivity and intensity-based exclusion criteria.

    Notes
    -----
    - Inner contours of features are analyzed and tagged to determine adjacency at the merging level.
    - Adjacent features are merged only if:
        (1) They share at least `px_thresh` adjacent contour pixels.
        (2) The number of bright pixels per feature does not exceed `bright` above `top_thresh`.
        (3) The scaled intensity mean is below `ref_thresh`.
    - Features are grouped and iteratively merged using tag-combination logic.
    - Progress can be optionally visualized using a progress bar.
    """
    yo, xo = image.shape
    yl, xl = labels.shape
    n_labels = len(np.unique(labels))
    adc = np.array([0, 1, -1, 1, 0, -1, 1, -1])
    adr = np.array([1, -1, 0, 1, -1, 1, 0, -1])
    cb0 = 1 + adc
    cb1 = xo - 2 + adc
    rb0 = 1 + adr
    rb1 = yo - 2 + adr

    ### Para crear CCO ##############
    ref0 = labels[1:-1, 1:-1]
    cco = np.minimum(ref0, 0)

    for i in range(0, 8,2):
        cco = np.maximum(cco,np.int32(labels[rb0[i]:rb1[i] + 1, cb0[i]:cb1[i] + 1] != ref0)*ref0)
    cco = cco > 0

    merged = labels * np.int32(normed_labels >= ref_level) #np.multiply(labels, (normed_labels > ref_level))  # ref == lmer
    ictg = np.zeros_like(labels) - 1
    adtg = ictg.copy()

    ref0 = merged[1:-1, 1:-1]
    con = np.zeros([yl - 2, xl - 2], dtype='int32')
    adj = con.copy()

    for i in range(0,8,2):
        sur = merged[rb0[i]: rb1[i] + 1, cb0[i]:cb1[i] + 1]
        adj = np.maximum(adj,(sur != ref0)*sur) 
        con = np.maximum(con,(sur != ref0)*ref0)

    ictg[1: yl - 1, 1: xl - 1] = con.copy()
    adtg[1: yl - 1, 1: xl - 1] = (con != 0) * adj

    # NON-MERGING
    cmex = np.zeros(n_labels, dtype='int32')
    fmex = np.minimum(image, 0)
    misc = np.zeros(n_labels, dtype='float32')

    iscl = np.sum(image) / image.size

    for n in range(1, n_labels):
        wn = labels == n
        nwn = np.count_nonzero(wn)
        cmex[n] = np.count_nonzero(normed_labels[wn] >= top_thresh)  # top == ltop
        fmex[wn] = cmex[n]
        misc[n] = np.sum(image[wn]) / nwn / iscl

    # use (adtg),(ictg)
    cotg = np.zeros([n_labels + 1, n_labels + 1], dtype='int32')
    wa = np.where(adtg > 0)
    nwa = len(wa[0])
    for p in range(0, nwa):  # p=0L, nwa-1 do begin
        c = int(ictg[wa[0][p], wa[1][p]])
        a = int(adtg[wa[0][p], wa[1][p]])
        cotg[c, a] = cotg[c, a] + 1

    # require minimum number (nmer)
    cotg = np.int32(cotg >= ref_thresh)  # refth == nmer
    imex = bright
    ntop = px_thresh


    wi = np.maximum(np.where(np.logical_or(misc >= imex,cmex <= ntop)),0)

    cotg[wi, :] = 0
    cotg[:, wi] = 0

    # for n in range(1, n_labels + 1): cotg[n, n] = 1
    for n in range(1, n_labels): cotg[n, n] = 1

    while True:
        ptg = cotg.copy()
        for n in range(1,n_labels+1):
            wm, = np.where(cotg[n,:] == 1)
            mt = len(wm)
            for m in range(mt-1):
                cotg[wm[m+1:],wm[m]] = 1
        cotg = np.maximum(cotg,np.transpose(cotg))

        if np.min([ptg == cotg]) == True:
            break

        # ptg = cotg.copy()
        # if np.min([cotg == ptg]) == False:
        #     for n in range(1, n_labels + 1):
        #         wm = np.where(cotg[n, :] == 1)
        #         mt = len(wm[0])
        #         for m in range(0, mt - 1):
        #             cotg[wm[m + 1:], wm[m]] = 1
        #     cotg = np.maximum(cotg, np.transpose(cotg))
        # else:
        #      break

    mtag = np.zeros([n_labels + 1, 200], dtype='int32')
    dm = 2

    for n in range(1, n_labels + 1):
        wm = np.where(cotg[n, :] == 1)
        nm = len(wm[0])
        mtag[n, 1:nm + 1] = wm[0]
        mtag[n, 0] = nm
        dm = max(dm, nm)
    mtag = mtag[:, 0:dm + 3]
    del cotg

    ceco = np.minimum(labels, 0)

    mtag[:, 0] = 1

    wt, = np.where(mtag[:, 0] != 0)
    mt = len(wt)

    while mt > 0:
        n = int(wt[0])
        m = 1
        while mtag[n, m] != 0:
            ceco = np.maximum(ceco, n* np.int32(labels == mtag[n, m]))
            m += 1

        for t in range(0, mt):
            mtag[wt[t], 0] = np.max(mtag[wt[t], :] != mtag[n, :])
        wt, = np.where(mtag[:, 0] != 0)
        mt = len(wt)

    labels = ceco.copy()

    ref0 = ceco[1: yo - 1, 1:xo - 1]
    con = np.minimum(ref0, 0)

    for i in range(0, 8, 2):
        con = np.maximum(con,(ceco[rb0[i]:rb1[i] + 1, cb0[i]:cb1[i] + 1] != ref0)*ref0) 

    ceco = np.minimum(0, ceco)

    ceco[1: yo - 1, 1:xo - 1] = np.int32(con > 0) + np.int32(cco > 0)

    wt = np.where(labels > 0)
    nwt = len(wt[0])

    if nwt == 0:
        raise ValueError("EXIT: no feature left after merging")
    else:
        tg = np.zeros(32767, dtype='int32')  # Check why 32767
        tg[labels[wt]] = 1
        tg, = np.where(tg == 1)
    return ceco, fmex, tg, labels

def shrinking_labels(image, normed_labels, labels, tags, cutoff):


    n_labels = len(np.unique(labels))
    inom = np.minimum(normed_labels, 0)
    ftag = np.minimum(labels, 0)
    csiz = ftag.copy()

    # for t in range(self.n_labels):
    for t in range(len(tags)):
        wm = np.where(labels == tags[t])
        # nwm = len(wm[0])
        ii = img_as_float32(image)[wm]
        i0, i1 = np.min(ii), np.max(ii)

        if i0 < i1:
            inom[wm] = np.fix(0.5 + 100. * (ii - i0) / (i1 - i0))
        else:
            inom[wm] = 100

        wc = np.where(inom[wm] > cutoff)
        nwc = len(wc[0])

        csiz[wm] = nwc
        ftag[wm[0][wc[0]], wm[1][wc[0]]] = tags[t]

    return ftag, csiz, inom

def intergranular_levels(image,normed_labels,ntop,fmex,ceco,csiz,ftag,igpx,lcut):
    yo,xo = image.shape
    tmp = ftag.copy()
    adc = np.array([0, 1, -1, 1, 0, -1, 1, -1])
    adr = np.array([1, -1, 0, 1, -1, 1, 0, -1])
    cb0 = 1 + adc
    cb1 = xo - 2 + adc
    rb0 = 1 + adr
    rb1 = yo - 2 + adr

    wc = csiz <= igpx  # np.where(self.csiz > igpx)
    nwc = np.count_nonzero(wc)  # len(wc[0])

    if nwc > 0:
        sum1 = np.sum(normed_labels * np.int32(csiz > igpx))
        sum2 = np.count_nonzero(csiz > igpx)
        sum3 = np.count_nonzero(csiz <= igpx)
        sum4 = np.sum(normed_labels * np.int32(csiz <= igpx))

        c = max(1, sum1 / sum2 * sum3 / sum4) - 1
        lcig = np.int32(0.5 + lcut * (1 + c))
        tmp[wc] = ftag[wc] * np.int32(normed_labels[wc] > lcig)

    feco = np.zeros_like(image)
    ref = tmp[1:-1, 1:-1]
    adj = np.zeros_like(ref)
    con = np.zeros_like(adj)
    for i in range(8):

        sur = tmp[rb0[i]: rb1[i] + 1, cb0[i]:cb1[i] + 1]
        adj = np.maximum(adj,(sur != ref)*sur)

    feco[1: image.shape[0] - 1, 1: image.shape[1] - 1] = adj

    fseg = (feco == 0) * (90 * (tmp == 0) + 180 * (tmp > 0) + 35 * np.logical_and(tmp > 0, ceco == 1) + \
                          75 * np.logical_and(np.logical_and(fmex > 0, fmex <= ntop),normed_labels >= lcut))
    fseg[image == 0] = 0
    return fseg, feco, tmp


def apply_config_file(image,params):
    ## params is the list of all the parameters from config file
    ## image is the input image
    shared_state = SharedState()
    init_labels = labels_hmin(image,params[0],params[1])
    shared_state.old_labels = init_labels
    normed_labels = normalize_labels(image,init_labels)
    shared_state.inos = normed_labels
    ceco, fmex, tg, new_labels = merge_labels(image, normed_labels, init_labels, params[2], params[3], params[4], params[5], params[6])
    shared_state.ceco = ceco
    shared_state.fmex = fmex
    shared_state.tags = tg
    shared_state.new_labels = new_labels
    ftag, csiz, new_normed = shrinking_labels(image, normed_labels, new_labels, tg, params[7])
    shared_state.ftag = ftag
    shared_state.csiz = csiz
    shared_state.new_labels2 = new_normed
    fseg, feco,tmp = intergranular_levels(image, new_normed, params[3], fmex, ceco,csiz, ftag,params[8], params[9])
    shared_state.fseg = fseg
    shared_state.feco = feco
    shared_state.tmpo = tmp
    return fseg, feco,tmp

def best_contrast_image(hdu):
    from skimage.exposure import rescale_intensity
    """
    This function will return the best contrast image from the given cube
    :param hdu: HDU cube object
    :return: index of the best contrast image
    """
    contrast = []
    for i in range(hdu[0].data.shape[0]):
        print('Reading image -> ', i, end='\r')
        slice = np.s_[130:-130, 130:-130]
        im = hdu[0].data[i][slice]
        im = rescale_intensity(im)
        contrast.append(np.nanstd(im) / np.nanmean(im))
        del im
    return np.argmax(np.asarray(contrast))