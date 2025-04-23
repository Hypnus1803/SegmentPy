class SharedState:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedState, cls).__new__(cls, *args, **kwargs)
            cls._instance.hdu_cube = None
            cls._instance.full_image = None
            cls._instance.original_image = None
            cls._instance.masked_image = None
            cls._instance.markers = None
            cls._instance.old_labels = None
            cls._instance.new_labels = None
            cls._instance.new_labels2 = None
            cls._instance.inos = None
            cls._instance.tags = None
            cls._instance.csiz = None
            cls._instance.ftag = None
            cls._instance.inom = None
            cls._instance.fmex = None
            cls._instance.ceco = None
            cls._instance.fseg = None
            cls._instance.feco = None
            cls._instance.tmpo = None
            #cls._instance.props = None

            cls._instance.perc = None
            cls._instance.classes = None

            ####### params state Thresholding h-minima #######
            cls._instance.scale = None   # pixel scale size of the image
            cls._instance.hmin = None   # h-minima thresholding
            cls._instance.radius = None   # radius h-minima
            cls._instance.lmer = None   # lmer variable based in MLT4 (normalized reference level required for merging)
            cls._instance.ltop = None   # ltop variable based in MLT4 (normalized level in [%] to count ntop > ltop in vector (cmex) for each tag)
            cls._instance.imex = None   # imex variable based in MLT4 (scaled intensity mean below max. value to merge features)
            cls._instance.ntop = None   # ntop variable based in MLT4 (minumum number of pixels exceeding (ltop)  to merge features)
            cls._instance.nmer = None   # nmer variable based in MLT4 (minimum number of adjacent contour px  to merge features)
            cls._instance.lcut = None   # lcut variable based in MLT4 (shrunk features after level cut below threshold)
            cls._instance.igpx = None   # igpx variable based in MLT4 (rate features as INTER-GRANULAR if <(igpx))
            cls._instance.lcut2 = None  # igpy variable based in MLT4 (shrink such features by increased cut-off)



        return cls._instance
