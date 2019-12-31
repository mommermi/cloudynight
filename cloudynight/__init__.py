""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This is the setup file for your allsky camera. Right now, it is setup in such a
way that it can be used with the example image data provided with
cloudynight.

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""

import os
import logging
from scipy.stats import uniform, randint

from astropy.visualization import ZScaleInterval

class ConfExample():
    """Allsky camera configuration class."""

    def __init__(self):
    
        # define base directory structure
        # each directory gets `AllskyCamera.night` appended
        # [here, directories are defined relative to the scripts/ directory]

        self.DIR_BASE = os.path.join(
            ' ', *os.path.abspath(__file__).split('/')[:-2]).strip()
        # location of module base (for example data)

        self.DIR_RAW = os.path.join(self.DIR_BASE, 'example_data')

        self.DIR_ARCHIVE = os.path.join(self.DIR_BASE, 'workbench')

        # data directory location on host machine (where to pull FITS files from)
        self.CAMHOST_NAME = ''
        self.CAMHOST_BASEDIR = ''

        # FITS file prefix and suffix used by allsky images
        self.FITS_PREFIX = ''
        self.FITS_SUFFIX = 'fits.bz2'

        # SEP parameters
        self.SEP_SIGMA = 1.5
        self.SEP_MAXFLAG = 7
        self.SEP_MINAREA = 3
        self.SEP_DEBLENDN = 32
        self.SEP_DEBLENDV = 0.005
        self.SEP_BKGBOXSIZE = 15
        self.SEP_BKGXRANGE = 3
        self.SEP_BKGYRANGE = 5

        # max solar elevation for processing (deg)
        self.MAX_SOLAR_ELEVATION = -6;  

        # image crop ranges (ROI must be square)
        self.X_CROPRANGE = (220, 1190)
        self.Y_CROPRANGE = (0, 960)

        # define subregion properties
        self.N_RINGS = 4
        self.N_RINGSEGMENTS = 8

        # define thumbnail properties
        self.THUMBNAIL_WIDTH = 4 # inch
        self.THUMBNAIL_HEIGHT = 4 # inch
        self.THUMBNAIL_DPI = 150
        self.THUMBNAIL_SCALE = ZScaleInterval

        # mask file
        self.MASK_FILENAME = os.path.abspath(os.path.join(self.DIR_RAW,
                                                       'mask.fits'))

        # database URL and credentials
        self.DB_URL = 'http://127.0.0.1:8000/'
        self.DB_USER = 'writer'
        self.DB_PWD = 'writecloud'

        # url for retrieving training data
        self.TRAINDATA_URL = 'http://127.0.0.1:8000/getAllLabeled/'
        self.UNTRAINDATA_URL = 'http://127.0.0.1:8000/getAllUnlabeled/'

        self.LGBMODEL_PARAMETERS = {
            'max_depth': 5,
            'n_estimators': 500,
            'learning_rate': 0.25,
            'num_leaves': 30,
            'min_child_samples': 100,
            'reg_alpha': 10,
            'reg_lambda': 100}

        self.LGBMODEL_PARAMETER_DISTRIBUTIONS = {
            'max_depth': randint(low=3, high=47),
            'n_estimators': randint(low=100, high=1400),
            'learning_rate': uniform(loc=0.1, scale=0.9),
            'feature_fraction': uniform(loc=0.1, scale=0.9),
            'num_leaves': randint(low=3, high=97),
            'min_child_samples': randint(low=10, high=190),
            'reg_alpha': [1, 5, 10, 50, 100],
            'reg_lambda': [1, 5, 10, 50, 100, 500, 1000]}

        self.LGBMODEL_FILE = os.path.join(self.DIR_ARCHIVE,
                                             'lightgbm.pickle')

    def update_directories(self, night):
        """prepare directory structure for a given night, provided as string
           in the form "%Y%m%d"""

        # make sure base directories exist
        os.mkdir(self.DIR_RAW) if not os.path.exists(self.DIR_RAW) else None
        os.mkdir(self.DIR_ARCHIVE) if not os.path.exists(self.DIR_ARCHIVE) else None
        self.DIR_RAW = os.path.join(self.DIR_RAW, night)
        os.mkdir(self.DIR_RAW) if not os.path.exists(self.DIR_RAW) else None

        self.DIR_ARCHIVE = os.path.join(self.DIR_ARCHIVE, night)
        os.mkdir(self.DIR_ARCHIVE) if not os.path.exists(self.DIR_ARCHIVE) else None

        self.setupLogger(night)

    def setupLogger(self, night=''):
        # setup logging
        logging.basicConfig(
            filename=os.path.join(self.DIR_ARCHIVE, night)+'.log',
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger=logging.getLogger(__name__)

        
conf = ConfExample()


from .cloudynight import AllskyImage, AllskyCamera, LightGBMModel, ServerError

__all__ = ['conf', 'AllskyImage', 'AllskyCamera', 'LightGBMModel',
           'ServerError']
