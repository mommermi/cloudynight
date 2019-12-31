""" Licensed under a 3-clause BSD style license - see LICENSE.rst

cloudynight - Tools for automated cloud detection in all-sky camera data

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""


import os
import requests
import shlex
import subprocess
import datetime
from joblib import dump, load
from collections import OrderedDict

import numpy as np
from scipy.signal import convolve2d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import sep
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch
from skimage import measure
from lightgbm import LGBMClassifier
from sklearn.model_selection import (train_test_split, cross_validate,
                                     RandomizedSearchCV)
from sklearn.metrics import f1_score, confusion_matrix
from astroplan import Observer

# define configuration class here
from . import conf

# define observatory
observatory = Observer.at_site('Lowell Observatory')

class AllskyImageError(Exception):
    pass

class ServerError(Exception):
    pass


class AllskyImage():
    """Class for handling individual FITS image files."""

    def __init__(self, filename, data, header):
        self.filename = filename  # raw image filename
        self.datetime = None      # date and time of observation
        self.thumbfilename = None # filename for thumbnail image
        self.data = data          # image data array
        self.header = header      # image header
        self.subregions = None    # subregion arrays
        self.features = None      # extracted features
        
    @classmethod
    def read_fits(cls, filename):
        """Create `~AllskyImage` instance from FITS image file.

        :return: self
        """
        hdu = fits.open(filename)[0]

        self = cls(filename.split(os.path.sep)[-1],
                   hdu.data.astype(np.float), hdu.header)

        try:
            self.datetime = Time(self.header['DATE-OBS'], format='isot')
        except (ValueError, KeyError):
            conf.logger.warning(('No time information for image file '
                                '{}.').format(filename))
            pass
            
        return self

    def write_fits(self, filename):
        """Write `~AllskyImage` instance to FITS image file"""
        hdu = fits.PrimaryHDU(self.data)
        hdu.writeto(filename, overwrite=True)

    def create_overlay(self, overlaytype='srcdens', regions=None):
        """Create overlay for thumbnail image. Requires self.subregions to be
        initialized. An overlay is an array with the same dimensions as
        self.data` in which certain subregions get assigned certain values as
        defined by `overlaytype`.

        :param overlaytype: define data source from `self.features` from which
                            overlay should be generated, default: 'srcdens'
        :param regions: list of length=len(self.subregions), highlights
                        subregions with list element value > 0; requires
                        `overlaytype='subregions'`, default: None

        :return: overlay array
        """
        map = np.zeros(self.data.shape)

        for i, sub in enumerate(self.subregions):
            if overlaytype == 'srcdens':
                map += sub*self.features['srcdens'][i]
            elif overlaytype == 'bkgmedian':
                map += sub*self.features['bkgmedian'][i]
            elif overlaytype == 'bkgmean':
                map += sub*self.features['bkgmean'][i]
            elif overlaytype == 'bkgstd':
                map += sub*self.features['bkgstd'][i]
            elif overlaytype == 'subregions':
                if regions[i]:
                    map += sub
            else:
                raise AllskyImageError('overlaytype "{}" unknown.'.format(
                    overlaytype))
                    
        map[map == 0] = np.nan
        return map
    
        
    def write_image(self, filename, overlay=None, mask=None,
                    overlay_alpha=0.3, overlay_color='Reds'):
        """Write `~AllskyImage` instance as scaled png thumbnail image file.

        :param filename: filename of image to be written, relative to cwd
        :param overlay: provide overlay or list of overlays, optional
        :param mask: apply image mask before writing image file
        :param overlay_alpha: alpha value to be applied to overlay
        :param overlay_color: colormap to be used with overlay

        :return: None
        """

        conf.logger.info('writing thumbnail "{}"'.format(filename))
        
        data = self.data

        # derive image scaling and stretching
        if mask is not None:
            norm = ImageNormalize(data[mask.data == 1],
                                  conf.THUMBNAIL_SCALE(),
                                  stretch=LinearStretch())
            data[mask.data == 0] = 0
        else:
            norm = ImageNormalize(data, conf.THUMBNAIL_SCALE(),
                                  stretch=LinearStretch())

        # create figure
        f, ax = plt.subplots(figsize=(conf.THUMBNAIL_WIDTH,
                                      conf.THUMBNAIL_HEIGHT))

        # plot image
        img = ax.imshow(data, origin='lower',
                        norm=norm, cmap='gray',
                        extent=[0, self.data.shape[1],
                                0, self.data.shape[0]])

        # plot overlay(s)
        if overlay is not None:
            if not isinstance(overlay, list):
                overlay = [overlay]
                overlay_color = [overlay_color]
            overlay_img = []
            for i in range(len(overlay)):
                overlay_img.append(ax.imshow(overlay[i], cmap=overlay_color[i],
                                             origin='lower', vmin=0, 
                                             alpha=overlay_alpha,
                                             extent=[0, overlay[i].shape[1],
                                                     0, overlay[i].shape[0]]))
                overlay_img[i].axes.get_xaxis().set_visible(False)
                overlay_img[i].axes.get_yaxis().set_visible(False)

        # remove axis labels and ticks
        plt.axis('off')
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        # save thumbnail image
        plt.savefig(filename, bbox_inches='tight', dpi=conf.THUMBNAIL_DPI,
                    pad_inches=0)
        plt.close()

        # let thumbfilename consist of <night>/<filename>
        self.thumbfilename = os.path.join(*filename.split(os.path.sep)[-2:])

    def apply_mask(self, mask):
        """Apply `~AllskyImage` mask to this instance"""
        self.data = self.data * mask.data
        
    def crop_image(self):
        """Crop this `~AllskyImage` instance to the ranges defined by
        ``conf.X_CROPRANGE`` and ``conf.Y_CROPRANGE``.
        """
        self.data = self.data[conf.Y_CROPRANGE[0]:conf.Y_CROPRANGE[1],
                              conf.X_CROPRANGE[0]:conf.X_CROPRANGE[1]]
        
    def extract_features(self, subregions, mask=None):
        """Extract image features for each subregion. Image should be cropped
        and masked.

        :param subregions: subregions to be used
        :param mask: mask to be applied in source extraction, optional

        :return: None
        """
        # set internal pixel buffer
        sep.set_extract_pixstack(10000000)

        # extract time from header and derive frame properties
        try:
            time = Time(self.header['DATE-OBS'], format='isot')
            features = OrderedDict([
                ('time', time.isot),
                ('filename', self.filename.split(os.path.sep)[-1]),
                ('moon_alt', observatory.moon_altaz(time).alt.deg),
                ('sun_alt', observatory.sun_altaz(time).alt.deg),
                ('moon_phase', 1-observatory.moon_phase(time).value/np.pi),
            ])
        except KeyError as e:
            conf.logger.error('missing time data in file {}: {}.'.format(
                self.filename, e))
            return False

        # derive and subtract sky background
        bkg = sep.Background(self.data.astype(np.float64),
                             bw=conf.SEP_BKGBOXSIZE, bh=conf.SEP_BKGBOXSIZE,
                             fw=conf.SEP_BKGXRANGE, fh=conf.SEP_BKGYRANGE)
        data_sub = self.data - bkg.back()

        # if mask is provided, it is applied in the proper derivation of
        # source brightness thresholds
        if mask is not None:
            threshold = (np.ma.median(np.ma.array(data_sub,
                                                  mask=(1-mask))) +
                         np.median(bkg.rms())*conf.SEP_SIGMA)
            src = sep.extract(data_sub, threshold, minarea=conf.SEP_MINAREA,
                              mask=(1-mask),
                              deblend_nthresh=conf.SEP_DEBLENDN,
                              deblend_cont=conf.SEP_DEBLENDV)
        else:
            threshold = (np.median(data_sub) +
                         np.median(bkg.rms())*conf.SEP_SIGMA)
            src = sep.extract(data_sub, threshold, minarea=conf.SEP_MINAREA,
                              mask=mask,
                              deblend_nthresh=conf.SEP_DEBLENDN,
                              deblend_cont=conf.SEP_DEBLENDV)

        # apply max_flag cutoff (reject flawed sources)
        src = src[src['flag'] <= conf.SEP_MAXFLAG]

        # feature extraction per subregion
        features['srcdens'] = []
        features['bkgmedian'] = []
        features['bkgmean'] = []
        features['bkgstd'] = []
        for i, sub in enumerate(subregions):
            features['srcdens'].append(len(
                src[sub[src['y'].astype(np.int),
                        src['x'].astype(np.int)]])/np.sum(sub[mask== 1]))
            features['bkgmedian'].append(np.median(bkg.back()[sub]))
            features['bkgmean'].append(np.mean(bkg.back()[sub]))
            features['bkgstd'].append(np.std(bkg.back()[sub]))

        self.subregions = subregions
        self.features = features

    def write_to_database(self):
        """Write extracted features to database."""
        session = requests.Session()
        post_headers = {'Content-Type': 'application/json'}

        try:
            data = {'date': self.features['time'],
                    'night': int(self.datetime.strftime('%Y%m%d')),
                    'filearchivepath': self.thumbfilename,
                    'moonalt': self.features['moon_alt'],
                    'sunalt': self.features['sun_alt'],
                    'moonphase': self.features['moon_phase'],
                    'srcdens': self.features['srcdens'],
                    'bkgmean' : self.features['bkgmean'],
                    'bkgmedian' : self.features['bkgmedian'],
                    'bkgstd': self.features['bkgstd'],
            }
        except KeyError:
            conf.logger.error('data incomplete for file {}; reject.'.format(
                self.filename))
            return None

        post_request = session.post(
            conf.DB_URL+'data/Unlabeled/',
            headers=post_headers, auth=(conf.DB_USER, conf.DB_PWD),
            json=data)

        if not ((post_request.status_code == requests.codes.ok) or
        (post_request.status_code == requests.codes.created)):
            conf.logger.error('upload to database failed with code {}; {}'.format(
                post_request.status_code, post_request.text))
            raise ServerError('upload to database failed with code {}'.format(
                post_request.status_code))

class AllskyCamera():
    """Class for handling data from an all-sky camera."""

    def __init__(self, night=None):
        if night is None:       # date of the night or directory name
            self.night = self.get_current_night()
        else:
            self.night = night
        self.imgdata = None     # image data array
        self.maskdata = None    # mask data array
        self.subregions = None  # subregion array
        self.polygons = None    # subregion outline polygons

        conf.update_directories(self.night)  # update directory structure

        conf.logger.info('setting up analysis for night {}'.format(night))
        
    def get_current_night(self, previous_night=False):
        """derive current night's or following night's directory name based on
        current time"""
        if not previous_night:
            now = datetime.datetime.utcnow()
        else:
            now = (datetime.datetime.utcnow() - datetime.timedelta(days=1))
            
        return now.strftime('%Y%m%d')
        
        
    def download_latest_data(self):
        """Download latest data from camera computer's self.night directory
        using rsync."""

        # build rsync command
        commandline = 'rsync -avr {}:{} {}'.format(
            conf.CAMHOST_NAME,
            os.path.join(conf.CAMHOST_BASEDIR, self.night,
                         '*.{}'.format(conf.FITS_SUFFIX)), conf.DIR_RAW)

        # download data
        conf.logger.info('attempting download with %s', commandline)
        try:
            run = subprocess.Popen(shlex.split(commandline),
                                   close_fds=True)
            run.wait()
        except Exception as e:
            conf.logger.error('download failed: {}'.format(e))
        else:
            conf.logger.info('download succeeded')


    def get_latest_unlabeled_date(self, night):
        """Retrieve latest unlabeled image for a given night from
        database."""
        session = requests.Session()
        get_request = session.get(
            conf.DB_URL+'latest_unlabeled', params={'night': night},
            auth=(conf.DB_USER, conf.DB_PWD))

        conf.logger.info('retrieving latest update of unlabeled database')
        
        if not ((get_request.status_code == requests.codes.ok) or
                (get_request.status_code == requests.codes.created)):
            conf.logger.error("error retrieving latest unlabeled date.",
                              get_request.text, get_request.status_code)
            return None
        
        return get_request.json()
        
    def read_data_from_directory(self, only_new_data=True, crop=True,
                                 batch_size=None, last_image_idx=None):
        """Read in images from a directory.

        :param only_new_data: only consider data that have not yet been
                              processed.
        :param crop: crop images
        :param batch_size: how many images to process in one batch
        :param last_image_idx: index of image where to start processing

        """

        conf.logger.info('reading data from directory "{}"'.format(
            conf.DIR_RAW))

        last_image_night = self.night
        if only_new_data:
            last_image = self.get_latest_unlabeled_date(self.night)
            if last_image is None:
                last_image_idx = 0
            else:
                last_image_night = last_image['night']
                if (str(last_image_night) in conf.DIR_RAW and
                    last_image_idx is None):
                    last_image_idx = int(last_image['filearchivepath'].split(
                        os.path.sep)[-1][len(conf.FITS_PREFIX):
                                         -len(conf.FITS_SUFFIX)-1])

            conf.logger.info(('  ignore frames in night {} with indices lower '
                              'than {}.').format(
                                  last_image_night, last_image_idx))

        data = []
        for i, filename in enumerate(sorted(os.listdir(conf.DIR_RAW))):
            # check file type
            if (not filename.startswith(conf.FITS_PREFIX) or
                not filename.endswith(conf.FITS_SUFFIX)):
                continue

            # extract image index (camera-specific) and reject if only_new_data
            file_idx = int(filename[len(conf.FITS_PREFIX):
                                    -len(conf.FITS_SUFFIX)-1])

            if only_new_data:
                if file_idx <= last_image_idx:
                    continue

            img = AllskyImage.read_fits(os.path.join(conf.DIR_RAW, filename))

            if img.datetime is None:
                continue

            # check solar elevation
            if observatory.sun_altaz(img.datetime).alt.deg >= -6:
                continue
            
            if crop:
                conf.logger.info('  cropping {}'.format(filename))
                img.crop_image()

            data.append(img)
            
            if batch_size is not None and len(data) >= batch_size:
                break

        conf.logger.info('{} images read for processing'.format(len(data)))

        self.imgdata = data
        
    
    def generate_mask(self, mask_gt=None, mask_lt=None,
                  return_median=False, convolve=None,
                  gaussian_blur=None, filename='mask.fits'):
        """Generate an image mask to cover the horizon.

        :param mask_gt: float, mask pixel values greater than this value
        :param mask_lt: float, mask pixel values less than this value
        :param return_median: boolean, if True, return simply the median of all
                              images
        :param convolve: int or None, convolve mask image for additional
                         smoothing, defines kernel edge length
        :param gaussian_blur: int or None, apply Gaussian filter to blur small
                              features before applying thresholds, defines
                              kernel edge length
        :param filename: mask file name

        :return: mask array (0 is masked, 1 is not masked)
        """
        mask = np.median([img.data for img in self.imgdata], axis=0)

        conf.logger.info('generating image mask from {} images.'.format(len(
            self.imgdata)))

        if gaussian_blur is not None:
            conf.logger.info(('  apply gaussian blur with kernel size {'
                             '}.').format(gaussian_blur))
            mask = gaussian_filter(mask, gaussian_blur)
        
            if not return_median:
                newmask = np.ones(mask.shape)
                if mask_gt is not None:
                    conf.logger.info(('  mask pixels with values greater '
                                      'than {}.').format(mask_gt))
                    newmask[mask < mask_gt] = 0
                elif mask_lt is not None:
                    conf.logger.info(('  mask pixels with values less '
                                      'than {}.').format(mask_lt))

                    newmask[mask > mask_lt] = 0
                mask = newmask

            if convolve is not None:
                conf.logger.info(('  convolve mask with kernel '
                                  'size {}.').format(convolve))
                mask = np.clip(convolve2d(mask, np.ones((convolve, convolve)),
                                          mode='same'), 0, 1)

        # masked regions have value 1, unmasked regions value 0
        mask = mask+1
        mask[mask ==2] = 0

        mask = AllskyImage('mask', mask, {})
        mask.write_fits(os.path.join(conf.DIR_ARCHIVE, filename))

        conf.logger.info('  mask file written to {}.'.format(
            os.path.join(conf.DIR_ARCHIVE, filename)))

        return mask

    def read_mask(self, filename):
        """Read in mask FITS file."""
        if filename is None:
            filename = conf.MASK_FILENAME
        conf.logger.info('read mask image file "{}"'.format(filename))
        self.maskdata = AllskyImage.read_fits(filename)
        
    def process_and_upload_data(self, no_upload=False):
        """Wrapper method to automatically process images and upload data to
        the database. This method also creates thumbnail images that are
        saved to the corresponding archive directory."""
        conf.logger.info('processing image files')

        for dat in self.imgdata:
            conf.logger.info('extract features from file "{}"'.format(
                dat.filename))
            file_idx = int(dat.filename[len(conf.FITS_PREFIX):
                                        -len(conf.FITS_SUFFIX)-1])
            extraction = dat.extract_features(self.subregions,
                                              mask=self.maskdata.data)
            if extraction is False:
                conf.logger.error('ignore results for image "{}".'.format(
                    dat.filename))
                continue

            filename = dat.filename[:dat.filename.find(conf.FITS_SUFFIX)]+'png'
            
            dat.write_image(os.path.join(conf.DIR_ARCHIVE, filename),
                            mask=self.maskdata)

            if not no_upload:
                dat.write_to_database()

        return file_idx
        
    def generate_subregions(self):
        """Create subregions array. This array consists of N_subregions
        arrays, each with the same dimensions as self.maskdata.
        """

        shape = np.array(self.maskdata.data.shape)
        center_coo = shape/2
        radius_borders = np.linspace(0, min(shape)/2,
                                     conf.N_RINGS + 2)
        azimuth_borders = np.linspace(-np.pi, np.pi,
                                      conf.N_RINGSEGMENTS + 1)
        n_subregions = conf.N_RINGS*conf.N_RINGSEGMENTS+1


        # build templates for radius and azimuth
        y, x = np.indices(shape)
        r_map = np.sqrt((x-center_coo[0])**2 +
                        (y-center_coo[1])**2).astype(np.int)
        az_map = np.arctan2(y-center_coo[1],
                            x-center_coo[0])

        # subregion maps
        subregions = np.zeros([n_subregions, *shape], dtype=np.bool)
        
        # polygons around each source region in original image dimensions
        polygons = []
        
        for i in range(conf.N_RINGS+1):
            for j in range(conf.N_RINGSEGMENTS):
                if i == 0 and j==0:
                    subregions[0][(r_map < radius_borders[i+1])] = True
                    # find contours
                    contours = measure.find_contours(
                        subregions[0], 0.5)
                elif i==0 and j>0:
                    break
                else:
                    subregions[(i-1)*conf.N_RINGSEGMENTS+j+1][
                        ((r_map > radius_borders[i]) &
                         (r_map < radius_borders[i+1]) &
                         (az_map > azimuth_borders[j]) &
                         (az_map < azimuth_borders[j+1]))] = True
                    contours = measure.find_contours(
                        subregions[(i-1)*conf.N_RINGSEGMENTS+j+1], 0.5)
                # downscale number of vertices
                polygons.append((contours[0][:,0][::10],
                                 contours[0][:,1][::10]))
                
        self.subregions = subregions
        self.polygons = np.array(polygons)

        return len(self.subregions)

    
class LightGBMModel():
    """Class for use of lightGBM model."""

    def __init__(self):
        self.data_X = None      # pandas DataFrame
        self.data_y = None      # pandas DataFrame
        self.model = None       # model implementation
        self.filename = None    # model pickle filename
        self.train_score = None # model training score
        self.test_score = None  # model test score
        self.val_score = None   # model validation sample score
        self.f1_score_val = None  # model validation sample f1 score

    def retrieve_training_data(self, size_limit=None):
        """Retrieves feature data from webapp database."""
        n_subregions = conf.N_RINGS*conf.N_RINGSEGMENTS+1
        
        get = requests.get(conf.TRAINDATA_URL)
        if get.status_code != requests.codes.ok:
            raise ServerError('could not retrieve training data from server')
        raw = pd.DataFrame(get.json())

        data = pd.DataFrame()
        for j in range(len(raw['moonalt'])):
            frame = pd.DataFrame(OrderedDict(
                (('moonalt', [raw['moonalt'][j]]*n_subregions),
                 ('sunalt', [raw['sunalt'][j]]*n_subregions),
                 ('moonphase', [raw['moonphase'][j]]*n_subregions),
                 ('subid', range(n_subregions)),
                 ('srcdens', raw['srcdens'][j]),
                 ('bkgmean', raw['bkgmean'][j]),
                 ('bkgmedian', raw['bkgmedian'][j]),
                 ('bkgstd', raw['bkgstd'][j]),
                 ('srcdens_3min', raw['srcdens_3min'][j]),
                 ('bkgmean_3min', raw['bkgmean_3min'][j]),
                 ('bkgmedian_3min', raw['bkgmedian_3min'][j]),
                 ('bkgstd_3min', raw['bkgstd_3min'][j]),
                 ('srcdens_15min', raw['srcdens_15min'][j]),
                 ('bkgmean_15min', raw['bkgmean_15min'][j]),
                 ('bkgmedian_15min', raw['bkgmedian_15min'][j]),
                 ('bkgstd_15min', raw['bkgstd_15min'][j]),
                 ('cloudy', raw['cloudy'][j]))))
            data = pd.concat([data, frame]) 

        self.data_X = data.drop(['cloudy'], axis=1)
        self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

        # limit data set size to size_limit subregions
        if size_limit is not None:
            self.data_X = self.data_X[:size_limit]
            self.data_y = self.data_y[:size_limit]
        
        return len(self.data_y)

    def load_data(self, filename):
        """Load feature data from file."""

        data = pd.read_csv(filename, index_col=0)

        # split features and target
        self.data_X = data.drop(['cloudy'], axis=1)
        self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

        return len(self.data_y)

    def train(self, parameters=conf.LGBMODEL_PARAMETERS, cv=5):
        """Train """

        # split data into training and validation sample
        X_cv, X_val, y_cv, y_val = train_test_split(
            self.data_X, self.data_y, test_size=0.1, random_state=42)

        # define model
        lgb = LGBMClassifier(objective='binary', random_state=42,
                             n_jobs=-1, **parameters)
        # train model
        lgb.fit(X_cv, y_cv)
        self.model = lgb

        # derive cv scores
        cv_results = cross_validate(lgb, X_cv, y_cv, cv=cv,
                                    return_train_score=True)
        self.train_score = np.max(cv_results['train_score'])
        self.test_score = np.max(cv_results['test_score'])
        self.parameters = parameters
        self.val_score = self.model.score(X_val, y_val)
        self.f1_score_val = f1_score(y_val, self.model.predict(X_val))

        return self.val_score


    def train_randomizedsearchcv(self, n_iter=100,
        distributions=conf.LGBMODEL_PARAMETER_DISTRIBUTIONS,
        cv=3, scoring="accuracy"):
        """Train the lightGBM model using a combined randomized
        cross-validation search."""

        # split data into training and validation sample
        X_grid, X_val, y_grid, y_val = train_test_split(
            self.data_X, self.data_y, test_size=0.1, random_state=42)

        # initialize model
        lgb = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

        # initialize random search + cross-validation
        lgbrand = RandomizedSearchCV(lgb, distributions, cv=cv, scoring=scoring,
                                     n_iter=n_iter, return_train_score=True)

        # fit model
        lgbrand.fit(X_grid, y_grid)

        self.cv_results = lgbrand.cv_results_
        self.model = lgbrand.best_estimator_

        # derive scores
        self.train_score = lgbrand.cv_results_['mean_train_score'][lgbrand.best_index_]
        self.test_score = lgbrand.cv_results_['mean_test_score'][lgbrand.best_index_]
        self.parameters = lgbrand.cv_results_['params'][lgbrand.best_index_]
        self.val_score = self.model.score(X_val, y_val)
        self.f1_score_val = f1_score(y_val, self.model.predict(X_val))

        return self.val_score

    def write_model(self,
                    filename=os.path.join(conf.DIR_ARCHIVE+'model.pickle')):
        """Write trained model to file."""
        self.filename = filename
        dump(self.model, filename)

    def read_model(self,
                   filename=os.path.join(conf.DIR_ARCHIVE+'model.pickle')):
        """Read trained model from file."""
        self.filename = filename
        self.model = load(filename)
        
    def predict(self, X):
        """Predict cloud coverage for feature data."""
        return self.model.predict(X)

