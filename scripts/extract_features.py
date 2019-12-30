""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This script shows how to extract features from raw images.
The use of this script requires a mask file,
which has to be created with the script generate_mask.py

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""
import os
import requests
import numpy as np
import cloudynight


# instantiate AllskyCamera object and define example image repository
# relative to base directory (defined in __init__.py: example_data/)
cam = cloudynight.AllskyCamera('images')
# this will create a directory `workbench/images` in the repository root;
# `images` is named after the raw image directory (could be a night directory)

# read in mask file; has to be created with generate_mask.fits!
cam.read_mask(filename='../workbench/images/mask.fits')

# read in image data
cam.read_data_from_directory(only_new_data=False)
# this will automatically crop the images
# only_new_data=True is necessary to read all data in the directory

# generate subregions
cam.generate_subregions()

# use wrapper to process all images
# `no_upload=True` can be removed if the webapp is setup properly
cam.process_and_upload_data(no_upload=True)

# plot background median values per subregion for all images
for img in cam.imgdata:
    sourcedens_overlay = img.create_overlay(overlaytype='bkgmedian')
    img.write_image(overlay=sourcedens_overlay, mask=cam.maskdata,
                    filename=
                    os.path.join(cloudynight.conf.DIR_ARCHIVE,
                                 '{}_bkgmedian.png'.format(
                            img.filename[:img.filename.find('.fit')])))
