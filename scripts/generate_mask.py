""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This script creates a mask image that can be used to mask the local
horizon in allsky camera images. If you want to use masking, which is highly
recommended, use this script first!

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""

import cloudynight

# instantiate AllskyCamera object and define example image repository
# relative to base directory (defined in __init__.py: example_data/)
cam = cloudynight.AllskyCamera('images')
# this will create a directory `workbench/images` in the repository root;
# `images` is named after the raw image directory (could be a night directory)

# read in image data
cam.read_data_from_directory(only_new_data=False)
# this will automatically crop the images
# only_new_data=True is necessary to read all data in the directory

# create median image of stack for inspection
median = cam.generate_mask(return_median=True, filename='median.fits')
# open the image with a fits viewer: any value less than ~3400 is most likely a
# local horizon feature, we use this value as cutoff

# generate mask image; apply in the following order
# gaussian_blur=10: blur the image with kernel size 10 to get rid of small
#                   features (like hot pixels)
# mask_lt=3400: mask every pixel with a value less than 3400
# convolve=20: convolve mask with a kernel size of 20 to smoothen mask
mask = cam.generate_mask(mask_lt=3400, gaussian_blur=10, convolve=20,
                         filename='mask.fits')