Example Image Data
==================

This directory contains 20 random raw all-sky camera images. They can be used
for testing feature extraction or to train the ResNet model (although the
sample size is much too small to obtain a useful model).

Each all-sky camera image file is provided in FITS format and using bzip2
compression. The filename contains the image identification number.

The file ``y_train.dat`` provides the locations of clouds that have been
found in these images. Each row in this file contains the image idx (as
provided in the image file name) and a whitespace-separated list of 33
symbols, each representing one subregion; 1 indicates the presence of a cloud
in this subregion, 0 indicates clear sky.

The file ``mask.fits`` contains an image mask, covering background and lens
features. It can be used with the example scripts provided.