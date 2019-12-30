Feature-Extracted Example Data
==============================

The file ``fulltrainingsample_features.dat`` holds the training data utilized
in our `lightGBM` model approach.

The file contains a header line and 65175 data lines using the
comma-separated values (csv) format. Each data line represents one of
33 subregions in 1975 all-sky camera images and contains
the following 16 features and one target variable:

* ``moonalt``: the elevation of the Moon (deg)
* ``sunalt``: the elevation of the Sun (deg)
* ``moonphase``: the phase of the Moon ([0, 1])
* ``subid``: the subregion identifier of the current subregion ([0,33])
* ``srcdens``: the source density in the current subregion
* ``bkgmean``: the background mean value in the current subregion
* ``bkgmedian``: the background median value in the current subregion
* ``bkgstd``: the background standard deviation in the current subregion
* ``srcdens_3min``: the difference in source density in the current subregion
  between the current image and the image taken 3 minutes ago
* ``bkgmean_3min``: the difference in background mean in the current subregion
  between the current image and the image taken 3 minutes ago
* ``bkgmedian_3min``: the difference in background median in the current
  subregion between the current image and the image taken 3 minutes ago
* ``bkgstd_3min``: the difference in background standard deviation in the
  current subregion between the current image and the image taken 3 minutes ago
* ``srcdens_15min``: the difference in source density in the current subregion
  between the current image and the image taken 15 minutes ago
* ``bkgmean_15min``: the difference in background mean in the current
  subregion between the current image and the image taken 15 minutes ago
* ``bkgmedian_15min``: the difference in background median in the current
  subregion between the current image and the image taken 15 minutes ago
* ``bkgstd_15min``: the difference in background standard deviation in the
  current subregion between the current image and the image taken 15 minutes
  ago
* ``cloudy``: flag whether a cloud has been found in the current subregion
  (bool)
