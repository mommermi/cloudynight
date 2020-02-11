Cloudynight - The All-sky Camera Cloud Detector
===============================================


.. image:: https://zenodo.org/badge/230988144.svg
   :target: https://zenodo.org/badge/latestdoi/230988144

This repository contains code built for
`Mommert (2020) <http://doi.org/10.3847/1538-3881/ab744f>`_:
*Cloud Identification from All-sky Camera Data with Machine Learning*
published by The Astronomical Journal.
For details on ``cloudynight``, please refer to this publication.

The system consists of several parts:

* the ``cloudynight`` Python module, which contains tools for data handling and
  preparation, feature extraction, model training and prediction
  (see directory ``cloudynight/``);
* a Python Django web server application for database management, data
  visualization, and manual labeling (see directory ``webapp/``);
* example data used in this work (see directory ``example_data/``);
* a number of scripts for testing the functionality on the example data
  (see directory ``scripts/``).

Please that ``cloudynight`` only utilizes the more efficient ``lightgbm``
classifier. The ResNet code is also included in ``scripts/`` for the sake
completeness.

Requirements
------------

``cloudynight`` requires the following Python modules to be available
(version numbers in parentheses signify the version numbers that were used in
building this code):

* numpy (1.16.3)
* scipy (1.2.1)
* matplotlib (3.0.3)
* pandas (0.25.3)
* sep (1.0.3)
* astropy (4.0)
* scikit-image (0.15.0)    
* lightgbm (2.2.3)
* scikit-learn (0.22.1)  
* astroplan (0.4)

The use of the ResNet implementation requires additional modules:

* pytorch (1.3.1)
* tqdm (4.36.1)

  
Use
---

``cloudynight`` contains all the parts necessary to build an automated cloud
detector, but it is not intended as a plug-and-play software.

First, install the ``cloudynight`` module:

  >>> python setup.py install

and run the provided example scripts to get familiar with the module.

Then, install the `web application <webapp/README.rst>`_.

To use the software for real-time cloud detection, write a script that
utilizes ``cloudynight.AllskyCamera.download_latest_data()`` to download data
from the camera computer. Then, utilize
``cloudynight.AllskyCamera.process_and_upload_data`` to extract features and
upload them to the database. Both tasks can be automated with cron jobs.

The ``label/`` task of the web application can be used for manual labeling
and training data generation. Once enough data are available, a modified
version of ``scripts/model_lightgbm.py`` can be used to tune model parameters
and fit a model. With a model being available, the ``check/`` task can be
utilized for faster manual labeling.

Cloud coverage can be predicted for the latest image obtained from the camera
using the web API with task ``predictLatestUnlabeled``.


Citing cloudynight
------------------

If you decide to use code elements from this repository, please reference
`Mommert (2020) <http://doi.org/10.3847/1538-3881/ab744f>`_:
*Cloud Identification from All-sky Camera Data with Machine Learning*
published by The Astronomical Journal.

Acknowledgements
----------------

The author would like to thank Ryan J. Kelly and the NAU/NASA Arizona Space Grant program
for enabling a case study for this project.

License
-------

This software is distributed under a `3-clause BSD license <LICENSE.rst>`_.


