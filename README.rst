Cloudynight - The All-sky Camera Cloud Detector
===============================================

This repository contains code built for Mommert (2020): `Cloud Identification
from All-sky Camera Data with Machine Learning`, submitted

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

You can reference Mommert (2020): `Cloud Identification
from All-sky Camera Data with Machine Learning`, submitted.

Acknowledgements
----------------

The author would like to thank Ryan J. Kelly and the NAU/NASA Arizona Space Grant program
for enabling a case study for this project.

License
-------

This software is distributed under a `3-clause BSD license <LICENSE.rst>`_.


