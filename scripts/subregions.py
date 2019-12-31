""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This script shows how subregions are created, visualized, and uploaded to a
database for use in the webapp. The use of this script requires a mask file,
which has to be created with the script generate_mask.py

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""
import os
import requests
import numpy as np
import cloudynight
import matplotlib.pyplot as plt


# instantiate AllskyCamera object and define example image repository
# relative to base directory (defined in __init__.py: example_data/)
cam = cloudynight.AllskyCamera('images')
# this will create a directory `workbench/images` in the repository root;
# `images` is named after the raw image directory (could be a night directory)

# read in mask file; has to be created with generate_mask.fits!
cam.read_mask(filename='../workbench/images/mask.fits')

# create subregions as defined in __init__.py
cam.generate_subregions()

print(len(cam.subregions), 'subregions were created.')

# for visualization, we create plots of the individual subregions to show their
# locations
for subi in range(len(cam.subregions)):
    print('plotting subregion', subi)

    plt.imshow(cam.subregions[subi], origin='lower', vmin=0, vmax=1)
    plt.savefig(os.path.join(cloudynight.conf.DIR_ARCHIVE,
                             'subregion_{:02d}.png'.format(subi)))
    plt.close()

# # !!! this part of the script will only work if the webapp is setup properly
#
# # setup server credentials
# url = conf.HOST_NAME+conf.HOST_BASEDIR+'data/Subregion/' # for use with test server
# user = 'writer'
# pwd = '' # add password here
#
# session = requests.Session()
# post_headers = {'Content-Type': 'application/json'}
#
# # upload subregion information to webapp database (required for use of the
# # webapp)
# for subi in range(len(cam.subregions)):
#     print('uploading subregion', subi)
#
#     # scale polygon coordinates to image size used in webapp
#     # factors at the end are image sizes used in the webapp
#     x = cam.polygons[subi][0]/cam.maskdata.data.shape[0]*460
#     y = cam.polygons[subi][1]/cam.maskdata.data.shape[1]*465
#
#     # rearrange polygon vertices
#     v = np.empty(len(x)+len(y), dtype=np.int)
#     v[0::2] = x
#     v[1::2] = y
#     post_request = session.post(
#         url, headers=post_headers, auth=(user, pwd),
#         json={'id': subi,
#               'polygon_xy': ",".join([str(int(val)) for val in v]),
#               'polygon_x': x.astype(np.int).tolist(),
#               'polygon_y': y.astype(np.int).tolist()})
#
#     if not ((post_request.status_code == requests.codes.ok) or
#             (post_request.status_code == requests.codes.created)):
#         print(post_request.text, post_request.status_code)
