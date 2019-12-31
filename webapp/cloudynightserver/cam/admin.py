"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
from django.contrib import admin
from .models import Labeled, Unlabeled, Subregion

admin.site.register(Labeled)
admin.site.register(Unlabeled)
admin.site.register(Subregion)
