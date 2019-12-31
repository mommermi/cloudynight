"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
from rest_framework import serializers
from .models import Labeled, Unlabeled, Subregion


class LabeledSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Labeled
        fields = ('id', 'date', 'night', 'filearchivepath', 'moonalt',
                  'sunalt',
                  'moonphase', 'srcdens', 'bkgmean', 'bkgmedian',
                  'bkgstd', 'srcdens_3min', 'bkgmean_3min', 'bkgmedian_3min',
                  'bkgstd_3min', 'srcdens_15min', 'bkgmean_15min',
                  'bkgmedian_15min', 'bkgstd_15min', 'cloudy',
                  'labeled_by', 'timestamp')


class UnlabeledSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Unlabeled
        fields = ('date', 'night', 'filearchivepath', 'moonalt', 'sunalt',
                  'moonphase', 'srcdens', 'bkgmean', 'bkgmedian',
                  'bkgstd', 'timestamp')


class SubregionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Subregion
        fields = ('id', 'polygon_xy', 'polygon_x', 'polygon_y')
