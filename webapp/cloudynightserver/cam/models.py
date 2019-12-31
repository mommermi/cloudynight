"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.utils import timezone


class Subregion(models.Model):
    id = models.IntegerField(help_text='subregion index',
                             null=False, db_index=True, primary_key=True)
    polygon_xy = models.TextField(help_text='polygon nodes: x,y,x,y,x,y,x,...',
                                  null=False, db_index=False)
    polygon_x = ArrayField(models.IntegerField(),
                           help_text='x coordinates of polygon nodes',
                           null=False, db_index=False)
    polygon_y = ArrayField(models.IntegerField(),
                           help_text='y coordinates of polygon nodes',
                           null=False, db_index=False)


class Unlabeled(models.Model):
    date = models.DateTimeField(help_text='date of observations', null=False,
                                db_index=True)
    night = models.IntegerField(help_text='night of observation (YYYYMMDD, UT)',
                                null=False, db_index=True, default=0)
    filearchivepath = models.TextField(
        help_text='thumbnail filename and archive path', null=False,
        db_index=False, default='')
    moonalt = models.FloatField(help_text='altitude of the Moon',
                                null=False, db_index=True, default=-99)
    sunalt = models.FloatField(help_text='altitude of the Sun',
                               null=False, db_index=True, default=-99)
    moonphase = models.FloatField(help_text='illumination of the Moon',
                                  null=False, db_index=True, default=-1)
    srcdens = ArrayField(models.FloatField(
        help_text='source density per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmean = ArrayField(models.FloatField(
        help_text='background mean value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmedian = ArrayField(models.FloatField(
        help_text='background median value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgstd = ArrayField(models.FloatField(
        help_text='background standard deviation per subregion',
        null=False, db_index=False, default=-1), null=True)
    timestamp = models.DateTimeField(default=timezone.now,
                                     help_text='time of ingestion', null=False,
                                     db_index=False)

    class Meta:
        get_latest_by = 'date'


class Labeled(models.Model):
    id = models.IntegerField(help_text='labeling id', null=False,
                             db_index=True,
                             primary_key=True)
    date = models.DateTimeField(help_text='date of observations', null=False,
                                db_index=True)
    night = models.IntegerField(help_text='night of observation (YYYYMMDD, UT)',
                                null=False, db_index=True, default=0)
    filearchivepath = models.TextField(
        help_text='thumbnail filename and archive path', null=False,
        db_index=False, default='')
    moonalt = models.FloatField(help_text='altitude of the Moon',
                                null=False, db_index=True, default=-99)
    sunalt = models.FloatField(help_text='altitude of the Sun',
                               null=False, db_index=True, default=-99)
    moonphase = models.FloatField(help_text='illumination of the Moon',
                                  null=False, db_index=True, default=-1)
    srcdens = ArrayField(models.FloatField(
        help_text='source density per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmean = ArrayField(models.FloatField(
        help_text='background mean value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmedian = ArrayField(models.FloatField(
        help_text='background median value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgstd = ArrayField(models.FloatField(
        help_text='background standard deviation per subregion',
        null=False, db_index=False, default=-1), null=True)
    srcdens_3min = ArrayField(models.FloatField(
        help_text='source density per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmean_3min = ArrayField(models.FloatField(
        help_text='background mean value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmedian_3min = ArrayField(models.FloatField(
        help_text='background median value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgstd_3min = ArrayField(models.FloatField(
        help_text='background standard deviation per subregion',
        null=False, db_index=False, default=-1), null=True)
    srcdens_15min = ArrayField(models.FloatField(
        help_text='source density per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmean_15min = ArrayField(models.FloatField(
        help_text='background mean value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgmedian_15min = ArrayField(models.FloatField(
        help_text='background median value per subregion',
        null=False, db_index=False, default=-1), null=True)
    bkgstd_15min = ArrayField(models.FloatField(
        help_text='background standard deviation per subregion',
        null=False, db_index=False, default=-1), null=True)
    cloudy = ArrayField(models.BooleanField(
        help_text='indicates presence of clouds per subregion',
        null=False, db_index=False, default=False), null=True)
    labeled_by = models.GenericIPAddressField(
        help_text='IP address of labeler', null=True,
        db_index=False)
    timestamp = models.DateTimeField(default=timezone.now,
                                     help_text='time of ingestion', null=False,
                                     db_index=False)

    class Meta:
        get_latest_by = 'date'
