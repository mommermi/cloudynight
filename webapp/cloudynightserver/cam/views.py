"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
import datetime
import numpy as np
from collections import OrderedDict
from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import JsonResponse, HttpResponse
from django.db.models.aggregates import Count, Max
from rest_framework import viewsets
from .models import Labeled, Unlabeled, Subregion
from .serializers import (LabeledSerializer, UnlabeledSerializer,
                          SubregionSerializer)
from django.forms import model_to_dict
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource
from astropy.time import Time
import pandas as pd
from cloudynight import LightGBMModel, conf


# Table viewsets
class LabeledViewSet(viewsets.ModelViewSet):
    queryset = Labeled.objects.all()
    serializer_class = LabeledSerializer


class UnlabeledViewSet(viewsets.ModelViewSet):
    queryset = Unlabeled.objects.all()
    serializer_class = UnlabeledSerializer


class SubregionViewSet(viewsets.ModelViewSet):
    queryset = Subregion.objects.all()
    serializer_class = SubregionSerializer


class projectHome(TemplateView):
    """Introduction page."""
    template_name = 'index.html'


def getRandomUnlabeled(request):
    """Return a random unlabeled image as json object."""

    queryset = Unlabeled.objects.all()
    count = queryset.aggregate(count=Count('date'))['count']
    data = queryset.values()[np.random.randint(0, count-1)]
    # data = queryset.values()[0]

    response = JsonResponse(data)

    # return response directly
    return response


def getLatestUnlabeled(request):
    """Return the latest unlabeled image as json object."""

    night = request.GET.get('night')

    if night is None:
        data = model_to_dict(Unlabeled.objects.latest())
    else:
        data = model_to_dict(
            Unlabeled.objects.filter(night__iexact=night).latest())
    return JsonResponse(data)


# def labeled(request):
#     if request.method == 'GET':
#         unlabeled_id = request.GET.get('id')
#         cloudy_subregion = request.GET.get('clouds')

#         if unlabeled_id is None or cloudy_subregion is None:
#             return HttpResponse('GET parameters incomplete.')

#         # corresponding unlabeled data set
#         unlabeled_data = Unlabeled.objects.get(pk=unlabeled_id)

#         # identify unlabeled data sets T-3min and T-15min
#         unlabeled_data_past = Unlabeled.objects.filter(
#             date__lte=unlabeled_data.date).filter(
#                 date__gte=(unlabeled_data.date -
#                            datetime.timedelta(minutes=16)))

#         timediff_minutes = np.array(
#             [(unlabeled_data.date-unlabeled_data_past[i].date).seconds//60 for
#              i in range(len(unlabeled_data_past))])

#         # T-3 min
#         unlabeled_data_3min = unlabeled_data_past[
#             int(np.argmin(np.abs(timediff_minutes-3)))]
#         if np.min(np.abs(timediff_minutes-3)) > 1.5:
#             return HttpResponse('no unlabeled data available 3 min into past.')

#         # T-15 min
#         unlabeled_data_15min = unlabeled_data_past[
#             int(np.argmin(np.abs(timediff_minutes-15)))]
#         if np.min(np.abs(timediff_minutes-15)) > 1.5:
#             return HttpResponse('no unlabeled data available 15 min into past.')

#         # derive array of cloudy subregion
#         cloudy = np.zeros(len(unlabeled_data.srcdens)).astype(np.bool)
#         if cloudy_subregion != '':
#             cloudy[np.array([int(s) for s in
#                              cloudy_subregion.split(',')])] = True
#         cloudy = list(cloudy)

#         # get trainer's IP address
#         x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
#         if x_forwarded_for:
#             ip = x_forwarded_for.split(',')[0]
#         else:
#             ip = request.META.get('REMOTE_ADDR')

#         # create labeled object and save it
#         labeled_data = Labeled(
#             id=Labeled.objects.aggregate(Max('id'))['id__max']+1,
#             date=unlabeled_data.date,
#             night=unlabeled_data.night,
#             filearchivepath=unlabeled_data.filearchivepath,
#             moonalt=unlabeled_data.moonalt,
#             sunalt=unlabeled_data.sunalt,
#             moonphase=unlabeled_data.moonphase,
#             srcdens=unlabeled_data.srcdens,
#             bkgmean=unlabeled_data.bkgmean,
#             bkgmedian=unlabeled_data.bkgmedian,
#             bkgstd=unlabeled_data.bkgstd,
#             srcdens_3min=unlabeled_data_3min.srcdens,
#             bkgmean_3min=unlabeled_data_3min.bkgmean,
#             bkgmedian_3min=unlabeled_data_3min.bkgmedian,
#             bkgstd_3min=unlabeled_data_3min.bkgstd,
#             srcdens_15min=unlabeled_data_15min.srcdens,
#             bkgmean_15min=unlabeled_data_15min.bkgmean,
#             bkgmedian_15min=unlabeled_data_15min.bkgmedian,
#             bkgstd_15min=unlabeled_data_15min.bkgstd,
#             cloudy=cloudy,
#             labeled_by=ip
#         )
#         labeled_data.save()

#         # # check whether meteor flag has been set
#         # meteor = request.GET.get('meteor')

#         # if meteor == "true":
#         #     meteor_data = Meteor(
#         #         date=unlabeled_data.date,
#         #         night=unlabeled_data.night,
#         #         filearchivepath=unlabeled_data.filearchivepath,
#         #         labeled_by=ip
#         #     )
#         #     meteor_data.save()

#         # remove unlabeled object
#         unlabeled_data.delete()

#         return HttpResponse("Success!")

#     else:
#         return HttpResponse('Please provide training data per GET request.')


def dashboard(request):
    """Create plots and content for the dashboard page."""

    unlabeled = np.array(Unlabeled.objects.values_list("date",
                                                       "sunalt",
                                                       "moonalt",
                                                       "moonphase",
                                                       )).transpose()
    labeled = np.array(Labeled.objects.values_list("date",
                                                   "sunalt",
                                                   "moonalt",
                                                   "moonphase",
                                                   "cloudy",
                                                   "timestamp")).transpose()

    # number of data points
    n_unlabeled = unlabeled.shape[1]
    n_labeled = labeled.shape[1]
    n_labeled_cloudy = np.sum(np.hstack(labeled[4]).astype(np.int))
    n_labeled_clear = -np.sum(np.hstack(labeled[4]).astype(np.int)-1)

    # solar elevation
    sunalt_unlabeled_hist, bins = np.histogram(unlabeled[1], bins=45)
    sunalt_labeled_hist, bins = np.histogram(labeled[1], bins=bins)

    source = ColumnDataSource(data=dict(
        x=bins[:-1]+(bins[1]-bins[0])/2,
        labeled=sunalt_labeled_hist,
        unlabeled=sunalt_unlabeled_hist,
    ))

    plot = figure(x_axis_label='Solar Elevation (deg)',
                  plot_width=500,
                  plot_height=300)

    plot.varea_stack(['labeled', 'unlabeled'],
                     x='x', color=("red", "blue"),
                     alpha=0.5, source=source)

    sunalt_script, sunalt_div = components(plot, CDN)

    # lunar elevation
    moonalt_unlabeled_hist, bins = np.histogram(unlabeled[2], bins=45)
    moonalt_labeled_hist, bins = np.histogram(labeled[2], bins=bins)

    source = ColumnDataSource(data=dict(
        x=bins[:-1]+(bins[1]-bins[0])/2,
        labeled=moonalt_labeled_hist,
        unlabeled=moonalt_unlabeled_hist,
    ))

    plot = figure(x_axis_label='Lunar Elevation (deg)',
                  plot_width=500,
                  plot_height=300)

    plot.varea_stack(['labeled', 'unlabeled'], x='x', color=("red", "blue"),
                     alpha=0.5, source=source)

    moonalt_script, moonalt_div = components(plot, CDN)

    # lunar phase
    moonphase_unlabeled_hist, bins = np.histogram(unlabeled[3], bins=50)
    moonphase_labeled_hist, bins = np.histogram(labeled[3], bins=bins)

    source = ColumnDataSource(data=dict(
        x=(bins[:-1]+(bins[1]-bins[0])/2)*100,
        labeled=moonphase_labeled_hist,
        unlabeled=moonphase_unlabeled_hist,
    ))

    plot = figure(x_axis_label='Lunar Illumination (%)',
                  plot_width=500,
                  plot_height=300)

    plot.varea_stack(['labeled', 'unlabeled'], x='x', color=("red", "blue"),
                     alpha=0.5, source=source)

    moonphase_script, moonphase_div = components(plot, CDN)

    # date
    labeled[0] = Time(labeled[0]).decimalyear
    unlabeled[0] = Time(unlabeled[0]).decimalyear

    date_unlabeled_hist, bins = np.histogram(unlabeled[0], bins=100)
    date_labeled_hist, bins = np.histogram(labeled[0], bins=bins)

    source = ColumnDataSource(data=dict(
        x=bins[:-1]+(bins[1]-bins[0])/2,
        labeled=date_labeled_hist,
        unlabeled=date_unlabeled_hist,
    ))

    plot = figure(x_axis_label='Year',
                  plot_width=500,
                  plot_height=300)

    plot.varea_stack(['labeled', 'unlabeled'], x='x', color=("red", "blue"),
                     alpha=0.5, source=source)

    date_script, date_div = components(plot, CDN)

    # training timestamp date
    labeled[5] = Time(labeled[5]).decimalyear

    date_labeled_hist, bins = np.histogram(labeled[5], bins=100)

    source = ColumnDataSource(data=dict(
        x=bins[:-1]+(bins[1]-bins[0])/2,
        labeled=date_labeled_hist,
        unlabeled=date_unlabeled_hist,
    ))

    plot = figure(x_axis_label='Training Year',
                  plot_width=500,
                  plot_height=300)

    plot.varea_stack(['labeled'], x='x', color=("red"),
                     alpha=0.5, source=source)

    labelingdate_script, labelingdate_div = components(plot, CDN)

    return render(request, 'dashboard.html',
                  {'n_labeled': n_labeled,
                   'n_unlabeled': n_unlabeled,
                   'n_labeled_cloudy': n_labeled_cloudy,
                   'n_labeled_clear': n_labeled_clear,
                   'sunalt_script': sunalt_script,
                   'sunalt_div': sunalt_div,
                   'moonalt_script': moonalt_script,
                   'moonalt_div': moonalt_div,
                   'moonphase_script': moonphase_script,
                   'moonphase_div': moonphase_div,
                   'date_script': date_script,
                   'date_div': date_div,
                   'labeling_script': labelingdate_script,
                   'labeling_div': labelingdate_div
                   })


def labeler(request):
    """Provide a random Unlabeled image to the user for manual labeling."""

    # retrieve subregion information
    subregion = Subregion.objects.all().order_by('id').values()

    ids = []
    polygons_xy = []
    polygons_x = []
    polygons_y = []
    for sub in subregion:
        ids.append(sub['id'])
        polygons_xy.append(sub['polygon_xy'])
        polygons_x.append(sub['polygon_x'])
        polygons_y.append(sub['polygon_y'])
    n_subregions = len(polygons_y)

    # pick a random frame
    # grab the max id in the database
    max_id = Unlabeled.objects.order_by('-id')[0].id
    random_id = np.random.randint(1, max_id + 1)

    unlabeled_data = Unlabeled.objects.filter(id__gte=random_id)[0]
    frame = model_to_dict(unlabeled_data)

    return render(request, 'label.html',
                  {'unlabeled_id': frame['id'],
                   'date': frame['date'],
                   'moonalt': int(frame['moonalt']),
                   'moonphase': int(100*frame['moonphase']),
                   'sunalt': int(frame['sunalt']),
                   'night': frame['night'],
                   'filearchivepath': frame['filearchivepath'],
                   'n_subregions': n_subregions,
                   'polygons_xy': polygons_xy,
                   'polygons_x': polygons_x,
                   'polygons_y': polygons_y,
                   'cloudy': []
                   })


def checker(request):
    """Predict cloud coverage for a random Unlabeled image and let the user 
    check and correct the classification results."""

    # retrieve subregion information
    subregion = Subregion.objects.all().order_by('id').values()

    ids = []
    polygons_xy = []
    polygons_x = []
    polygons_y = []
    for sub in subregion:
        ids.append(sub['id'])
        polygons_xy.append(sub['polygon_xy'])
        polygons_x.append(sub['polygon_x'])
        polygons_y.append(sub['polygon_y'])
    n_subregions = len(polygons_y)

    # read in latest model
    model = LightGBMModel()
    try:
        model.read_model(conf.LGBMODEL_FILE)
    except FileNotFoundError:
        return HttpResponse(('No trained model available in {}. You have '
                             'to train a model before this feature is '
                             'available.').format(
            conf.LGBMODEL_FILE))

    # pick a random frame
    # grab the max id in the database
    max_id = Unlabeled.objects.order_by('-id')[0].id
    random_id = np.random.randint(1, max_id + 1)

    unlabeled_data = Unlabeled.objects.filter(id__gte=random_id)[0]

    # identify unlabeled data sets T-3min and T-15min
    unlabeled_data_past = Unlabeled.objects.filter(
        date__lte=unlabeled_data.date).filter(
            date__gte=(unlabeled_data.date -
                       datetime.timedelta(minutes=16)))

    timediff_minutes = np.array(
        [(unlabeled_data.date-unlabeled_data_past[i].date).seconds//60 for
         i in range(len(unlabeled_data_past))])

    # T-3 min
    unlabeled_data_3min = unlabeled_data_past[
        int(np.argmin(np.abs(timediff_minutes-3)))]
    if np.min(np.abs(timediff_minutes-3)) > 1.5:
        # if no data available, set differences to zero
        unlabeled_data_3min = unlabeled_data

    # T-15 min
    unlabeled_data_15min = unlabeled_data_past[
        int(np.argmin(np.abs(timediff_minutes-15)))]
    if np.min(np.abs(timediff_minutes-15)) > 1.5:
        # if no data available, set differences to zero
        unlabeled_data_15min = unlabeled_data

    frame = model_to_dict(unlabeled_data)

    # build feature vector for model
    X = pd.DataFrame(OrderedDict(
        (('moonalt', [frame['moonalt']]*n_subregions),
         ('sunalt', [frame['sunalt']]*n_subregions),
         ('moonphase', [frame['moonphase']]*n_subregions),
         ('subid', range(n_subregions)),
         ('srcdens', frame['srcdens']),
         ('bkgmean', frame['bkgmean']),
         ('bkgmedian', frame['bkgmedian']),
         ('bkgstd', frame['bkgstd']),
         ('srcdens_3min', unlabeled_data_3min.srcdens),
         ('bkgmean_3min', unlabeled_data_3min.bkgmean),
         ('bkgmedian_3min', unlabeled_data_3min.bkgmedian),
         ('bkgstd_3min', unlabeled_data_3min.bkgstd),
         ('srcdens_15min', unlabeled_data_15min.srcdens),
         ('bkgmean_15min', unlabeled_data_15min.bkgmean),
         ('bkgmedian_15min', unlabeled_data_15min.bkgmedian),
         ('bkgstd_15min', unlabeled_data_15min.bkgstd))))

    cloud_pred = model.predict(X)

    return render(request, 'label.html',
                  {'unlabeled_id': frame['id'],
                   'date': frame['date'],
                   'moonalt': int(frame['moonalt']),
                   'moonphase': int(100*frame['moonphase']),
                   'sunalt': int(frame['sunalt']),
                   'night': frame['night'],
                   'filearchivepath': frame['filearchivepath'],
                   'n_subregions': n_subregions,
                   'polygons_xy': polygons_xy,
                   'polygons_x': polygons_x,
                   'polygons_y': polygons_y,
                   'cloudy': list(np.arange(max(ids)+1).astype(
                       np.int)[cloud_pred > 0])
                   })


def assignLabels(request):
    """Retrieve Unlabeled id and cloud coverage array; remove data point
    from Unlabeled and add to Labeled together with classification
    results, user IP, and timestamp.
    """

    if request.method == 'GET':
        unlabeled_id = request.GET.get('id')
        cloudy_subregions = request.GET.get('clouds')

        if unlabeled_id is None or cloudy_subregions is None:
            return HttpResponse('GET parameters incomplete.')

        # corresponding unlabeled data set
        unlabeled_data = Unlabeled.objects.get(pk=unlabeled_id)

        # identify unlabeled data sets T-3min and T-15min
        unlabeled_data_past = Unlabeled.objects.filter(
            date__lte=unlabeled_data.date).filter(
                date__gte=(unlabeled_data.date -
                           datetime.timedelta(minutes=16)))

        timediff_minutes = np.array(
            [(unlabeled_data.date-unlabeled_data_past[i].date).seconds//60 for
             i in range(len(unlabeled_data_past))])

        # T-3 min
        unlabeled_data_3min = unlabeled_data_past[
            int(np.argmin(np.abs(timediff_minutes-3)))]
        if np.min(np.abs(timediff_minutes-3)) > 1.5:
            # if no data available, set differences to zero
            unlabeled_data_3min = unlabeled_data

        # T-15 min
        unlabeled_data_15min = unlabeled_data_past[
            int(np.argmin(np.abs(timediff_minutes-15)))]
        if np.min(np.abs(timediff_minutes-15)) > 1.5:
            # if no data available, set differences to zero
            unlabeled_data_15min = unlabeled_data

        # derive array of cloudy subregions
        cloudy = np.zeros(len(unlabeled_data.srcdens)).astype(np.bool)
        if cloudy_subregions != '':
            cloudy[np.array([int(s) for s in
                             cloudy_subregions.split(',')])] = True
        cloudy = list(cloudy)

        # get labeler's IP address
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')

        # create labeled object and save it
        try:
            labeled_id = Labeled.objects.aggregate(Max('id'))['id__max']+1
        except TypeError:
            labeled_id = 0

        labeled_data = Labeled(
            id=labeled_id,
            date=unlabeled_data.date,
            night=unlabeled_data.night,
            filearchivepath=unlabeled_data.filearchivepath,
            moonalt=unlabeled_data.moonalt,
            sunalt=unlabeled_data.sunalt,
            moonphase=unlabeled_data.moonphase,
            srcdens=unlabeled_data.srcdens,
            bkgmean=unlabeled_data.bkgmean,
            bkgmedian=unlabeled_data.bkgmedian,
            bkgstd=unlabeled_data.bkgstd,
            srcdens_3min=unlabeled_data_3min.srcdens,
            bkgmean_3min=unlabeled_data_3min.bkgmean,
            bkgmedian_3min=unlabeled_data_3min.bkgmedian,
            bkgstd_3min=unlabeled_data_3min.bkgstd,
            srcdens_15min=unlabeled_data_15min.srcdens,
            bkgmean_15min=unlabeled_data_15min.bkgmean,
            bkgmedian_15min=unlabeled_data_15min.bkgmedian,
            bkgstd_15min=unlabeled_data_15min.bkgstd,
            cloudy=cloudy,
            labeled_by=ip
        )
        labeled_data.save()

        # remove unlabeled object
        unlabeled_data.delete()

        return HttpResponse("Success!")

    else:
        return HttpResponse('Please provide labeling data per GET request.')


def predictLatestUnlabeled(request):
    """Predict cloud coverage for a random image and return results as 
    json object."""

    subregions = Subregion.objects.all().order_by('id').values()
    n_subregions = len(subregions)

    print('base:', conf.DIR_BASE)
    print('archive:', conf.DIR_ARCHIVE)
    print('raw:', conf.DIR_RAW)

    print('model:', conf.LGBMODEL_FILE)

    # read in latest model
    model = LightGBMModel()

    try:
        model.read_model(conf.LGBMODEL_FILE)
    except FileNotFoundError:
        return HttpResponse(('No trained model available in {}. You have '
                             'to train a model before this feature is '
                             'available.').format(
            conf.LGBMODEL_FILE))

    # pick a random frame
    # grab the max id in the database
    max_id = Unlabeled.objects.order_by('-id')[0].id

    unlabeled_data = Unlabeled.objects.filter(id__gte=max_id)[0]

    # identify unlabeled data sets T-3min and T-15min
    unlabeled_data_past = Unlabeled.objects.filter(
        date__lte=unlabeled_data.date).filter(
            date__gte=(unlabeled_data.date -
                       datetime.timedelta(minutes=16)))

    timediff_minutes = np.array(
        [(unlabeled_data.date-unlabeled_data_past[i].date).seconds//60 for
         i in range(len(unlabeled_data_past))])

    # T-3 min
    unlabeled_data_3min = unlabeled_data_past[
        int(np.argmin(np.abs(timediff_minutes-3)))]
    if np.min(np.abs(timediff_minutes-3)) > 1.5:
        # if no data available, set differences to zero
        unlabeled_data_3min = unlabeled_data

    # T-15 min
    unlabeled_data_15min = unlabeled_data_past[
        int(np.argmin(np.abs(timediff_minutes-15)))]
    if np.min(np.abs(timediff_minutes-15)) > 1.5:
        # if no data available, set differences to zero
        unlabeled_data_15min = unlabeled_data

    frame = model_to_dict(unlabeled_data)

    # build feature vector for model
    X = pd.DataFrame(OrderedDict(
        (('moonalt', [frame['moonalt']]*n_subregions),
         ('sunalt', [frame['sunalt']]*n_subregions),
         ('moonphase', [frame['moonphase']]*n_subregions),
         ('subid', range(n_subregions)),
         ('srcdens', frame['srcdens']),
         ('bkgmean', frame['bkgmean']),
         ('bkgmedian', frame['bkgmedian']),
         ('bkgstd', frame['bkgstd']),
         ('srcdens_3min', unlabeled_data_3min.srcdens),
         ('bkgmean_3min', unlabeled_data_3min.bkgmean),
         ('bkgmedian_3min', unlabeled_data_3min.bkgmedian),
         ('bkgstd_3min', unlabeled_data_3min.bkgstd),
         ('srcdens_15min', unlabeled_data_15min.srcdens),
         ('bkgmean_15min', unlabeled_data_15min.bkgmean),
         ('bkgmedian_15min', unlabeled_data_15min.bkgmedian),
         ('bkgstd_15min', unlabeled_data_15min.bkgstd))))

    cloud_pred = model.predict(X)

    data = {'unlabeled_id': frame['id'],
            'date': frame['date'],
            'moonalt': int(frame['moonalt']),
            'moonphase': int(100*frame['moonphase']),
            'sunalt': int(frame['sunalt']),
            'night': frame['night'],
            'filearchivepath': frame['filearchivepath'],
            'cloudy': [int(v) for v in cloud_pred]
            }

    return JsonResponse(data)


def getAllLabeled(request):
    """Return all labeled data from database as json object."""

    # number of objects to be retrieved
    n = request.GET.get('n')

    data = Labeled.objects.all().values_list()

    if n is not None:
        data = data[:int(n)]

    data = list(map(list, zip(*data)))
    results = {f.name: data[i] for i, f in enumerate(Labeled._meta.fields)}

    return JsonResponse(results)


def getAllUnlabeled(request):
    """Return all unlabeled data from database as json object."""

    # number of objects to be retrieved
    n = request.GET.get('n')

    data = Unlabeled.objects.all().values_list()

    if n is not None:
        data = data[:int(n)]

    data = list(map(list, zip(*data)))
    results = {f.name: data[i] for i, f in enumerate(Unlabeled._meta.fields)}

    return JsonResponse(results)
