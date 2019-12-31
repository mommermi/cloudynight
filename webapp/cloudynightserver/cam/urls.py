"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
from django.conf.urls import include, url
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'Labeled', views.LabeledViewSet)
router.register(r'Unlabeled', views.UnlabeledViewSet)
router.register(r'Subregion', views.SubregionViewSet)

urlpatterns = [
    # actual pages
    url(r'^$', views.projectHome.as_view()),
    url(r'^dashboard[/]$', views.dashboard, name='dashboard'),
    url(r'^data[/]', include(router.urls)),
    url(r'^label[/]', views.labeler),
    url(r'^check[/]', views.checker),
    # API data retrieval tools
    url(r'^predictLatestUnlabeled[/]$', views.predictLatestUnlabeled),
    url(r'^getAllUnlabeled[/]$', views.getAllUnlabeled),
    url(r'^getAllLabeled[/]$', views.getAllLabeled),
    url(r'^getRandomUnlabeled[/]', views.getRandomUnlabeled),
    url(r'^getLatestUnlabeled[/]', views.getLatestUnlabeled),
    # interface for manual labeling
    url(r'^assignLabels[/]', views.assignLabels),

]
