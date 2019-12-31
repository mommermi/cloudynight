"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

This file can be copied directly into your Django project.
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'', include('cam.urls'))
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns()
