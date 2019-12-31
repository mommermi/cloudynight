"""Licensed under a 3-clause BSD style license - see LICENSE.rst

This file is part of 
cloudynight (c) Michael Mommert (mommermiscience@gmail.com), 2020

Update your own file to match the settings shown here; code lines
that you don't have to change were omitted.
"""

# ...

ALLOWED_HOSTS = ['*']

# ...

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'cam',
    'django_extensions',
]

# ...

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

# ...

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'cloudynight',
        'USER': 'cloud',
        'PASSWORD': 'cloud',
        'HOST': 'localhost',
        'PORT': '',
    }
}

# ...

STATIC_URL = '/static/'

MEDIA_ROOT = '/home/mommermi/lowell/projects/cloudynight_paper/cloudynight/workbench/'
MEDIA_URL = '/media/'


DATA_UPLOAD_MAX_NUMBER_FIELDS = 10000
