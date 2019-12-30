# Installation and Setup

## PostgreSQL Database
(https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04)

* install postgresql
  >>> sudo apt install libpq-dev postgresql postgresql-contrib

* create database
  >>> sudo su - postgres
  >>> psql
  >>> CREATE DATABASE cloudynight;
  >>> CREATE USER cloud WITH PASSWORD '###';
  >>> ALTER ROLE cloud SET client_encoding TO 'utf8';
  >>> ALTER ROLE cloud SET default_transaction_isolation TO 'read committed';
  >>> ALTER ROLE cloud SET timezone TO 'UTC';
  >>> GRANT ALL PRIVILEGES ON DATABASE cloudynight TO cloud;
  >>> \q
  >>> exit
  
## Django

* install anaconda as root (perform as root)
  download Anaconda3 installer to /usr/local/
  >>> bash Anaconda
  install path: /usr/local/anaconda3
  add the following lines to /root/.bashrc and the .bashrc files of all users:
  """
  export PATH=/usr/local/anaconda3/bin:$PATH
  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$('/usr/local/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
  else
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin:$PATH"
    fi
  fi 
  unset __conda_setup
  # <<< conda initialize <<<
  """
  
* create a root conda environment for `cloudynight` (perform as root)
  >>> conda create -n cloudynight python=3.7
  >>> conda activate cloudynight
  >>> pip install django djangorestframework numpy requests django_tables2 /
  ...     Pillow psycopg2 django-extensions django-toolbelt bokeh astropy
  
* setup django in /var/www/cloudynight-server/
  >>> django-admin startproject cloudynight

* create admin user:
  >>> python manage.py createsuperuser
  cloud/lo-clouds  

* create "writer" user under http://127.0.0.1:8000/admin

* populate subregions in database (in cloudynight/scripts)
  >>> python populate_subregions.py 


  apt install apache2-dev
  pip install mod_wsgi
  sudo a2enmod wsgi

## Deployment

* make sure cloudynights-server is located in /var/www/cloudynights-server,
  data in /data/
  
* install apache2 
  >>> sudo apt-get update
  >>> sudo apt-get install libexpat1 apache2 apache2-utils apache2-dev ssl-cert 

* install mod-wsgi (as root)
  >>> pip install mod_wsgi
  >>> mod_wsgi-express module-config
  output line "WSGIPythonHome" goes into /etc/apache2/mods-available/wsgi.conf
  output line "WSGIScriptAlias" goes into /etc/apache2/mods-available/wsgi.load
  
* activate mod-wsgi and restart apache2 (as root)
  >>> a2enmod wsgi
  >>> service apache2 restart

* modify /etc/apache2/apache2.conf to include:
  """
  WSGIScriptAlias / /var/www/cloudynight-server/cloudynight/wsgi.py
  WSGIPythonPath /var/www/cloudynight-servercloudynight

  <Directory /var/www/cloudynight-server/cloudynight>
  <Files wsgi.py>
  Require all granted
  </Files>
  </Directory>
  """
  
* modify /etc/apache2/mods-available/wsgi.conf to include:
  """
  WSGIDaemonProcess cloudynight python-home=/usr/local/anaconda3/envs/cloudynight python-path=/var/www/cloudynight-server/
  WSGIProcessGroup cloudynight

  Alias /robots.txt /var/www/static/robots.txt
  Alias /favicon.ico /var/www/static/favicon.ico

  Alias /static/ /var/www/static/
  Alias /media/ /data/archive/

  <Directory /var/www/static>
  Require all granted
  </Directory>

  <Directory /data>
  Require all granted
  </Directory>

  WSGIScriptAlias / /var/www/cloudynight-server/cloudynight/wsgi.py

  <Directory /var/www/cloudynight-server/cloudynight>
    <Files wsgi.py>
      Require all granted
    </Files>
  </Directory>  
  """
  
* deactivate apache2 default page
  >>> sudo a2dissite 000-default.conf

* restart apache2 server
  >>> sudo service apache2 restart

