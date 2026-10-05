#!/bin/sh
set -e

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Building Tailwind CSS..."
python manage.py tailwind build

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Supervisor..."
exec supervisord -c supervisord.conf