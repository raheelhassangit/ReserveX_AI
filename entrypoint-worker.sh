#!/bin/sh
set -e

echo "Waiting for database..."
python manage.py wait_for_db

exec "$@"