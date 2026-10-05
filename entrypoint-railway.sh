# git add entrypoint-railway.sh
# git commit -m "Build Tailwind CSS before collecting static files"
# git push origin main
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