git push https://github.com/Ahmetereni/greenflask.git master
git push -f origin master
gunicorn -w 4 -b 0.0.0.0:8000 'application:create_app()'
