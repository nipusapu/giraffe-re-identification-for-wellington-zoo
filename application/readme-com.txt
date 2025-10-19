
cd D:\My Lecs\Project\model\finalmodel\application\
==========================
conda activate reid-env

===========================
pip cache purge

===========================
pip install -r requirements.txt

=============================

================ venv=======================
"D:\My Lecs\Project\model\finalmodel\application\.venv\Scripts\Activate.ps1"

========= Django =============

python manage.py runserver

================api key==================

python manage.py create_api_key nextjs-ui

=========== celery worker==========

celery -A config worker -l info -P solo

=========== migration=============

python manage.py makemigrations api
python manage.py migrate


=================docker==========

docker compose build

============ up down ==============

docker compose up -d

docker compose down -v 

========== specific===========

docker compose build api worker --no-cache

docker compose up -d api worker


===================================
docker compose exec api python manage.py migrate

docker compose exec api python manage.py create_api_key "nextjs-ui"