# Run local to test
1. pip freeze > requirements.txt
2. echo "web: python app.py" > Procfile
3. python app.py

# To deploy in "Render"
| require | command |
| ------- | ------- |
| Build Command | pip install -r requirements.txt |
| Start Command | gunicorn app:app |
