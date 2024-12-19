# Run local to test
1. pip freeze > requirements.txt
2. echo "web: python app.py" > Procfile
3. https://drive.google.com/drive/folders/15rjIHMKBdn9cvtlZtg0Na2_k-PDHTNtu?usp=sharing
download saved_models and put it under ML_website folder
4. python app.py

# To deploy in "Render"
| require | command |
| ------- | ------- |
| Build Command | pip install -r requirements.txt |
| Start Command | gunicorn app:app |

