FROM python:3.10.7

WORKDIR /app

copy requirements.txt requirements.txt

run pip install -r requirements.txt

copy . .

expose 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
