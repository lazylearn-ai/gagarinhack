FROM python3.8-alpine

WORKDIR /app

copy requirements.txt requirements.txt

run pip install -r requirements.txt

copy . .

expose 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
