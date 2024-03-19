FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --ignore-installed \
  pandas==1.5.3 \
  numpy==1.23.5 \
  joblib==1.3.2 \
  Flask scikit-learn

EXPOSE 3000

CMD ["python3", "api.py"]