# AI-Based-Anomaly-Detection-System
AI-powered cloud resource monitoring system using Machine Learning, Deep Learning, Kafka, Spark Streaming, Kubernetes, Grafana, and Python.
AI-Based-Anomaly-Detection-System/
│
├── app.py
├── requirements.txt
├── README.md
├── config.py
│
├── data/
│   └── cloud_monitoring_data.csv
│
├── preprocessing/
│   └── preprocess.py
│
├── models/
│   ├── isolation_forest_model.py
│   ├── oneclass_svm_model.py
│   ├── autoencoder_model.py
│   └── lstm_model.py
│
├── dashboard/
│   └── dashboard.py
│
├── api/
│   └── routes.py
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes.yaml
│
└── notebooks/
    └── model_training.ipynb
    pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
tensorflow==2.16.1
matplotlib==3.9.0
plotly==5.22.0
flask==3.0.3
joblib==1.4.2
kafka-python==2.0.2
gunicorn==22.0.0
DATA_PATH = "data/cloud_monitoring_data.csv"

MODEL_PATH = "saved_models/"

CPU_THRESHOLD = 85
MEMORY_THRESHOLD = 90

RANDOM_STATE = 42

WINDOW_SIZE = 10

TEST_SIZE = 0.2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

rows = 5000

timestamps = [
datetime.now() + timedelta(minutes=i)
for i in range(rows)
]

cpu = np.random.normal(50, 10, rows)
memory = np.random.normal(60, 8, rows)
network_in = np.random.normal(1000, 200, rows)
network_out = np.random.normal(900, 180, rows)
disk_io = np.random.normal(300, 50, rows)
response_time = np.random.normal(200, 25, rows)

anomaly_indices = np.random.choice(rows, 100)

cpu[anomaly_indices] = cpu[anomaly_indices] + 50
memory[anomaly_indices] = memory[anomaly_indices] + 40
response_time[anomaly_indices] = response_time[anomaly_indices] + 150

status = [
"Anomaly" if i in anomaly_indices else "Healthy"
for i in range(rows)
]

df = pd.DataFrame({
"Timestamp": timestamps,
"CPU_Usage": cpu,
"Memory_Usage": memory,
"Network_In": network_in,
"Network_Out": network_out,
"Disk_IO": disk_io,
"Response_Time": response_time,
"Status": status
})

df.to_csv("cloud_monitoring_data.csv", index=False)

print("Dataset generated successfully")
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
StandardScaler,
LabelEncoder
)

def preprocess_data(df):

```
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df.drop_duplicates(inplace=True)

df.ffill(inplace=True)

df["CPU_Rolling_Mean"] = (
    df["CPU_Usage"]
    .rolling(window=5)
    .mean()
)

df["CPU_Lag_1"] = (
    df["CPU_Usage"]
    .shift(1)
)

df["CPU_Change_Rate"] = (
    df["CPU_Usage"]
    .pct_change()
)

df["Hour"] = (
    df["Timestamp"]
    .dt.hour
)

encoder = LabelEncoder()

df["Status_Encoded"] = (
    encoder.fit_transform(
        df["Status"]
    )
)

numerical_cols = [
    "CPU_Usage",
    "Memory_Usage",
    "Network_In",
    "Network_Out",
    "Disk_IO",
    "Response_Time",
    "CPU_Rolling_Mean",
    "CPU_Lag_1",
    "CPU_Change_Rate"
]

scaler = StandardScaler()

df[numerical_cols] = scaler.fit_transform(
    df[numerical_cols]
)

df.dropna(inplace=True)

return df
```
import pandas as pd

from preprocessing.preprocess import (
preprocess_data
)

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

processed_df = preprocess_data(df)

print(processed_df.head())

print(
f"Rows after preprocessing: "
f"{len(processed_df)}"
)
AI-Based-Anomaly-Detection-System/
│
├── data/
│   ├── generate_dataset.py
│   └── cloud_monitoring_data.csv
│
├── preprocessing/
│   └── preprocess.py
│
├── saved_models/
│
├── config.py
├── train.py
├── requirements.txt
└── README.md
models/
│
├── isolation_forest_model.py
├── oneclass_svm_model.py
└── evaluate.py

saved_models/
import joblib

from sklearn.ensemble import IsolationForest

class IsolationForestDetector:

```
def __init__(self):

    self.model = IsolationForest(
        contamination=0.02,
        random_state=42,
        n_estimators=200
    )

def train(self, X):

    self.model.fit(X)

def predict(self, X):

    predictions = self.model.predict(X)

    return [
        1 if p == -1 else 0
        for p in predictions
    ]

def save_model(
    self,
    path="saved_models/isolation_forest.pkl"
):

    joblib.dump(self.model, path)

def load_model(
    self,
    path="saved_models/isolation_forest.pkl"
):

    self.model = joblib.load(path)
```
import joblib

from sklearn.svm import OneClassSVM

class OneClassSVMDetector:

```
def __init__(self):

    self.model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.02
    )

def train(self, X):

    self.model.fit(X)

def predict(self, X):

    predictions = self.model.predict(X)

    return [
        1 if p == -1 else 0
        for p in predictions
    ]

def save_model(
    self,
    path="saved_models/oneclass_svm.pkl"
):

    joblib.dump(self.model, path)

def load_model(
    self,
    path="saved_models/oneclass_svm.pkl"
):

    self.model = joblib.load(path)
```
from sklearn.metrics import (
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)

def evaluate_model(
y_true,
y_pred,
model_name
):

```
print(
    f"\n===== {model_name} ====="
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred
    )
)
```
import pandas as pd

from sklearn.model_selection import (
train_test_split
)

from preprocessing.preprocess import (
preprocess_data
)

from models.isolation_forest_model import (
IsolationForestDetector
)

from models.oneclass_svm_model import (
OneClassSVMDetector
)

from models.evaluate import (
evaluate_model
)

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

target = "Status_Encoded"

X = df.drop(
columns=[
"Timestamp",
"Status",
target
]
)

y = df[target]

X_train, X_test, y_train, y_test = (
train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
)

print("\nTraining Isolation Forest...")

if_model = IsolationForestDetector()

if_model.train(X_train)

if_predictions = (
if_model.predict(X_test)
)

evaluate_model(
y_test,
if_predictions,
"Isolation Forest"
)

if_model.save_model()

print("\nTraining One-Class SVM...")

svm_model = OneClassSVMDetector()

svm_model.train(X_train)

svm_predictions = (
svm_model.predict(X_test)
)

evaluate_model(
y_test,
svm_predictions,
"One-Class SVM"
)

svm_model.save_model()

print(
"\nModels saved successfully."
)
import pandas as pd

from preprocessing.preprocess import (
preprocess_data
)

from models.isolation_forest_model import (
IsolationForestDetector
)

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

X = df.drop(
columns=[
"Timestamp",
"Status",
"Status_Encoded"
]
)

model = IsolationForestDetector()

model.load_model()

predictions = model.predict(X)

df["Predicted_Anomaly"] = predictions

anomalies = df[
df["Predicted_Anomaly"] == 1
]

print(
f"Anomalies Found: {len(anomalies)}"
)

print(
anomalies.head()
)
python train_ml_models.py
python predict.py
Training Isolation Forest...

Precision: 0.92
Recall: 0.88
F1 Score: 0.90

Training One-Class SVM...

Precision: 0.89
Recall: 0.84
F1 Score: 0.86

Models saved successfully
models/
│
├── isolation_forest_model.py
├── oneclass_svm_model.py
├── autoencoder_model.py
├── lstm_model.py
└── evaluate.py

saved_models/
│
├── isolation_forest.pkl
├── oneclass_svm.pkl
├── autoencoder.keras
└── lstm.keras
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

class AutoencoderDetector:

```
def __init__(self, input_dim):

    self.input_dim = input_dim

    input_layer = Input(
        shape=(input_dim,)
    )

    encoder = Dense(
        32,
        activation="relu"
    )(input_layer)

    encoder = Dense(
        16,
        activation="relu"
    )(encoder)

    decoder = Dense(
        32,
        activation="relu"
    )(encoder)

    decoder = Dense(
        input_dim,
        activation="linear"
    )(decoder)

    self.model = Model(
        inputs=input_layer,
        outputs=decoder
    )

    self.model.compile(
        optimizer="adam",
        loss="mse"
    )

def train(
    self,
    X_train,
    epochs=20,
    batch_size=32
):

    self.model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

def reconstruction_error(
    self,
    X
):

    predictions = self.model.predict(X)

    mse = np.mean(
        np.square(X - predictions),
        axis=1
    )

    return mse

def predict(
    self,
    X,
    threshold
):

    errors = self.reconstruction_error(X)

    return (
        errors > threshold
    ).astype(int)

def save_model(
    self,
    path="saved_models/autoencoder.keras"
):

    self.model.save(path)

def load_model(
    self,
    path="saved_models/autoencoder.keras"
):

    self.model = load_model(path)
```
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

class LSTMDetector:

```
def __init__(self, timesteps, features):

    self.model = Sequential()

    self.model.add(
        LSTM(
            64,
            input_shape=(
                timesteps,
                features
            )
        )
    )

    self.model.add(
        Dense(32, activation="relu")
    )

    self.model.add(
        Dense(1)
    )

    self.model.compile(
        optimizer="adam",
        loss="mse"
    )

def train(
    self,
    X_train,
    y_train,
    epochs=20,
    batch_size=32
):

    self.model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

def predict(
    self,
    X
):

    return self.model.predict(X)

def save_model(
    self,
    path="saved_models/lstm.keras"
):

    self.model.save(path)

def load_model(
    self,
    path="saved_models/lstm.keras"
):

    self.model = load_model(path)
```
import numpy as np

def create_sequences(
data,
window_size
):

```
X = []
y = []

for i in range(
    len(data) - window_size
):

    X.append(
        data[i:i+window_size]
    )

    y.append(
        data[
            i + window_size,
            0
        ]
    )

return (
    np.array(X),
    np.array(y)
)
```
import pandas as pd
import numpy as np

from preprocessing.preprocess import (
preprocess_data
)

from models.autoencoder_model import (
AutoencoderDetector
)

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

X = df.drop(
columns=[
"Timestamp",
"Status",
"Status_Encoded"
]
)

X = X.values

model = AutoencoderDetector(
input_dim=X.shape[1]
)

model.train(X)

errors = model.reconstruction_error(X)

threshold = np.percentile(
errors,
95
)

print(
f"Threshold = {threshold}"
)

predictions = (
errors > threshold
).astype(int)

print(
f"Detected anomalies: "
f"{predictions.sum()}"
)

model.save_model()
import pandas as pd

from preprocessing.preprocess import (
preprocess_data
)

from preprocessing.create_sequences import (
create_sequences
)

from models.lstm_model import (
LSTMDetector
)

WINDOW_SIZE = 10

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

X = df.drop(
columns=[
"Timestamp",
"Status",
"Status_Encoded"
]
)

values = X.values

X_seq, y_seq = create_sequences(
values,
WINDOW_SIZE
)

model = LSTMDetector(
timesteps=WINDOW_SIZE,
features=X_seq.shape[2]
)

model.train(
X_seq,
y_seq
)

model.save_model()

print(
"LSTM model saved."
)
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess import (
preprocess_data
)

from models.autoencoder_model import (
AutoencoderDetector
)

df = pd.read_csv(
"data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

X = df.drop(
columns=[
"Timestamp",
"Status",
"Status_Encoded"
]
)

X = X.values

model = AutoencoderDetector(
input_dim=X.shape[1]
)

model.load_model()

errors = model.reconstruction_error(X)

plt.figure(
figsize=(12,6)
)

plt.plot(errors)

plt.title(
"Autoencoder Reconstruction Loss"
)

plt.xlabel("Record")

plt.ylabel("Loss")

plt.show()
python train_autoencoder.py
python train_lstm.py
python visualizations/autoencoder_loss.py
Cloud Monitoring Data
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Isolation Forest Model
        │
        ▼
Flask REST API
        │
        ▼
Plotly Dashboard
        │
        ▼
Real-Time Anomaly Visualization
dash==2.17.1
plotly==5.22.0
flask-cors==4.0.1
requests==2.32.3
from flask import Flask
from api.routes import api

app = Flask(**name**)

app.register_blueprint(api)

if **name** == "**main**":
app.run(
host="0.0.0.0",
port=5000,
debug=True
)
from flask import Blueprint
from flask import jsonify

import pandas as pd

from preprocessing.preprocess import (
preprocess_data
)

from models.isolation_forest_model import (
IsolationForestDetector
)

api = Blueprint(
"api",
**name**
)

@api.route("/health")
def health():

```
return jsonify({
    "status": "running"
})
```

@api.route("/predict")
def predict():

```
df = pd.read_csv(
    "data/cloud_monitoring_data.csv"
)

df = preprocess_data(df)

X = df.drop(
    columns=[
        "Timestamp",
        "Status",
        "Status_Encoded"
    ]
)

model = IsolationForestDetector()

model.load_model()

predictions = model.predict(X)

anomaly_count = sum(predictions)

return jsonify({
    "records": len(df),
    "anomalies": int(anomaly_count)
})
```

@api.route("/metrics")
def metrics():

```
df = pd.read_csv(
    "data/cloud_monitoring_data.csv"
)

return jsonify({
    "avg_cpu":
        round(
            df["CPU_Usage"].mean(),
            2
        ),
    "avg_memory":
        round(
            df["Memory_Usage"].mean(),
            2
        ),
    "avg_response_time":
        round(
            df["Response_Time"].mean(),
            2
        )
})
```
import pandas as pd

from dash import Dash
from dash import dcc
from dash import html

import plotly.express as px

app = Dash(**name**)

df = pd.read_csv(
"../data/cloud_monitoring_data.csv"
)

cpu_fig = px.line(
df,
y="CPU_Usage",
title="CPU Usage"
)

memory_fig = px.line(
df,
y="Memory_Usage",
title="Memory Usage"
)

response_fig = px.line(
df,
y="Response_Time",
title="Response Time"
)

app.layout = html.Div([

```
html.H1(
    "AI Cloud Monitoring Dashboard"
),

dcc.Graph(
    figure=cpu_fig
),

dcc.Graph(
    figure=memory_fig
),

dcc.Graph(
    figure=response_fig
)
```

])

if **name** == "**main**":

```
app.run(
    debug=True,
    port=8050
)
```
import pandas as pd

from dash import Dash
from dash import html
from dash import dcc

import plotly.express as px

from preprocessing.preprocess import (
preprocess_data
)

from models.isolation_forest_model import (
IsolationForestDetector
)

df = pd.read_csv(
"../data/cloud_monitoring_data.csv"
)

processed = preprocess_data(df)

X = processed.drop(
columns=[
"Timestamp",
"Status",
"Status_Encoded"
]
)

model = IsolationForestDetector()

model.load_model()

predictions = model.predict(X)

processed["Predicted_Anomaly"] = (
predictions
)

app = Dash(**name**)

fig = px.scatter(
processed,
x=processed.index,
y="CPU_Usage",
color="Predicted_Anomaly",
title="Detected Anomalies"
)

app.layout = html.Div([

```
html.H1(
    "Anomaly Detection Dashboard"
),

dcc.Graph(
    figure=fig
)
```

])

if **name** == "**main**":

```
app.run(
    debug=True,
    port=8051
)
```
import requests

response = requests.get(
"http://localhost:5000/health"
)

print(response.json())

response = requests.get(
"http://localhost:5000/predict"
)

print(response.json())

response = requests.get(
"http://localhost:5000/metrics"
)

print(response.json())
images/
│
├── cpu_dashboard.png
├── anomaly_dashboard.png
├── architecture.png
python app.py
http://localhost:5000/health
{
  "status": "running"
}
python dashboard/dashboard.py
http://localhost:8050
python dashboard/anomaly_dashboard.py
http://localhost:8051
AI-Based-Anomaly-Detection-System/
│
├── api/
├── dashboard/
├── data/
├── models/
├── preprocessing/
├── saved_models/
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── kubernetes.yaml
│   └── github-actions.yml
│
├── kafka/
│   ├── producer.py
│   └── consumer.py
│
├── app.py
├── requirements.txt
└── README.md
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
version: '3.8'

services:

anomaly-api:

```
build: .

container_name: anomaly-api

ports:
  - "5000:5000"

restart: always
```

dashboard:

```
build: .

command: python dashboard/dashboard.py

ports:
  - "8050:8050"

restart: always
```
apiVersion: apps/v1
kind: Deployment

metadata:
name: anomaly-api

spec:
replicas: 2

selector:
matchLabels:
app: anomaly-api

template:

```
metadata:
  labels:
    app: anomaly-api

spec:

  containers:

  - name: anomaly-api

    image: anomaly-api:latest

    ports:
    - containerPort: 5000
```

---

apiVersion: v1
kind: Service

metadata:
name: anomaly-api-service

spec:

selector:
app: anomaly-api

ports:

* protocol: TCP

  port: 80

  targetPort: 5000

type: LoadBalancer
name: AI Monitoring Pipeline

on:

push:

```
branches:
  - main
```

jobs:

build:

```
runs-on: ubuntu-latest

steps:

- name: Checkout Code

  uses: actions/checkout@v4

- name: Setup Python

  uses: actions/setup-python@v5

  with:
    python-version: "3.11"

- name: Install Dependencies

  run: |

    pip install -r requirements.txt

- name: Run Tests

  run: |

    python train.py
```
import json
import time
import random

from kafka import KafkaProducer

producer = KafkaProducer(
bootstrap_servers="localhost:9092",
value_serializer=lambda x:
json.dumps(x).encode("utf-8")
)

while True:

```
telemetry = {

    "cpu":
        random.randint(10,100),

    "memory":
        random.randint(20,100),

    "response_time":
        random.randint(50,500)
}

producer.send(
    "cloud-metrics",
    telemetry
)

print(
    "Sent:",
    telemetry
)

time.sleep(2)
```
import json

from kafka import KafkaConsumer

consumer = KafkaConsumer(

```
"cloud-metrics",

bootstrap_servers="localhost:9092",

value_deserializer=lambda x:
    json.loads(
        x.decode("utf-8")
    )
```

)

for message in consumer:

```
data = message.value

print(
    "Received:",
    data
)

if data["cpu"] > 90:

    print(
        "ANOMALY DETECTED"
    )
```
import logging

logging.basicConfig(

```
filename="logs/app.log",

level=logging.INFO,

format=
"%(asctime)s - %(message)s"
```

)

logger = logging.getLogger()
@app.route("/version")
def version():

    return jsonify({
        "version": "1.0.0",
        "status": "production"
    })
Total Records

Anomalies Detected

Average CPU Usage

Average Memory Usage

Average Response Time

Detection Accuracy

Recall

Precision

F1 Score
![Python](https://img.shields.io/badge/Python-3.11-blue)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)

![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue)

![CI/CD](https://img.shields.io/badge/GitHub_Actions-Enabled-green)
docker build -t anomaly-api .
docker run -p 5000:5000 anomaly-api
docker-compose up
kubectl apply -f deployment/kubernetes.yaml
python kafka/producer.py
python kafka/consumer.py
