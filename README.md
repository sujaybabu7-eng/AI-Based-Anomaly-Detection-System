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
