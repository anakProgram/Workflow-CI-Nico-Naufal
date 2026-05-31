import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Experiment
mlflow.set_experiment("Stock_Prediction")

# Wajib untuk kriteria Basic
mlflow.sklearn.autolog(log_models=True)

# Load dataset
df = pd.read_csv(
    "dataset_preprocessing/hasil_preprocessing.csv"
)

# Features & target
X = pd.get_dummies(
    df.drop("LastPrice", axis=1),
    drop_first=True
)

y = df["LastPrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

# Optional metric tambahan
mlflow.log_metric("mse", mse)

print(f"MSE: {mse}")
print("Training selesai")