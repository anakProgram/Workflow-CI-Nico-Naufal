import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("Stock_Prediction")

mlflow.sklearn.autolog(log_models=True)

df = pd.read_csv(
    "Workflow-CI/dataset_preprocessing/hasil_preprocessing.csv"
)

X = pd.get_dummies(
    df.drop("LastPrice", axis=1),
    drop_first=True
)

y = df["LastPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_metric("mse", mse)

    print(f"MSE: {mse}")