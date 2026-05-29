import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Log your model training code here
mlflow.start_run()

# Your training logic
print("Training model...")

mlflow.end_run()