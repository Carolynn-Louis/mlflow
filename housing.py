import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_data(data_path):
    data = pd.read_csv(data_path)

    # Identify the target variable
    target_variable = "median_house_value"

    # Separate features and target
    X = data.drop(columns=[target_variable])
    y = data[[target_variable]]

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create a preprocessing pipeline for the categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'  # Keep other columns (numerical ones) as they are
    )

    # Apply the preprocessing pipeline to the features
    X_transformed = preprocessor.fit_transform(X)

    # Split the data into training and test sets (75% training, 25% test)
    train_x, test_x, train_y, test_y = train_test_split(X_transformed, y, test_size=0.25, random_state=42)

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Path to the house price dataset
    data_path = os.path.join("datasets", "housing", "housing.csv")
    train_x, train_y, test_x, test_y = load_data(data_path)

    with mlflow.start_run():
        # Train a HistGradientBoostingRegressor model
        model = HistGradientBoostingRegressor()
        model.fit(train_x, train_y.values.ravel())

        # Predict and evaluate
        predicted_prices = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_prices)

        print("HistGradientBoostingRegressor model:")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log metrics and model to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
