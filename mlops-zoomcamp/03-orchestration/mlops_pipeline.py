import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

dv = DictVectorizer(sparse=True)
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year=None, month=None):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet"
    print(f"Reading data from {filename}")
    df = pd.read_parquet(filename)
    return df

def preprocess_data(df):
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df

def feature_engineering(df_train, df_val, dv):
    target = 'duration'
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values
    return X_train, X_val, y_train, y_val

def train_model(X_train, X_val, y_train, y_val, dv):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    with mlflow.start_run():
    
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

def find_best_model(
    X_train, X_val, y_train, y_val, dv, max_evals=10
):
    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)

            booster = xgb.train(
                params=params,
                dtrain=xgb.DMatrix(X_train, label=y_train),
                num_boost_round=50,
                evals=[(xgb.DMatrix(X_val, label=y_val), 'validation')],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            y_pred = booster.predict(xgb.DMatrix(X_val))
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            return {'loss': rmse, 'status': STATUS_OK}

    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'min_child_weight': hp.uniform('min_child_weight', 1e-5, 10),
        'objective': 'reg:squarederror',
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'seed': 42
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    return best

def main(train_year,train_month, val_year,val_month):
    df_train = read_dataframe(train_year, train_month)
    df_val = read_dataframe(val_year, val_month)
    print(f"Train data shape: {df_train.shape}, Validation data shape: {df_val.shape}")
    df_train = preprocess_data(df_train)
    df_val = preprocess_data(df_val)

    # train_size = int(len(df) * 0.8)
    # df_train = df[:train_size]
    # df_val = df[train_size:]

    X_train, X_val, y_train, y_val = feature_engineering(df_train, df_val, dv)

    best_params = find_best_model(X_train, X_val, y_train, y_val, dv, max_evals=10)
    print("Best parameters:", best_params)

    train_model(X_train, X_val, y_train, y_val, dv)

if __name__ == "__main__":
    main(2021, "01", 2021, "02")  # Example: train on January 2021, validate on February 2021