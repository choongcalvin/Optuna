import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna
from time import time

# Load the data set
df = pd.read_csv("ml_house_data_set.csv")

# Remove unwanted fields
df.drop(['house_number', 'unit_number', 'street_name', 'zip_code'], axis=1, inplace=True)

# One-hot encode categorical data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])
X, y = features_df.drop('sale_price', axis=1).values, df['sale_price'].values

# Split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

start = time()

model = ensemble.GradientBoostingRegressor()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 4, 6),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 17),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_features': trial.suggest_float('max_features', 0.1, 0.6),
        'loss': trial.suggest_categorical('loss', ['absolute_error', 'squared_error', 'huber'])
    }

    model.set_params(**params)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return mae

study = optuna.create_study()
study.optimize(objective, n_trials=100)

speed = {'GradientBoosting': np.round(time() - start, 3)}

best_params = study.best_params
model.set_params(**best_params)

mae_train = mean_absolute_error(y_train, model.predict(X_train))
mae_test = mean_absolute_error(y_test, model.predict(X_test))

print("Best parameters:", best_params)
print(f"Run time: {speed['GradientBoosting']}s")
print("MAE on training data:", mae_train)
print("MAE on testing data:", mae_test)
