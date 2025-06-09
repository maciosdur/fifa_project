import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Wczytanie danych
try:
    df = pd.read_csv('players_22.csv', encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)

# 2. Przygotowanie danych
features = [
    'age', 'overall', 'potential', 'height_cm', 'weight_kg',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'preferred_foot', 'player_positions', 'value_eur'
]
df = df[features].dropna(subset=['value_eur'])
df = df[~df['value_eur'].isin([np.inf, -np.inf])]

# 3. Definicja cech
numeric_features = ['age', 'overall' , 'potential', 'height_cm', 'weight_kg',
                   'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
categorical_features = ['preferred_foot', 'player_positions']
target = 'value_eur'

# 4. Podział danych na zbiór treningowy i testowy
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipeline przetwarzania
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 6. GridSearchCV dla DecisionTreeRegressor
dt_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])
dt_param_grid = {
    'model__max_depth': [3, 5, 7, 10, None],
    'model__min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(dt_pipe, dt_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print("Najlepsze parametry dla DecisionTreeRegressor:", dt_grid.best_params_)

# 7. GridSearchCV dla RandomForestRegressor
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
rf_param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Najlepsze parametry dla RandomForestRegressor:", rf_grid.best_params_)

# 8. Ewaluacja najlepszych modeli z GridSearch
results = []
for name, model in [
    ('Decision Tree (GridSearch)', dt_grid.best_estimator_),
    ('Random Forest (GridSearch)', rf_grid.best_estimator_)
]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })
    print(f"{name:30} | MSE: {mse:>8.4f} | RMSE: {rmse:>6.4f} | R2: {r2:.3f}")

# 8b. Ewaluacja modeli bez strojenia hiperparametrów (domyślne parametry)
dt_default = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])
rf_default = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

for name, model in [
    ('Decision Tree (Default)', dt_default),
    ('Random Forest (Default)', rf_default)
]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })
    print(f"{name:30} | MSE: {mse:>8.4f} | RMSE: {rmse:>6.4f} | R2: {r2:.3f}")

# 9. Podsumowanie wyników
if results:
    results_df = pd.DataFrame(results)
    print("\nPodsumowanie wyników:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
else:
    print("Żaden model nie został pomyślnie wytrenowany.")

print("""
Dlaczego przeszukiwanie hiperparametrów jest trudne?
- Liczba możliwych kombinacji rośnie wykładniczo z liczbą parametrów i ich wartości.
- Każda kombinacja wymaga osobnego treningu i walidacji modelu, co jest kosztowne obliczeniowo.
- Wyniki mogą być niestabilne przy małych zbiorach lub dużej liczbie parametrów.
- Często istnieje kompromis między dokładnością a złożonością modelu (overfitting/underfitting).
""")