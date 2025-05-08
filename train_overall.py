# -*- coding: utf-8 -*-
"""
Ostateczna, działająca wersja skryptu do trenowania modeli FIFA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

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


# 4. Definicja cech
numeric_features = ['age', 'value_eur', 'potential', 'height_cm', 'weight_kg',
                   'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
categorical_features = ['preferred_foot', 'player_positions']
target = 'overall'

# 5. Podział danych na zbiór treningowy i testowy
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Pipeline przetwarzania
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

# 7. Definicja modeli
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# 8. Trening i ewaluacja
results = []

for name, model in models.items():
    try:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
        
        print(f"{name:25} | MSE: {mse:>8.4f} | RMSE: {rmse:>6.4f} | R2: {r2:.3f}")
    except Exception as e:
        print(f"Błąd w modelu {name}: {str(e)}")

# 9. Podsumowanie wyników
if results:
    results_df = pd.DataFrame(results)
    print("\nPodsumowanie wyników:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Zapisz najlepszy model
    best_model_idx = results_df['R2'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[best_model_name])
    ])
    
    final_pipeline.fit(X, y)  # Trening na wszystkich danych
    joblib.dump(final_pipeline, 'fifa_overall_predictor.pkl')
    print(f"\nZapisano najlepszy model ({best_model_name}) do pliku 'fifa_overall_predictor.pkl'")
else:
    print("Żaden model nie został pomyślnie wytrenowany.")