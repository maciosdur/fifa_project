# -*- coding: utf-8 -*-
"""
Analiza underfittingu i overfittingu dla modelu FIFA - wersja poprawiona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Wczytanie danych
try:
    df = pd.read_csv('players_22.csv', encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)

# 2. Przygotowanie danych
features = ['age', 'value_eur', 'potential', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
target = 'overall'

df_clean = df[features + [target]].dropna()

# Wybieramy jedną cechę dla wizualizacji (np. 'value_eur')
feature_to_plot = 'value_eur'
X = df_clean[[feature_to_plot]].values
y = df_clean[target].values

# Sortujemy dane dla lepszej wizualizacji
sort_idx = np.argsort(X[:, 0])
X = X[sort_idx]
y = y[sort_idx]

# 3. Definiujemy stopnie wielomianu do przetestowania
degrees = [1, 2, 3, 5]

# # 4. Przygotowanie wykresu
# plt.figure(figsize=(14, 5))
# for i, degree in enumerate(degrees):
#     ax = plt.subplot(1, len(degrees), i + 1)
#     plt.setp(ax, xticks=(), yticks=())

#     # Tworzenie pipeline z PolynomialFeatures
#     polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
#     linear_regression = LinearRegression()
#     pipeline = Pipeline([
#         ("polynomial_features", polynomial_features),
#         ("linear_regression", linear_regression),
#     ])
    
#     # Trening modelu
#     pipeline.fit(X, y)
    
#     # Ocena modelu przy użyciu walidacji krzyżowej
#     scores = cross_val_score(
#         pipeline, X, y, scoring="neg_mean_squared_error", cv=5
#     )
    
#     # Generowanie punktów do wykresu
#     X_test = np.linspace(X.min(), X.max(), 100)
#     plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
#     plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
#     plt.xlabel(feature_to_plot)
#     plt.ylabel(target)
#     plt.legend(loc="best")
#     plt.title(
#         "Stopień {}\nMSE = {:.2e}(+/- {:.2e})".format(
#             degree, -scores.mean(), scores.std()
#         )
#     )

# plt.tight_layout()
# plt.show()
# Analiza underfitting/overfitting dla wielu cech
# X_all = df_clean[features].values
# y_all = df_clean[target].values

# train_errors = []
# test_errors = []
# cv_errors = []

# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# for degree in degrees:
#     pipeline = Pipeline([
#         ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
#         ("reg", LinearRegression())
#     ])
#     pipeline.fit(X_train, y_train)
#     train_pred = pipeline.predict(X_train)
#     test_pred = pipeline.predict(X_test)
#     train_errors.append(mean_squared_error(y_train, train_pred))
#     test_errors.append(mean_squared_error(y_test, test_pred))
#     cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
#     cv_errors.append(-cv_scores.mean())

# plt.figure(figsize=(8, 5))
# plt.plot(degrees, train_errors, 'bo-', label='Train error')
# plt.plot(degrees, test_errors, 'ro-', label='Test error')
# plt.plot(degrees, cv_errors, 'go-', label='CV error')
# plt.xlabel('Stopień wielomianu')
# plt.ylabel('Mean Squared Error')
# plt.title('Underfitting vs Overfitting (wiele cech)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Analiza wpływu liczby cech
feature_subsets = [
    ['age', 'value_eur', 'potential'],  # minimalny zestaw
    ['age', 'value_eur', 'potential', 'pace', 'shooting', 'passing'],  # średni zestaw
    features  # wszystkie cechy numeryczne
]
X_all = df_clean[features].values
y_all = df_clean[target].values

plt.figure(figsize=(10, 6))
for i, features_subset in enumerate(feature_subsets):
    X_subset = df_clean[features_subset]
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_subset, y_all, test_size=0.2, random_state=42)
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X_train_sub, y_train_sub)
    train_score = mean_squared_error(y_train_sub, pipeline.predict(X_train_sub))
    test_score = mean_squared_error(y_test_sub, pipeline.predict(X_test_sub))
    plt.bar([f'Subset {i+1} - Train'], [train_score], color=plt.cm.tab10(i))
    plt.bar([f'Subset {i+1} - Test'], [test_score], color=plt.cm.tab10(i), alpha=0.5)
plt.ylabel('Mean Squared Error')
plt.title('Wpływ liczby cech na błąd modelu')
plt.grid(True)
plt.show()

# Analiza wpływu wielkości zbioru danych
sample_sizes = [len(df_clean)//10*1,len(df_clean)//10*2,len(df_clean)//10*3,len(df_clean)//10*4,len(df_clean)//10*5,len(df_clean)//10*6,len(df_clean)//10*7,len(df_clean)//10*8,len(df_clean)//10*9 , len(df_clean)]
train_size_errors = []
test_size_errors = []

repeats = 20
for size in sample_sizes:
    train_mses = []
    test_mses = []
    for seed in range(repeats):
        df_sample = df_clean.sample(size)
        X_sample = df_sample[features]
        y_sample = df_sample[target]
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sample, y_sample, test_size=0.2)
        pipeline = make_pipeline(StandardScaler(), LinearRegression())
        pipeline.fit(X_train_s, y_train_s)
        train_mses.append(mean_squared_error(y_train_s, pipeline.predict(X_train_s)))
        test_mses.append(mean_squared_error(y_test_s, pipeline.predict(X_test_s)))
    train_size_errors.append(np.mean(train_mses))
    test_size_errors.append(np.mean(test_mses))
    
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes[:len(train_size_errors)], train_size_errors, 'b-o', label='Train error')
plt.plot(sample_sizes[:len(test_size_errors)], test_size_errors, 'r-o', label='Test error')
plt.xlabel('Liczba próbek w zbiorze treningowym')
plt.ylabel('Mean Squared Error')
plt.title('Wpływ wielkości zbioru danych na błąd modelu')
plt.legend()
plt.grid(True)
plt.show()