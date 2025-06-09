import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
import copy

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iter=1000, batch_size=32, l2_alpha=0.0):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2_alpha = l2_alpha
        self.loss_history = []

    def _compute_gradient(self, X_batch, y_batch):
        n_samples = X_batch.shape[0]
        predictions = np.dot(X_batch, self.weights) + self.bias
        errors = predictions - y_batch
        dw = (2/n_samples) * np.dot(X_batch.T, errors) + 2 * self.l2_alpha * self.weights
        db = (2/n_samples) * np.sum(errors)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                dw, db = self._compute_gradient(X_batch, y_batch)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            predictions = self.predict(X)
            mse = np.mean((predictions - (y if isinstance(y, np.ndarray) else y.to_numpy()))**2)
            mse_reg = mse + self.l2_alpha * np.sum(self.weights ** 2)
            self.loss_history.append(mse_reg)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
try:
    df = pd.read_csv('players_22.csv', encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)    

features = [
    'age', 'value_eur', 'potential', 'height_cm', 'weight_kg',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'preferred_foot'
]
numeric_features = [
    'age', 'value_eur', 'potential', 'height_cm', 'weight_kg',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'
]
categorical_features = ['preferred_foot']
target = 'overall'

df_clean = df[features + [target]].dropna()
X = df_clean[features]
y = df_clean[target]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # <-- poprawka
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Dodaj to:
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()


# Gradient prosty bez regularyzacji
lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32, l2_alpha=0.0)
lr_gd.fit(X_train_processed, y_train)
y_pred_gd = lr_gd.predict(X_test_processed)
mse_gd = mean_squared_error(y_test, y_pred_gd)

# Gradient prosty z regularyzacją L2 (Ridge)
lr_gd_l2 = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32, l2_alpha=0.1)
lr_gd_l2.fit(X_train_processed, y_train)
y_pred_gd_l2 = lr_gd_l2.predict(X_test_processed)
mse_gd_l2 = mean_squared_error(y_test, y_pred_gd_l2)

# Scikit-learn
from sklearn.linear_model import LinearRegression
lr_sklearn = LinearRegression(fit_intercept=True)
lr_sklearn.fit(X_train_processed, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_processed)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

print("MSE - Gradient prosty:", mse_gd)
print("MSE - Gradient prosty + L2:", mse_gd_l2)
print("MSE - Scikit-learn:", mse_sklearn)

# Wypisz nazwy cech i odpowiadające im wagi
feature_names = preprocessor.get_feature_names_out()
print("\nWagi gradientu prostego:")
for name, weight in zip(feature_names, lr_gd.weights):
    print(f"{name}: {weight:.4f}")

print("\nWagi gradientu prostego + L2:")
for name, weight in zip(feature_names, lr_gd_l2.weights):
    print(f"{name}: {weight:.4f}")

print("\nWagi scikit-learn:")
for name, weight in zip(feature_names, lr_sklearn.coef_):
    print(f"{name}: {weight:.4f}")

import matplotlib.pyplot as plt

plt.plot(lr_gd.loss_history)
plt.title('Proces uczenia (Gradient prosty)')
plt.xlabel('Iteracja')
plt.ylabel('MSE')
plt.show()

# Przygotuj dane (bez podziału train_test_split)
X_full = df_clean[features]
y_full = df_clean[target].values

# Nie przetwarzaj od razu! Przetwarzaj w każdej fold osobno:
kf = KFold(n_splits=3, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X_full), 1):
    X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
    y_train, y_test = y_full[train_index], y_full[test_index]

    # Pipeline fit/transform tylko na train!
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
    lr_gd.fit(X_train_processed, y_train)
    y_pred = lr_gd.predict(X_test_processed)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    print(f"Fold {fold} - MSE: {mse:.4f}, R²: {r2:.4f}")

print(f"\nŚrednie MSE (3-fold): {np.mean(mse_scores):.4f} (±{np.std(mse_scores):.4f})")
print(f"Średnie R² (3-fold): {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")


# --- Analiza wpływu liczby cech ---
feature_subsets = [
    ['age', 'overall', 'potential'],  # minimalny zestaw
    ['age', 'overall', 'potential', 'pace', 'shooting', 'passing'],  # średni zestaw
    features  # wszystkie cechy
]

for i, features_subset in enumerate(feature_subsets):
    X_sub = df_clean[features_subset]
    y_sub = df_clean[target]
    # Wyodrębnij cechy numeryczne i kategoryczne dla tego podzbioru
    numeric = [col for col in features_subset if col in numeric_features]
    categorical = [col for col in features_subset if col in categorical_features]
    # Zbuduj nowy preprocessor dla tego podzbioru
    preproc = ColumnTransformer([
        ('num', numeric_transformer, numeric),
        ('cat', categorical_transformer, categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    X_train_processed = preproc.fit_transform(X_train)
    X_test_processed = preproc.transform(X_test)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()
    lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
    lr_gd.fit(X_train_processed, y_train.to_numpy())
    train_pred = lr_gd.predict(X_train_processed)
    test_pred = lr_gd.predict(X_test_processed)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    print(f"Subset {i+1}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

# --- Analiza wpływu wielkości zbioru danych ---
sample_sizes = [len(df_clean)//10*i for i in range(1, 11)]
train_size_errors = []
test_size_errors = []
repeats = 5
for size in sample_sizes:
    train_mses = []
    test_mses = []
    for seed in range(repeats):
        df_sample = df_clean.sample(size, random_state=seed)
        X_sample = df_sample[features]
        y_sample = df_sample[target]
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=seed)
        X_train_s_processed = preprocessor.fit_transform(X_train_s)
        X_test_s_processed = preprocessor.transform(X_test_s)
        if hasattr(X_train_s_processed, "toarray"):
            X_train_s_processed = X_train_s_processed.toarray()
        if hasattr(X_test_s_processed, "toarray"):
            X_test_s_processed = X_test_s_processed.toarray()
        lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
        lr_gd.fit(X_train_s_processed, y_train_s.to_numpy())
        train_mses.append(mean_squared_error(y_train_s, lr_gd.predict(X_train_s_processed)))
        test_mses.append(mean_squared_error(y_test_s, lr_gd.predict(X_test_s_processed)))
    train_size_errors.append(np.mean(train_mses))
    test_size_errors.append(np.mean(test_mses))

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, train_size_errors, 'b-o', label='Train error')
plt.plot(sample_sizes, test_size_errors, 'r-o', label='Test error')
plt.xlabel('Liczba próbek w zbiorze treningowym')
plt.ylabel('Mean Squared Error')
plt.title('Wpływ wielkości zbioru danych na błąd modelu (Gradient Descent)')
plt.legend()
plt.grid(True)
plt.show()