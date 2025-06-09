import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class LinearRegressionClosedForm:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Zamień X na tablicę NumPy jeśli to sparse matrix
        if hasattr(X, "toarray"):
            X = X.toarray()
        # Dodajemy kolumnę jedynek dla biasu
        X_aug = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Obliczanie parametrów: theta = (XTX)_inv*XTy
        XTX = np.dot(X_aug.T, X_aug)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_aug.T, y)
        theta = np.dot(XTX_inv, XTy)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.dot(X, self.weights) + self.bias


class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iter=1000, batch_size=32):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.loss_history = []
    
    def _compute_gradient(self, X_batch, y_batch):
        n_samples = X_batch.shape[0]
        predictions = np.dot(X_batch, self.weights) + self.bias
        errors = predictions - y_batch
        
        dw = (2/n_samples) * np.dot(X_batch.T, errors)
        db = (2/n_samples) * np.sum(errors)
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices].to_numpy()

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                dw, db = self._compute_gradient(X_batch, y_batch)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            predictions = self.predict(X)
            mse = np.mean((predictions - y.to_numpy())**2)
            self.loss_history.append(mse)
    
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

# Trenowanie modeli
# Zamknięta formuła
lr_closed = LinearRegressionClosedForm()
lr_closed.fit(X_train_processed, y_train.to_numpy())  # <-- ważne!
y_pred_closed = lr_closed.predict(X_test_processed)
mse_closed = mean_squared_error(y_test, y_pred_closed)

# Gradient prosty
lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
lr_gd.fit(X_train_processed, y_train)
y_pred_gd = lr_gd.predict(X_test_processed)
mse_gd = mean_squared_error(y_test, y_pred_gd)

# Scikit-learn
from sklearn.linear_model import LinearRegression
lr_sklearn = LinearRegression(fit_intercept=True)
lr_sklearn.fit(X_train_processed, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_processed)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

# Wyniki
print("MSE - Zamknięta formuła:", mse_closed)
print("MSE - Gradient prosty:", mse_gd)
print("MSE - Scikit-learn:", mse_sklearn)

# Porównanie wag
print("\nPorównanie wag:")
print("Zamknięta formuła:", lr_closed.weights)
print("Gradient prosty:", lr_gd.weights)
print("Scikit-learn:", lr_sklearn.coef_)


import matplotlib.pyplot as plt

plt.plot(lr_gd.loss_history)
plt.title('Proces uczenia (Gradient prosty)')
plt.xlabel('Iteracja')
plt.ylabel('MSE')
plt.show()