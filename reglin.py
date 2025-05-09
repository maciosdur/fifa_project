import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionClosedForm:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Dodajemy kolumnę jedynek dla biasu
        X_aug = np.c_[np.ones(X.shape[0]), X]
        
        # Obliczanie parametrów: theta = (XTX)_inv*XTy
        XTX = np.dot(X_aug.T, X_aug)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_aug.T, y)
        theta = np.dot(XTX_inv, XTy)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def predict(self, X):
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
            # Podział na batch'e
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                dw, db = self._compute_gradient(X_batch, y_batch)
                
                # Aktualizacja parametrów
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Śledzenie funkcji kosztu (MSE)
            predictions = self.predict(X)
            mse = np.mean((predictions - y)**2)
            self.loss_history.append(mse)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    
    
    
    
    
try:
    df = pd.read_csv('players_22.csv', encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)    

features = ['age', 'value_eur', 'potential', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
target = 'overall'

df_clean = df[features + [target]].dropna()

X = df_clean[features].values
y = df_clean[target].values

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trenowanie modeli
# 1. implementacja zamkniętej formuły
lr_closed = LinearRegressionClosedForm()
lr_closed.fit(X_train, y_train)
y_pred_closed = lr_closed.predict(X_test)
mse_closed = mean_squared_error(y_test, y_pred_closed)

# 2. implementacja gradientu prostego
lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
lr_gd.fit(X_train, y_train)
y_pred_gd = lr_gd.predict(X_test)
mse_gd = mean_squared_error(y_test, y_pred_gd)

# 3. Scikit-learn
from sklearn.linear_model import LinearRegression
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)
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