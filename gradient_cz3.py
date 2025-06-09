import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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


lr_gd = LinearRegressionGradientDescent(learning_rate=0.005, n_iter=100, batch_size=32)
lr_gd.fit(X_train, y_train)
y_pred_gd = lr_gd.predict(X_test)
mse_gd = mean_squared_error(y_test, y_pred_gd)

plt.plot(lr_gd.loss_history)
plt.title('Proces uczenia (Gradient prosty)')
plt.xlabel('Iteracja')
plt.ylabel('MSE')
plt.show()