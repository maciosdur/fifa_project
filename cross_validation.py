import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred) 


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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare for 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []
for train_index, test_index in kf.split(X_scaled):
    # Split data into training and test sets for this fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train and evaluate the model
    lr = LinearRegressionClosedForm()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # Calculate and store MSE for this fold
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"Fold MSE: {mse:.4f}")
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    print(f"R²: {r2:.4f}")

# Calculate average performance across all folds
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
print(f"Average MSE: {np.mean(mse_scores):.4f} (±{np.std(mse_scores):.4f})")
print(f"Average R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
