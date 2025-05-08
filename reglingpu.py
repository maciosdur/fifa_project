# -*- coding: utf-8 -*-
"""
Poprawiona implementacja regresji liniowej w PyTorch dla danych FIFA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
print(f"Czy CUDA jest dostępne: {torch.cuda.is_available()}")
print(f"Nazwa karty graficznej: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Brak GPU'}")
# 1. Wczytanie danych
try:
    df = pd.read_csv('players_22.csv', encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)

# 2. Przygotowanie danych
features = [
    'age', 'potential', 'height_cm', 'weight_kg',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'preferred_foot', 'player_positions'
]
target = 'overall'

df = df[features + [target]].dropna()

# 3. Definicja cech
numeric_features = ['age', 'potential', 'height_cm', 'weight_kg',
                   'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
categorical_features = ['preferred_foot', 'player_positions']

# 4. Pipeline przetwarzania
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

# 5. Przygotowanie danych
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przetwarzanie danych
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obliczenie wymiarowości danych wejściowych
input_dim = X_train_processed.shape[1]  # Definiujemy input_dim przed użyciem

# 6. Konwersja do tensorów PyTorch
X_train_tensor = torch.FloatTensor(X_train_processed.toarray())
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_processed.toarray())
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

# 7. Definicja modelu
class LinearRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# 8. Funkcja treningowa
def train_model(model, X_train, y_train, X_test, y_test, device='cpu', 
                batch_size=64, epochs=100, learning_rate=0.01):
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Przygotowanie DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    test_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()  # Obliczenie gradientów
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Średni loss dla epoki
        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)
        
        # Ewaluacja na testowym
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device))
            test_loss_history.append(test_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_loss.item():.4f}')
    
    return loss_history, test_loss_history

# 9. Inicjalizacja modeli
model_cpu = LinearRegressionPyTorch(input_dim)  # Teraz input_dim jest zdefiniowane

# 10. Trening na CPU
print("\nTrening na CPU...")
start_time = time.time()
loss_history_cpu, test_loss_cpu = train_model(model_cpu, X_train_tensor, y_train_tensor,
                                            X_test_tensor, y_test_tensor, device='cpu')
cpu_time = time.time() - start_time
print(f"Czas treningu na CPU: {cpu_time:.2f} sekund")

# 11. Trening na GPU
if torch.cuda.is_available():
    model_gpu = LinearRegressionPyTorch(input_dim)
    
    X_train_gpu = X_train_tensor.to('cuda')
    y_train_gpu = y_train_tensor.to('cuda')
    X_test_gpu = X_test_tensor.to('cuda')
    y_test_gpu = y_test_tensor.to('cuda')
    
    print("\nTrening na GPU...")
    start_time = time.time()
    loss_history_gpu, test_loss_gpu = train_model(model_gpu, X_train_gpu, y_train_gpu, 
                                                X_test_gpu, y_test_gpu, device='cuda')
    gpu_time = time.time() - start_time
    print(f"Czas treningu na GPU: {gpu_time:.2f} sekund")

# 12. Porównanie wyników
if torch.cuda.is_available():
    print(f"\nPrzyspieszenie GPU vs CPU: {cpu_time/gpu_time:.2f}x")
else:
    print("\nGPU nie jest dostępne, pokazano tylko wyniki CPU")

# 13. Wykresy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history_cpu, label='CPU Train')
plt.plot(test_loss_cpu, '--', label='CPU Test')
if torch.cuda.is_available():
    plt.plot(loss_history_gpu, label='GPU Train')
    plt.plot(test_loss_gpu, '--', label='GPU Test')
plt.title('Funkcja straty podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
if torch.cuda.is_available():
    plt.bar(['CPU', 'GPU'], [cpu_time, gpu_time])
    plt.title('Czas treningu (sekundy)')
else:
    plt.bar(['CPU'], [cpu_time])
    plt.title('Czas treningu (CPU)')
plt.show()

# 14. Ewaluacja końcowa
def evaluate_model(model, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X.toarray()).to(device)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, r2

print("\nEwaluacja końcowa:")
mse_cpu, r2_cpu = evaluate_model(model_cpu, X_test_processed, y_test, device='cpu')
print(f"CPU - MSE: {mse_cpu:.4f}, R2: {r2_cpu:.4f}")

if torch.cuda.is_available():
    mse_gpu, r2_gpu = evaluate_model(model_gpu, X_test_processed, y_test, device='cuda')
    print(f"GPU - MSE: {mse_gpu:.4f}, R2: {r2_gpu:.4f}")