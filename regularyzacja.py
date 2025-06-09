import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Wczytaj dane
df = pd.read_csv('players_22.csv', encoding='latin-1', low_memory=False)
features = ['age','value_eur', 'potential', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
target = 'overall'
df_clean = df[features + [target]].dropna()

X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modele
models = {
    "Linear": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=1.0, max_iter=10000)
}

results = {}

for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # Pobierz wagi po skalowaniu
    coefs = pipeline.named_steps[model.__class__.__name__.lower()].coef_
    results[name] = {
        "mse": mse,
        "coefs": coefs
    }
    print(f"{name} - MSE: {mse:.2f}")
    print("Wagi cech:", dict(zip(features, coefs)))
    print("-" * 40)

# Porównanie wag
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
styles = ['-', '--', '-.']
for idx, (name, res) in enumerate(results.items()):
    plt.plot(features, res["coefs"], marker='o', label=name, linewidth=2, linestyle=styles[idx % len(styles)])
plt.xticks(rotation=45)
plt.ylabel("Wartość wagi")
plt.title("Porównanie wag cech dla różnych modeli")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from tabulate import tabulate
for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"Mean Squared Error (MSE): {res['mse']:.2f}")
    table = [[feature, f"{coef:.4f}"] for feature, coef in zip(features, res["coefs"])]
    print(tabulate(table, headers=["Cecha", "Waga"], tablefmt="github"))

print("\nPorównanie wag cech dla wszystkich modeli:")
all_table = []
for i, feature in enumerate(features):
    row = [feature]
    for name in models.keys():
        row.append(f"{results[name]['coefs'][i]:.4f}")
    all_table.append(row)
headers = ["Cecha"] + list(models.keys())
print(tabulate(all_table, headers=headers, tablefmt="github"))