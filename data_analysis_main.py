import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja folderów wyjściowych
output_dir = "output"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Wczytanie danych
try:
    df = pd.read_csv('players_22.csv')
    print("Dane wczytane pomyślnie!")
except FileNotFoundError:
    print("Błąd: Nie znaleziono pliku 'players_22.csv'")
    print("Upewnij się, że plik jest w tym samym folderze co skrypt")
    exit()

# Przygotowanie danych - wybór kolumn
analysis_cols = [
    'overall', 'potential', 'age', 'height_cm', 'weight_kg',
    'nationality_name', 'club_name', 'value_eur', 'wage_eur',
    'preferred_foot', 'player_positions'
]
df = df[analysis_cols].copy()

# Funkcja do zapisywania wykresów
def save_plot(fig, filename):
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    print(f"Zapisano wykres: {path}")

# 1. Analiza podstawowych statystyk
basic_stats = df.describe(include='all').transpose()
basic_stats.to_csv(os.path.join(output_dir, "basic_stats.csv"))

# 2. Wizualizacje
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='overall', bins=30, kde=True)
plt.title("Rozkład ocen ogólnych graczy")
save_plot(plt, "overall_rating.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='preferred_foot', y='overall')
plt.title("Porównanie ocen w zależności od preferowanej nogi")
save_plot(plt, "foot_preference.png")
plt.close()

# 3. Zaawansowana analiza
top_countries = df['nationality_name'].value_counts().head(5).index
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df[df['nationality_name'].isin(top_countries)],
    x='nationality_name',
    y='value_eur'
)
plt.title("Wartość rynkowa graczy z top 5 krajów")
plt.xticks(rotation=45)
save_plot(plt, "value_by_country.png")
plt.close()

print("\nAnaliza zakończona pomyślnie!")
print(f"Wyniki zapisane w folderze: {output_dir}")

# Heatmapa korelacji
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Korelacje między cechami numerycznymi")
save_plot(plt, "correlation_heatmap.png")

# Analiza pozycji
df['main_position'] = df['player_positions'].str.split(',').str[0]
position_stats = df['main_position'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(x=position_stats.index, y=position_stats.values)
plt.xticks(rotation=45)
plt.title("Rozkład głównych pozycji graczy")
save_plot(plt, "positions_distribution.png")