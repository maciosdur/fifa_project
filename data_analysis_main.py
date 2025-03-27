import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Metoda do wczytywania danych
def load_data(file_path):    
    try:
        data = pd.read_csv(file_path)
        print(f"Dane wczytane pomyślnie z pliku: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{file_path}'")
        exit()
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        exit()

# Konfiguracja folderów wyjściowych
output_dir = "output"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Wczytanie danych za pomocą metody
df = load_data('players_22.csv')

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

#Funkcja do obliczania statystyk numerycznych
def calculate_numerical_stats(df):
    numerical_cols = df.select_dtypes(include=np.number).columns
    stats = df[numerical_cols].agg(
        ['count', 'mean', 'median', 'min', 'max', 'std', 
         lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]
    ).transpose()
    stats.columns = [
        'count', 'mean', 'median', 'min', 'max', 'std', 
        '5th_percentile', '95th_percentile'
    ]
    stats['missing_values'] = len(df) - stats['count']
    return stats

# Funkcja do obliczania statystyk kategorialnych
def calculate_categorical_stats(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    stats = []
    
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True)
        stats.append({
            'feature': col,
            'unique_classes': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'most_common_class': df[col].mode()[0] if not df[col].mode().empty else np.nan,
            'most_common_proportion': value_counts.iloc[0] if not value_counts.empty else np.nan,
            'classes_distribution': dict(value_counts.head(5))  # Top 5 klas
        })
    
    return pd.DataFrame(stats)

# Obliczenie i zapis statystyk
print("\nObliczanie statystyk...")
numerical_stats = calculate_numerical_stats(df)
categorical_stats = calculate_categorical_stats(df)

# Zapis do plików CSV
numerical_stats.to_csv(os.path.join(output_dir, "numerical_stats.csv"), float_format='%.2f')
categorical_stats.to_csv(os.path.join(output_dir, "categorical_stats.csv"), index=False)

# Dodatkowe zapisanie rozkładu klas kategorialnych do osobnego pliku
with open(os.path.join(output_dir, "categorical_distributions.txt"), 'w') as f:
    for _, row in categorical_stats.iterrows():
        f.write(f"\n--- {row['feature']} ---\n")
        for cls, prop in row['classes_distribution'].items():
            f.write(f"{cls}: {prop:.2%}\n")

print("\nStatystyki zapisane w plikach:")
print(f"- numerical_stats.csv")
print(f"- categorical_stats.csv")
print(f"- categorical_distributions.txt")

# =============================================
# Zakres prac na ocenę 3.5 z części I
# =============================================

# HISTOGRAM: Rozkład ocen overall
plt.figure(figsize=(16, 8))
min_wage = int(df['overall'].min())
max_wage = int(df['overall'].max())
bin_width = 1
bins = range(min_wage, max_wage + bin_width, bin_width)
ax = sns.histplot(data=df, x='overall', bins=bins, kde=True,
                 color='#1f77b4', edgecolor='white', linewidth=0.5)
mean_val = df['overall'].mean()
median_val = df['overall'].median()
q1 = df['overall'].quantile(0.25)
q3 = df['overall'].quantile(0.75)
p5 = df['overall'].quantile(0.05)
p95 = df['overall'].quantile(0.95)
ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(median_val, color='green', linestyle='-', linewidth=2, alpha=0.8)
ax.axvline(q1, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
ax.axvline(q3, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
ax.axvline(p5, color='purple', linestyle='-.', linewidth=1, alpha=0.8)
ax.axvline(p95, color='purple', linestyle='-.', linewidth=1, alpha=0.8)
ax.text(mean_val, ax.get_ylim()[1]*0.9, f'Średnia: {mean_val:.1f}', 
        rotation=90, va='top', ha='right', color='red')
ax.text(median_val, ax.get_ylim()[1]*0.85, f'Mediana: {median_val:.1f}', 
        rotation=90, va='top', ha='right', color='green')
ax.text(q1, ax.get_ylim()[1]*0.8, f'Q1: {q1:.1f}', 
        rotation=90, va='top', ha='right', color='orange')
ax.text(q3, ax.get_ylim()[1]*0.8, f'Q3: {q3:.1f}', 
        rotation=90, va='top', ha='right', color='orange')
ax.text(p5, ax.get_ylim()[1]*0.75, f'5%: {p5:.1f}', 
        rotation=90, va='top', ha='right', color='purple')
ax.text(p95, ax.get_ylim()[1]*0.75, f'95%: {p95:.1f}', 
        rotation=90, va='top', ha='right', color='purple')
plt.title("Rozkład ocen overall z kluczowymi statystykami", fontsize=16, pad=20)
plt.xlabel("Ocena overall", fontsize=12)
plt.ylabel("Gęstość", fontsize=12)
plt.xticks(range(40, 101, 2))
plt.xlim(40, 100)
plt.grid(axis='y', alpha=0.2)
save_plot(plt, "overall_distribution_clean.png")
plt.close()


# BOXPLOT 1: Porównanie ocen overall dla różnych pozycji (top 5 pozycji)
df['main_position'] = df['player_positions'].str.split(',').str[0]
position_stats = df['main_position'].value_counts()

top_positions = df['main_position'].value_counts().head(5).index
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=df[df['main_position'].isin(top_positions)],
    x='main_position',
    y='overall',
    hue='main_position', 
    palette='viridis',
    dodge=False 
)
plt.title('Rozkład ocen overall dla głównych pozycji (boxplot)', fontsize=14)
plt.xlabel('Pozycja', fontsize=12)
plt.ylabel('Ocena overall', fontsize=12)
plt.xticks(rotation=45)
save_plot(plt, "boxplot_positions.png")
plt.close()

# BOXPLOT 2: Wartość rynkowa vs preferowana noga
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x='preferred_foot',
    y='value_eur',
    hue='preferred_foot',  
    showfliers=False,
    palette='coolwarm'
)
plt.title('Wartość rynkowa wg preferowanej nogi (boxplot)', fontsize=14)
plt.xlabel('Preferowana noga', fontsize=12)
plt.ylabel('Wartość (EUR)', fontsize=12)
save_plot(plt, "boxplot_foot_value.png")
plt.close()

# VIOLINPLOT 1: Rozkład wieku dla top 5 narodowości
top_countries = df['nationality_name'].value_counts().head(5).index
plt.figure(figsize=(14, 7))
sns.violinplot(
    data=df[df['nationality_name'].isin(top_countries)],
    x='nationality_name',
    y='age',
    hue='nationality_name',  
    palette='magma',
    inner='quartile',
    legend=False
)
plt.title('Rozkład wieku graczy dla top 5 narodowości (violinplot)', fontsize=14)
plt.xlabel('Narodowość', fontsize=12)
plt.ylabel('Wiek', fontsize=12)
plt.xticks(rotation=45)
save_plot(plt, "violinplot_nationality_age.png")
plt.close()

# VIOLINPLOT 2: Porównanie potencjału dla pozycji
plt.figure(figsize=(16, 8))
sns.violinplot(
    data=df[df['main_position'].isin(top_positions)],
    x='main_position',
    y='potential',
    hue='preferred_foot', 
    split=True,
    palette='Set2',
    inner='stick'
)
plt.title('Potencjał graczy wg pozycji i preferowanej nogi (violinplot)', fontsize=14)
plt.xlabel('Pozycja', fontsize=12)
plt.ylabel('Potencjał', fontsize=12)
plt.legend(title='Preferowana noga')
plt.xticks(rotation=45)
save_plot(plt, "violinplot_position_potential.png")
plt.close()

# =============================================
# Zakres prac na ocenę 4.0 z części I
# =============================================
# 1. Error bars dla cech numerycznych
df['age_group'] = pd.cut(df['age'], bins=[15, 20, 25, 30, 35, 40, 50])
plt.figure(figsize=(14, 7))
sns.pointplot(
    data=df,
    x='age_group',
    y='overall',
    hue='preferred_foot',
    estimator=np.median,
    errorbar='sd',
    capsize=0.15,
    palette={'Left': '#2ca02c', 'Right': '#d62728'},
    markers=['^', 'v'],
    linestyles=[':', '-.']
)
plt.title('Mediana ocen overall w grupach wiekowych')
plt.xlabel('Grupa wiekowa')
plt.ylabel('Mediana overall')
save_plot(plt, "error_bars_age_groups.png")
plt.close()


# 2. Histogramy dla cech numerycznych
numeric_features = ['overall', 'potential', 'age', 'value_eur', 'wage_eur']

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=feature,
        bins=30,
        color='skyblue',
        edgecolor='black'  # Lepsza widoczność słupków
    )
    plt.title(f'Rozkład wartości dla cechy: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Liczba graczy')
    save_plot(plt, f"histogram_{feature}.png")
    plt.close()

# Histogram dla value_eur bez skrajnych 10% wartości
plt.figure(figsize=(12, 6))
threshold = df['value_eur'].quantile(0.9)
filtered_data = df[df['value_eur'] <= threshold]
min_val = 0
max_val = int(filtered_data['value_eur'].max())
bin_width = 250000 
bins = range(min_val, max_val + bin_width, bin_width)

ax = sns.histplot(
    data=filtered_data,
    x='value_eur',
    bins=bins,
    color='#1f77b4',
    edgecolor='white',
    linewidth=0.5
)
ax.set_xticks(bins[::4]) 
ax.set_xticklabels([f'€{x/1000:.0f}k' for x in bins[::4]])
plt.title('Rozkład wartości rynkowej (bez górnych 10% danych)')
plt.xlabel('Wartość rynkowa (EUR)')
plt.ylabel('Liczba graczy')
plt.grid(axis='y', alpha=0.3)
plt.text(0.9, 0.9, f'Pominięto wartości > €{threshold:,.0f}', 
         ha='right', va='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))
save_plot(plt, "histogram_value_eur_trimmed.png")
plt.close()

# Histogram dla wage_eur bez skrajnych 10% wartości
plt.figure(figsize=(12, 6))
threshold = df['wage_eur'].quantile(0.90)
filtered_data = df[df['wage_eur'] <= threshold]
min_wage = 0
max_wage = int(filtered_data['wage_eur'].max())
bin_width = 2000 
bins = range(min_wage, max_wage + bin_width, bin_width)

ax = sns.histplot(
    data=filtered_data,
    x='wage_eur',
    bins=bins,
    color='#ff7f0e',
    edgecolor='white',
    linewidth=0.5
)
ax.set_xticks(bins[::4]) 
ax.set_xticklabels([f'€{x/1000:.0f}k' for x in bins[::4]])
plt.title('Rozkład tygodniowych płac (bez górnych 10% danych)')
plt.xlabel('Tygodniowe zarobki (EUR)')
plt.ylabel('Liczba graczy')
plt.grid(axis='y', alpha=0.3)
plt.text(0.95, 0.95, f'Pominięto wartości > €{threshold:,.0f}/tydz.', 
         ha='right', va='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

save_plot(plt, "histogram_wage_eur_trimmed.png")
plt.close()


# 3. Histogramy warunkowane (z hue)
plt.figure(figsize=(12, 6))
min_wage = int(df['overall'].min())
max_wage = int(df['overall'].max())
bin_width = 1
bins = range(min_wage, max_wage + bin_width, bin_width)
ax = sns.histplot(
    data=df,
    x='overall',
    hue='preferred_foot',
    bins=bins,
    kde=True,
    palette={'Left': 'blue', 'Right': 'orange'},
    alpha=0.6,
    element='step'
)

plt.title('Rozkład ocen overall z podziałem na preferowaną nogę')
plt.xlabel('Ocena overall')
plt.ylabel('Liczba graczy')
save_plot(plt, "conditional_histogram_foot.png")
plt.close()

# Histogram warunkowany dla wieku vs wartość rynkowa
sns.histplot(
    data=df,
    x='age',
    y='value_eur',
    bins=(25, 25),  
    cbar=True,
    cbar_kws={'label': 'Liczba zawodników'}
)
# Formatowanie
plt.title('Rozkład wartości rynkowej w zależności od wieku', pad=20, fontsize=16)
plt.xlabel('Wiek', fontsize=14)
plt.ylabel('Wartość rynkowa (EUR)', fontsize=14)
plt.legend()
save_plot(plt, "age_vs_value_distribution.png")
plt.close()
# =============================================
#Zakres prac na ocenę 4.5 z części I
# =============================================

# Heatmapa korelacji
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Korelacje między cechami numerycznymi")
save_plot(plt, "correlation_heatmap.png")

# =============================================
# ANALIZA REGRESJI LINIOWEJ DLA OCENY 5.0
# =============================================


plt.figure(figsize=(12, 8))
sns.regplot(
    data=df,
    x='overall',
    y='value_eur',
    scatter_kws={'alpha':0.3, 'color':'#1f77b4'},
    line_kws={'color':'red', 'linewidth':2},
    ci=95,
    truncate=False  
)
plt.title('Liniowa zależność: ocena overall a wartość rynkowa)', pad=20)
plt.xlabel('Ocena overall')
plt.ylabel('Wartość rynkowa (EUR)')
plt.grid(True, alpha=0.3)
save_plot(plt, "linear_regression_overall_value.png")
plt.close()

# Najprostsza wersja
plt.figure(figsize=(10, 6))
sns.lmplot(
    x="value_eur",
    y="wage_eur",
    data=df,
    height=6,
    aspect=1.5
)
plt.title('Relacja: wartość rynkowa vs zarobki')
save_plot(plt, "simple_regression_value_wage.png")
plt.close()

# Wersja z podziałem na preferowaną nogę
sns.lmplot(
    x="value_eur",
    y="wage_eur",
    hue="preferred_foot",
    data=df,
    height=6,
    aspect=1.5,
    palette={'Left': 'blue', 'Right': 'orange'}
)
plt.title('Relacja wartość-zarobki z podziałem na preferowaną nogę')
save_plot(plt, "regression_value_wage_by_foot.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.regplot(x='overall', y='value_eur', data=df, 
            scatter_kws={'alpha':0.3}, 
            line_kws={'color':'red'})
plt.yscale('log')  # Skala logarytmiczna = regresja wykładnicza
plt.title('Regresja wykładnicza (skala log)')
save_plot(plt,'simple_exp_regression.png')
plt.close()






















# 3. Zaawansowana analiza



print("\nAnaliza zakończona pomyślnie!")
print(f"Wyniki zapisane w folderze: {output_dir}")




