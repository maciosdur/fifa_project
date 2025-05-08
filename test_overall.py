import pandas as pd
import joblib
from tabulate import tabulate

# 1. Wczytaj model
model = joblib.load('fifa_overall_predictor.pkl')

# 2. Przykładowe dane wejściowe
new_data = pd.DataFrame({
    'age': [25, 30, 22, 28],
    'value_eur': [50_000_000, 100_000_000, 20_000_000, 35_000_000],  
    'potential': [88, 92, 89, 88],
    'height_cm': [180, 175, 185, 178],
    'weight_kg': [75, 70, 80, 72],
    'pace': [80, 85, 90, 82],
    'shooting': [85, 90, 82, 88],
    'passing': [90, 92, 85, 89],
    'dribbling': [92, 95, 88, 90],
    'defending': [40, 35, 50, 45],
    'physic': [75, 70, 80, 78],
    'preferred_foot': ['Right', 'Left', 'Right', 'Right'],
    'player_positions': ['CAM', 'ST', 'CB', 'CM']
})



# 4. Wykonaj predykcję
predictions = model.predict(new_data)

# 5. Formatowanie wyników
results = pd.DataFrame({
    'ID': range(1, len(new_data)+1),
    'Pozycja': new_data['player_positions'],
    'Wartość (mln €)': [f"€{x/1_000_000:,.1f}" for x in new_data['value_eur']],
    'Wiek': new_data['age'],
    'Przewidywany overall': [f"{x:.2f}" for x in predictions],
})

# Wyświetl wyniki
print("\nPRZEWIDYWANE OCENY OVERALL ZAWODNIKÓW")
print(tabulate(results, headers='keys', tablefmt='pretty', showindex=False))

# 2. Dane rzeczywiste zawodników (z Twojego pliku CSV)
real_players = pd.DataFrame({
    'short_name': ['L. Messi', 'R. Lewandowski', 'Cristiano Ronaldo'],
    'age': [34, 32, 36],
    'overall': [93, 92, 91],
    'potential': [93, 92, 91],
    'height_cm': [170, 185, 187],
    'weight_kg': [72, 81, 83],
    'pace': [85, 78, 87],
    'shooting': [92, 92, 94],
    'passing': [91, 79, 80],
    'dribbling': [95, 86, 88],
    'defending': [34, 44, 34],
    'physic': [65, 82, 75],
    'preferred_foot': ['Left', 'Right', 'Right'],
    'player_positions': ['RW, ST, CF', 'ST', 'ST, LW'],
    'value_eur': [78000000, 119500000, 45000000]
})

# 3. Przygotowanie danych do predykcji
prediction_data = real_players.drop(['short_name', 'overall'], axis=1)

# 4. Wykonaj predykcję
predictions = model.predict(prediction_data)

# 5. Formatowanie wyników
results = pd.DataFrame({
    'Zawodnik': real_players['short_name'],
    'Pozycja': real_players['player_positions'],
    'Wartość (€)': [f"€{x:,.2f}" for x in real_players['value_eur']],
    'Wiek': real_players['age'],
    'Rzeczywista ocena': real_players['overall'],
    'Przewidywana ocena': predictions,
    'Różnica (%)': [f"{(predictions[i] - real_players['overall'][i])/real_players['overall'][i]*100:.1f}%" 
                    for i in range(len(predictions))]
})

print("\nPORÓWNANIE RZECZYWISTYCH I PRZEWIDYWANYCH WARTOŚCI ZAWODNIKÓW")
print(tabulate(results, headers='keys', tablefmt='pretty', showindex=False))