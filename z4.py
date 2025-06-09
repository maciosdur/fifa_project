import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Wczytaj dane
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

#target binarny: 0 = Left, 1 = Right
df_clean = df[features].dropna()
df_clean = df_clean[df_clean['preferred_foot'].isin(['Left', 'Right'])].copy()
df_clean['foot_bin'] = (df_clean['preferred_foot'] == 'Right').astype(int)
X = df_clean[features].drop(columns=['preferred_foot'])
y = df_clean['foot_bin']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

print("Rozkład klas w oryginalnym zbiorze:", np.bincount(y_train))

# Oversampling (SMOTE)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_processed, y_train)
print("Rozkład klas po SMOTE:", np.bincount(y_train_sm))

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_processed, y_train)
print("Rozkład klas po undersamplingu:", np.bincount(y_train_rus))

# ewaluacja
def evaluate(X_tr, y_tr, X_te, y_te, desc):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    print(f"\n{desc}")
    print(classification_report(y_te, y_pred, digits=4))

# Oryginalny zbiór
evaluate(X_train_processed, y_train, X_test_processed, y_test, "Oryginalny zbiór")

# Oversampling (SMOTE)
evaluate(X_train_sm, y_train_sm, X_test_processed, y_test, "Oversampling (SMOTE)")

# Undersampling
evaluate(X_train_rus, y_train_rus, X_test_processed, y_test, "Undersampling")