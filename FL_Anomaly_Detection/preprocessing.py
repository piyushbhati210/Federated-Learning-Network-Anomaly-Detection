import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['label'].value_counts())
print(df['attack_cat'].value_counts())
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Removed {before - len(df)} duplicate rows")
print("New shape:", df.shape)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"Removed {before - len(df)} rows with missing/infinite values")
df.drop(columns=['id', 'attack_cat'], inplace=True)
print("Shape after dropping columns:", df.shape)
le = LabelEncoder()
categorical_cols = ['proto', 'service', 'state']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded: {col}")

print(df.head())
X = df.drop(columns=['label'])
y = df['label']

print("X shape:", X.shape)
print("y shape:", y.shape)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("Scaling done!")
print(X_scaled.describe().loc[['min','max']])
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

NUM_CLIENTS = 4
client_size = len(X_train) // NUM_CLIENTS

for i in range(NUM_CLIENTS):
    start = i * client_size
    end = (i + 1) * client_size if i < NUM_CLIENTS - 1 else len(X_train)
    
    client_df = X_train.iloc[start:end].copy()
    client_df['label'] = y_train.iloc[start:end].values
    
    client_df.to_csv(f"data/clients/client_{i+1}_train.csv", index=False)
    print(f"Client {i+1}: {len(client_df)} samples saved")
    train_df = X_train.copy()
train_df['label'] = y_train.values
train_df.to_csv("data/processed/train_cleaned.csv", index=False)

test_df = X_test.reset_index(drop=True).copy()
test_df['label'] = y_test.reset_index(drop=True).values
test_df.to_csv("data/processed/test_cleaned.csv", index=False)

print("✅ All files saved!")
print("Train:", train_df.shape)
print("Test:", test_df.shape)

