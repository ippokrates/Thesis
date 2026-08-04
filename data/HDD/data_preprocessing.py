import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

csvColumns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('./heart.csv')

# replace missing values (?)
df = df.replace('?', np.nan)

# remove rows with empty values
df = df.dropna()

# transform 'ca', 'thal' to num values
print("Type:", type(df['ca']))
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df = df.dropna()


# target variable
# 0 healthy / 1 heart desease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

# transform only the train set to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save the scaler so the Streamlit app can reuse it
joblib.dump(scaler, "scaler.pkl")

# restore to pandas DataFrame for SHAP/LIME
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# save the preprocessed data
X_train_scaled_df.to_csv("X_train_ready.csv", index=False)
X_test_scaled_df.to_csv("X_test_ready.csv", index=False)
y_train.to_csv("y_train_ready.csv", index=False)
y_test.to_csv("y_test_ready.csv", index=False)


#print(f"Διαστάσεις X_train: {X_train_scaled_df.shape}")
#print(f"Διαστάσεις X_test: {X_test_scaled_df.shape}")
