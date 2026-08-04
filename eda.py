import pandas as pd

df = pd.read_csv('data/HDD/heart.csv')  # ή όποιο είναι το filename σου

# 1. Βασική επισκόπηση
print(df.shape)  # (αριθμός εγγραφών, αριθμός στηλών)
print(df.columns.tolist())
df.head()

# 2. Ελλιπείς τιμές
print(df.isnull().sum())

# 3. Κατανομή της μεταβλητής-στόχου
print(df['target'].value_counts())  # άλλαξε 'target' αν λέγεται αλλιώς η στήλη σου
print(df['target'].value_counts(normalize=True) * 100)  # ποσοστά

# 4. Στατιστικά χαρακτηριστικών
df.describe()

# 5. Τύποι δεδομένων (ποια είναι κατηγορικά/αριθμητικά)
df.dtypes
print(df.duplicated().sum())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - Cleveland Heart Disease Dataset')
plt.savefig('correlation_heatmap.png')
plt.show()