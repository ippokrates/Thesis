import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/HDD/heart.csv')

df = df.drop_duplicates()
print(f"Shape μετά το dedup: {df.shape}")

print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True) * 100)

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - Cleveland Heart Disease Dataset (302 unique records)')
plt.savefig('outputs/eda/correlation_heatmap.png')
plt.show()
print("Saved to: outputs/eda/correlation_heatmap.png")

print("-"*10)
print(df['ca'].value_counts())
print(df['thal'].value_counts())
