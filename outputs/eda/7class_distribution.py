# %%
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load metadata CSV
df = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')

# Map short codes to full readable names
dx_names = {
    'nv': 'Melanocytic Nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign Keratosis (bkl)',
    'bcc': 'Basal Cell Carcinoma (bcc)',
    'akiec': 'Actinic Keratoses (akiec)',
    'vasc': 'Vascular Lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}

counts = df['dx'].map(dx_names).value_counts()

# 2. Plot horizontal or vertical Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.barh(counts.index, counts.values, color='#4C72B0')

# Add numbers next to each bar
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 50, bar.get_y() + bar.get_height()/2, f'{xval:,}', 
             va='center', fontsize=10, fontweight='bold')

plt.title('HAM10000 Dataset - Disease Category Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Images', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()  # Highest count at top
plt.tight_layout()

# Save figure
plt.savefig('outputs/eda/ham10000_7class_distribution.png', dpi=150)
plt.show()
print("Saved bar chart to: outputs/eda/ham10000_7class_distribution.png")
