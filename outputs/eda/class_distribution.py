# %%
import os
import matplotlib.pyplot as plt

# 1. Path to your dataset directory
data_dir = 'data/HAM10000/skin_cancer_data'

# 2. Count images in each folder
counts = {}
for category in ['benign', 'malignant']:
    folder_path = os.path.join(data_dir, category)
    if os.path.exists(folder_path):
        counts[category.capitalize()] = len(os.listdir(folder_path))

# 3. Plot the Bar Chart
plt.figure(figsize=(7, 5))
bars = plt.bar(counts.keys(), counts.values(), color=['#2A9D8F', '#E63946'], width=0.5)

# Add exact numbers above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, f'{yval:,}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('HAM10000 Dataset - Class Distribution (Binary)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Lesion Type', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
plt.savefig('outputs/eda/skin_lesions_class_distribution.png', dpi=150)
plt.show()
print("Saved bar chart to: outputs/eda/skin_lesions_class_distribution.png")
