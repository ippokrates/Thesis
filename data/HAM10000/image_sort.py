import pandas as pd
import os
import shutil

# setting paths
csv_path = 'data/HAM10000/HAM10000_metadata.csv'
source_images_folder = 'data/HAM10000/all_ham_images'
base_output_folder = 'data/HAM10000/skin_cancer_data'

malignant_dir = os.path.join(base_output_folder, 'malignant')
benign_dir = os.path.join(base_output_folder, 'benign')
os.makedirs(malignant_dir, exist_ok=True)
os.makedirs(benign_dir, exist_ok=True)


malignant_codes = ['mel', 'bcc', 'akiec'] # cancer medical codes
df = pd.read_csv(csv_path)

copy_count = 0
for index, row in df.iterrows():
    image_name = row['image_id'] + '.jpg'
    diagnosis = row['dx']
    source_path = os.path.join(source_images_folder, image_name)
    
    if os.path.exists(source_path):
        if diagnosis in malignant_codes:
            destination_path = os.path.join(malignant_dir, image_name)
        else:
            destination_path = os.path.join(benign_dir, image_name)
            
        shutil.copy(source_path, destination_path)
        copy_count += 1
        
        # progress update every 1000 images
        if copy_count % 1000 == 0:
            print(f"Sorted {copy_count} images")

print(f"\nSorted {copy_count} images")