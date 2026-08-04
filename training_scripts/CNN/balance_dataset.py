import os
import random
import shutil

# --- Διαδρομές ---
source_dir = 'data/HAM10000/skin_cancer_data'
target_dir = 'data/HAM10000/balanced_data'

# Καθαρισμός του παλιού φακέλου αν υπάρχει (για να μη διπλογραφούνται)
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Δημιουργία των νέων φακέλων
os.makedirs(os.path.join(target_dir, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'benign'), exist_ok=True)

# --- Λήψη Λίστας Αρχείων ---
malignant_files = os.listdir(os.path.join(source_dir, 'malignant'))
benign_files = os.listdir(os.path.join(source_dir, 'benign'))

num_malignant = len(malignant_files)
print(f"Βρέθηκαν {num_malignant} Malignant εικόνες.")

# ΕΔΩ ΕΙΝΑΙ Η ΜΑΓΕΙΑ (Undersampling): Διαλέγουμε τυχαία ΤΟΝ ΙΔΙΟ αριθμό Benign
print(f"Επιλογή {num_malignant} τυχαίων Benign εικόνων για απόλυτη ισορροπία...")
selected_benign = random.sample(benign_files, num_malignant)

# --- Αντιγραφή Αρχείων ---
print("Αντιγραφή Malignant εικόνων...")
for f in malignant_files:
    shutil.copy(os.path.join(source_dir, 'malignant', f), os.path.join(target_dir, 'malignant', f))

print("Αντιγραφή Benign εικόνων...")
for f in selected_benign:
    shutil.copy(os.path.join(source_dir, 'benign', f), os.path.join(target_dir, 'benign', f))

print(f"\nΕΠΙΤΥΧΙΑ! Το νέο dataset είναι έτοιμο: {num_malignant} Benign και {num_malignant} Malignant.")