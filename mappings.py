FEATURE_LABELS_EL = {
    'age': 'Ηλικία',
    'sex': 'Φύλο',
    'cp': 'Τύπος θωρακικού πόνου',
    'trestbps': 'Αρτηριακή πίεση σε κατάσταση ηρεμίας (mm Hg)',
    'chol': 'Χοληστερίνη (mg/dl)',
    'fbs': 'Σάκχαρο Νηστείας (> 120 mg/dl)',
    'restecg': 'Ηλεκτροκαρδιογράφημα Ηρεμίας',
    'thalach': 'Μέγιστη Καρδιακή Συχνότητα',
    'exang': 'Πρόκληση πόνου κατά την άσκηση​',
    'oldpeak': 'Κατάσπαση διαστήματος ST',
    'slope': 'Κλίση διαστήματος ST',
    'ca': 'Αριθμός Στεφανιαίων Αγγείων​',
    'thal': 'Αποτέλεσμα δοκιμασίας κόπωσης με θάλλιο​'
}

FEATURE_LABELS_EN = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Oldpeak)',
    'slope': 'Slope of ST Segment',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia',
}

sex_mapping = {
    0: "Γυναίκα / Female (0)",
    1: "Άνδρας / Male (1)"
}

cp_mapping = {
    0: "Ασυμπτωματικός / Asymptomatic (0)",
    1: "Άτυπη στηθάγχη / Atypical angina (1)",
    2: "Μη στηθαγχικός πόνος / Non-anginal pain (2)",
    3: "Τυπική στηθάγχη / Typical angina (3)"
}

fbs_mapping = {
    0: "Όχι - Μικρότερο από 120 mg/dl (0) / No - Below 120 mg/dl (0)",
    1: "Ναι - Μεγαλύτερο από 120 mg/dl (1) / Yes - Above 120 mg/dl (1)"
}

restecg_mapping = {
    0: "Υπερτροφία αριστερής κοιλίας / Left ventricular hypertrophy (0)",
    1: "Φυσιολογικό / Normal (1)",
    2: "Ανωμαλία κύματος ST-T / ST-T wave abnormality (2)"
}

exang_mapping = {
    0: "Όχι / No (0)",
    1: "Ναι / Yes (1)"
}

slope_mapping = {
    0: "Καθοδική / Downsloping (0)",
    1: "Επίπεδη / Flat (1)",
    2: "Ανοδική / Upsloping (2)"
}

thal_mapping = {
    1: "Σταθερό ελάττωμα / Fixed defect (1)",
    2: "Φυσιολογική / Normal (2)",
    3: "Αναστρέψιμο ελάττωμα / Reversible defect (3)"
}