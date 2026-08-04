import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


X_test = pd.read_csv("data/HDD/X_test_ready.csv")
rf_model = joblib.load("saved_models/rf_model.pkl")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# shape error fix
if isinstance(shap_values, list):
    # older SHAP versions
    shap_values_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    #bnewer SHAP versions
    shap_values_class1 = shap_values[:, :, 1]
else:
    shap_values_class1 = shap_values

# SHAP summary Plot
shap.summary_plot(shap_values_class1, X_test, feature_names=X_test.columns, show=False)

plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()