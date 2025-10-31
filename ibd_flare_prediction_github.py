import zipfile
import os

# Create folder structure
os.makedirs('IBD_Flare_GitHub/images', exist_ok=True)

# Write a sample notebook with embedded results
notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# IBD Flare-up Prediction Project\n",
    "\n",
    "**Research Question:** Can we predict inflammatory bowel disease (IBD) flare-ups 90 days in advance using clinical, lab, medication, and procedure data?\n",
    "\n",
    "**Abstract:** This project develops a logistic regression model to predict flare-ups in IBD patients, using features such as lab measurements (CRP, ESR, Hb, Hct, WBC, albumin, ALT, AST, fecal calprotectin), medication use, and procedures. The goal is to alert patients and doctors for timely intervention.\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Simulate data (embedding)\n",
    "np.random.seed(42)\n",
    "n_samples = 61329\n",
    "df = pd.DataFrame({\n",
    "    'count_crohn': np.random.randint(0,3,n_samples),\n",
    "    'count_uc': np.random.randint(0,3,n_samples),\n",
    "    'max_crp': np.random.normal(8,5,n_samples),\n",
    "    'max_esr': np.random.normal(18,8,n_samples),\n",
    "    'min_hb': np.random.normal(13,1.5,n_samples),\n",
    "    'max_hct': np.random.normal(40,5,n_samples),\n",
    "    'max_wbc': np.random.normal(7,2,n_samples),\n",
    "    'min_albumin': np.random.normal(4,0.5,n_samples),\n",
    "    'max_alt': np.random.normal(25,10,n_samples),\n",
    "    'max_ast': np.random.normal(22,8,n_samples),\n",
    "    'max_fcp': np.random.normal(150,50,n_samples),\n",
    "    'count_medications': np.random.randint(0,2,n_samples),\n",
    "    'count_procedures': np.random.randint(0,2,n_samples)\n",
    "})\n",
    "\n",
    "# Define target\n",
    "df['flare_up'] = ((df['count_procedures']>0)|(df['count_medications']>0)|(df['max_crp']>10)|(df['max_esr']>20)).astype(int)\n",
    "\n",
    "# Split data\n",
    "X = df.drop(columns=['flare_up'])\n",
    "y = df['flare_up']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Impute and scale\n",
    "imputer = SimpleImputer(strategy='mean')\n",
    "X_train_scaled = StandardScaler().fit_transform(imputer.fit_transform(X_train))\n",
    "X_test_scaled = StandardScaler().fit_transform(imputer.transform(X_test))\n",
    "\n",
    "# Train model\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "y_prob = model.predict_proba(X_test_scaled)[:,1]\n",
    "\n",
    "# Metrics\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "roc = roc_auc_score(y_test, y_prob)\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "report = classification_report(y_test, y_pred)\n",
    "\n",
    "print('Accuracy:', acc)\n",
    "print('ROC-AUC:', roc)\n",
    "print('Confusion Matrix:\n', cm)\n",
    "print('Classification Report:\n', report)\n",
    "\n",
    "# Confusion matrix plot\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.savefig('IBD_Flare_GitHub/images/confusion_matrix.png')\n",
    "plt.show()\n",
    "\n",
    "# ROC curve\n",
    "fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "plt.figure(figsize=(6,4))\n",
    "plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc:.3f})')\n",
    "plt.plot([0,1],[0,1],'k--')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curve')\n",
    "plt.legend()\n",
    "plt.savefig('IBD_Flare_GitHub/images/roc_curve.png')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5
}"""

with open('IBD_Flare_GitHub/flare_prediction.ipynb', 'w') as f:
    f.write(notebook_content)

# Create a markdown index file
index_content = """
# IBD Flare-up Prediction Project

**Research Question:** Can we predict IBD flare-ups 90 days in advance?

**Abstract:** Logistic regression model using lab, medication, procedure, and condition features to alert patients/doctors.

**Notebook:** [flare_prediction.ipynb](flare_prediction.ipynb)

**Images:** 
- Confusion Matrix: images/confusion_matrix.png
- ROC Curve: images/roc_curve.png
"""
with open('IBD_Flare_GitHub/index.md', 'w') as f:
    f.write(index_content)

# Zip the folder
zipf = zipfile.ZipFile('IBD_Flare_GitHub.zip', 'w', zipfile.ZIP_DEFLATED)
for root, dirs, files in os.walk('IBD_Flare_GitHub'):
    for file in files:
        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), 'IBD_Flare_GitHub'))
zipf.close()

print('ZIP file IBD_Flare_GitHub.zip created successfully!')
