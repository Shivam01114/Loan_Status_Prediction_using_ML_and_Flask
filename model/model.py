import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv("../dataset.csv")
df = df.dropna()

# Encoding
df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
df.replace(to_replace='3+', value=4, inplace=True)

df.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

df = df.infer_objects(copy=False)

# Split
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Balanced pipeline (BEST VERSION)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(
        kernel='linear',
        class_weight='balanced',
        C=0.3
    ))
])

pipeline.fit(X_train, y_train)

# Evaluation
print("Train Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, pipeline.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, pipeline.predict(X_test)))

# Save
pickle.dump(pipeline, open("loan_model.pkl", "wb"))
print("✅ FINAL BALANCED MODEL SAVED")
