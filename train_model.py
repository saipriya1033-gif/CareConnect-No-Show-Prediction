

import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
     r"C:\Users\DELL\Downloads\KaggleV2-May-2016.csv\KaggleV2-May-2016.csv"

)

print("----- DATASET SHAPE -----")
print(df.shape)


# ==========================================
# 2. COLUMN NAMES
# ==========================================

print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())


# ==========================================
# 3. FIRST 5 ROWS
# ==========================================

print("\n----- FIRST 5 ROWS -----")
print(df.head())


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())


# ==========================================
# 5. DUPLICATE ROWS
# ==========================================

print("\n----- DUPLICATE ROWS -----")
print(df.duplicated().sum())


# ==========================================
# 6. NO-SHOW DISTRIBUTION
# ==========================================

print("\n----- NO-SHOW DISTRIBUTION -----")
print(df["No-show"].value_counts())


# ==========================================
# 7. NO-SHOW PERCENTAGE
# ==========================================

print("\n----- NO-SHOW PERCENTAGE -----")
print(
    df["No-show"].value_counts(normalize=True) * 100
)


# ==========================================
# 8. AGE CHECK
# ==========================================

print("\n----- AGE INFORMATION -----")
print("Minimum age:", df["Age"].min())
print("Maximum age:", df["Age"].max())


# ==========================================
# 9. DATE CONVERSION
# ==========================================

df["ScheduledDay"] = pd.to_datetime(
    df["ScheduledDay"]
)

df["AppointmentDay"] = pd.to_datetime(
    df["AppointmentDay"]
)

print("\n----- DATE CONVERSION -----")
print("Dates converted successfully!")


# ==========================================
# 10. CREATE LEAD TIME
# ==========================================

df["LeadTime"] = (
    df["AppointmentDay"] -
    df["ScheduledDay"]
).dt.days

print("\n----- LEAD TIME CREATED -----")
print(df["LeadTime"].describe())


# ==========================================
# 11. REMOVE INVALID AGES
# ==========================================

df = df[
    (df["Age"] >= 0) &
    (df["Age"] <= 120)
]

print("\n----- AFTER AGE CLEANING -----")
print(df.shape)


# ==========================================
# 12. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\n----- AFTER DUPLICATE REMOVAL -----")
print(df.shape)


# ==========================================
# 13. FINAL CHECK
# ==========================================

print("\n----- FINAL DATASET -----")
print(df.head())

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\n----- CLEANING COMPLETED -----")

# ==========================================
# PREPROCESSING
# ==========================================

from sklearn.model_selection import train_test_split


# Convert dates
df["ScheduledDay"] = pd.to_datetime(
    df["ScheduledDay"]
)

df["AppointmentDay"] = pd.to_datetime(
    df["AppointmentDay"]
)


# Create Lead Time
df["LeadTime"] = (
    df["AppointmentDay"] -
    df["ScheduledDay"]
).dt.days

# Remove negative values
df["LeadTime"] = df["LeadTime"].clip(
    lower=0
)


# Create appointment weekday
df["AppointmentWeekday"] = (
    df["AppointmentDay"].dt.dayofweek
)


# Encode Gender
df["Gender"] = df["Gender"].map({
    "F": 0,
    "M": 1
})


# Encode target
df["Target"] = df["No-show"].map({
    "No": 0,
    "Yes": 1
})


# Select features
features = [
    "Gender",
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "LeadTime",
    "AppointmentWeekday"
]


X = df[features]

y = df["Target"]


# Handle missing values
X = X.fillna(0)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n----- PREPROCESSING COMPLETED -----")

print("Features used:")
print(features)

print("\nTraining data:", X_train.shape)

print("Testing data:", X_test.shape)

print("\n----- READY FOR ML -----")

# ==========================================
# TRAIN RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(
    X_train,
    y_train
)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n----- MODEL TRAINING COMPLETED -----")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# Classification report
print("\n----- CLASSIFICATION REPORT -----")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# Confusion matrix
print("\n----- CONFUSION MATRIX -----")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n----- MODEL EVALUATION -----")

print("Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

import joblib

joblib.dump(
    model,
    "../careconnect_model.pkl"
)

print("\n----- MODEL SAVED -----")
print("careconnect_model.pkl created successfully!")


# ==========================================
# CARECONNECT - TEST SAVED ML MODEL
# ==========================================

import joblib
import pandas as pd


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    "careconnect_model.pkl"
)

print("\n====================================")
print("   CARECONNECT ML MODEL TEST")
print("====================================")


# ==========================================
# 2. GET PATIENT INFORMATION
# ==========================================

print("\nEnter patient information:\n")


# Gender
gender = input(
    "Gender (Female/Male): "
).strip().lower()

if gender == "female":
    gender_value = 0
else:
    gender_value = 1


# Age
age = int(
    input("Age: ")
)


# Scholarship
scholarship = input(
    "Scholarship (Yes/No): "
).strip().lower()

scholarship_value = (
    1 if scholarship == "yes" else 0
)


# Hypertension
hypertension = input(
    "Hypertension (Yes/No): "
).strip().lower()

hypertension_value = (
    1 if hypertension == "yes" else 0
)


# Diabetes
diabetes = input(
    "Diabetes (Yes/No): "
).strip().lower()

diabetes_value = (
    1 if diabetes == "yes" else 0
)


# Alcoholism
alcoholism = input(
    "Alcoholism (Yes/No): "
).strip().lower()

alcoholism_value = (
    1 if alcoholism == "yes" else 0
)


# Handicap
handicap = input(
    "Handicap (Yes/No): "
).strip().lower()

handicap_value = (
    1 if handicap == "yes" else 0
)


# SMS received
sms = input(
    "SMS reminder received (Yes/No): "
).strip().lower()

sms_value = (
    1 if sms == "yes" else 0
)


# Lead time
lead_time = int(
    input(
        "Days between scheduling and appointment: "
    )
)


# Appointment weekday
print("\nAppointment Day:")

print("0 - Monday")
print("1 - Tuesday")
print("2 - Wednesday")
print("3 - Thursday")
print("4 - Friday")
print("5 - Saturday")
print("6 - Sunday")

weekday = int(
    input("Enter number (0-6): ")
)


# ==========================================
# 3. CREATE INPUT DATAFRAME
# ==========================================

patient_data = pd.DataFrame([{

    "Gender": gender_value,

    "Age": age,

    "Scholarship": scholarship_value,

    "Hipertension": hypertension_value,

    "Diabetes": diabetes_value,

    "Alcoholism": alcoholism_value,

    "Handcap": handicap_value,

    "SMS_received": sms_value,

    "LeadTime": lead_time,

    "AppointmentWeekday": weekday

}])


# ==========================================
# 4. DISPLAY INPUT
# ==========================================

print("\n====================================")
print("PATIENT INFORMATION")
print("====================================")

print(
    patient_data.to_string(
        index=False
    )
)


# ==========================================
# 5. MAKE PREDICTION
# ==========================================

prediction = model.predict(
    patient_data
)[0]


# ==========================================
# 6. GET PROBABILITY
# ==========================================

probability = model.predict_proba(
    patient_data
)[0][1]


probability_percentage = (
    probability * 100
)


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n====================================")
print("       PREDICTION RESULT")
print("====================================")


if prediction == 1:

    print(
        "\n⚠️ HIGHER NO-SHOW RISK"
    )

else:

    print(
        "\n✅ LOWER NO-SHOW RISK"
    )


print(
    f"\nEstimated No-show Probability: "
    f"{probability_percentage:.2f}%"
)


# ==========================================
# 8. REMINDER SUGGESTION
# ==========================================

print("\n------------------------------------")

if probability_percentage >= 50:

    print(
        "🔔 Recommendation:"
    )

    print(
        "Send an additional appointment "
        "reminder to the patient."
    )

else:

    print(
        "ℹ️ Recommendation:"
    )

    print(
        "Normal appointment reminder "
        "is sufficient."
    )


print("\n====================================")
print("          TEST COMPLETED")
print("====================================")

