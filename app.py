import streamlit as st
from datetime import date, time
import joblib
import pandas as pd

# Load trained ML model
model = joblib.load("careconnect_model.pkl")

st.set_page_config(
    page_title="CareConnect",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 CareConnect")

st.write(
    "Patient Follow-Up & Health Management System"
)

st.divider()

st.header("Welcome 👋")

st.write(
    "Manage your appointments, medicines, tests "
    "and follow-ups in one place."
)

st.subheader("🔔 Today's Reminders")

st.info("💊 Medicine reminder — 8:00 PM")

st.warning("📅 Doctor appointment — Tomorrow at 10:30 AM")

st.info("🧪 Blood test — Due in 3 days")

st.divider()

# -------------------------------
# PATIENT PROFILE
# -------------------------------

st.header("👤 Patient Profile")

name = st.text_input("Patient Name")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

condition = st.selectbox(
    "Health Condition",
    [
        "None",
        "Diabetes",
        "Hypertension",
        "Heart Disease",
        "Asthma",
        "Other"
    ]
)

if condition == "Other":
    condition = st.text_input(
        "Please specify your health condition"
    )

if st.button("Save Patient Information"):

    st.success("✅ Patient information saved!")

    st.write("### Patient Details")

    st.write("**Name:**", name)
    st.write("**Age:**", age)
    st.write("**Gender:**", gender)
    st.write("**Health Condition:**", condition)

    # -------------------------------
# APPOINTMENT DETAILS
# -------------------------------

st.divider()

st.header("📅 Appointment Details")

doctor = st.text_input(
    "Doctor Name"
)

department = st.text_input(
    "Department / Hospital"
)

appointment_date = st.date_input(
    "Appointment Date"
)

from datetime import time

appointment_time = st.time_input(
    "Appointment Time",
    value=time(10, 30)
)

purpose = st.text_input(
    "Purpose of Visit"
)

if st.button("Save Appointment"):

    st.success("✅ Appointment saved successfully!")

    st.write("### Appointment Details")

    st.write("**Doctor:**", doctor)
    st.write("**Department / Hospital:**", department)
    st.write(
        "**Date:**",
        appointment_date.strftime("%d %B %Y")
    )
    st.write(
        "**Time:**",
        appointment_time.strftime("%I:%M %p")
    )
    st.write("**Purpose:**", purpose)

    # -------------------------------
# MEDICINE DETAILS
# -------------------------------

st.divider()

st.header("💊 Medicine Schedule")

medicine_name = st.text_input(
    "Medicine Name"
)

medicine_time = st.time_input(
    "Medicine Time",
    value=time(8, 0)
)

frequency = st.selectbox(
    "How often?",
    [
        "Once a day",
        "Twice a day",
        "Three times a day",
        "As prescribed"
    ]
)

if st.button("Save Medicine"):

    st.success("✅ Medicine reminder saved!")

    st.write("### Medicine Details")

    st.write("**Medicine:**", medicine_name)

    st.write(
        "**Time:**",
        medicine_time.strftime("%I:%M %p")
    )

    st.write(
        "**Frequency:**",
        frequency
    )

    # -------------------------------
# TESTS & FOLLOW-UPS
# -------------------------------

st.divider()

st.header("🧪 Tests & Follow-ups")

test_name = st.text_input(
    "Test / Follow-up Name"
)

test_date = st.date_input(
    "Test / Follow-up Date"
)

doctor_instruction = st.text_input(
    "Doctor's Instruction (Optional)"
)

if st.button("Save Test / Follow-up"):

    st.success("✅ Test / Follow-up saved successfully!")

    st.write("### Test Details")

    st.write(
        "**Test / Follow-up:**",
        test_name
    )

    st.write(
        "**Date:**",
        test_date.strftime("%d %B %Y")
    )

    if doctor_instruction:
        st.write(
            "**Doctor's Instruction:**",
            doctor_instruction
        )

        # -------------------------------
# REMINDERS
# -------------------------------

st.divider()

st.header("🔔 Reminders")

# Appointment reminder
if 'appointment_date' in locals():

    days_left = (
        appointment_date - date.today()
    ).days

    if days_left == 0:
        st.warning("⚠️ Your appointment is TODAY!")

    elif days_left == 1:
        st.warning("⚠️ Your appointment is TOMORROW!")

    elif days_left > 1:
        st.info(
            f"📅 Your appointment is in {days_left} days."
        )

# Medicine reminder
if 'medicine_name' in locals():

    if medicine_name:

        st.info(
            f"💊 Medicine reminder: "
            f"{medicine_name} at "
            f"{medicine_time.strftime('%I:%M %p')}"
        )

# -------------------------------
# REMINDERS
# -------------------------------

st.divider()

st.header("🔔 Reminders")

days_left = (
    appointment_date - date.today()
).days

if days_left == 0:

    st.warning("⚠️ Your appointment is TODAY!")

elif days_left == 1:

    st.warning("⚠️ Your appointment is TOMORROW!")

elif days_left > 1:

    st.info(
        f"📅 Your appointment is in {days_left} days."
    )

    # ==========================================
# ML NO-SHOW PREDICTION
# ==========================================

st.divider()

st.header("🤖 Appointment No-Show Prediction")

st.write(
    "Estimate the probability that an appointment may be missed "
    "using the trained machine learning model."
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

scholarship = st.selectbox(
    "Scholarship",
    ["No", "Yes"]
)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

diabetes = st.selectbox(
    "Diabetes",
    ["No", "Yes"]
)

alcoholism = st.selectbox(
    "Alcoholism",
    ["No", "Yes"]
)

handicap = st.selectbox(
    "Handicap",
    ["No", "Yes"]
)

sms_received = st.selectbox(
    "SMS Reminder Received",
    ["No", "Yes"]
)

lead_time = st.number_input(
    "Days between scheduling and appointment",
    min_value=0,
    max_value=365,
    value=7
)

appointment_day = st.selectbox(
    "Appointment Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

if st.button("🔮 Predict No-Show Risk"):

    gender_value = 0 if gender == "Female" else 1

    scholarship_value = (
        1 if scholarship == "Yes" else 0
    )

    hypertension_value = (
        1 if hypertension == "Yes" else 0
    )

    diabetes_value = (
        1 if diabetes == "Yes" else 0
    )

    alcoholism_value = (
        1 if alcoholism == "Yes" else 0
    )

    handicap_value = (
        1 if handicap == "Yes" else 0
    )

    sms_value = (
        1 if sms_received == "Yes" else 0
    )

    weekday_value = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }[appointment_day]

    input_data = pd.DataFrame([{
        "Gender": gender_value,
        "Age": age,
        "Scholarship": scholarship_value,
        "Hipertension": hypertension_value,
        "Diabetes": diabetes_value,
        "Alcoholism": alcoholism_value,
        "Handcap": handicap_value,
        "SMS_received": sms_value,
        "LeadTime": lead_time,
        "AppointmentWeekday": weekday_value
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability = probability * 100

    st.subheader("Prediction Result")

    if prediction == 1:

        st.warning(
            f"⚠️ Higher No-Show Risk: {probability:.2f}%"
        )

        st.info(
            "Consider sending an additional appointment reminder."
        )

    else:

        st.success(
            f"✅ Lower No-Show Risk: {probability:.2f}%"
        )