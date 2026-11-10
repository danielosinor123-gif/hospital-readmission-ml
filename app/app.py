"""
Streamlit app: "Will this patient be readmitted within 30 days?"

WHAT THIS FILE DOES
-------------------
1. Loads the trained LightGBM model and the feature list produced by notebook 02.
2. Shows a form with the patient details a nurse could actually know at discharge.
3. Fills every remaining model feature with the most common value seen in training
   (so the model always receives a complete, valid row).
4. Predicts the probability of readmission within 30 days and shows a risk band.
5. Validates every input and refuses to predict on nonsense.

RUN IT
------
    python -m streamlit run app/app.py

IMPORTANT: this is a learning/decision-support demo. It must NOT be used for
real clinical decisions -- see the disclaimer printed inside the app.
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------- paths
APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
MODEL_PATH = PROJECT / "models" / "lightgbm.joblib"
FEATURES_PATH = PROJECT / "data" / "processed" / "feature_columns.json"
TRAIN_PATH = PROJECT / "data" / "processed" / "X_train.csv"

# --------------------------------------------------------------------- setup
st.set_page_config(page_title="Hospital Readmission Risk (demo)", page_icon="🏥")

st.title("🏥 30-Day Readmission Risk — demonstration app")
st.write(
    "Enter what is known about the patient **at discharge**. The app returns the "
    "model's estimated probability that this patient is readmitted within 30 days."
)

# --------------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features_and_defaults():
    """Read the feature list and the most common value of every feature.

    Using the training data's mode as the default means features the user does
    not touch hold a *typical* value instead of a made-up one.
    """
    features = json.loads(FEATURES_PATH.read_text())
    train = pd.read_csv(TRAIN_PATH, usecols=features)
    defaults = {c: train[c].mode().iloc[0] for c in features}
    return features, defaults

missing_files = [str(p) for p in (MODEL_PATH, FEATURES_PATH, TRAIN_PATH) if not p.exists()]
if missing_files:
    st.error(
        "Model files are missing. Run the notebooks first "
        "(01 then 02), then restart this app.\n\nMissing:\n" + "\n".join(missing_files)
    )
    st.stop()

try:
    model = load_model()
    features, defaults = load_features_and_defaults()
except Exception as exc:                       # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.stop()

# --------------------------------------------------------------------- the form
with st.form("patient_form"):
    st.subheader("Patient details (all fields are known at discharge)")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age (years)", 0, 100, 65, help="Used as the bracket midpoint.")
        gender = st.radio("Gender", ["Female", "Male"], index=0, horizontal=True)
        race = st.selectbox(
            "Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
        )
        time_in_hospital = st.number_input(
            "Days in hospital this stay", min_value=1, max_value=14, value=3, step=1
        )

    with c2:
        num_lab_procedures = st.number_input(
            "Lab procedures this stay", min_value=0, max_value=200, value=40, step=1
        )
        num_procedures = st.number_input(
            "Medical procedures this stay", min_value=0, max_value=20, value=1, step=1
        )
        num_medications = st.number_input(
            "Medications this stay", min_value=0, max_value=100, value=15, step=1
        )
        number_diagnoses = st.number_input(
            "Number of diagnoses", min_value=0, max_value=20, value=8, step=1
        )

    with c3:
        number_inpatient = st.number_input(
            "Inpatient stays in the previous year",
            min_value=0, max_value=21, value=0, step=1,
            help="The strongest single risk signal in this dataset.",
        )
        number_emergency = st.number_input(
            "Emergency visits in the previous year",
            min_value=0, max_value=76, value=0, step=1,
        )
        number_outpatient = st.number_input(
            "Outpatient visits in the previous year",
            min_value=0, max_value=56, value=0, step=1,
        )
        prior_visits_total = st.number_input(
            "Total health-system visits in the previous year (if known)",
            min_value=0, max_value=100, value=0, step=1,
        )

    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    with c4:
        diag_1 = st.selectbox(
            "Primary diagnosis (group)",
            ["circulatory", "diabetes", "digestive", "genitourinary", "injury",
             "musculoskeletal", "neoplasms", "other", "respiratory", "supplementary"],
        )
        discharge = st.selectbox(
            "Discharge destination",
            ["Discharged to home",
             "Discharged/transferred to another short term hospital",
             "Discharged/transferred to SNF",
             "Discharged/transferred to another type of inpatient care institution",
             "Discharged/transferred/referred to this institution for outpatient services",
             "Discharged/transferred to home with home health service",
             "Discharged/transferred to another rehab fac",
             "Hospice / home", "Hospice / medical facility"],
        )
    with c5:
        a1c = st.selectbox("HbA1c test result", ["Not measured", "Norm", ">7", ">8"])
        glu = st.selectbox("Glucose serum test", ["Not measured", "Norm", ">200", ">300"])
        specialty = st.selectbox(
            "Attending specialty",
            ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
             "Cardiology", "Surgery-General", "Nephrology", "Other"],
        )
    with c6:
        diabetes_med = st.radio("Diabetes medication prescribed?", ["Yes", "No"], horizontal=True)
        dose_changed = st.radio("Was any dose changed this stay?", ["No", "Yes"], horizontal=True)
        insulin = st.selectbox("Insulin dose this stay", ["No", "Steady", "Up", "Down"])
        n_dose_changes = st.number_input(
            "Number of drugs whose dose was changed", min_value=0, max_value=10, value=0, step=1
        )

    submitted = st.form_submit_button("Predict readmission risk", type="primary")

# --------------------------------------------------------------------- build the row
# The model expects one value for every feature. We start from the training
# defaults, then overwrite the ones the user actually controls.
def build_feature_row() -> pd.DataFrame:
    row = dict(defaults)

    # direct numeric / engineered inputs
    row.update({
        "age_mid": float(age),
        "time_in_hospital": float(time_in_hospital),
        "num_lab_procedures": float(num_lab_procedures),
        "num_procedures": float(num_procedures),
        "num_medications": float(num_medications),
        "number_diagnoses": float(number_diagnoses),
        "number_inpatient": float(number_inpatient),
        "number_emergency": float(number_emergency),
        "number_outpatient": float(number_outpatient),
        "prior_visits_total": float(prior_visits_total or
                                    (number_inpatient + number_emergency + number_outpatient)),
        "n_dose_changes": float(n_dose_changes),
        "diabetesMed": 1 if diabetes_med == "Yes" else 0,
        "change": 1 if dose_changed == "Yes" else 0,
        "gender": 0 if gender == "Female" else 1,
        "insulin": {"No": 0, "Steady": 1, "Up": 2, "Down": 3}[insulin],
    })

    # binary flags: keep all drug columns at their default, set insulin separately
    for c in features:
        if c.startswith("race_"):
            row[c] = 1 if c == f"race_{race}" else 0
        elif c.startswith("diag_1_grp_"):
            row[c] = 1 if c == f"diag_1_grp_{diag_1}" else 0
        elif c.startswith("A1Cresult_"):
            row[c] = 1 if c == f"A1Cresult_{a1c}" else 0
        elif c.startswith("max_glu_serum_"):
            row[c] = 1 if c == f"max_glu_serum_{glu}" else 0
        elif c.startswith("medical_specialty_"):
            row[c] = 1 if c == f"medical_specialty_{specialty}" else 0

    # the discharge codes the training data actually saw
    discharge_codes = {
        "Discharged to home": 1,
        "Discharged/transferred to another short term hospital": 2,
        "Discharged/transferred to SNF": 3,
        "Discharged/transferred to another type of inpatient care institution": 5,
        "Discharged/transferred/referred to this institution for outpatient services": 17,
        "Discharged/transferred to home with home health service": 6,
        "Discharged/transferred to another rehab fac": 22,
        "Hospice / home": 13,
        "Hospice / medical facility": 14,
    }[discharge]
    for c in features:
        if c.startswith("discharge_disposition_id_"):
            row[c] = 1 if c == f"discharge_disposition_id_{discharge_codes}" else 0

    return pd.DataFrame([row], columns=features)

# --------------------------------------------------------------------- predict
if submitted:
    # --- input validation -----------------------------------------------------
    problems = []
    if age < 0 or age > 100:
        problems.append("Age must be between 0 and 100.")
    if time_in_hospital < 1:
        problems.append("Days in hospital must be at least 1.")
    for label, value in [("Lab procedures", num_lab_procedures),
                         ("Procedures", num_procedures),
                         ("Medications", num_medications),
                         ("Diagnoses", number_diagnoses),
                         ("Prior inpatient stays", number_inpatient),
                         ("Prior emergency visits", number_emergency),
                         ("Prior outpatient visits", number_outpatient)]:
        if value < 0:
            problems.append(f"{label} cannot be negative.")

    if problems:
        st.error("Please fix the input before predicting:\n\n- " + "\n- ".join(problems))
    else:
        X_row = build_feature_row()
        proba = float(model.predict_proba(X_row)[0, 1])

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Probability of readmission within 30 days", f"{proba:.1%}")

        if proba < 0.10:
            band = ("LOW", "green")
            advice = "Below the dataset average. Standard discharge process."
        elif proba < 0.25:
            band = ("MODERATE", "orange")
            advice = "Around the dataset average. Worth a follow-up call / appointment."
        else:
            band = ("HIGH", "red")
            advice = "Well above average. Flag for a follow-up plan before discharge."

        st.markdown(f"### Risk band: :{band[1]}[{band[0]}]")
        st.write(advice)

        with st.expander("What the model used (input echo)"):
            st.json({
                "age": age, "gender": gender, "race": race,
                "prior_inpatient_stays": number_inpatient,
                "discharge_destination": discharge,
                "primary_diagnosis": diag_1,
                "HbA1c": a1c, "insulin": insulin,
                "number_of_diagnoses": number_diagnoses,
            })

        st.info(
            "**Disclaimer:** this is a student project trained on 1999–2008 insurance "
            "records. It is a demonstration of the pipeline only and must not inform "
            "real clinical decisions. Real use requires validation on current local "
            "data, clinical review, and regulatory approval."
        )
else:
    st.caption("Fill the form and press **Predict readmission risk**.")
