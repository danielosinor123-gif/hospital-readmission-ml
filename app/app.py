"""
Streamlit app: "Will this patient be readmitted within 30 days?"

WHAT THIS FILE DOES
-------------------
1. Loads the FINAL tuned model, the deployed decision threshold, the exact
   feature list, and the default value of every feature. All four files are
   small and committed to the repository, so the app runs after a plain clone
   (including on Streamlit Cloud) with no retraining.
2. Shows a form with the patient details a nurse could actually know at discharge.
3. Fills untouched features with the training-data mode (a valid row, never a
   made-up one).
4. Predicts the probability of readmission within 30 days, shows the alert
   decision and a risk band.
5. Validates every input and refuses to predict on nonsense.

RUN IT
------
    python -m streamlit run app/app.py

IMPORTANT: this is a learning/decision-support demo. It must NOT be used for
real clinical decisions -- see the disclaimer printed inside the app.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------- paths
APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
MODEL_PATH = PROJECT / "models" / "final_model.joblib"          # tuned model (notebook 03)
SUMMARY_PATH = PROJECT / "data" / "processed" / "final_model_summary.json"  # deployed threshold
FEATURES_PATH = PROJECT / "data" / "processed" / "feature_columns.json"
DEFAULTS_PATH = PROJECT / "data" / "processed" / "feature_defaults.json"    # mode per feature

# --------------------------------------------------------------------- setup
st.set_page_config(page_title="Hospital Readmission Risk (demo)", page_icon="🏥")

st.title("🏥 30-Day Readmission Risk — demonstration app")
st.write(
    "Enter what is known about the patient **at discharge**. The app returns the "
    "model's estimated probability of readmission within 30 days and whether it "
    "would raise an alert."
)

# --------------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_config():
    """Feature list, default value per feature, and the deployed threshold.

    The defaults come from feature_defaults.json -- the most common value of each
    feature in the training data -- computed once by src/make_feature_defaults.py
    so the app needs no large data files.
    """
    features = json.loads(FEATURES_PATH.read_text())
    defaults = json.loads(DEFAULTS_PATH.read_text())
    summary = json.loads(SUMMARY_PATH.read_text())
    return features, defaults, float(summary["chosen_threshold"])

missing = [str(p) for p in (MODEL_PATH, SUMMARY_PATH, FEATURES_PATH, DEFAULTS_PATH)
           if not p.exists()]
if missing:
    st.error(
        "Model files are missing. Run the notebooks in order "
        "(01, 02, 03), then `python src/make_feature_defaults.py`, and restart "
        "this app.\n\nMissing:\n" + "\n".join(missing)
    )
    st.stop()

try:
    model = load_model()
    features, defaults, THRESHOLD = load_config()
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
def build_feature_row() -> pd.DataFrame:
    """Start from the training defaults, then apply what the user entered.

    One-hot columns are overwritten group by group so exactly one category in
    each group is 1 -- a valid categorical encoding, not arbitrary zeros.
    """
    row = dict(defaults)

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

    one_hot_choice = {
        "race_": race, "diag_1_grp_": diag_1, "A1Cresult_": a1c,
        "max_glu_serum_": glu, "medical_specialty_": specialty,
    }
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
        for prefix, choice in one_hot_choice.items():
            if c.startswith(prefix):
                row[c] = 1 if c == prefix + choice else 0
        if c.startswith("discharge_disposition_id_"):
            row[c] = 1 if c == f"discharge_disposition_id_{discharge_codes}" else 0

    return pd.DataFrame([row], columns=features)

# --------------------------------------------------------------------- predict
if submitted:
    problems = []
    if not 0 <= age <= 100:
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
        alert = proba >= THRESHOLD

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Probability of readmission within 30 days", f"{proba:.1%}")

        if proba < THRESHOLD / 2:
            band, colour = "LOW", "green"
            advice = f"Well below the alert threshold ({THRESHOLD:.0%}). Standard discharge process."
        elif proba < THRESHOLD:
            band, colour = "MODERATE", "orange"
            advice = f"Below the alert threshold ({THRESHOLD:.0%}) but above half of it. Worth a follow-up call."
        else:
            band, colour = "HIGH", "red"
            advice = (f"At or above the alert threshold ({THRESHOLD:.0%}) -- the model would "
                      "raise an alert and request a follow-up plan.")

        st.markdown(f"### Risk band: :{colour}[{band}]")
        st.write(advice)
        if alert:
            st.warning("⚠️ This profile crosses the deployed threshold: the model flags it for follow-up.")
        else:
            st.success("No alert: below the deployed threshold.")

        with st.expander("What the model used (input echo)"):
            st.json({
                "age": age, "gender": gender, "race": race,
                "prior_inpatient_stays": number_inpatient,
                "discharge_destination": discharge,
                "primary_diagnosis": diag_1,
                "HbA1c": a1c, "insulin": insulin,
                "number_of_diagnoses": number_diagnoses,
                "deployed_threshold": THRESHOLD,
            })

        st.info(
            "**Disclaimer:** this is a student project trained on 1999–2008 insurance "
            "records. On the held-out test set the model catches about 44% of true "
            "readmissions at 21% precision. It is a demonstration of the pipeline only "
            "and must not inform real clinical decisions -- real use requires validation "
            "on current local data, clinical review, and regulatory approval."
        )
else:
    st.caption("Fill the form and press **Predict readmission risk**.")