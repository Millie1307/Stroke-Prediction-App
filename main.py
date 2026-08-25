import streamlit as st
import joblib
import pandas as pd

# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "stroke_prediction_pipeline.joblib"
METADATA_PATH = "stroke_prediction_metadata.joblib"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Stroke Prediction App")
st.write(
    "Enter the patient's information below to estimate their risk of stroke."
)


# ============================================================
# 3. LOAD MODEL AND METADATA
# ============================================================

try:
    loaded_model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    st.success("Model and metadata loaded successfully!")

except FileNotFoundError as e:
    st.error("Model or metadata file was not found.")
    st.write("Make sure these files are in the same folder as your app.py:")
    st.code(
        "stroke_prediction_pipeline.joblib\n"
        "stroke_prediction_metadata.joblib"
    )
    st.exception(e)
    st.stop()

except Exception as e:
    st.error("Could not load the model or metadata.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. GET FEATURES FROM METADATA
# ============================================================

numerical_features = metadata.get("numerical_features", [])
categorical_features = metadata.get("categorical_features", [])

classification_threshold = metadata.get(
    "classification_threshold", 
    0.5
)

# Display information about the model
with st.expander("Model Information"):
    st.write("Numerical features:", numerical_features)
    st.write("Categorical features:", categorical_features)
    st.write(
        f"Classification threshold: {classification_threshold}"
    )


# ============================================================
# 5. COLLECT PATIENT INFORMATION
# ============================================================

input_data = {}

# ------------------------------------------------------------
# Numerical Features
# ------------------------------------------------------------

st.header("📊 Patient Information")

col1, col2, col3 = st.columns(3)

for feature in numerical_features:

    if feature == "age":
        with col1:
            input_data[feature] = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=45.0,
                step=0.1
            )

    elif feature == "avg_glucose_level":
        with col2:
            input_data[feature] = st.number_input(
                "Average Glucose Level",
                min_value=50.0,
                max_value=300.0,
                value=100.0,
                step=0.1
            )

    elif feature == "bmi":
        with col3:
            input_data[feature] = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1
            )


# ============================================================
# 6. CATEGORICAL FEATURES
# ============================================================

st.header("🏥 Medical and Lifestyle Information")

cat_col1, cat_col2 = st.columns(2)

for feature in categorical_features:

    if feature == "gender":
        with cat_col1:
            input_data[feature] = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

    elif feature == "hypertension":
        with cat_col2:
            input_data[feature] = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

    elif feature == "heart_disease":
        with cat_col1:
            input_data[feature] = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

    elif feature == "ever_married":
        with cat_col2:
            input_data[feature] = st.selectbox(
                "Ever Married",
                ["Yes", "No"]
            )

    elif feature == "work_type":
        with cat_col1:
            input_data[feature] = st.selectbox(
                "Work Type",
                [
                    "Private",
                    "Self-employed",
                    "Govt_job",
                    "children",
                    "Never_worked"
                ]
            )

    elif feature == "Residence_type":
        with cat_col2:
            input_data[feature] = st.selectbox(
                "Residence Type",
                ["Urban", "Rural"]
            )

    elif feature == "smoking_status":
        with cat_col1:
            input_data[feature] = st.selectbox(
                "Smoking Status",
                [
                    "never smoked",
                    "formerly smoked",
                    "smokes",
                    "Unknown"
                ]
            )


# ============================================================
# 7. PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Stroke Risk",
    use_container_width=True
):

    try:

        # Convert input into DataFrame
        input_df = pd.DataFrame([input_data])

        # Display submitted information
        with st.expander("View Patient Input"):
            st.dataframe(input_df)

        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probability = loaded_model.predict_proba(
            input_df
        )[0, 1]

        # ----------------------------------------------------
        # Apply classification threshold
        # ----------------------------------------------------

        prediction = int(
            probability >= classification_threshold
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.subheader("📋 Prediction Result")

        st.metric(
            "Estimated Stroke Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:

            st.error(
                "⚠️ HIGH RISK OF STROKE"
            )

            st.warning(
                "The model has classified this individual "
                "as being at increased risk of stroke."
            )

        else:

            st.success(
                "✅ LOW RISK OF STROKE"
            )

            st.info(
                "The model has classified this individual "
                "as being at lower risk of stroke."
            )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)