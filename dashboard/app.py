import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Personal Loan Acceptance Prediction",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


# ============================================================
# Custom Premium CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #061826 0%, #0b3c5d 45%, #061826 100%);
        color: white;
    }

    .main-title {
        font-size: 46px;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }

    .sub-title {
        font-size: 18px;
        color: #d4af37;
        text-align: center;
        margin-bottom: 35px;
    }

    .lux-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 22px;
        padding: 25px;
        box-shadow: 0 8px 35px rgba(0,0,0,0.35);
        margin-bottom: 20px;
    }

    .metric-card {
        background: rgba(255,255,255,0.10);
        padding: 22px;
        border-radius: 20px;
        border: 1px solid rgba(212, 175, 55, 0.35);
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.30);
    }

    .metric-value {
        font-size: 34px;
        font-weight: 900;
        color: #d4af37;
    }

    .metric-label {
        font-size: 15px;
        color: #ffffff;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #d4af37, #f5d76e);
        color: #061826;
        font-weight: 800;
        border-radius: 14px;
        border: none;
        padding: 12px 28px;
        width: 100%;
        font-size: 17px;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #f5d76e, #d4af37);
        color: #061826;
    }

    section[data-testid="stSidebar"] {
        background-color: #061826;
    }

    h1, h2, h3 {
        color: white;
    }

    label {
        color: white !important;
        font-weight: 600 !important;
    }

    .stDataFrame {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Load Models and Data
# ============================================================

@st.cache_resource
def load_models():
    personal_model = joblib.load(os.path.join(MODELS_DIR, "personal_loan_model.pkl"))
    personal_columns = joblib.load(os.path.join(MODELS_DIR, "personal_loan_columns.pkl"))

    approval_model = joblib.load(os.path.join(MODELS_DIR, "loan_approval_model.pkl"))
    approval_columns = joblib.load(os.path.join(MODELS_DIR, "loan_approval_columns.pkl"))

    return personal_model, personal_columns, approval_model, approval_columns


@st.cache_data
def load_data():
    personal_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "personal_loan_cleaned.csv"))
    approval_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "loan_approval_cleaned.csv"))
    results_df = pd.read_csv(os.path.join(REPORTS_DIR, "model_results.csv"))

    return personal_df, approval_df, results_df


try:
    personal_model, personal_columns, approval_model, approval_columns = load_models()
    personal_df, approval_df, results_df = load_data()
except Exception:
    st.error("Pehle model train karein: python src/train_model.py")
    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    "<div class='main-title'>Personal Loan Acceptance Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Premium AI Dashboard for Banking & Loan Intelligence</div>",
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Personal Loan Prediction",
        "Loan Approval Prediction",
        "EDA & Insights",
        "Model Performance"
    ]
)


# ============================================================
# Home Page
# ============================================================

if page == "Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{personal_df.shape[0]}</div>
                <div class='metric-label'>Personal Loan Records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{approval_df.shape[0]}</div>
                <div class='metric-label'>Loan Approval Records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>2</div>
                <div class='metric-label'>Machine Learning Models</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='lux-card'>
            <h3>Project Objective</h3>
            <p>
            Is project ka objective banking customers ka data use kar ke predict karna hai
            ke kaunsa customer personal loan offer accept kar sakta hai.
            Is ke sath second dataset par loan approval prediction bhi perform ki gayi hai.
            </p>
            <p>
            Project mein data cleaning, EDA, machine learning model training,
            evaluation, business insights aur premium Streamlit dashboard included hai.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='lux-card'>
            <h3>Algorithms Used</h3>
            <p><b>Personal Loan Acceptance:</b> Logistic Regression</p>
            <p><b>Loan Approval Prediction:</b> Decision Tree Classifier</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Personal Loan Prediction Page
# ============================================================

elif page == "Personal Loan Prediction":

    st.markdown("## Personal Loan Acceptance Prediction")

    st.markdown(
        """
        <div class='lux-card'>
            Customer ki information enter karein. Model predict karega ke customer
            personal loan offer accept karega ya nahi.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        experience = st.number_input("Experience", min_value=0, max_value=60, value=10)
        income = st.number_input("Income", min_value=0, max_value=500, value=80)

    with col2:
        family = st.selectbox("Family Members", [1, 2, 3, 4])
        ccavg = st.number_input("Average Credit Card Spending", min_value=0.0, max_value=20.0, value=2.5)
        education = st.selectbox(
            "Education",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Undergraduate",
                2: "Graduate",
                3: "Advanced / Professional"
            }[x]
        )

    with col3:
        mortgage = st.number_input("Mortgage", min_value=0, max_value=1000, value=0)
        securities_account = st.selectbox("Securities Account", [0, 1])
        cd_account = st.selectbox("CD Account", [0, 1])
        online = st.selectbox("Online Banking", [0, 1])
        credit_card = st.selectbox("Credit Card", [0, 1])

    input_data = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities_account,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": credit_card
    }])

    input_data = input_data[personal_columns]

    if st.button("Predict Personal Loan Acceptance"):
        prediction = personal_model.predict(input_data)[0]
        probability = personal_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"Customer is likely to ACCEPT the personal loan offer. Probability: {probability:.2%}")
        else:
            st.error(f"Customer is NOT likely to accept the personal loan offer. Probability: {probability:.2%}")


# ============================================================
# Loan Approval Prediction Page
# ============================================================

elif page == "Loan Approval Prediction":

    st.markdown("## Loan Approval Prediction")

    st.markdown(
        """
        <div class='lux-card'>
            Applicant ki financial information enter karein.
            Model predict karega ke loan approve hoga ya reject.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Person Income", min_value=0, max_value=1000000, value=60000)
        person_emp_exp = st.number_input("Employment Experience", min_value=0, max_value=60, value=5)
        loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=100000, value=15000)

    with col2:
        loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=50.0, value=12.5)
        loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.25)
        cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, max_value=50, value=5)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

    with col3:
        person_gender_male = st.selectbox("Gender Male", [0, 1])
        previous_default_yes = st.selectbox("Previous Loan Default Yes", [0, 1])

        education = st.selectbox(
            "Education Level",
            ["Bachelor", "Doctorate", "High School", "Master"]
        )

        home = st.selectbox(
            "Home Ownership",
            ["OTHER", "OWN", "RENT"]
        )

        intent = st.selectbox(
            "Loan Intent",
            ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
        )

    approval_input = pd.DataFrame(columns=approval_columns)
    approval_input.loc[0] = 0

    approval_input["person_age"] = person_age
    approval_input["person_income"] = person_income
    approval_input["person_emp_exp"] = person_emp_exp
    approval_input["loan_amnt"] = loan_amnt
    approval_input["loan_int_rate"] = loan_int_rate
    approval_input["loan_percent_income"] = loan_percent_income
    approval_input["cb_person_cred_hist_length"] = cb_person_cred_hist_length
    approval_input["credit_score"] = credit_score
    approval_input["person_gender_male"] = person_gender_male
    approval_input["previous_loan_defaults_on_file_Yes"] = previous_default_yes

    education_col = f"person_education_{education}"
    home_col = f"person_home_ownership_{home}"
    intent_col = f"loan_intent_{intent}"

    if education_col in approval_input.columns:
        approval_input[education_col] = 1

    if home_col in approval_input.columns:
        approval_input[home_col] = 1

    if intent_col in approval_input.columns:
        approval_input[intent_col] = 1

    approval_input = approval_input[approval_columns]

    if st.button("Predict Loan Approval"):
        prediction = approval_model.predict(approval_input)[0]

        if prediction == 1:
            st.success("Loan is likely to be APPROVED.")
        else:
            st.error("Loan is likely to be REJECTED.")


# ============================================================
# EDA & Insights Page
# ============================================================

elif page == "EDA & Insights":

    st.markdown("## Exploratory Data Analysis & Business Insights")

    tab1, tab2 = st.tabs(["Personal Loan Dataset", "Loan Approval Dataset"])

    with tab1:
        st.markdown("### Personal Loan Dataset Overview")
        st.dataframe(personal_df.head())

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                personal_df,
                x="Age",
                color="Personal Loan",
                title="Age Distribution by Loan Acceptance",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                personal_df,
                x="Income",
                color="Personal Loan",
                title="Income Distribution by Loan Acceptance",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            personal_df,
            x="Education",
            y="Income",
            color="Personal Loan",
            title="Education, Income and Loan Acceptance",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class='lux-card'>
            <h3>Business Insights</h3>
            <ul>
                <li>High income customers personal loan accept karne ke zyada chances rakhte hain.</li>
                <li>CD Account wale customers banking products mein zyada interested ho sakte hain.</li>
                <li>Credit card average spending aur income important factors hain.</li>
                <li>Education level customer loan acceptance behavior ko affect kar sakta hai.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown("### Loan Approval Dataset Overview")
        st.dataframe(approval_df.head())

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                approval_df,
                x="person_age",
                color="loan_status",
                title="Age Distribution by Loan Status",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                approval_df,
                x="person_income",
                color="loan_status",
                title="Income Distribution by Loan Status",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            approval_df,
            x="loan_amnt",
            y="person_income",
            color="loan_status",
            title="Loan Amount vs Income",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class='lux-card'>
            <h3>Business Insights</h3>
            <ul>
                <li>High credit score wale applicants ke approval chances zyada hote hain.</li>
                <li>Previous default hone par rejection ka risk zyada hota hai.</li>
                <li>Loan amount agar income ke comparison mein zyada ho to approval chances kam ho sakte hain.</li>
                <li>Interest rate aur credit history loan decision mein important role play karte hain.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# Model Performance Page
# ============================================================

elif page == "Model Performance":

    st.markdown("## Model Performance")

    st.dataframe(results_df)

    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        color="Algorithm",
        title="Model Accuracy Comparison",
        template="plotly_dark",
        text="Accuracy"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class='lux-card'>
        <h3>Final Conclusion</h3>
        <p>
        Personal Loan Acceptance model ne 95% se zyada accuracy achieve ki.
        Loan Approval model ne bhi strong performance show ki.
        </p>
        <p>
        Business point of view se banks high-income, strong credit profile,
        no previous default aur active banking customers ko better target kar sakte hain.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )