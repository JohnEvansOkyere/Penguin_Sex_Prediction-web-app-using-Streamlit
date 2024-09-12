import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Set page configuration with title, icon, and layout
st.set_page_config(page_title="Penguin Prediction App", page_icon="🐧", layout="wide")

# Custom CSS to style the app
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #cfe2f3;
        padding: 20px;
        border-radius: 15px;
    }
    .stSlider > div > div {
        color: #05668D;
    }
    h1, h2, h3, h4 {
        color: #028090;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.write("""
# 🐧 Penguin Prediction App
This app predicts the **sex** of a penguin based on user input or CSV file data. 🐧  

""")

# Sidebar header and input instructions
st.sidebar.header("Input Features")
st.sidebar.markdown("Upload your CSV file or use the sliders below to manually input data.")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])

# Manual input in case the file is not uploaded
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.subheader("Manual Input Parameters")
        species = st.sidebar.selectbox('Species', ('Adelie', 'Chinstrap', 'Gentoo'))
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4203.0)

        data = {
            'species': species,
            'island': island,
            "bill_length_mm": bill_length_mm,
            "bill_depth_mm": bill_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g
        }

        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

# Load penguins dataset for encoding purposes
penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['sex'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding the categorical columns
encode = ['species', 'island']
for col in encode:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(col, axis=1, inplace=True)

# Select the first row (user input) for prediction
df = df[:1]

# Display user input features
st.subheader("User Input Features")
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write("Awaiting CSV file to be uploaded. Displaying manual input parameters:")
    st.write(input_df)

# Load the pre-trained model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply the model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Prediction results section
st.subheader("Prediction Results")
penguins_sex = np.array(['Male', 'Female'])
st.write(f"**Predicted Sex**: {penguins_sex[prediction][0]}")

# Display prediction probability
st.subheader("Prediction Probability")
st.write(f"**Probability (Male/Female)**: {prediction_proba[0][0]:.2f} / {prediction_proba[0][1]:.2f}")

