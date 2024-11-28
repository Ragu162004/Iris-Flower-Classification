import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Iris Flower Classification", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Iris.csv")
    return data

data = load_data()

# Preprocess the data
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.markdown("<h1 style='text-align: center; color: #3E4E55;'>Iris Flower Classification</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #6A737D; font-size: 16px;'>
    Predict the species of an Iris flower based on its features using a Random Forest Classifier.
    </p>
    """,
    unsafe_allow_html=True,
)

# Sidebar input sliders for the features
st.sidebar.header("Input Flower Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['SepalLengthCm'].min()), float(X['SepalLengthCm'].max()), float(X['SepalLengthCm'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['SepalWidthCm'].min()), float(X['SepalWidthCm'].max()), float(X['SepalWidthCm'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['PetalLengthCm'].min()), float(X['PetalLengthCm'].max()), float(X['PetalLengthCm'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['PetalWidthCm'].min()), float(X['PetalWidthCm'].max()), float(X['PetalWidthCm'].mean()))

# Predict based on user inputs
features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)

# Output columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Flower Species (Class)")
    st.markdown(
        f"<p style='font-size: 20px; color: white;'><b>{le.inverse_transform([prediction[0]])[0]}</b></p>",
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    st.table(proba_df.style.highlight_max(axis=1, color='rgb(0, 215, 130)'))

# Footer or additional details
st.markdown("""
<hr style="border: 1px solid #ddd;">
<p style="text-align: center; color: #6A737D;">
    Created by <a href="https://github.com/Ragu162004" target="_blank" style="color: #3E4E55; text-decoration: none;">Ragu</a>
</p>
""", unsafe_allow_html=True)
