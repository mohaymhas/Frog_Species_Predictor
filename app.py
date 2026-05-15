
import streamlit as st
import pandas as pd
import numpy as np
import pickle   
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')



# Page Configuration
st.set_page_config(
    page_title="Frog Classification using MFCCs",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)
# place this near the top of your app (after st.set_page_config)

with st.sidebar:
    st.image("Frogs_MFCCs.png", caption=None, use_container_width=False)
   # st.markdown("---")           # optional line separator
   # st.subheader("Navigation & Filters")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)
# load models names

# Load Data Function
@st.cache_data


# Load the data
def load_data():
    """Load and preprocess the sleep health data"""
    models = pd.read_csv('model.csv')
    return models
species = [
    'AdenomeraAndre', 'AdenomeraHylaedactylus', 'Ameeregatrivittata', 
    'HylaMinuta', 'HypsiboasCinerascens', 'HypsiboasCordobae', 
    'LeptodactylusFuscus', 'OsteocephalusOophagus', 
    'Rhinellagranulosa', 'ScinaxRuber'
]

Genus = [
    'Adenomera', 'Adenomera', 'Ameerega', 
    'Dendropsophus', 'Hypsiboas', 'Hypsiboas', 
    'Leptodactylus', 'Osteocephalus', 
    'Rhinella', 'Scinax'
]

Family = [
    'Leptodactylidae', 'Leptodactylidae', 'Dendrobatidae', 
    'Hylidae', 'Hylidae', 'Hylidae', 
    'Leptodactylidae', 'Hylidae', 
    'Bufonidae', 'Hylidae'
]


 
models = load_data()
# Sidebar - Navigation and Filters
#st.sidebar.title("🎯 Navigation & Filters")
#page = st.sidebar.radio(
#    "Select Analysis Page:",
#    ["🏠 Overview", "📊 Financial Analysis", "🔍 Provider Analysis", "📋 Data Explorer"]
#)
st.sidebar.subheader("Frog Classification using MFCCs")

#sliders for MFCCs inputs
MFCCs_1 = st.sidebar.slider(
    label="MFCCs_1",
    min_value=-0.251179153694868,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_2 = st.sidebar.slider(
    label="MFCCs_2",
    min_value=-0.673025383989239,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_3 = st.sidebar.slider(
    label="MFCCs_3",
    min_value=-0.436027557631229,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_4 = st.sidebar.slider(
    label="MFCCs_4",
    min_value=-0.472676234325039,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_5 = st.sidebar.slider(
    label="MFCCs_5",
    min_value=-0.636012475323151,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_6 = st.sidebar.slider(
    label="MFCCs_6",
    min_value=-0.410416976545143,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_7 = st.sidebar.slider(
    label="MFCCs_7",
    min_value=-0.538981700470428,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_8 = st.sidebar.slider(
    label="MFCCs_8",
    min_value=-0.576506174491891,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_9 = st.sidebar.slider(
    label="MFCCs_9",
    min_value=-0.58731343775337,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_10 = st.sidebar.slider(
    label="MFCCs_10",
    min_value=-0.952265655557829,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_11 = st.sidebar.slider(
    label="MFCCs_11",
    min_value=-0.901988854570569,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_12 = st.sidebar.slider(
    label="MFCCs_12",
    min_value=-0.799441302280563,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_13 = st.sidebar.slider(
    label="MFCCs_13",
    min_value=-0.644115927164735,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_14 = st.sidebar.slider(
    label="MFCCs_14",
    min_value=-0.590380219762154,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_15 = st.sidebar.slider(
    label="MFCCs_15",
    min_value=-0.717155589773686,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_16 = st.sidebar.slider(
    label="MFCCs_16",
    min_value=-0.498675403830133,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_17 = st.sidebar.slider(
    label="MFCCs_17",
    min_value=-0.421479728327128,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_18 = st.sidebar.slider(
    label="MFCCs_18",
    min_value=-0.759321553990284,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_19 = st.sidebar.slider(
    label="MFCCs_19",
    min_value=-0.680745390359785,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_20 = st.sidebar.slider(
    label="MFCCs_20",
    min_value=-0.361648778290323,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_21 = st.sidebar.slider(
    label="MFCCs_21",
    min_value=-0.430811776008093,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)
MFCCs_22 = st.sidebar.slider(
    label="MFCCs_22",
    min_value=-0.379304321533521,
    max_value=1.000000,
    value=0.000000,
    step=0.000001,
    format="%f"
)




st.sidebar.markdown("---")
st.sidebar.write("Select model to apply")

# Filters
#st.sidebar.subheader("🔧 Filters")


model_filter = st.sidebar.multiselect(
    "model:",
    options = models['Model'].unique(),
    default = models['Model'].unique()
)

    
    


# Apply filters
filtered_models = models[
    (models['Model'].isin(model_filter))  
]
loop = list(filtered_models['id'])
st.sidebar.markdown("---")
st.sidebar.info(f"**Filtered models:** {len(filtered_models)} / {len(models)}")

# Download filtered data

# Main Content Area


#speech_vector = vectorizer.fit_transform([speech])
if st.button("Predict"):
    #print('good')
    Required_MFCCs = [MFCCs_1, MFCCs_2, MFCCs_3, MFCCs_4, MFCCs_5, MFCCs_6, MFCCs_7, MFCCs_8, MFCCs_9, MFCCs_10, MFCCs_11, MFCCs_12, MFCCs_13, MFCCs_14, MFCCs_15, MFCCs_16, MFCCs_17, MFCCs_18, MFCCs_19, MFCCs_20, MFCCs_21, MFCCs_22]
    input_data = np.array(Required_MFCCs).reshape(1, -1)
    for model_id in loop:
        st.write(f"Model: {model_id} - {filtered_models[filtered_models['id'] == model_id]['Model'].values[0]}")
        match model_id:
            case 1:
                filename = filtered_models['filename'].loc[filtered_models['id'] == model_id].values[0]
                with open("knn_model.pkl", "rb") as f:
                    model = pickle.load(f)  # 'bundle' is the model, so load it directly into 'model'

                prediction = model.predict(input_data)
                st.write(f"Prediction: {prediction[0]}")
                st.write(f"Species: {species[prediction[0]]}") 
                st.write(f"Genus: {Genus[prediction[0]]}")
                st.write(f"Family: {Family[prediction[0]]}")

        # Here you would load the model and make a prediction based on the input text
        # For example:
        # model = load_model(model_id)
        # prediction = model.predict(speech)
        # st.write(f"Prediction: {prediction}")
            case 2:
                filename = filtered_models['filename'].loc[filtered_models['id'] == model_id].values[0]
                with open("SVC_poly_model.pkl", "rb") as f:
                    model = pickle.load(f) 
                prediction = model.predict(input_data)
                st.write(f"Prediction: {prediction[0]}")
                st.write(f"Species: {species[prediction[0]]}")
                st.write(f"Genus: {Genus[prediction[0]]}")
                st.write(f"Family: {Family[prediction[0]]}")

            case 3:
                filename = filtered_models['filename'].loc[filtered_models['id'] == model_id].values[0]
                with open("RandomForest_model.pkl", "rb") as f:
                    model = pickle.load(f) 
                prediction = model.predict(input_data)
                st.write(f"Prediction: {prediction[0]}")
                st.write(f"Species: {species[prediction[0]]}")
                st.write(f"Genus: {Genus[prediction[0]]}")
                st.write(f"Family: {Family[prediction[0]]}")

            case 4:
                filename = filtered_models['filename'].loc[filtered_models['id'] == model_id].values[0]
                with open("lgb_model.pkl", "rb") as f:
                    model = pickle.load(f) 
                prediction = model.predict(input_data)
                st.write(f"Prediction: {prediction[0]}")
                st.write(f"Species: {species[prediction[0]]}")
                st.write(f"Genus: {Genus[prediction[0]]}")
                st.write(f"Family: {Family[prediction[0]]}")

st.image("frog_graph.png", caption=None, use_container_width=True)
st.markdown("""
<style>
/* KPI card */
[data-testid="stMetric"] {
  background: rgba(050,050,050,0.06);   /* optional: subtle card bg on dark theme */
  border: 1px solid rgba(200,200,200,0.12);
  border-radius: 12px;
  padding: 14px 16px;
}

/* label (small text above) */
[data-testid="stMetricLabel"] > div {
  color: #000000 !important;
  opacity: 0.85;                         /* keep a bit dimmer than value */
}

/* main value */
[data-testid="stMetricValue"] {
  color: #000000 !important;
  opacity: 0.85;
  font-size: 1.4rem;  /* Smaller than 1.6rem */
  font-weight: 700;    
}

/* delta text + badge */
[data-testid="stMetricDelta"] {
  color: #000000 !important;             /* make delta text white */
}
[data-testid="stMetricDelta"] svg {      /* make the arrow white too */
  filter: brightness(0) invert(1);
}

/* optional: make all plotly charts stretch instead of using use_container_width */
.user-select-none svg { max-width: 100%; }
</style>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Frog Classification using MFCCs | Built with Streamlit</p>
    <p>Data Source: Frog Classification using MFCCs  Dataset from Kaggle</p>
</div>
""", unsafe_allow_html=True)