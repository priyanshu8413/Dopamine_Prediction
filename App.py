import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
st.set_page_config(layout="wide",page_title="Dopamine Prediction",page_icon="⚕️")


page_bg_img = '''
        <style>
            .stApp {
                background: url("https://i.imgur.com/niEIKu3.jpeg") no-repeat center center fixed;
                background-size: cover;
            }
            .center-text {
                text-align: center;
                font-size: 30px !important;
                font-weight: bold;
                color: black;
               
            }
            .title-text {
            font-size: 80px !important;
            font-weight: bold !important;
            color: yellow;
            white-space: nowrap;
            text-align: left;
            display: block; /* Ensures the text block respects alignment */
               line-height: 1.1; 
        }
        div[data-testid="stForm"] button {
        font-size: 90px !important;
        font-weight: bold !important;
        padding: 20px 10px !important;
        background-color:#fff3cd  !important;
        color: red !important;
        border-radius: 5px !important;
        border: none !important;
        width: 100% !important;
        display: block;
        margin: auto;
    }
      input[type="text"] {
                 font-size: 50px !important; /* Adjust the size as needed */
                  font-weight: bold !important;
                  
                }
        </style>
        '''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<p class="title-text">Dopamine Prediction App </p>', unsafe_allow_html=True)
model_filename = "decision_tree_model.pkl"  # Path to uploaded model
with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)
dataset_filename = "train data i0_i _final.csv"  # Path to uploaded dataset
df = pd.read_csv(dataset_filename)
feature_columns = df.columns[:-1]  # All columns except the last one (target)
X_train = df[feature_columns]

# Fit the StandardScaler on the dataset
scaler = StandardScaler()
scaler.fit(X_train)
scaler_filename = "scaler.pkl"
with open(scaler_filename, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
emissions_file = "emissions.csv"

with st.form("Dopamine Form"):
    col1,col2=st.columns([2,1])
    with col1:
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">Concentration Of Dopamine(µm)</p>',
            unsafe_allow_html=True)
        Dopamine= st.text_input("Enter Value", key="DA Conc.", placeholder="Enter Dopamine Content",
                                 label_visibility="collapsed")
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">Concentration  Of ANWM-GQDS(mg/ml)</p>',
            unsafe_allow_html=True)
        Concentration = st.text_input("Enter Value", key="Concentration", placeholder="Enter Concentration of ANWM-GQDS(mg/ml)",
                                 label_visibility="collapsed")
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">Temperature(°c)</p>',
            unsafe_allow_html=True)
        Temp = st.text_input("Enter Value", key="Temp ", placeholder="Enter The Temperature",
                                 label_visibility="collapsed")
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">pH</p>',
            unsafe_allow_html=True)
        pH = st.text_input("Enter Value", key="pH", placeholder="Enter pH Value",
                                 label_visibility="collapsed")
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">Peak position(nm)</p>',
            unsafe_allow_html=True)
        Peak = st.text_input("Enter Value", key="Peak", placeholder="Enter Peak Content",
                                 label_visibility="collapsed")
        st.markdown(
            '<p style="font-size: 40px; font-weight: bold; color: yellow; margin-bottom:-5px">Intensity</p>',
            unsafe_allow_html=True)
        Intensity = st.text_input("Enter Value", key="Intensity", placeholder="Enter Intensity Content",
                                 label_visibility="collapsed")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        submit = st.form_submit_button(label="Submit")

if submit:
    try:
        # Convert input values to float
        input_values = np.array(
            [float(Dopamine), float( Concentration), float( Temp), float(pH), float(Peak), float( Intensity)]).reshape(1,
                                                                                                                    -1)

        # Scale input using the scaler fitted on the dataset
        # Ensure input values have correct feature names
        input_df = pd.DataFrame(input_values, columns=feature_columns)

        # Apply the scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction

        predicted_ratio = model.predict(input_scaled)[0]

        # Display prediction result
        print(f"Predicted Intensity Ratio: **{predicted_ratio:.5f}**")

        st.markdown(
            f"""
                       <div style="
                           padding: 40px;
                           background-color: #E8F5E9;
                           color: #1B5E20;
                           border-radius: 10px;
                           text-align: center;
                           font-size: 100px;
                           font-weight: bold;
                           border: 2px solid #1B5E20;
                       ">
                           Predicted Intensity Ratio ( Io/I)  : {predicted_ratio} 
                       </div>
                       """,
            unsafe_allow_html=True
        )

        emissions_df = pd.read_csv(emissions_file)
        emissions_df.columns = emissions_df.columns.str.strip()
        emissions_df = emissions_df[['Carbon Emissions (kg CO2)', 'Energy Consumption (kWh)', 'Runtime (s)']]




        st.markdown("<h2 style='color:yellow; text-align:center font-size:90px;'>Sustainability Checking</h2>",
                            unsafe_allow_html=True)
        table_html = f"""
                                  <style>


                                      table {{

                                          width: 3000px;
                                          border-collapse: collapse;
                                          text-align: center ;
                                          font-size: 30px;
                                          border-radius: 20px;
                                          overflow: hidden;

                                      }}
                                      th, td {{
                                          padding: 50px;
                                          border: 1px solid black;
                                          color: black;
                                          font-size:40px !important;
                                          font-weight:bold !important;

                                      }}
                                      th {{
                                          font-weight: bold;
                                         justify-content: center;
                                         align-items: center; 


                                      }}

                                      th:nth-child(1), td:nth-child(1) {{ background-color: #ffcccb; }}  /* Light Red */
                                      th:nth-child(2), td:nth-child(2) {{ background-color: #d4edda; }}  /* Light Green */
                                      th:nth-child(3), td:nth-child(3) {{ background-color: #cce5ff; }}  /* Light Blue */
                                      th:nth-child(4), td:nth-child(4) {{ background-color: #fff3cd; }}  /* Light Yellow */


                                  </style>
                              <table>




                                  """

                # Display the styled table
        st.markdown(table_html , unsafe_allow_html=True)
                # Display table with bold headers
        st.markdown(emissions_df.to_html(index=False, escape=False), unsafe_allow_html=True)



    except ValueError:
        st.error("Please enter valid numerical values for all fields.")





