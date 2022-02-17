import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
sns.set(style = 'darkgrid', font_scale = 6)


st.title('Diamond Pricing App')
st.write("""From the Data below we built a machine learning-based pricing model 
         to get diamond price depending on is specifications""")

st.sidebar.title("Diamond Pricing App")
st.sidebar.info("Change Parameter to see how diamond pricing change")
st.sidebar.title("Parameter")

carat = st.sidebar.slider('Carat', 0.00, 10.00, 2.00, step = 0.01)
depth = st.sidebar.slider('depth', 40.00, 80.00, 45.00,step = 0.01)
table = st.sidebar.slider('table', 40.00, 100.00, 50.00,step = 0.01)
x = st.sidebar.slider('X', 0.00, 12.00, 1.00)
y = st.sidebar.slider('Y', 0.00, 60.00, 1.00)
z = st.sidebar.slider('Z', 0.00, 32.00, 1.00)

    
cut = st.sidebar.selectbox("Cut",['cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good'])

if cut == 'cut_Fair':
    cut_list = [1, 0, 0, 0,0]
elif cut == 'cut_Good':
    cut_list = [0, 1, 0, 0,0]
elif cut == 'cut_Ideal':
    cut_list = [0, 0, 1, 0,0]
elif cut == 'cut_Premium':
    cut_list = [0, 0, 0, 1,0]
elif cut == 'cut_Very Good':
    cut_list = [0, 0, 0,0,1]
    
color = st.sidebar.selectbox("Color",['color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I',
       'color_J'])
    
if color == 'color_D':
    col_list = [1, 0, 0, 0,0,0,0]
elif color == 'color_E':
    col_list = [0, 1, 0, 0,0,0,0]
elif color == 'color_F':
    col_list = [0, 0, 1, 0,0,0,0]
elif color == 'color_G':
    col_list = [0, 0, 0, 1,0,0,0]
elif color == 'color_H':
    col_list = [0, 0, 0,0,1,0,0]
elif color == 'color_I':
    col_list = [0, 0, 0,0,0,1,0]
elif color == 'color_J':
    col_list = [0, 0, 0,0,0,0,1]
                                         
st.subheader("Output Diamond Price")


filename = 'diammodel.sav'
loaded_model = joblib.load(filename)

prediction = np.round(loaded_model.predict([[carat, depth, table, x, y,z] + cut_list + col_list])[0],2)
st.write(f"Suggested Diamond Price is: {prediction}")