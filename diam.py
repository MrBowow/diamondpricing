import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
sns.set_theme(style="whitegrid")


diamonddata = pd.read_csv("diamonds_regression.csv")
diamdata = diamonddata.drop(columns = ['Unnamed: 0'])

if st.sidebar.checkbox('Show DataFrame'):
       st.write(diamdata.head(20))

st.title('Diamond Pricing App')
st.write("""From the Data below we built a machine learning-based pricing model 
         to get diamond price depending on is specifications""")

st.sidebar.title("Diamond Pricing App")
st.sidebar.info("Change Parameter to see how diamond pricing change")
st.sidebar.title("Parameter")

carat = st.sidebar.slider('Carat', 0.00, 6.00, 2.00, step = 0.01)
depth = st.sidebar.slider('depth', 40.00, 80.00, 45.00,step = 0.01)
table = st.sidebar.slider('table', 40.00, 100.00, 50.00,step = 0.01)
x = st.sidebar.slider('X', 0.00, 12.00, 1.00)
y = st.sidebar.slider('Y', 0.00, 60.00, 1.00)
z = st.sidebar.slider('Z', 0.00, 32.00, 1.00)

    
cut = st.sidebar.selectbox("Cut",['Fair', 'Good', 'Ideal', 'Premium', 'Very Good'])

if cut == 'Fair':
    cut_list = [1, 0, 0, 0,0]
elif cut == 'Good':
    cut_list = [0, 1, 0, 0,0]
elif cut == 'Ideal':
    cut_list = [0, 0, 1, 0,0]
elif cut == 'Premium':
    cut_list = [0, 0, 0, 1,0]
elif cut == 'Very Good':
    cut_list = [0, 0, 0,0,1]
    
color = st.sidebar.selectbox("Color",['D', 'E', 'cF', 'cG', 'H', 'I',
       'J'])
    
if color == 'D':
    col_list = [1, 0, 0, 0,0,0,0]
elif color == 'E':
    col_list = [0, 1, 0, 0,0,0,0]
elif color == 'F':
    col_list = [0, 0, 1, 0,0,0,0]
elif color == 'G':
    col_list = [0, 0, 0, 1,0,0,0]
elif color == 'H':
    col_list = [0, 0, 0,0,1,0,0]
elif color == 'I':
    col_list = [0, 0, 0,0,0,1,0]
elif color == 'J':
    col_list = [0, 0, 0,0,0,0,1]
                                         
st.subheader("Output Diamond Price")


filename = 'diammodel.sav'
loaded_model = joblib.load(filename)

prediction = np.round(loaded_model.predict([[carat, depth, table, x, y,z] + cut_list + col_list])[0],2)
st.write(f"Suggested Diamond Price is: {prediction}")

        
fig1 = sns.catplot(data=diamonddata,
            x='color',
            y='price',
            kind='violin')

st.subheader("Diamond Price Charts")

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
fig2 = sns.scatterplot(x="carat", y="price",
                hue="clarity", size="depth",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=diamonddata, ax=ax)

f2, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f2, left=True, bottom=True)
cut_ranking = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
fig3 = sns.scatterplot(data=diamonddata,
            x="carat", y="price",
            hue="cut", size="depth",
            hue_order=cut_ranking,
            sizes=(1, 5))

plt.setp(fig2.get_legend().get_texts(), fontsize='8') 
plt.setp(fig3.get_legend().get_texts(), fontsize='8') 

st.pyplot(fig1)
st.pyplot(f)
st.pyplot(f2)
