import streamlit as st
import pandas as pd

from utils import load_model,make_prediction


st.title("Iris Flower Classification")
st.write("This is a simple iris flower classification app")




cols=st.columns(4)

features=["sepal_length","sepal_width","petal_length","petal_width"]

features_values=[]

for i,col in enumerate(cols):
    with col:
        st.write("Input for "+features[i])
        features_values.append(st.number_input(features[i],min_value=0.,max_value=10.,step=0.5))

features_values=pd.DataFrame([features_values],columns=features)


model=load_model("xgb_best_model.pkl")

predict=st.button("Give me a prediction")

if predict:
    prediction=make_prediction(model,features_values)
    st.write(f"The prediction is: {prediction.item()}")
    










