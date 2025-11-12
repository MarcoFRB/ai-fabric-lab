import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo (asegúrate de que 'modelo.pkl' esté en la misma carpeta)
try:
    with open('modelo.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Archivo 'modelo.pkl' no encontrado. Asegúrate de que esté en el repositorio.")
    st.stop()

# Título y subtítulo
st.title("🤖Predictor Grupo #5, Estrategia de BA")
st.markdown("Esta app demuestra cómo el 'AI Factory' (Hugging Face) despliega un modelo de ML con una UI de Streamlit, todo disparado por GitOps.")

# --- UI de Entrada (Sliders) ---
st.sidebar.header("Introduce las características de la flor:")

def user_inputs():
    sepal_length = st.sidebar.slider('Largo del Sépalo (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Ancho del Sépalo (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Largo del Pétalo (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Ancho del Pétalo (cm)', 0.1, 2.5, 0.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_inputs()

# Mostrar las entradas del usuario
st.subheader('Características seleccionadas:')
st.dataframe(input_df, use_container_width=True)

# --- Predicción y Salida ---
if st.sidebar.button('¡Predecir tipo de Iris!'):
    # Convertir el dataframe a un array numpy para el modelo
    features_array = np.array(input_df)
    
    # Hacer la predicción
    prediction = model.predict(features_array)
    prediction_proba = model.predict_proba(features_array)
    
    # Mapear el resultado
    iris_map = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
    species = iris_map[prediction[0]]
    
    # Mostrar el resultado
    st.subheader('Resultado de la Predicción')
    st.success(f'La flor es una **{species}**.')
    
    # Mostrar confianza (probabilidades)
    st.subheader('Confianza de la Predicción')
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    proba_df = proba_df.rename(columns=iris_map).T
    proba_df.columns = ['Probabilidad']
    st.bar_chart(proba_df)
