import streamlit as st
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2

from image_feature_extractor import ImageFeatureExtractor
from plot_processor import transformer
from dataset.manager import ds

class StreamlitHandler:
    def main(self):
        st.title("Visualizador de carácteristicas visuales de posters de películas")

        # Sidebar para cargar o generar datos
        st.sidebar.header("Imagen elegida")

        modo = st.radio("Selecciona fuente de input", ["Seleccionar poster", "Subir imagen"])

        if modo == "Seleccionar poster":
            (imagen_pil, imagen_nombre) = self.seleccionar_poster()
        else:
            (imagen_pil, imagen_nombre) = self.subir_imagen()
        
        if imagen_pil is None:
            self.dibujar_plot(None, None)
        else:
            # Mostrar la imagen en la app
            st.sidebar.image(imagen_pil, caption="Imagen", use_container_width=True)

            # Convertir a formato OpenCV (numpy array con BGR)
            imagen_np = np.array(imagen_pil)
            imagen_cv2 = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
            imagen_cv2 = cv2.resize(imagen_cv2, (128, 192))

            imagen_features = ImageFeatureExtractor.extract_all_descriptors(imagen_cv2)
            imagen_features = imagen_features[:562]
            self.dibujar_plot(imagen_features, imagen_nombre)

    def seleccionar_poster(self) -> (Image, str):
        rows = ds.test_initial_df[["movieId", "title"]]
        rows = rows[rows["movieId"].isin(ds.test_df["movieId"].tolist())]
        opciones = [f"{rows.iloc[i]["movieId"]}: {rows.iloc[i]["title"]}" for i in range(len(rows))]

        pelicula_elegida = st.selectbox(
            "Selecciona uno o más tipos de contenido:",
            opciones
        )

        pelicula_split = pelicula_elegida.split(":")
        pelicula_id = pelicula_split[0]
        title = ":".join(pelicula_split[1:])

        imagen = ds.get_poster_image(pelicula_id)

        if imagen is None:
            st.error("Ocurrio un error al buscar la imagen")
            return (None, None)

        return (imagen, title)

    def subir_imagen(self) -> (Image, str):
        archivo = st.file_uploader("Sube tu poster", type=["png", "jpg", "bmp"])
            
        if archivo is None:
            st.warning("Por favor sube una imagen para continuar")
            return (None, None)
        
        imagen = Image.open(archivo).convert("RGB")
        return (imagen, archivo.name)
    
    def dibujar_plot(self, query_point, query_name):
        X_train_2d = transformer.fit_transform(ds.X_train)
        train_rows = ds.train_initial_df[["movieId", "title"]]
        train_ids = [f"{train_rows.iloc[i]["movieId"]}: {train_rows.iloc[i]["title"]}" for i in range(len(train_rows))]
        
        x = X_train_2d[:, 0]
        y = X_train_2d[:, 1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            name='Train data',
            x=x, y=y, text=train_ids, mode='markers',
            marker=dict(size=6, color='cyan', opacity=0.5)
        ))
        
        if query_point is not None:
            X_query = transformer.transform(query_point.reshape(1, -1))
            query_x = X_query[:, 0]
            query_y = X_query[:, 1]

            fig.add_trace(go.Scatter(
                name='Query point',
                text=query_name,
                x=query_x, y=query_y, mode='markers',
                marker=dict(size=6, color='red', opacity=1)
            ))

        fig.update_layout(
            title='Scatter Plot desde numpy.ndarray',
            xaxis_title='PCA 1',
            yaxis_title='PCA 2'
        )

        st.plotly_chart(fig, use_container_width=True)