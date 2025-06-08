import streamlit as st
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from PIL import Image
import cv2

from image_feature_extractor import ImageFeatureExtractor
from plot_processor import PlotProcessor
from clustering import AgglomerativeManual
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from dataset.manager import ds

class StreamlitHandler:
    def main(self):
        st.title("Visualizador de carácteristicas visuales de posters de películas")

        # Sidebar para cargar o generar datos
        st.sidebar.header("Opciones del modelo")

        modo = st.radio("Selecciona fuente de input", ["Seleccionar poster", "Subir imagen"])

        if modo == "Seleccionar poster":
            (imagen_pil, imagen_nombre) = self.seleccionar_poster()
        else:
            (imagen_pil, imagen_nombre) = self.subir_imagen()
        
        reduction_method = st.sidebar.radio("Método de reducción de dimensionalidad", ["pca", "svd", "lda", "pca+lda"])
        cluster_model = st.sidebar.radio("Modelo de clustering", ["kmeans", "agglo"])
        num_clusters = st.sidebar.number_input("Número de clusters", min_value=1, step=1, value=4)
        subspace = st.sidebar.radio("Subespacio", ["full", "vt", "kbest"])

        st.sidebar.header("Imagen elegida")

        if imagen_pil is None:
            medoid_indices = self.dibujar_plot(None, None, reduction_method, cluster_model, num_clusters, subspace)
        else:
            # Mostrar la imagen en la app
            st.sidebar.image(imagen_pil, caption="Imagen", use_container_width=True)

            # Convertir a formato OpenCV (numpy array con BGR)
            imagen_np = np.array(imagen_pil)
            imagen_cv2 = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
            imagen_cv2 = cv2.resize(imagen_cv2, (128, 192))

            imagen_features = ImageFeatureExtractor.extract_all_descriptors(imagen_cv2)
            medoid_indices = self.dibujar_plot(imagen_features, imagen_nombre, reduction_method, cluster_model, num_clusters, subspace)

        self.dibujar_representantes(medoid_indices)

    def seleccionar_poster(self) -> (Image, str):
        rows = ds.test_initial_df[["movieId", "title"]]
        rows = rows[rows["movieId"].isin(ds.test_df["movieId"].tolist())]
        opciones = [f"{rows.iloc[i]["movieId"]}: {rows.iloc[i]["title"]}" for i in range(len(rows))]

        pelicula_elegida = st.selectbox(
            "Selecciona una pelicula del testing dataset:",
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
    
    def dibujar_plot(self, query_point, query_name, reduction_method="pca", cluster_model="kmeans", num_clusters=4, subspace="full"):
        # points metadata
        train_rows = ds.train_initial_df[["movieId", "title"]]
        train_ids = np.array([
            f"{train_rows.iloc[i]["movieId"]}: {train_rows.iloc[i]["title"]}"
            for i in range(len(train_rows))
            if train_rows.iloc[i]["movieId"] in ds.get_all_ids()
        ])

        # Etiquetas de género
        local_train_df = ds.train_initial_df.copy()
        local_train_df = local_train_df[local_train_df["movieId"].isin(ds.get_all_ids())]
        local_train_df["primary_genre"] = local_train_df["genres"].apply(lambda g: g.split("|")[0] if isinstance(g, str) else None)
        genre_map = local_train_df.set_index("movieId")["primary_genre"].to_dict()
        y_train = ds.train_df["movieId"].map(genre_map).values

        # model
        transformer = PlotProcessor(
            reduction_method=reduction_method,
            cluster_model=cluster_model,
            num_clusters=num_clusters,
            subspace=subspace,
            y_train=y_train
        )

        # transforming
        X_train_2d = transformer.fit_transform(ds.X_train)
        X_train_full, train_labels = transformer.fit_predict(ds.X_train)
        train_x = X_train_2d[:, 0]
        train_y = X_train_2d[:, 1]

        qual_colors = qualitative.Plotly  # Similar to tab10
        cluster_colors = [qual_colors[i % len(qual_colors)] for i in range(num_clusters)]
        
        X_query_full = None
        if query_point is not None:
            query_point_reshaped = query_point.reshape(1, -1)

            # full projection
            X_query_full = transformer.full_transform(query_point_reshaped)

            # 2d projection
            X_query_2d = transformer.transform(query_point_reshaped)
            query_2d_x = X_query_2d[:, 0]
            query_2d_y = X_query_2d[:, 1]        

        if X_query_full is not None:
            self.generar_recomendaciones(X_query_full, X_train_full)
        else:
            st.subheader("Peliculas recomendadas:")
            st.write("Elige una pelicula para generar recomendaciones!")

        medoid_indices = self.generar_medoides(X_train_full, train_labels)

        fig = go.Figure()

        for cluster_id in range(num_clusters):
            mask = (train_labels == cluster_id)
            cluster_indices = np.where(mask)[0]
            medoid_idx = medoid_indices[cluster_id]

            # Exclude medoid from base cluster points
            base_cluster_indices = [i for i in cluster_indices if i != medoid_idx]

            fig.add_trace(go.Scatter(
                name=f'Cluster {cluster_id}',
                x=train_x[base_cluster_indices],
                y=train_y[base_cluster_indices],
                mode='markers',
                text=train_ids[base_cluster_indices],
                marker=dict(
                    size=5,
                    color=cluster_colors[cluster_id],
                    opacity=0.5
                )
            ))

        for cluster_id in range(num_clusters):
            mask = (train_labels == cluster_id)
            cluster_indices = np.where(mask)[0]
            medoid_idx = medoid_indices[cluster_id]
            base_cluster_indices = [i for i in cluster_indices if i != medoid_idx]
            # Draw medoid separately
            fig.add_trace(go.Scatter(
                x=[train_x[medoid_idx]],
                y=[train_y[medoid_idx]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=cluster_colors[cluster_id],
                    opacity=1.0,
                    line=dict(width=2, color='black')
                ),
                name=f"Medoid {cluster_id}",
                text=train_ids[medoid_indices[cluster_id]],
                showlegend=True
            ))

        if query_point is not None:
            fig.add_trace(go.Scatter(
                name='Query point',
                text=query_name,
                x=query_2d_x, y=query_2d_y, mode='markers',
                marker=dict(
                    size=13,
                    color='red',
                    line=dict(
                        color='black', # Outline color
                        width=2 # Thickness of the outline
                    ),
                    opacity=1
                )
            ))

        fig.update_layout(
            title=transformer.create_title(),
            xaxis_title='Componente 1',
            yaxis_title='Componente 2'
        )
        st.subheader("Visualizacion de las peliculas proyectadas en espacio 2D")
        st.plotly_chart(fig, use_container_width=True)
        return medoid_indices

    def generar_recomendaciones(self, X_query, X_train):
        k = 10
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_query.reshape(1, -1))
        indices = indices[0]

        # Get the recommended movieIds in order
        movie_ids = np.array([int(ds.train_df.iloc[i]["movieId"]) for i in indices])

        # Filter movie info and preserve order
        movie_df_all = ds.train_initial_df.set_index("movieId")
        movie_df = movie_df_all.loc[movie_ids].reset_index()

        # Get array of (movieId, title) pairs in the correct order
        recomendaciones = movie_df[["movieId", "title"]].to_numpy()

        st.subheader("Peliculas recomendadas:")
        st.write("Recomendaciones generadas por proximidada en el subespacio")
        cols1 = st.columns(k//2)
        cols2 = st.columns(k//2)

        for index, recomendacion in enumerate(recomendaciones):
            movie_id = recomendacion[0]
            movie_title = recomendacion[1]
            image = ds.get_poster_image(movie_id)
            if index < k//2:
                with cols1[index]:
                    c1 = st.container(height=30, border=False)
                    c2 = st.container(height=200, border=False)
                    c1.markdown(
                        f"""
                        <div style="
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            max-width: 200px;  /* adjust width as needed */ 
                        " title="{movie_title}">
                        {index+1}: {movie_title}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    c2.image(image, use_container_width=True)
            else:
                with cols2[index-k//2]:
                    c1 = st.container(height=30, border=False)
                    c2 = st.container(height=200, border=False)
                    c1.markdown(
                        f"""
                        <div style="
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            max-width: 200px;  /* adjust width as needed */ 
                        " title="{movie_title}">
                        {index+1}: {movie_title}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    c2.image(image, use_container_width=True)
    
    def generar_medoides(self, X_train, labels):
        unique_labels = np.unique(labels)
        medoid_indices = []

        for label in unique_labels:
            # Get indices of points in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_points = X_train[cluster_indices]

            # Compute pairwise distances within the cluster
            distances = pairwise_distances(cluster_points, metric='euclidean')
            total_distances = distances.sum(axis=1)

            # Find the point with minimum total distance (medoid)
            medoid_local_index = np.argmin(total_distances)
            medoid_global_index = cluster_indices[medoid_local_index]

            medoid_indices.append(medoid_global_index)

        return np.array(medoid_indices)
    
    def dibujar_representantes(self, medoid_indices):
        movie_ids = np.array([int(ds.train_df.iloc[i]["movieId"]) for i in medoid_indices])

        # Filter movie info and preserve order
        movie_df_all = ds.train_initial_df.set_index("movieId")
        movie_df = movie_df_all.loc[movie_ids].reset_index()

        # Get array of (movieId, title) pairs in the correct order
        movie_medoids = movie_df[["movieId", "title"]].to_numpy()

        st.subheader("Representantes de cada cluster")
        st.write("Representantes obtenidos mediante la extraccion de medoides de cada cluster")

        cols = st.columns(len(medoid_indices))
        for index, medoid in enumerate(movie_medoids):
            movie_id = medoid[0]
            movie_title = medoid[1]
            image = ds.get_poster_image(movie_id)
            with cols[index]:
                st.image(image, caption=f"Cluster {index}: {movie_title}", use_container_width=True)