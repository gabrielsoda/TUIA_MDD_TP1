# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: tp1-mineria
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

# %%
# Carga el dataset en un dataframe
df = pd.read_csv('SmartFarmingCropYield.csv')

# Revisa si hay filas duplicadas
df.duplicated().sum() # 0 filas duplicadas

df.head()

# %%
df.info()

# %% [markdown]
# ## Limpieza y preprocesamiento

# %%
df.columns

# %%
df["tipoCultivo"].unique()

# %%
# Elimina espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()

# Revisa los valores únicos en las columnas categóricas
print(df['tipoRiego'].unique())
print(df['tipoFertilizante'].unique())
print(df['estadoEnfermedadesCultivo'].unique())

# %%
# Convierte la categoria 'Moderate' a 'Moderado' para mantener consistencia en el idioma
df['estadoEnfermedadesCultivo'] = df['estadoEnfermedadesCultivo'].replace('Moderate', 'Moderado')

# %%
df.isnull().sum()

# %% [markdown]
# ### Tratamiento de valores faltantes

# %% [markdown]
# #### Variable *estadoEnfermedadesCultivo*

# %%
df['estadoEnfermedadesCultivo'].value_counts()

# %%
df['estadoEnfermedadesCultivo'].isnull().sum()

# %% [markdown]
# Como no necesariamente el cultivo debe estar enfermo, vamos a considerar que los valores faltantes de la variable corresponden a cultivos saludables.

# %%
df['estadoEnfermedadesCultivo'] = df['estadoEnfermedadesCultivo'].fillna('Saludable')

# %% [markdown]
# #### Variable *tipoRiego*

# %%
df["tipoRiego"].value_counts()

# %%
df["tipoRiego"].isnull().sum()

# %%
print(f"El {df["tipoRiego"].isnull().sum() / df.shape[0] * 100:.2f} % de los valores de tipoRiego son nulos")

# %% [markdown]
# Como el porcentaje de valores faltantes de la variable es alto no vamos a imputar para no introducir sesgo por overfitting. Creamos y asignamos la categoría "Desconocido" para identificarlos.

# %%
df["tipoRiego"] = df["tipoRiego"].fillna('Desconocido')

# %% [markdown]
# ## EDA

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sns.countplot(data=df, x="tipoRiego", palette="muted", hue="tipoRiego", ax=axes[0])
sns.countplot(data=df, x="tipoFertilizante", palette="muted", hue="tipoFertilizante", ax=axes[1])
sns.countplot(data=df, x="estadoEnfermedadesCultivo", palette="muted", hue="estadoEnfermedadesCultivo", ax=axes[2])

plt.suptitle("Distribución de variables categóricas")

for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 3, figsize=(16, 12))

sns.histplot(data=df, x="humedadSuelo(%)",bins=35, color=sns.color_palette("muted")[0], ax=axes[0,0], kde=True)
sns.histplot(data=df, x="pHSuelo",bins=35, color=sns.color_palette("muted")[1], ax=axes[0,1], kde=True)
sns.histplot(data=df, x="mlPesticida",bins=35, color=sns.color_palette("muted")[2], ax=axes[0,2], kde=True)
sns.histplot(data=df, x="horasLuzSolar",bins=35, color=sns.color_palette("muted")[3], ax=axes[1,0], kde=True)
sns.histplot(data=df, x="humedad(%)",bins=35, color=sns.color_palette("muted")[4], ax=axes[1,1], kde=True)
sns.histplot(data=df, x="precipitacion(mm)",bins=35, color=sns.color_palette("muted")[5], ax=axes[1,2], kde=True)
sns.histplot(data=df, x="diasTotales",bins=35, color=sns.color_palette("muted")[6], ax=axes[2,0], kde=True)
sns.histplot(data=df, x="temperatura(°C)",bins=35, color=sns.color_palette("muted")[7], ax=axes[2,1], kde=True)
sns.histplot(data=df, x="rendimientoKg_hectarea",bins=35, color=sns.color_palette("muted")[8], ax=axes[2,2], kde=True)
sns.histplot(data=df, x="indiceNDVI", bins=35, color=sns.color_palette("muted")[9], ax=axes[3,0], kde=True)

plt.suptitle("Distribución de variables numéricas")

fig.delaxes(axes[3,1])
fig.delaxes(axes[3,2])
for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 3, figsize=(16, 12))

sns.boxplot(data=df, x="humedadSuelo(%)", palette='dark:#4878d0', hue="tipoCultivo", ax=axes[0,0])
sns.boxplot(data=df, x="pHSuelo", palette='dark:#ee854a', hue="tipoCultivo", ax=axes[0,1])
sns.boxplot(data=df, x="mlPesticida", palette='dark:#6acc64', hue="tipoCultivo", ax=axes[0,2])
sns.boxplot(data=df, x="horasLuzSolar", palette='dark:#d65f5f', hue="tipoCultivo", ax=axes[1,0])
sns.boxplot(data=df, x="humedad(%)", palette='dark:#956cb4', hue="tipoCultivo", ax=axes[1,1])
sns.boxplot(data=df, x="precipitacion(mm)", palette='dark:#8c613c', hue="tipoCultivo", ax=axes[1,2])
sns.boxplot(data=df, x="diasTotales", palette='dark:#dc7ec0', hue="tipoCultivo", ax=axes[2,0])
sns.boxplot(data=df, x="temperatura(°C)", palette='dark:#797979', hue="tipoCultivo", ax=axes[2,1])
sns.boxplot(data=df, x="rendimientoKg_hectarea", palette='dark:#d5bb67', hue="tipoCultivo", ax=axes[2,2])
sns.boxplot(data=df, x="indiceNDVI", palette='dark:#82c6e2', hue="tipoCultivo", ax=axes[3,0])

plt.suptitle("Distribución de variables numéricas")

fig.delaxes(axes[3,1])
fig.delaxes(axes[3,2])
for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# De los gráficos de boxplots no se observan outliers en ninguna variable 

# %%
# Correlación entre variables numéricas
variables_numericas = ["humedadSuelo(%)", "pHSuelo", "mlPesticida", "horasLuzSolar", "humedad(%)", "precipitacion(mm)", "diasTotales", "temperatura(°C)", "rendimientoKg_hectarea", "indiceNDVI"]

plt.figure(figsize=(16, 9))

sns.heatmap(df[variables_numericas].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8}, vmin=-1, vmax=1)

plt.title("Matriz de correlación entre variables numéricas")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 3, figsize=(16, 12))

sns.kdeplot(data=df, x="humedadSuelo(%)", hue="tipoRiego", common_norm=False, alpha=0.7, palette="muted", ax=axes[0,0])
sns.kdeplot(data=df, x="pHSuelo", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[0,1])
sns.kdeplot(data=df, x="mlPesticida", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[0,2])
sns.kdeplot(data=df, x="horasLuzSolar", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[1,0])
sns.kdeplot(data=df, x="humedad(%)", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[1,1])
sns.kdeplot(data=df, x="precipitacion(mm)", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[1,2])
sns.kdeplot(data=df, x="diasTotales", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[2,0])
sns.kdeplot(data=df, x="temperatura(°C)", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[2,1])
sns.kdeplot(data=df, x="rendimientoKg_hectarea", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[2,2])
sns.kdeplot(data=df, x="indiceNDVI", hue="tipoRiego", common_norm=False, alpha=0.5, palette="muted", ax=axes[3,0])

plt.suptitle("Distribución de variables numéricas según tipo de riego")

fig.delaxes(axes[3,1])
fig.delaxes(axes[3,2])
for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# La distribución de las variables númericas para registros con tipo de riesgo desconocido es similar a las distribuciones que presentan los registros con tipos de riego conocidos. 

# %% [markdown]
# ## Generación de variables dummys

# %%
df['riego_goteo'] = (df['tipoRiego'] == 'Goteo').astype(int)
df['riego_aspersor'] = (df['tipoRiego'] == 'Aspersor').astype(int)
df['riego_manual'] = (df['tipoRiego'] == 'Manual').astype(int)

df = df.drop(columns=['tipoRiego'])

# %%
df['fertilizante_mixto'] = (df['tipoFertilizante'] == 'Mixto').astype(int)
df['fertilizante_organico'] = (df['tipoFertilizante'] == 'Organico').astype(int)

df = df.drop(columns=['tipoFertilizante'])

# %%
df['enfermedad_moderada'] = (df['estadoEnfermedadesCultivo'] == 'Moderado').astype(int)
df['enfermedad_severa'] = (df['estadoEnfermedadesCultivo'] == 'Severo').astype(int)
df['enfermedad_leve'] = (df['estadoEnfermedadesCultivo'] == 'Leve').astype(int)

df = df.drop(columns=['estadoEnfermedadesCultivo'])

# %%
df

# %% [markdown]
# ## Estandarización

# %%
scaler = StandardScaler()

df[variables_numericas] = scaler.fit_transform(df[variables_numericas])
# Dataframe sin tipoCultivo (variable target)
x = df.drop(columns=['tipoCultivo'])

# %%
x_num = df[variables_numericas]
y = df['tipoCultivo']

# %% [markdown]
# # PCA
#

# %%
pca = PCA(n_components=x.shape[1]).set_output(transform="pandas")

pca_df = pca.fit_transform(x)

# %%
pd.DataFrame(pca.components_,
             columns = [f'pca{i}' for i in range(x.shape[1])],
             index=x.columns) 


# %%
# Creamos función para acumular la varianza
def acumular(numbers):
     sum = 0
     var_c = []
     for num in numbers:
        sum += num
        var_c.append(sum)
     return var_c


# %%
var_c = acumular(pca.explained_variance_ratio_)
pca_rtd = pd.DataFrame({'Eigenvalues':pca.explained_variance_,
                        'Proporción de variancia explicada':pca.explained_variance_ratio_,
                        'Proporción acumulado de variancia explicada': var_c})
pca_rtd

# %%
n_components = np.arange(pca.n_components) + 1
plt.figure(figsize=(16, 9))
plt.bar(range(1,pca.n_components + 1), pca.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(1, pca.n_components + 1), np.cumsum(pca.explained_variance_ratio_),
         where='mid',
         color='red')
plt.title('Varianza acumulada')
plt.ylabel('Proporción de variancia explicada')
plt.xlabel('Componente principales')

plt.xticks(n_components)
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Explicada Acumulada por PCA')
plt.grid(True)
plt.show()

# %%

figure = plt.figure(figsize=(16, 9))

PC_values = np.arange(pca.n_components_) + 1

sns.lineplot(x=PC_values, y=pca.explained_variance_ratio_, marker='o')

plt.title('Gráfico de codo')
plt.xlabel('Componentes principales')
plt.ylabel('Proporción de variancia explicada')

plt.xticks(n_components)
plt.show()

# %%
fig = plt.figure(figsize=(16, 9))

ax = sns.heatmap(
    pca_df.corr(),
    vmin=-1, vmax=1, center=0,
    cmap="vlag",
    annot = True,
    square= True,
    annot_kws = {'size': 11},
)

plt.tight_layout()
plt.show()

# %%
import plotly.express as px

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig = px.scatter(pca_df, x='pca0', y='pca1', color=y,
                 title="Distribución de los cultivos en 2 dimensiones")
fig.update_layout(width=1600, height=900)    
fig.show()

fig = px.scatter_3d(pca_df, x='pca0', y='pca1', z='pca2',
              color=y,
              title="Distribución de los cultivos en 3 dimensiones")
fig.update_layout(width=1600, height=900)   
fig.show()

# %% [markdown]
# ### Criterio de Kaiser
#

# %%
eigenvalues = pca.explained_variance_
kaiser_components = np.sum(eigenvalues > 1)

print(f"Número de componentes principales según criterio de Kaiser: {kaiser_components}")


# %% [markdown]
# # ISOMAP
#
#

# %%
# Compute residual variances for different dimensions
residual_variances = []
max_d = 14  # Adjust this to the maximum number of dimensions you want to test

for d in range(1, max_d + 1):
    iso = Isomap(n_neighbors=3, n_components=d)
    embedding = iso.fit_transform(x)
    
    # Geodesic distances from Isomap
    d_geo = iso.dist_matrix_
    
    # Euclidean distances in the embedding space
    d_emb = pairwise_distances(embedding)
    
    # Correlation between distances
    cor = np.corrcoef(d_emb.ravel(), d_geo.ravel())[0, 1]
    
    # Residual variance
    rv = 1 - cor ** 2
    residual_variances.append(rv)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_d + 1), residual_variances, marker='o', linestyle='--')
plt.xlabel('Número de Dimensiones')
plt.ylabel('Varianza Residual')
plt.title('Gráfico de Codo para Isomap')
plt.grid(True)
plt.show()

# %%
# Valores de vecinos a probar
vecinos_a_probar = [6, 7, 8, 9, 10]

for k in vecinos_a_probar:
    # Aplicar Isomap con 8 componentes y k vecinos
    isomap = Isomap(n_neighbors=k, n_components=8)

    isomap_df = pd.DataFrame(isomap.fit_transform(x), columns=['DIM1', 'DIM2', 'DIM3', 'DIM4', 'DIM5', 'DIM6', 'DIM7', 'DIM8'])
    isomap_df['Tipo de cultivo'] = y  # agregar las etiquetas

    # Graficar con plotly
    fig_isomap_2d = px.scatter(isomap_df, x='DIM1', y='DIM2', color = y,
                           labels = {'color':'Tipo de cultivo'},
                           title='ISOMAP 2 Componentes y ' + str(k) +" vecinos")


    fig_isomap_2d.update_layout(width=1000, height=1000)
    fig_isomap_2d.show()

# %% [markdown]
# Si bien ninguna combinación de hiperparámetro coincide con una separación de tipo de cultivo, podemos visualizar agrupaciones más marcadas con 6 dimensiones.

# %% [markdown]
# # t-SNE
#

# %%
tsne = TSNE(n_components=3, init='random', method='exact', random_state=42, perplexity=20)

tsne_df = pd.DataFrame(tsne.fit_transform(x), columns=['DIM1', 'DIM2', 'DIM3'])

# %%
fig = plt.figure(figsize=(16, 9))

ax = sns.heatmap(
    tsne_df.corr(),
    vmin=-1, vmax=1, center=0,
    cmap="vlag",
    annot = True,
    square= True,
    annot_kws = {'size': 11},
)

plt.tight_layout()
plt.show()

# %%
fig_tsne_2d = px.scatter(tsne_df, x='DIM1', y='DIM2', color = y,
                    labels = {'color':'tipo de cultivo'},
                    title='t-SNE 2 Componentes')

fig_tsne_2d.update_layout(width=1600, height=900)
fig_tsne_2d.show()

# %%
fig_tsne_3d = px.scatter_3d(tsne_df, x='DIM1', y='DIM2', z='DIM3', color = y,
                    labels = {'color':'tipo de cultivo'},
                    title='TSNE 3 Componentes')

fig_tsne_3d.update_layout(width=1600, height=900)
fig_tsne_3d.show()

# %% [markdown]
# Luego de intentar con distintos valores de perplexidad, encontramos que en 20 se separan más los datos. 

# %% [markdown]
# # K-Means

# %%
cols = [f'pca{i}' for i in range(0, 8)]  # ['pca1', 'pca2', ..., 'pca8']
pca_8 = pca_df[cols]
kmeans_df = pca_8.copy()

# %% [markdown]
# Se utiliza la reducción de dimensiones obtenidas con PCA

# %%
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=k, random_state=42).fit(kmeans_df) for k in Nc]

# La suma de residuos cuadrados intra grupos de kMeans en sklearn se guarda en
# el atributo inertia
inertias = [model.inertia_ for model in kmeans]

plt.figure(figsize=(8, 3.5))
plt.plot(Nc,inertias, "bo-")
plt.xlabel('Número de Clusters')
plt.ylabel('RSS dentro de los grupos')
plt.title('Gráfica de codo')
plt.grid()
plt.show()


# %% [markdown]
# No hay un codo claro; la curva es suave y casi lineal, con una disminución inicial más rápida y luego más lenta, a partir del 9 clúster aproximadamente.

# %% [markdown]
# Elección de cantidad de clústeres con GAP Statistic

# %%
def calculate_intra_cluster_dispersion(X, k):
    kmeans = KMeans(n_clusters=k) # Aplica K-means
    kmeans.fit(X)
    return kmeans.inertia_ # Retorna la suma de distancias al centroide


# %%
gaps = []
max_k = 15

# Calcula el Gap Statistic para determinar el número óptimo de clusters

for k in range(5, max_k + 1):
    # Calculo la inercia real sobre mis datos reales
    real_inertia = calculate_intra_cluster_dispersion(kmeans_df, k)
    #Calculo la inercia de datos aleatorios con la misma estructura que mis datos originales
    inertia_list = []
    for _ in range(200):
      random_data = np.random.rand(*x.shape)
      intra_cluster_dispersion = calculate_intra_cluster_dispersion(random_data, k)
      inertia_list.append(intra_cluster_dispersion)

    reference_inertia = np.mean(inertia_list)

    #Aplico la funcion de gap
    gap = np.log(reference_inertia) - np.log(real_inertia)
    gaps.append(gap)

#se selecciona el valor de k (número de clusters) que maximiza el Gap Statistic.
optimal_k = np.argmax(gaps) + 1


# %%
print("Número óptimo de clusters según el Gap Statistic:", optimal_k)

plt.figure(figsize=(16, 9))
plt.plot(range(5, max_k + 1), gaps, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic para determinar el número de clusters')
plt.grid()
plt.show()

# %%
kmeans = KMeans(n_clusters=11, random_state=42,
                init='k-means++', n_init=5, algorithm='lloyd')
kmeans.fit(kmeans_df) #Entrenamos el modelo
y_pred = kmeans.predict(kmeans_df)

# %%
np.array_equal(y_pred, kmeans.labels_)

# %%
# El metodo labels_ nos da a que cluster corresponde cada observacion
kmeans_df['Etiquetas KMeans'] = kmeans.labels_
kmeans_df['Etiquetas KMeans'] = kmeans_df['Etiquetas KMeans'].astype('category')
kmeans_df.head()

# %%
np.set_printoptions(precision=6)
kmeans.cluster_centers_
# caracteristicas normalizadas que tendria el centroide de ese cluster.

# %%
fig = px.scatter_3d(kmeans_df, x='pca0', y='pca1', z='pca2',
                    color='Etiquetas KMeans',
                    title='Dispersión de los tipos de cultivo (K-means)')
fig.show()

# %% [markdown]
# No encontramos diferenciación marcada entre los clústeres generados.

# %% [markdown]
# 3 clústeres:

# %%
kmeans = KMeans(n_clusters=3, random_state=42,
                init='k-means++', n_init=5, algorithm='lloyd')
kmeans.fit(kmeans_df) #Entrenamos el modelo
y_pred = kmeans.predict(kmeans_df)

# %%
np.array_equal(y_pred, kmeans.labels_)

# %%
# El metodo labels_ nos da a que cluster corresponde cada observacion
kmeans_df['Etiquetas KMeans'] = kmeans.labels_
kmeans_df['Etiquetas KMeans'] = kmeans_df['Etiquetas KMeans'].astype('category')
kmeans_df.head()

# %%
np.set_printoptions(precision=6)
kmeans.cluster_centers_
# caracteristicas normalizadas que tendria el centroide de ese cluster.

# %%
fig = px.scatter_3d(kmeans_df, x='pca0', y='pca1', z='pca2',
                    color='Etiquetas KMeans',
                    title='Dispersión de los tipos de cultivo (K-means)')
fig.show()

# %%
fig = px.scatter_3d(kmeans_df,  x='pca0', y='pca1', z='pca2',
                    color=y,
                    title='Dispersión de los tipos de cultivo (original)')
fig.show()

# %% [markdown]
# No se visualiza correspondencia con los clústeres generados y los tipos de cultivos.

# %% [markdown]
# # Clustering jerárquico

# %%
# Importa la función linkage de SciPy para realizar el clustering jerárquico
Z = linkage(x, "ward")

# %%
# Dibuja un dendrograma para visualizar el resultado del clustering jerárquico
plt.figure(figsize=(26, 9)) 
dendrogram(Z)               
plt.show()


# %%
# Dibuja un dendrograma truncado (solo muestra las últimas p hojas)
plt.figure(figsize=(16, 9)) 
dendrogram(Z,  truncate_mode = 'lastp', p = 10, show_leaf_counts = True,
           show_contracted = True)
plt.axhline(y=110, c='k', linestyle='dashed')
plt.xlabel("Numero de puntos en el nodo")
plt.show()

# %%
from scipy.spatial.distance import cdist

distancias=[]
for i in range(1, 10):
    clustering = AgglomerativeClustering(n_clusters=i) # Aplica clustering jerárquico con i clusters
    clustering.fit(x)

    # Calcula la matriz de distancias por pares entre los puntos
    pairwise_distances = cdist(x, x, 'euclidean')

    # Calcula la distancia total dentro de los clusters
    distancia_total = 0
    for j in range(i):
        cluster_indices = np.where(clustering.labels_ == j)
        # Encuentra los índices de los puntos en el cluster j
        distancia_total += pairwise_distances[cluster_indices][:, cluster_indices].sum()
        # Suma las distancias dentro del cluster


    distancias.append(distancia_total)
    # Almacena la distancia total para el número de clusters i

# %%
# Grafica la distancia total en función del número de clusters
plt.plot(range(1, 10), distancias, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Distancia Total')
plt.title('Método del Codo para Clustering Jerárquico')
plt.grid()
plt.show()

# %%
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters)

cluster_assignments = clustering.fit_predict(x) # Asigna los clusters a los datos

x['Etiquetas jerarquico'] = cluster_assignments # Añade la columna con el cluster asignado a cada punto

x.head()

# %%
fig = plt.figure(figsize=(10, 6))
ax = Axes3D(fig, auto_add_to_figure=False) # Crea un gráfico 3D
fig.add_axes(ax)
labels = np.unique(x['Etiquetas jerarquico']) # Obtiene los clusters únicos
palette = sns.color_palette("husl", len(labels)) # Define una paleta de colores
for label, color in zip(labels, palette):
  df1 = x[x['Etiquetas jerarquico'] == label]
  ax.scatter(df1['pHSuelo'], df1['diasTotales'], df1['indiceNDVI'],
             s=40, marker='o', color=color, alpha=1, label=label)
# x='pHSuelo', y='diasTotales', z='indiceNDVI'

# %% [markdown]
# ### Validación clústering jerárquico

# %% [markdown]
# #### GAP STATISTIC

# %%
def calculate_intra_cluster_dispersion(X_scaled, k, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = clustering.fit_predict(X_scaled)

    # Calcula los centroides de los clústeres como la media de los puntos dentro de cada clúster
    centroids = np.array([np.mean(X_scaled[labels == i], axis=0) for i in range(k)])

    # Calcula la dispersión intraclúster sumando las distancias al cuadrado entre los puntos y sus centroides
    # np.linalg.norm calcula la norma (distancia euclidiana) entre los puntos y el centroide correspondiente
    intra_cluster_dispersion = np.sum(np.linalg.norm(X_scaled[labels] - centroids[labels], axis=1)**2)
    return intra_cluster_dispersion



# %%
def calculate_intra_cluster_dispersion(X, k, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = clustering.fit_predict(X)

    # Convert X to a NumPy array for integer-based indexing
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X

    # Calcula los centroides de los clústeres como la media de los puntos dentro de cada clúster
    centroids = np.array([np.mean(X_np[labels == i], axis=0) for i in range(k)])

    # Calcula la dispersión intraclúster sumando las distancias al cuadrado entre los puntos y sus centroides
    # np.linalg.norm calcula la norma (distancia euclidiana) entre los puntos y el centroide correspondiente
    intra_cluster_dispersion = np.sum(np.linalg.norm(X_np[labels] - centroids[labels], axis=1)**2)
    return intra_cluster_dispersion

gaps = []
max_k = 10
for k in range(1, max_k + 1):
    # Calcula la dispersión intraclúster en los datos reales para 'k' clústeres
    X_std_numerical = x.select_dtypes(include=np.number)
    real_inertia = calculate_intra_cluster_dispersion(X_std_numerical, k, linkage='ward')

    inertia_list = []
    for _ in range(10):
      random_data = np.random.rand(*X_std_numerical.shape)
      intra_cluster_dispersion = calculate_intra_cluster_dispersion(random_data, k)
      inertia_list.append(intra_cluster_dispersion)

    reference_inertia = np.mean(inertia_list)

    gap = np.log(reference_inertia) - np.log(real_inertia)
    gaps.append(gap)

optimal_k = np.argmax(gaps) + 1

# %%
print("Número seleccionado de clusters según el Gap Statistic:", optimal_k)

plt.figure(figsize=(16, 9))
plt.plot(range(1, max_k + 1), gaps, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic para determinar el número óptimo de clusters (Clustering Jerárquico)')
plt.grid()
plt.show()

# %% [markdown]
# #### Coeficiente de Silhouette

# %%
# Calcula el coeficiente de Silhouette promedio
silhouette_score(x, cluster_assignments)


# %%
def calculate_silhouette(X_scaled, k, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = clustering.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    return silhouette_avg, sample_silhouette_values

max_k = 15

silhouette_scores = []
for k in range(2, max_k + 1):
    silhouette_avg, _ = calculate_silhouette(x, k)
    silhouette_scores.append(silhouette_avg)

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silhouette')
plt.title('Coeficiente de Silhouette para determinar el número óptimo de clusters (Clustering Jerárquico)')
plt.show()

# %% [markdown]
# # Conclusión
#
# Tras aplicar los métodos de reducción de dimensionalidad (PCA, Isomap, t-SNE) y técnicas de clustering (k-means y clustering jerárquico) a un conjunto de datos con 311 muestras y 14 variables relacionadas con cultivos (como humedad del suelo, pH, temperatura, precipitación, entre otras), no se logró obtener visualizaciones que reflejaran una separación clara de los clústeres correspondientes a los tipos de cultivo (Trigo, Soja, Maíz). Este resultado puede atribuirse a varios factores fundamentales:
#
# 1. **Volumen limitado de datos**: El conjunto de datos, con solo 311 muestras, probablemente no proporciona suficiente información para que los algoritmos identifiquen patrones robustos que diferencien los cultivos. La alta dimensionalidad inicial (14 variables) agrava esta limitación, ya que los métodos de reducción y clustering requieren un mayor número de observaciones para capturar estructuras significativas.
#
# 2. **Ausencia de correlaciones relevantes**: El análisis de correlaciones mostró un valor máximo de -0.20 (ph del suelo con ml. de pesticidas), lo que indica que ninguna de las variables tiene una relación fuerte con la variable objetivo (`tipoCultivo`) ni entre sí. Esta falta de correlaciones relevantes sugiere que las variables incluidas en el conjunto de datos no son suficientemente discriminantes para distinguir entre los tipos de cultivo, lo que dificulta tanto la reducción de dimensionalidad como la formación de clústeres coherentes.
#
# 3. **Variables ruidosas o irrelevantes**: El preprocesamiento aplicado para codificar variables categóricas (como one-hot encoding) afectó la calidad de los datos y los resultados de los métodos aplicados. El dataset contaba con demasiadas dimensioens y además se sumaron las variables categóricas que fueron transformadas en dummies, que terminaron generando un total inicial de 17 dimensiones, de las cuales 9 tenían valores binarios.
#
# 4. **Métodos utilizados**: Los métodos de reducción de dimensionalidad empleados tienen supuestos que podrían no alinearse con la estructura de los datos. En PCA, por ejemplo, la baja correlación lineal (máximo -0.20) podría explicar la necesidad de generar muchas componentes principales para poder explicar un porcentaje suficiente de varianza. En cambio, ISOMAP y t-SNE, aunque diseñados para capturar estructuras no lineales, podrían no haber sido efectivos debido al pequeño tamaño del conjunto de datos, incluso habiendo intentado con distintas combinaciones de hiperparámetros. De manera similar, k-means y el clustering jerárquico podrían no ser ideales si los datos no presentan una estructura clara de agrupamiento.
#
# 5. **Solapamiento entre clases**: La falta de correlaciones relevantes y la incapacidad de los métodos para formar clusters significativos sugieren un alto grado de solapamiento entre las clases de cultivos en el espacio de características. Esto podría indicar que las variables medidas no capturan diferencias agronómicas o biológicas clave entre Trigo, Soja y Maíz, posiblemente debido a condiciones ambientales similares.
#
#
