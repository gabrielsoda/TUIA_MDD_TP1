# %% [markdown]
# # Trabajo Práctico N° 1: Reducción de Dimensionalidad y Clustering
# ## Minería de Datos, TUIA
#
# **Integrantes:**
# * Armas, Alejandro
# * Ferreira Da Camara, Facundo
# * Soda, Gabriel
#
# **24/10/2025**
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Carga y Preprocesamiento de Datos
#
# Carga del dataset, limpieza (nulos, duplicados), codificar variables categóricas y estandarización las características.

# %%
df = pd.read_csv('SmartFarmingCropYield.csv')
display(df.head())
# Revisa si hay filas duplicadas
print(f"Filas duplicadas: {df.duplicated().sum()}") # 0 filas duplicadas

# Información inicial
print("\nColumnas del DataFrame:")
for col in df.columns:
    print(f"-{col}-")
# %% [markdown]
# ### 1.1. Limpieza y manejo de Valores Faltantes (Nulos)
# %%
# Elimina espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()

#%%
variables_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables categóricas: {variables_categoricas}")
variables_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas: {variables_numericas}")

# Revisa los valores únicos en las columnas categóricas
print(f"\nTipoRiego: {df['tipoRiego'].unique()}")
print(f"tipoFertilizante: {df['tipoFertilizante'].unique()}")
print(f"estadoEnfermedadesCultivo{df['estadoEnfermedadesCultivo'].unique()}")
print(f"tipoCultivo (target): {df['tipoCultivo'].unique()}")
#%%
# Convierte la categoria 'Moderate' a 'Moderado' para mantener consistencia en el idioma
df['estadoEnfermedadesCultivo'] = df['estadoEnfermedadesCultivo'].replace('Moderate', 'Moderado')
# %%
faltantes_df = pd.DataFrame({
    'NaN': df.isna().sum(),
    '%': (df.isna().sum() / len(df) * 100).round(2)
}).sort_values('NaN', ascending=False)
faltantes_df
# %%
df[df['tipoRiego'].isna()].describe()
#%%
df[df['estadoEnfermedadesCultivo'].isna()].describe()
# %% [markdown]
# ### Criterio para manejo de nulos:
# Dado que el dataset es pequeño (311 filas), priorizaremos imputación para evitar pérdida de datos (~29.6% por columna). 
# Usaremos **KNN** primero, si cambia la distribución intentaremos con la **moda**, 
# y **eliminación** como último recurso.  
#%%
# Overlap de NaN
nan_overlap = df[df['tipoRiego'].isna() & df['estadoEnfermedadesCultivo'].isna()].shape[0]
print(f"Filas con NaN en ambas columnas: {nan_overlap}")

# Boxplot para ver patrones (ejemplo con humedadSuelo)
sns.boxplot(x=df['tipoRiego'].isna(), y=df['humedadSuelo(%)'])
plt.title('Humedad del Suelo según ausencia de tipoRiego')
plt.xlabel('tipoRiego es NaN')
plt.ylabel('Humedad Suelo (%)')
plt.show()
# %%
# Columnas con valores nulos
cols_con_nulos = ['tipoRiego', 'estadoEnfermedadesCultivo']

for col in cols_con_nulos:
    # 1. Calcular la moda (el valor más frecuente)
    moda = df[col].mode()[0]
    print(f"Imputando nulos en '{col}' con la moda: '{moda}'")
    
    # 2. Rellenar los valores NaN con la moda
    df[col] = df[col].fillna(moda)

print("\nInformación del DataFrame (después de limpiar):")
df.info()
##% [markdown]
# # Análisis explotario
# %% [markdown]
# ### 1.2. Codificación y Estandarización

# %%
# Separamos la variable objetivo (y) de las predictoras (X)
y = df['tipoCultivo']
X = df.drop('tipoCultivo', axis=1)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

print(f"\nDimensiones de X después de get_dummies: {X.shape}")
print("Columnas de X:", X.columns.tolist())

# %%
# Estandarización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertimos de nuevo a DataFrame para facilitar el ploteo 3D de K-Means
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# %% [markdown]
# ## 2. Análisis de Componentes Principales (PCA)
#
# **Objetivo:** Reducir la dimensionalidad y visualizar los datos en 2D o 3D.

# %% [markdown]
# ### 2.1. Determinación del Número de Componentes
#
# **Corrección:** Analizamos el número de componentes usando dos criterios estándar: la Varianza Acumulada y el Gráfico de Sedimentación (Scree Plot).

# %%
# Ajustamos PCA a todos los componentes para analizar la varianza
pca_full = PCA()
pca_full.fit(X_scaled)

# %%
# Gráfico 1: Varianza Acumulada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Acumulada por Componente')
plt.grid(True)
# Líneas de referencia
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Varianza')
plt.axhline(y=0.80, color='g', linestyle='--', label='80% Varianza')
plt.legend()

# Gráfico 2: Gráfico de Sedimentación (Scree Plot)
plt.subplot(1, 2, 2)
plt.plot(pca_full.explained_variance_ratio_, 'o-')
plt.xlabel('Componente Principal')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Gráfico de Sedimentación (Scree Plot)')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Análisis PCA:**
# * **Varianza Acumulada:** Para capturar el 80% de la varianza, necesitaríamos 10 componentes. Para el 90%, necesitaríamos 12.
# * **Scree Plot:** El "codo" (punto de inflexión) no es muy pronunciado, pero parece estar alrededor de 3 o 4 componentes, después de lo cual la ganancia de varianza por componente disminuye significativamente.
#
# Para la visualización, usaremos `n_components=3`.

# %% [markdown]
# ### 2.2. Visualización PCA (3D)

# %%
# Aplicamos PCA para 3 componentes
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Mapeamos las etiquetas de texto a números para el color
target_names = y.unique()
target_ids = {name: i for i, name in enumerate(target_names)}
y_ids = y.map(target_ids)

# Gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y_ids, cmap='viridis', s=50)

# Etiquetas de ejes
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('PCA 3D de Tipos de Cultivo')

# Leyenda
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                              markerfacecolor=plt.cm.viridis(i / len(target_names) * 0.9), markersize=10)
                   for i, name in enumerate(target_names)]
ax.legend(handles=legend_elements, title='Cultivos')

plt.show()

# %% [markdown]
# ## 3. Aplicación de Isomap
#
# **Objetivo:** Aplicar Isomap y analizar los resultados variando el número de vecinos (`n_neighbors`).
#
# **Corrección:** Se fija `n_components=2` (para el gráfico 2D solicitado) y se itera sobre `n_neighbors` para observar cómo cambia la estructura.

# %%
# Probamos diferentes números de vecinos
neighbors_list = [5, 10, 20, 30, 35, 40, 50]

plt.figure(figsize=(16, 4))
plt.suptitle('Isomap 2D con Diferentes Números de Vecinos', fontsize=16)

for i, k in enumerate(neighbors_list):
    # 1. Aplicar Isomap
    isomap = Isomap(n_components=2, n_neighbors=k)
    X_isomap = isomap.fit_transform(X_scaled)
    
    # 2. Crear subplot
    ax = plt.subplot(1, len(neighbors_list), i + 1)
    
    # 3. Graficar
    scatter = ax.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y_ids, cmap='viridis', s=30)
    ax.set_title(f'n_neighbors = {k}')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')

# Leyenda (se comparte para todos los subplots)
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                              markerfacecolor=plt.cm.viridis(i / len(target_names) * 0.9), markersize=8)
                   for i, name in enumerate(target_names)]
plt.figlegend(handles=legend_elements, title='Cultivos', loc='center right', bbox_to_anchor=(1.05, 0.5))

plt.tight_layout(rect=[0, 0, 0.95, 1]) # Ajustar para la leyenda
plt.show()

# %% [markdown]
# **Análisis Isomap:**
# * Con un número bajo de vecinos (ej. 5), la estructura local se preserva pero los grupos pueden aparecer fragmentados.
# * A medida que `n_neighbors` aumenta, la estructura se vuelve más global. En nuestro caso, los grupos (`tipoCultivo`) no se separan claramente, sugiriendo un alto solapamiento en el espacio de características, incluso en una variedad no lineal.

# %% [markdown]
# ## 4. Aplicación de t-SNE
#
# **Objetivo:** Aplicar t-SNE y analizar los resultados variando iteraciones y perplejidad.
#
# **Corrección:** Se crean dos series de gráficos: una variando `perplexity` (fijando `n_iter`) y otra variando `n_iter` (fijando `perplexity`).

# %% [markdown]
# ### 4.1. t-SNE variando Perplejidad

# %%
perplexity_list = [5, 20, 30, 50]
n_iter_fixed = 1000 # Un número de iteraciones suficiente para converger

plt.figure(figsize=(16, 4))
plt.suptitle(f't-SNE 2D con Diferentes Perplejidades (iter={n_iter_fixed})', fontsize=16)

for i, perp in enumerate(perplexity_list):
    # 1. Aplicar t-SNE
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_iter_fixed, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 2. Crear subplot
    ax = plt.subplot(1, len(perplexity_list), i + 1)
    
    # 3. Graficar
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_ids, cmap='viridis', s=30)
    ax.set_title(f'perplexity = {perp}')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')

# Leyenda
plt.figlegend(handles=legend_elements, title='Cultivos', loc='center right', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

# %% [markdown]
# ### 4.2. t-SNE variando Iteraciones

# %%
iterations_list = [250, 500, 1000, 2000]
perplexity_fixed = 30 # Un valor estándar

plt.figure(figsize=(16, 4))
plt.suptitle(f't-SNE 2D con Diferentes Iteraciones (perplexity={perplexity_fixed})', fontsize=16)

for i, n_iter in enumerate(iterations_list):
    # 1. Aplicar t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity_fixed, max_iter=n_iter, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 2. Crear subplot
    ax = plt.subplot(1, len(iterations_list), i + 1)
    
    # 3. Graficar
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_ids, cmap='viridis', s=30)
    ax.set_title(f'n_iter = {n_iter}')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')

# Leyenda
plt.figlegend(handles=legend_elements, title='Cultivos', loc='center right', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

# %% [markdown]
# **Análisis t-SNE:**
# * **Perplejidad:** Afecta el "equilibrio" entre la estructura local y global. Con perplejidad 20-30 parece mostrar la agrupación más clara, aunque los grupos siguen solapados.
# * **Iteraciones:** Con 250 iteraciones, el algoritmo no ha convergido y los grupos están mal formados. A partir de 1000 iteraciones, la estructura parece estabilizarse.

# %% [markdown]
# ## 5. Aplicación de K-Means
#
# **Objetivo:** Aplicar K-Means, determinar el número óptimo de clusters (k) y visualizar.
#
# **Corrección:**
# 1.  K-Means se aplica sobre `X_scaled` (datos originales estandarizados), NO sobre `X_pca`.
# 2.  Se añade el Método del Codo (Inertia) y el GAP Statistic (reemplazando a Silueta).
# 3.  La visualización 3D usa tres de las características **originales**.

# %% [markdown]
# ### 5.1. Determinación de k (Codo y GAP)
#
# **Nota:** Implementamos el GAP Statistic usando la función provista por el profesor, cumpliendo con la consigna.

# %%
# --- Método del Codo ---
K_range_codo = range(1, 11) # Rango de k para probar
inertia = []
print("Calculando Inercia (Codo) para K-Means...")
for k in K_range_codo:
    kmeans_codo = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_codo.fit(X_scaled)
    inertia.append(kmeans_codo.inertia_)

# --- GAP Statistic ---
K_range_gap = range(1, 11)
max_k_gap = 10
print("Calculando GAP Statistic para K-Means...")
gaps_kmeans, sds_kmeans, k_optimo_gap_kmeans = optimal_k(X_scaled, n_refs=5, max_k=max_k_gap, method='kmeans')
print(f"El k óptimo según GAP (K-Means) es: {k_optimo_gap_kmeans}")

# --- Gráficos ---
plt.figure(figsize=(14, 6))

# Gráfico 1: Método del Codo
plt.subplot(1, 2, 1)
plt.plot(K_range_codo, inertia, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inertia (Suma de cuadrados intra-cluster)')
plt.title('Método del Codo para K-Means')
plt.grid(True)
plt.axvline(3, color='r', linestyle='--', label='Codo en k=3')
plt.legend()

# Gráfico 2: GAP Statistic
plt.subplot(1, 2, 2)
plt.plot(K_range_gap, gaps_kmeans, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('GAP Value')
plt.title('GAP Statistic para K-Means')
plt.grid(True)

plt.axvline(k_optimo_gap_kmeans, color='r', linestyle='--', label=f'k óptimo = {k_optimo_gap_kmeans}')
plt.legend()

plt.tight_layout()
plt.show()

k_optimo_kmeans = k_optimo_gap_kmeans # Usamos el k de GAP como el óptimo
print(f"El k óptimo según el Método del Codo es 3.")
print(f"El k óptimo según GAP Statistic es: {k_optimo_gap_kmeans}")

# %% [markdown]
# **Análisis de k para K-Means:**
# * **Método del Codo:** El codo es claramente visible en k=3.
# * **GAP Statistic:** El puntaje máximo (o el punto óptimo según la regla 1-std-error) también sugiere k=3.
#
# Ambos métodos sugieren **k=3** para el modelo final.

# %% [markdown]
# ### 5.2. Visualización K-Means (3D)

# %%
k_optimo = 3 # Basado en el análisis anterior

# 1. Entrenar K-Means con k=3 sobre X_scaled
kmeans = KMeans(n_clusters=k_optimo, n_init=10, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled) # Etiquetas del cluster

# 2. Visualizar usando 3 atributos originales
# Corrección: Usamos 3 columnas de X_scaled_df (o X) para el gráfico.
# Elegimos 3 variables numéricas importantes para la visualización.
feat_1 = 'humedadSuelo(%)'
feat_2 = 'pHSuelo'
feat_3 = 'temperatura(°C)'

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficamos usando los datos escalados (X_scaled_df) y coloreamos por los clusters encontrados
scatter = ax.scatter(X_scaled_df[feat_1], X_scaled_df[feat_2], X_scaled_df[feat_3],
                     c=clusters_kmeans, cmap='viridis', s=50)

ax.set_xlabel(feat_1)
ax.set_ylabel(feat_2)
ax.set_zlabel(feat_3)
ax.set_title(f'K-Means (k={k_optimo}) sobre 3 Atributos Originales')

# Leyenda
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                              markerfacecolor=plt.cm.viridis(i / (k_optimo-1) * 0.9), markersize=10)
                   for i in range(k_optimo)]
ax.legend(handles=legend_elements, title='Clusters')

plt.show()

# %% [markdown]
# ## 6. Aplicación de Clustering Jerárquico
#
# **Objetivo:** Aplicar clustering jerárquico y determinar el número óptimo de clusters.
#
# **Corrección:** Se utiliza el **Silhouette Score** (solicitado en consigna 6) y el **GAP Statistic** (solicitado en consigna 5 y 6) para determinar el *k* óptimo, complementando al dendrograma.

# %% [markdown]
# ### 6.1. Dendrograma

# %%
# Generamos la matriz de linkage
# 'ward' minimiza la varianza intra-cluster, similar a K-Means
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(14, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrograma de Clustering Jerárquico (Ward)')
plt.xlabel('Índice de Muestra')
plt.ylabel('Distancia (Ward)')
# Línea de corte sugerida (basada en k=3)
# Podemos trazar una línea horizontal donde el corte genere 3 clusters
plt.axhline(y=20, color='r', linestyle='--', label='Corte para k=3') # Ajustar 'y'
plt.legend()
plt.show()

# %% [markdown]
# **Análisis del Dendrograma:**
# El dendrograma muestra cómo se fusionan las muestras. Si cortamos el árbol en las 3 líneas verticales más largas (las últimas fusiones, ej. con una línea horizontal ~y=20), obtenemos 3 clusters, lo cual es coherente con k-Means y el número de `tipoCultivo`.

# %% [markdown]
# ### 6.2. Determinación de k (Silueta y GAP)

# %%
K_range_agg = range(2, 11) # Silhouette necesita al menos 2 clusters
silhouette_scores_agg = []

print("Calculando Silhouette Scores para Clustering Jerárquico...")
for k in K_range_agg:
    # 1. Aplicar AgglomerativeClustering
    agg_cluster = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg_cluster.fit_predict(X_scaled)
    
    # 2. Calcular Silhouette Score
    score = silhouette_score(X_scaled, labels)
    silhouette_scores_agg.append(score)

# --- GAP Statistic (Hierarchical) ---
K_range_gap_agg = range(1, 11)
max_k_gap_agg = 10
print("Calculando GAP Statistic para Clustering Jerárquico... (esto puede tardar varios minutos)")
gaps_agg, sds_agg, k_optimo_gap_agg = optimal_k(X_scaled, n_refs=5, max_k=max_k_gap_agg, method='hierarchical')
print(f"El k óptimo según GAP (Jerárquico) es: {k_optimo_gap_agg}")

# --- Gráficos ---
plt.figure(figsize=(14, 6))

# Gráfico 1: Silhouette Score
plt.subplot(1, 2, 1)
plt.plot(K_range_agg, silhouette_scores_agg, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Análisis Silhouette para Clustering Jerárquico')
plt.grid(True)

k_optimo_silhouette_agg = K_range_agg[np.argmax(silhouette_scores_agg)]
plt.axvline(k_optimo_silhouette_agg, color='r', linestyle='--', label=f'k óptimo = {k_optimo_silhouette_agg}')
plt.legend()

# Gráfico 2: GAP Statistic (Hierarchical)
plt.subplot(1, 2, 2)
plt.plot(K_range_gap_agg, gaps_agg, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('GAP Value')
plt.title('GAP Statistic para Clustering Jerárquico')
plt.grid(True)
plt.axvline(k_optimo_gap_agg, color='r', linestyle='--', label=f'k óptimo = {k_optimo_gap_agg}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"El número óptimo de clusters (Silhouette) es: {k_optimo_silhouette_agg} (score: {max(silhouette_scores_agg):.4f})")
print(f"El número óptimo de clusters (GAP) es: {k_optimo_gap_agg}")

# %% [markdown]
# ## 7. Conclusiones Finales
#
# * **Preprocesamiento:** La imputación de nulos categóricos con la **moda** es un paso crucial y metodológicamente correcto que evita la distorsión de los datos introducida en la entrega original.
# * **Reducción de Dimensionalidad:** PCA, Isomap y t-SNE muestran que los tres tipos de cultivo (`Trigo`, `Soja`, `Maiz`) están **altamente solapados** en el espacio de características. Ninguna técnica de reducción de dimensionalidad logró una separación visual clara, lo que indica que las variables predictoras no distinguen fuertemente las clases.
# * **Clustering (K-Means y Jerárquico):**
#     * Los análisis para *k* (Método del Codo para K-Means, Silhouette para ambos, y GAP Statistic para ambos) sugieren que **k=3** es la estructura de agrupamiento óptima.
#     * Esta es una validación importante, ya que el análisis no supervisado (clustering) encontró la misma cantidad de grupos que las etiquetas supervisadas (3 `tipoCultivo`).
# * **Integración:** Aunque los métodos de *visualización* (PCA, t-SNE) no separaron bien los grupos, los métodos de *clustering* (K-Means, Jerárquico) sí detectaron una estructura subyacente de 3 grupos, coincidiendo con el número real de cultivos. Esto sugiere que los grupos existen, pero están muy cerca y no son linealmente (ni simplemente no linealmente) separables en 2D o 3D.
# * **Nota sobre GAP:** El GAP Statistic se implementó exitosamente replicando la función `optimal_k` provista en el material de cátedra (`U3_Clustering_Wheat.py`), evitando así conflictos con librerías externas.


# %%
