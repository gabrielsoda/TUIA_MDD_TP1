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
import matplotlib.colors as mcolors
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Carga, análisis exploratorio y Preprocesamiento de Datos
#
# Carga del dataset, limpieza (nulos, duplicados), codificar variables categóricas y estandarización las características.

# %%
df = pd.read_csv('SmartFarmingCropYield.csv')
df.head()


#%%
# Revisa si hay filas duplicadas
print(f"Filas duplicadas: {df.duplicated().sum()}") # 0 filas duplicadas

# Información inicial
print("\nColumnas del DataFrame:")
for col in df.columns:
    print(f"-{col}-")
# %%
# Elimina espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()
#%%
# Identificamos las columnas numéricas y categóricas
variables_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
variables_categoricas = df.select_dtypes(include=['object']).columns.tolist()

# Revisa los valores únicos en las columnas categóricas
print(f"\nTipoRiego: {df['tipoRiego'].unique()}")
print(f"tipoFertilizante: {df['tipoFertilizante'].unique()}")
print(f"estadoEnfermedadesCultivo{df['estadoEnfermedadesCultivo'].unique()}")
print(f"tipoCultivo (target): {df['tipoCultivo'].unique()}")
# %%
df.describe(include='number', percentiles=[0.01, 0.1,0.25, 0.5, 0.75, 0.9, 0.99])
# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sns.countplot(data=df, x="tipoRiego", palette="muted", hue="tipoRiego", ax=axes[0], legend=False)
sns.countplot(data=df, x="tipoFertilizante", palette="muted", hue="tipoFertilizante", ax=axes[1], legend=False)
sns.countplot(data=df, x="estadoEnfermedadesCultivo", palette="muted", hue="estadoEnfermedadesCultivo", ax=axes[2], legend=False)

plt.suptitle("Distribución de variables categóricas")

for ax in axes.flat:
    ax.set_ylabel('')
plt.tight_layout()
plt.show()
#%% [markdown]
# **Tipo de Riego:**
# Los tipos de riego Aspersor y Manual tienen frecuencia de uso similar y mayor a la de Goteo.
#
# **Tipo de Fertilizante:**
# El fertilizante Inorganico tiene una frecuencia levemente mayor a Mixto e Inorgánico.
#
# **Estado de enfermedad:**
# El estado de enfermedad 'Severo' es el mas frecuente seguido de 'Leve' y luego 'Moderado'.

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

sns.boxplot(data=df, x="humedadSuelo(%)", palette='light:#d5bb67',
            hue="tipoCultivo", ax=axes[0,0])
handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,0].legend_.remove()

sns.boxplot(data=df, x="humedadSuelo(%)", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[0,0], legend=False)
sns.boxplot(data=df, x="pHSuelo", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[0,1], legend=False)
sns.boxplot(data=df, x="mlPesticida", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[0,2], legend=False)
sns.boxplot(data=df, x="horasLuzSolar", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[1,0], legend=False)
sns.boxplot(data=df, x="humedad(%)", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[1,1], legend=False)
sns.boxplot(data=df, x="precipitacion(mm)", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[1,2], legend=False)
sns.boxplot(data=df, x="diasTotales", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[2,0], legend=False)
sns.boxplot(data=df, x="temperatura(°C)", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[2,1], legend=False)
sns.boxplot(data=df, x="rendimientoKg_hectarea", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[2,2], legend=False)
sns.boxplot(data=df, x="indiceNDVI", palette='light:#d5bb67', hue="tipoCultivo", ax=axes[3,0], legend=False)

fig.delaxes(axes[3,1])
fig.delaxes(axes[3,2])

plt.suptitle("Distribución de variables numéricas")
fig.legend(handles, labels, title='tipoCultivo',loc='center right', )
for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# De los gráficos de boxplots no se observan outliers en ninguna variable.
# Notamos importante solapamiento en las distribuciones de las variables numéricas entre los distintos tipos de cultivo.
# Posiblemente esto haga que la tarea de clustering sea más compleja.

# %%
# Correlación entre variables numéricas
variables_numericas = ["humedadSuelo(%)", "pHSuelo", "mlPesticida", "horasLuzSolar", "humedad(%)", "precipitacion(mm)", "diasTotales", "temperatura(°C)", "rendimientoKg_hectarea", "indiceNDVI"]

plt.figure(figsize=(16, 9))

sns.heatmap(df[variables_numericas].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8}, vmin=-1, vmax=1)

plt.title("Matriz de correlación entre variables numéricas")

plt.tight_layout()
plt.show()
# %% [markdown]
# No se observan correlaciones muy fuertes entre las variables numéricas.
#%%
sns.pairplot(df,hue='tipoCultivo')
plt.show()
# %% [markdown]
# De los gráficos de pairplot no se observan relaciones lineales claras entre las variables numéricas, lo que 
# coincide con la baja correlación observada en la matriz de correlación.
















# %% [markdown]
# ### 1.1. Análisis, Limpieza y manejo de Valores Faltantes 
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
# Dado que el dataset es pequeño (311 filas), priorizaremos imputación con la **moda** para evitar pérdida de datos (~29.6% por columna). 
#%%
# Overlap de NaN
nan_overlap = df[df['tipoRiego'].isna() & df['estadoEnfermedadesCultivo'].isna()].shape[0]
print(f"Filas con NaN en ambas columnas: {nan_overlap}")
# %%
df_antes = df.copy()
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
# %%
# Separamos la variable objetivo (y) de las predictoras (X)
y = df['tipoCultivo']
X = df.drop('tipoCultivo', axis=1)

# Identificamos las columnas numéricas y categóricas
variables_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
variables_categoricas = X.select_dtypes(include=['object']).columns.tolist()

print(f"Variables numéricas: {variables_numericas}")
print(f"Variables categóricas: {variables_categoricas}")

# One-Hot Encoding solo para variables categóricas
X_encoded = pd.get_dummies(X, columns=variables_categoricas, drop_first=True)

print(f"\nDimensiones de X después de get_dummies: {X_encoded.shape}")
print("Columnas de X:", X_encoded.columns.tolist())

# %%
# Estandarización solo de las características numéricas
scaler = StandardScaler()
X_std = X_encoded.copy()

# Escalamos solo las columnas numéricas originales
X_std[variables_numericas] = scaler.fit_transform(X_encoded[variables_numericas])

print(f"\nDimensiones finales: {X_std.shape}")
print("\nPrimeras filas de X_std:")
X_std.head()
# %% [markdown]
# ## 2. Análisis de Componentes Principales (PCA)
#%%
# Obtener todas las componentes principales
pca = PCA(n_components=X_std.shape[1]).set_output(transform="pandas")
pca_df = pca.fit_transform(X_std)
pca_df['target'] = y
# %%
# dataframe de componentes
pca_df
# %%
# Eigenvectors
cols = pca_df.columns.to_list()
cols.remove('target')
pca_df[cols]


pd.DataFrame(pca.components_,
             columns=cols,
             index=X_std.columns)
# %%
# Función para acumular la varianza
def acumular(numeros):
     sum = 0
     var_c = []
     for num in numeros:
        sum += num
        var_c.append(sum)
     return var_c


var_c = acumular(pca.explained_variance_ratio_)
pca_rtd = pd.DataFrame({'Eigenvalues':pca.explained_variance_,
                        'Proporción de variancia explicada':pca.explained_variance_ratio_,
                        'Proporción acumulado de variancia explicada': var_c})
pca_rtd
# %%
# Scree Plot
var_exp = pca.explained_variance_ratio_
n = len(var_exp)

x = np.arange(1, n+1)
cum = np.cumsum(var_exp)

plt.figure(figsize=(8,5))
plt.bar(x, var_exp, alpha=0.8, align='center', color='blue', label='Varianza explicada por componente')
plt.step(x, cum, where='mid', color='r', label='Varianza explicada acumulada')
plt.axhline(y=0.80, color='gray', linestyle='--', linewidth=1, label='80%')
plt.ylabel('Proporción de varianza explicada')
plt.xlabel('Componentes principales')
plt.title('Scree Plot - PCA')
plt.legend()
plt.tight_layout()
plt.show()
# %% [markdown]
# Elegimos las componentes principales según **criterio de varianza acumulada** (+80%)
# - Las **Componentes Principales** seleccionadas son las primeras **9** que explican aproximadamente el **82.9%** de la varianza total.
# %%
# Gráfico de correlación de variables
corr = pca_df[['pca0', 'pca1', 'pca2','pca3','pca4','pca5','pca6','pca7','pca8']].corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws = {'size': 6}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()
# %%
# Seleccionamos las columnas de características
features = X_std.columns.to_list()

# Obtenemos las clases únicas de la variable objetivo
unique_target = y.unique()
num_colors = len(unique_target)

# Definimos una paleta de colores con el mismo número de clases
color_palette = plt.get_cmap('tab20', num_colors)

# Creamos un diccionario que asigna cada clase a un color en formato HEX
target_color_map = {cultivo: mcolors.to_hex(color_palette(i)) for i,
                       cultivo in enumerate(unique_target)}

# Calculamos las cargas (loadings) de cada variable en los componentes principales
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Gráfico 2D de PCA con Plotly (primeras dos componentes principales)
fig = px.scatter(pca_df, x='pca0', y='pca1', color=pca_df["target"],
                 labels={'color': 'target'},
                 color_discrete_map=target_color_map,
                 title="Distribución de las variedades en 2 dimensiones")
fig.show()

# Gráfico 3D de PCA con Plotly (primeras tres componentes principales)
fig = px.scatter_3d(pca_df, x='pca0', y='pca1', z='pca2',
              color=pca_df["target"], labels={'color': 'target'},
              color_discrete_map=target_color_map,
              title="Distribución de las variedades en 3 dimensiones")
fig.show()
# %% [markdown]
# ### Observaciones PCA:
# - En los gráficos 2D y 3D de PCA, se observa que las diferentes variedades de cultivos no están claramente separadas, lo que indica que las características seleccionadas no permiten una diferenciación clara entre las clases.
# - Esto sugiere que la tarea de clustering puede ser desafiante debido al solapamiento entre las variedades.
# - No son lineales las relaciones entre las variables, lo que puede limitar la efectividad de PCA.

# %% [markdown]
#  ## 3. Isomap
# %%
def aplicar_isomap(X_std, y, n_vecinos=5, n_componentes=2):

    isomap = Isomap(n_neighbors=n_vecinos, n_components=n_componentes)
    X_reduced = isomap.fit_transform(X_std)

    columnas = [f"PC{i+1}" for i in range(n_componentes)]
    df_isomap = pd.DataFrame(X_reduced, columns=columnas)
    df_isomap['tipoCultivo'] = y.reset_index(drop=True)

    if n_componentes >= 2:
        fig = px.scatter(
            df_isomap,
            x='PC1', y='PC2',
            color=y,
            labels={'color':'Tipo de cultivo', 'PC1':'PC1', 'PC2':'PC2'},
            title=f'ISOMAP ({n_vecinos} vecinos, {n_componentes} componentes)'
        )
        fig.show()

    return df_isomap
# %%

for n_componentes in [2, 3]:
    for n_vecinos in [5, 25, 50]:
        aplicar_isomap(X_std, y, n_vecinos, n_componentes)
# aplicar_isomap(X_std, y, 10, 3)
# aplicar_isomap(X_std, y, 30, 3)
# aplicar_isomap(X_std, y, 5, 2)
# aplicar_isomap(X_std, y, 50, 2)

# %% [markdown]
# **Análisis Isomap:**
# * Los grupos `tipoCultivo` no se separan claramente, 
# sugiriendo un alto solapamiento en el espacio de características, 
# incluso en una variedad no lineal.

# %% [markdown]
# ## 4. t-SNE

# %%
# Función para aplicar t-SNE y graficar
def plot_tsne(X, y, max_iter=1000, n_components=2, perplexity=30, title='t-SNE'):
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        max_iter=max_iter, 
        random_state=42, 
        n_iter_without_progress=300, 
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2'])
    tsne_df['tipoCultivo'] = y.reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=tsne_df, 
        x='Component 1', 
        y='Component 2', 
        hue='tipoCultivo', 
        palette='Set1', 
        s=50,
        alpha=0.7,       
        edgecolor='white',     
        linewidth=0.5
    )
    plt.title(f'{title} (Iter: {max_iter}, Perp: {perplexity}, Comp: {n_components})')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend(title='Tipo de Cultivo', frameon=True, shadow=True)
    plt.grid(alpha=0.3) 
    plt.tight_layout()
    plt.show()
    
    print(f"KL divergence final: {tsne.kl_divergence_:.4f}")  # 🔹 7. Métrica de calidad
    return X_tsne
# %%
plot_tsne(X_std, y, max_iter=500, perplexity=5, title='t-SNE Bajo Iter y Perplejidad Baja')
plot_tsne(X_std, y, max_iter=5000, perplexity=30, title='t-SNE Balanceado')
plot_tsne(X_std, y, max_iter=10000, perplexity=50, title='t-SNE Alto Iter y Perplejidad Alta')

# %% [markdown]
#  ### **Análisis t-SNE:**
# En todos los casos analizados, los puntos correspondientes a los distintos tipos de cultivo se superponen considerablemente, 
# sin formar agrupaciones claramente diferenciadas.
# Esto indica que t-SNE no logra separar las clases con las variables disponibles, 
# lo cual puede ser esperable considerando que:

# - t-SNE preserva principalmente relaciones locales (vecindarios cercanos), 
# no necesariamente estructura global de clusters
# - Los datos pueden no tener una estructura intrínsecamente separable en el espacio de características original
# - La superposición sugiere que las variables actuales no capturan diferencias 
# discriminativas suficientes entre tipos de cultivo

# %% [markdown]
# ## 5. K-Means

# %%
# %% [markdown]
# ### 5.1. Determinación de k (Codo y GAP)

#%%
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=k, random_state=42).fit(X_std) for k in Nc]

# La suma de residuos cuadrados intra grupos de kMeans en sklearn se guarda en el atributo inertia
inertias = [model.inertia_ for model in kmeans]

plt.figure(figsize=(8, 3.5))
plt.plot(Nc,inertias, "bo-")
plt.xlabel('Número de Clusters')
plt.ylabel('RSS dentro de los grupos')
plt.title('Gráfica de codo')
plt.grid()
plt.show()
# %% [markdown]
#  ### Análisis Método del Codo en K-Means:

# Observamos que en el gráfico del codo la inercia decrece de forma relativamente suave y continua, 
# sin un "codo" pronunciado que indique un número óptimo claro de clusters.
# Podríamos considerar k=6 o k=7 como candidatos, donde la curva sutilmente comienza a aplanarse, 
# aunque la mejora continúa gradualmente hasta aproximadamente k=12.
# Dado que nuestro problema tiene 3 tipos de cultivo, esperaríamos idealmente observar un codo marcado en 
# k=3 si los cultivos fueran naturalmente separables y una caída abrupta de inercia al pasar de 2 a 3 clusters.
# La ausencia de un codo pronunciado y la mejora continua con más clusters sugiere que:
# - Los 3 tipos de cultivo no forman grupos compactos y bien diferenciados en el espacio de características
# - Existe variabilidad significativa dentro de cada tipo de cultivo que K-means intenta capturar con subclusters
# - Nuevamente, las variables actuales no discriminan efectivamente entre las clases reales

# %%
# Aplicamos el modelo
kmeans = KMeans(n_clusters=12, random_state=42,
                init='k-means++', n_init=5, algorithm='lloyd')
kmeans.fit(X_std) #Entrenamos el modelo
y_pred = kmeans.predict(X_std)
# %%
# A que cluster corresponde cada observacion
X_std['Etiquetas KMeans'] = kmeans.labels_
X_std['Etiquetas KMeans'] = X_std['Etiquetas KMeans'].astype('category')
X_std.head()
# %%
fig = px.scatter_3d(X_std, x='humedadSuelo(%)', y='pHSuelo', z='temperatura(°C)',
                    color='Etiquetas KMeans',
                    title='Dispersión de las variedades de cultivo (K-means)')
fig.show()
# %%
fig = px.scatter_3d(X_std, x='humedadSuelo(%)', y='pHSuelo', z='temperatura(°C)',
                    color=y,
                    title='Dispersión de las variedades de cultivo (original)')
fig.show()
# %%
# Aplicamos el modelo, esta vez con k=6
kmeans = KMeans(n_clusters=6, random_state=42,
                init='k-means++', n_init=5, algorithm='lloyd')
kmeans.fit(X_std) #Entrenamos el modelo
y_pred = kmeans.predict(X_std)
# %%
# A que cluster corresponde cada observacion
X_std['Etiquetas KMeans'] = kmeans.labels_
X_std['Etiquetas KMeans'] = X_std['Etiquetas KMeans'].astype('category')
X_std.head()
# %%
fig = px.scatter_3d(X_std, x='humedadSuelo(%)', y='pHSuelo', z='temperatura(°C)',
                    color='Etiquetas KMeans',
                    title='Dispersión de las variedades de cultivo (K-means)')
fig.show()
# %%
fig = px.scatter_3d(X_std, x='humedadSuelo(%)', y='pHSuelo', z='temperatura(°C)',
                    color=y,
                    title='Dispersión de las variedades de cultivo (original)')
fig.show()
# %% [markdown]
# Los scatterplots 3D muestran que no existe una separación clara entre los tipos de cultivo 
# en el espacio de las variables originales. Los puntos se superponen considerablemente, 
# sin formar agrupaciones visualmente distinguibles.

# %%
# --- Función para calcular la inercia ---
def calculate_intra_cluster_dispersion(X, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.inertia_

# --- Calcular Gap Statistic ---
gaps = []
max_k = 15

for k in range(1, max_k + 1):
    # Inercia real con los datos originales
    real_inertia = calculate_intra_cluster_dispersion(X_std, k)

    # Inercia con datos de referencia aleatorios
    inertia_list = []
    for _ in range(10):  # número de replicaciones
        random_data = np.random.rand(*X_std.shape)
        inertia_list.append(calculate_intra_cluster_dispersion(random_data, k))

    reference_inertia = np.mean(inertia_list)

    # Gap Statistic
    gap = np.log(reference_inertia) - np.log(real_inertia)
    gaps.append(gap)

# Número óptimo de clusters
optimal_k = np.argmax(gaps) + 1
print("Número óptimo de clusters según GAP:", optimal_k)

# --- Gráfico de Gap Statistic ---
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, max_k + 1), gaps, "bo-")
plt.xlabel("Número de clusters")
plt.ylabel("Gap Statistic")
plt.title("Método GAP para selección de K")
plt.grid()
plt.show()

# --- Aplicar KMeans con k óptimo ---
kmeans = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++', n_init=5, algorithm='lloyd')
kmeans.fit(X_std)

X_std['Etiquetas KMeans'] = kmeans.labels_ #.astype('category')

# --- Gráfico 3D ---
fig = px.scatter_3d(
    X_std,
    x='humedadSuelo(%)',
    y='pHSuelo',
    z='temperatura(°C)',
    color='Etiquetas KMeans',
    title=f'Dispersión de las variedades de trigo (K-means con K={optimal_k})'
)
fig.show()


# %% [markdown]
# ### Análisis del Método GAP Statistic
# El gráfico del método GAP muestra que el valor máximo se alcanza en k=14 clusters, que sería el óptimo según este criterio estadístico. Sin embargo, observamos que:
# 1. A partir de k=6, se presenta una meseta donde GAP se mantiene relativamente alto y estable, sin aumentar significativamente más.
# 2. Este rango es consistente con el método del codo, que también sugería 6 clusters
# 
# Dado que k=14 parece excesivo para un dataset con solo 3 clases reales, y considerando que:
# - GAP continúa aumentando ligeramente sin un máximo claro antes de k=14
# - La mejora marginal después de k=6 es pequeña
# - Nuestro problema tiene 3 tipos de cultivo conocidos
# 
# Podríamos interpretar que ningún número de clusters captura adecuadamente la estructura real de los datos. Tanto el método del codo como GAP coinciden en que se necesitan muchos clusters (6-15) para modelar la variabilidad, lo cual refuerza que los 3 tipos de cultivo no forman grupos naturalmente compactos y separables con las variables actuales.

# %% [markdown]
## 6. **Clustering jerárquico**
# %%
Z = linkage(X_std, "ward")

dendrogram(Z)
plt.show()

# %%
# Dibuja un dendrograma truncado (solo muestra las últimas p hojas)
dendrogram(Z,  truncate_mode = 'lastp', p = 20, show_leaf_counts = True,
           show_contracted = True)
plt.axhline(y=110, c='k', linestyle='dashed')
plt.xlabel("Numero de puntos en el nodo")
plt.show()
# %%
columnas_booleanas = X_std.select_dtypes(include=['bool']).columns

# Convertir esas columnas a tipo entero (True=1, False=0)
X_std[columnas_booleanas] = X_std[columnas_booleanas].astype(int)

# Verificar los cambios
print(X_std.info())
#%%
distancias=[]
for i in range(1, 10):
    clustering = AgglomerativeClustering(n_clusters=i) # Aplica clustering jerárquico con i clusters
    clustering.fit(X_std)

    # Calcula la matriz de distancias por pares entre los puntos
    pairwise_distances = cdist(X_std, X_std, 'euclidean')

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
# %% [markdown]
# En el gráfico del codo para clustering jerárquico, observamos que el número óptimo de clusters se encuentra entre k=3 y k=4, ya que a partir de ese punto la curva comienza a aplanarse significativamente y la reducción de la distancia/inercia se vuelve marginal.
#%%
n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=n_clusters)

cluster_assignments = clustering.fit_predict(X_std) # Asigna los clusters a los datos

X_std['Etiquetas jerarquico'] = cluster_assignments # Añade la columna con el cluster asignado a cada punto

X_std.head()
# %%
# Gráfico 3D del clustering jerárquico
fig = px.scatter_3d(
    X_std,
    x='humedadSuelo(%)',
    y='pHSuelo',
    z='temperatura(°C)',
    color='Etiquetas jerarquico',
    title=f'Clustering Jerárquico en 3D (n_clusters={n_clusters})',
    labels={'Etiquetas jerarquico': 'Cluster'},
    color_discrete_sequence=px.colors.qualitative.Pastel  # Paleta Pastel
)
fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.show()
# %%
np.random.seed(42)
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
max_k = 15
for k in range(1, max_k + 1):
    # Calcula la dispersión intraclúster en los datos reales para 'k' clústeres
    X_std_numerical = X_std.select_dtypes(include=np.number)
    real_inertia = calculate_intra_cluster_dispersion(X_std, k, linkage='ward')

    inertia_list = []
    for _ in range(10):
      random_data = np.random.rand(*X_std.shape)
      intra_cluster_dispersion = calculate_intra_cluster_dispersion(random_data, k)
      inertia_list.append(intra_cluster_dispersion)

    reference_inertia = np.mean(inertia_list)

    gap = np.log(reference_inertia) - np.log(real_inertia)
    gaps.append(gap)

optimal_k = np.argmax(gaps) + 1
# %%
print("Número seleccionado de clusters según el Gap Statistic:", optimal_k)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), gaps, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic para determinar el número óptimo de clusters (Clustering Jerárquico)')
plt.show()
# %% [markdown]
# Al aplicar el método GAP al clustering jerárquico, observamos que el valor máximo de GAP se alcanza en k=12 clusters.
# Sin embargo, k=9 podría ser una opción más robusta y parsimoniosa, ya que
# el pico en k=9 parece cercano (~-2.91) al de k=12 (~-2.8).
# En general, los valores de la Gap Statistic muestran oscilaciones, en ningún momento se configura un máximo claro y pronunciado.
# Incluso en k=1 la Gap Statistic ya es relativamente alta, lo que sugiere que los datos no tienen una estructura de clusters bien definida.
# Esto refuerza la conclusión de que los tipos de cultivo no forman grupos naturalmente compactos y separados en el espacio de características. 
# mientras que dividir en más clusters no aporta una mejora significativa en la calidad del agrupamiento.
# %% [markdown]
# ### Coeficiente de Silhouette
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
    silhouette_avg, _ = calculate_silhouette(X_std, k)
    silhouette_scores.append(silhouette_avg)

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linewidth=2, markersize=7)
plt.xlabel('Número de Clusters (k)', fontsize=11)
plt.ylabel('Coeficiente de Silhouette', fontsize=11)
plt.title('Coeficiente de Silhouette para determinar el número óptimo de clusters (Clustering Jerárquico)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# %% [markdown]
# Después de k=2, el coeficiente cae bruscamente, 
# lo que indica que agregar más clusters (k=3, k=4, etc.) empeora la calidad de la agrupación; 
# los clusters se vuelven menos distinguibles entre sí.

# Aunque k=2 es la mejor opción en este gráfico, 
# un valor de ~0.350 no está muy cerca de 1. Esto sugiere que, si bien 2 es el número óptimo de grupos, 
# la separación entre ellos es solo moderada, no excelente. Los clusters no son "indiferentes" (cercanos a 0), 
# pero tampoco están "claramente distinguidos" (cercanos a 1).

# %%
