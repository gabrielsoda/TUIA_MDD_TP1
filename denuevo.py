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
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
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
variables_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables categóricas: {variables_categoricas}")
variables_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas: {variables_numericas}")

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
#%% [markdown]
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
#%% 
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
#  https://colab.research.google.com/drive/18hoaWX9tCcURS9z4rlOx57Zn68IYPRQz#scrollTo=QQL4F5OwpcvP
