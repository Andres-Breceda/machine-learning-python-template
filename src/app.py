#1- Recopilacion de datos Y definicion del problema. Que datos estan relacionados o afectan el precio de los Airbnb?, el tipo de inmueble?, el grupo inmobiliario? o las vistas que tienen los Airbnb.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

#print(data.columns)
print(data.shape)

data.describe()

#-Existen valores 0 en los valores minimos del precio, y en la disponibilidad que pueden afectar nuestro analisis. Se deben eliminar. Esto es muy necesario en caso de querer hacer analisis predictivos con los datos.

#2- Limpieza de datos, y exploracion.
#En este caso, es necesario eliminar los datos que contengan 0 en el precio, no existen rentas de inmuebles en 0 dolares o 0 pesos. Al eliminar los los ceros nos aseguramos de que el minimo de disponibilidad sea 1, y el precio no sea cero. 

 # Elimina las filas donde price == 0 o availability_365 == 0:
data = data[(data['price'] != 0) & (data['availability_365'] != 0)].reset_index(drop=True)
data.info()
data.describe()

#-Eliminamos datos irrelevantes para nuestro analisis y duplicados.

#Datos duplicados por columna
duplicados_por_columna = data.apply(lambda x: x.duplicated().sum())
print(duplicados_por_columna)
data.drop(["id", "name", "host_name", "last_review","reviews_per_month"], axis = 1, inplace = True)

#En este caso, al ser categorias puede haber datos duplicados, lo importante es que no haya ids duplicadas. host_id puede tener duplicados por que podrian tener registros multiples los propietarios.
data.head()

#3- Analisis de variables categoricas
# Visualización de variables categóricas

fig, axis = plt.subplots(1, 5, figsize=(15, 5))  # 1 fila, 3 columnas

sns.histplot(ax=axis[0], data=data, x="room_type")
axis[0].tick_params(axis='x', rotation=45)

sns.histplot(ax=axis[1], data=data, x="neighbourhood_group").set_xticks([])
sns.histplot(ax=axis[2], data=data, x="neighbourhood").set_xticks([])
sns.histplot(ax = axis[3], data = data, x = "host_id")
sns.histplot(ax = axis[4], data = data, x = "availability_365")

plt.tight_layout()
plt.show()# Guarda imagen

#-Existen mas casas completas que cuartos compartidos o cuartos privados.

#-Hay mas airbnb dependiendo del grupo.

#-Existen varios propietarios que rentan mas de un inmueble.

#-Hay casi igual numero de casas disponibles en menos de 30 dias que casas con disponibilidad completa de 365 dias.

# Visualización de variables numéricas (histogramas)
# Crear el grid de subgráficas
fig, axis = plt.subplots(4, 2, figsize=(10, 14), gridspec_kw={"height_ratios": [6, 1, 6, 1]})

# Histograma y boxplot para 'price'
sns.histplot(ax=axis[0, 0], data=data, x="price")
sns.boxplot(ax=axis[1, 0], data=data, x="price")

# Ajustar límites y rotación de etiquetas para 'minimum_nights'
sns.histplot(ax=axis[0, 1], data=data, x="minimum_nights").set_xlim(0, 200)
sns.boxplot(ax=axis[1, 1], data=data, x="minimum_nights")

# Histograma y boxplot para 'number_of_reviews'
sns.histplot(ax=axis[2, 0], data=data, x="number_of_reviews")
sns.boxplot(ax=axis[3, 0], data=data, x="number_of_reviews")

# Histograma y boxplot para 'calculated_host_listings_count'
sns.histplot(ax=axis[2, 1], data=data, x="calculated_host_listings_count")
sns.boxplot(ax=axis[3, 1], data=data, x="calculated_host_listings_count")

# Ajustar la rotación de las etiquetas del eje X
for ax in axis.flat:
    ax.tick_params(axis='x', rotation=45)

# Ajustar layout para evitar solapamientos
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Crear canvas de subplots con 5 filas 
fig, axis = plt.subplots(4, 2, figsize=(10, 20))  # Aumentamos la altura a 20

# Gráfico 1: minimum_nights vs price
sns.regplot(ax=axis[0, 0], data=data, x="minimum_nights", y="price")
sns.heatmap(data[["price", "minimum_nights"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False)

# Gráfico 2: number_of_reviews vs price
sns.regplot(ax=axis[0, 1], data=data, x="number_of_reviews", y="price").set(ylabel=None)
sns.heatmap(data[["price", "number_of_reviews"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1])

# Gráfico 3: calculated_host_listings_count vs price
sns.regplot(ax=axis[2, 0], data=data, x="calculated_host_listings_count", y="price").set(ylabel=None)
sns.heatmap(data[["price", "calculated_host_listings_count"]].corr(), annot=True, fmt=".2f", ax=axis[3, 0]).set(ylabel=None)

# Gráfico 4: availability_365 vs price
sns.regplot(ax=axis[2, 1], data=data, x="availability_365", y="price")
sns.heatmap(data[["price", "availability_365"]].corr(), annot=True, fmt=".2f", ax=axis[3, 1])

# Ajustar diseño
plt.tight_layout()

# Mostrar
plt.show()

#Mirando las correlaciones ponemos ver que:
#- No tiene una relacion la cantidad de reviews con el precio.
#- No existe relacion al aumentar el numero de noches con el precio.
#- Al aumentar el numero de dias disponibles, deberia de aumentar el precio de los Airbnb para poder encontrar una relacion entre estas dos variables.
#- No existe relacion entre la cantidad de propiedades que tiene un propietario con el precio.

#- Análisis de variables multivariadas

# Seleccionar solo variables numéricas
numeric_df = data.select_dtypes(include=['int64', 'float64'])

# Matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación entre variables numéricas')
plt.show()
plt.savefig("Ver correlaciones entre variables numéricas.png", dpi=300)

#Calculamos la correlacion entre las variables numericas. Como podemos ver no existe muha relacion entre variables numericas. procederemos a calcular la relacion entre las variables que mas nos interesan para nuestro analisis.

### Factorize the Room Type and Neighborhood Data
data1 = data.copy()
data2 = data.copy()
data1["room_type"] = pd.factorize(data["room_type"])[0]
data1["neighbourhood_group"] = pd.factorize(data["neighbourhood_group"])[0]
data1["neighbourhood"] = pd.factorize(data["neighbourhood"])[0]

# Ver el resultado

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(data1[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

#Existe poca relacion entre las varibles del dataset, como para sacar alguna conclucion.

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(5, 4))

# Colores personalizados por grupo de vecindario
colores = {
    "Brooklyn": "#1f77b4",     # Azul suave 
    "Manhattan": "#ffbf00",    # Amarillo dorado 
    "Queens": "#2ca02c",       # Verde fuerte 
    "Staten Island": "#d62728",# Rojo clásico 
    "Bronx": "#9467bd" 
}

# Gráfico de precios promedios por grupos de vecindarios
avg_price = data.groupby(["room_type", "neighbourhood_group"])["price"].mean().reset_index()
sns.barplot(
    data=avg_price,
    x="room_type",
    y="price",
    hue="neighbourhood_group",
    palette=colores,
    ax=ax
)

# Mostrar el gráfico
plt.show()

#- Este grafico nos muestra en promedio los precios mas caros por el tipo de habitacion. En este caso el grupo Manhattan tiene los precios mas caros por casa 
#  completa, cuarto privado o cuarto compartido.

fig, axis = plt.subplots(figsize = (5, 4))

# Gráfico2 de conteo de  grupos de vencindarios 
sns.countplot(data = data, x = "room_type", hue = "neighbourhood_group",   palette=colores)

# Show the plot
plt.show()

#1- Podemos concluir que el grupo Manhattan tiene mas inmuebles para rentar, en la categoria de casas completas o 
#   apartamentos.

#2- Brooklyn es el segundo grupo con mas propiedades y cuenta con mas cantidad de inmuebles por rentar de cuartos privados.

#3- No existe tantos inmuebles de cuartos compartidos, esto puede ser debido a la demanda.

#ANALISIS DE TODOS LOS DATOS

sns.pairplot(data = data)

#Valores atipicos
data.describe()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))

sns.boxplot(ax = axes[0, 0], data = data, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = data, y = "price")
sns.boxplot(ax = axes[0, 2], data = data, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = data, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = data, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = data, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = data, y = "room_type")

plt.tight_layout()

plt.show()

#"Distribución de precios por tipo de habitación y grupo de vecindario"
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="room_type", y="price", hue="neighbourhood_group")
plt.title("Distribución de precios por tipo de habitación y grupo de vecindario")
plt.xticks(rotation=45)  # Rotar etiquetas del eje x
plt.tight_layout()
plt.show()

#Podemos ver los precios mas lejanos por grupo inmobiliario y el tipo de airbnb.

estadisticos_precio = data["price"].describe()
estadisticos_precio

# Rango intercuartílico (IQR) para el precio

rango_iqr = estadisticos_precio["75%"] - estadisticos_precio["25%"]
limite_superior = estadisticos_precio["75%"] + 1.5 * rango_iqr
limite_inferior = estadisticos_precio["25%"] - 1.5 * rango_iqr

print(f"Los límites superior e inferior para detectar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(rango_iqr, 2)}")

#Limpieza de valores atipicos
data = data[data["price"] > 0]
count_0 = data[data["price"] == 0].shape[0]
count_1 = data[data["price"] == 1].shape[0]

print("Count of 0: ", count_0)
print("Count of 1: ", count_1)

#Valores atipicos minimum_nights

noches_stats = data["minimum_nights"].describe()
noches_stats

# Rango intercuartílico (IQR) para minimum_nights
rango_iqr_noches = noches_stats["75%"] - noches_stats["25%"]

limite_superior = noches_stats["75%"] + 1.5 * rango_iqr_noches
limite_inferior = noches_stats["25%"] - 1.5 * rango_iqr_noches

print(f"Los límites superior e inferior para detectar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(rango_iqr_noches, 2)}")

# Limpiar valores atipicos

data = data[data["minimum_nights"] <= 15]

count_0 = data[data["minimum_nights"] == 0].shape[0]
count_1 = data[data["minimum_nights"] == 1].shape[0]
count_2 = data[data["minimum_nights"] == 2].shape[0]
count_3 = data[data["minimum_nights"] == 3].shape[0]
count_4 = data[data["minimum_nights"] == 4].shape[0]


print("Count of 0: ", count_0)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)
print("Count of 3: ", count_3)
print("Count of 4: ", count_4)

#Detección de valores atípicos para number_of_reviews

review_stats = data["number_of_reviews"].describe()
review_stats

# Rango intercuartílico (IQR) para number_of_reviews

rango_iqr_reviews = review_stats["75%"] - review_stats["25%"]

limite_superior = review_stats["75%"] + 1.5 * rango_iqr_reviews
limite_inferior = review_stats["25%"] - 1.5 * rango_iqr_reviews

print(f"Los límites superior e inferior para detectar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(rango_iqr_reviews, 2)}")

#Detección de valores atípicos para calculated_host_listings_count

# Stats for calculated_host_listings_count

hostlist_stats = data["calculated_host_listings_count"].describe()
hostlist_stats

# Rango intercuartílico (IQR) para el conteo de anuncios del anfitrión calculado

hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]

limite_superior = hostlist_stats["75%"] + 1.5 * hostlist_iqr
limite_inferior = hostlist_stats["25%"] - 1.5 * hostlist_iqr

print(f"Los límites superior e inferior para detectar valores atípicos son {round(limite_superior, 2)} y {round(limite_inferior, 2)}, con un rango intercuartílico de {round(hostlist_iqr, 2)}")

count_04 = sum(1 for x in data["calculated_host_listings_count"] if x in range(0, 5))
count_1 = data[data["calculated_host_listings_count"] == 1].shape[0]
count_2 = data[data["calculated_host_listings_count"] == 2].shape[0]

print("Count of 0: ", count_04)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)

# limpiar datos atipicos

total_data = data[data["calculated_host_listings_count"] > 4]

# Count NaN
data.isnull().sum().sort_values(ascending = False)

##3- Escalado de características#################################

from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Codificamos las columnas categóricas con .map()
data["neighbourhood_group"] = data["neighbourhood_group"].map({
    'Bronx': 0, 
    'Brooklyn': 1, 
    'Manhattan': 2, 
    'Queens': 3, 
    'Staten Island': 4
})

data["room_type"] = data["room_type"].map({
    'Shared room': 0,
    'Private room': 1,
    'Entire home/apt': 2
})

# Lista de variables a escalar
num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]

# Escalado con MinMaxScaler
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(data[num_variables])

# Crear DataFrame escalado
df_scal = pd.DataFrame(scal_features, index = data.index, columns = num_variables)

# Añadir la variable dependiente
df_scal["price"] = data["price"]

# Mostrar primeras filas
df_scal.head()

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)