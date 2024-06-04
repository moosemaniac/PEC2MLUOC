# Carlos Expósito Carrera, ML UOC 2023-2024/2. PEC 2

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Cargamos los datos 
my_data = pd.read_csv('star_classification.csv')

# Creamos un array para los nombres de los atributos con el atributo class como útlimo
attribute_names = [ "obj_ID","alpha","delta","u","g","r","i","z","run_ID","rerun_ID","cam_col","field_ID","spec_obj_ID","redshift","plate","MJD","fiber_ID","class" ]

# Cuantos ejemplos hay (shape[0] nos indica la primera dimensión del array Numpy, que equivale a los ejemplos)
sample_num = my_data.shape[0]
print("número de ejemplos: ", sample_num)

# Cuantos atributos hay (shape[1] nos indica la segunda dimensión del array Numpy, que equivale a los atributos)
attribute_num = my_data.shape[1]
print("Número de atributos: ", attribute_num)

# Se busca el índice correspondiente a class y se indican cuantas ocurrencias únicas hay
class_num = len(np.unique(my_data['class']))
print("Número de clases: ",class_num)

# Ordenaremos los atributos, dejando la clase como el último atributo. 
class_data = my_data.pop('class')
my_data['class'] = class_data

# Mostramos los 10 primeros números. 
print(my_data.head(10))

# Mostramos cuantos objetos tenemos de cada clase
print(my_data['class'].value_counts())

# Normalizamos mediante estandarización. Para ello, separaremos la clase del dataset, normalizaremos, y la uniremos otra vez.
my_data.pop('class')
scaler = StandardScaler()
scaler.fit(my_data)
my_data_normalized = scaler.transform(my_data)

# Para añadir los datos, usaremos los dataframes de Panda
normalized_dataframe = pd.DataFrame(my_data_normalized, columns=attribute_names[:-1])
normalized_dataframe['class'] = class_data

# Mostramos los datos normalizados
print(normalized_dataframe.head(10))

'''
    Se decide realizar una selección de características univariante para evaluar la función de bonanza de las características.
    Posteriormente, analizaremos si alguna de las variables puede ser eliminada. En cualquier caso, una vez decididas las características, 
    normalizaremos los datos para su posterior tratamiento.
'''
# Usaremos sklearn para ello. Crearemos un objeto SelectKBest y escogeremos las mejores características
# (serán todas con una puntuación superior a 100, en este caso son 10)
# Documentación: https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
# separamos las características de la clase

Y= normalized_dataframe.pop('class')
X= normalized_dataframe
X_new = SelectKBest(f_classif, k=10).fit(X,Y)
feature_scores = X_new.scores_
selected_features_indices = X_new.get_support()


# Mostraremos las puntuaciones del algoritmo, y las características seleccionadas.
selected_features = np.array(attribute_names[:-1])[selected_features_indices]
print("Puntuaciones de las características:")
for feature, score in zip(attribute_names[:-1], feature_scores):
    print(f"{feature}: {score}")

print("\nCaracterísticas seleccionadas:")
for feature in selected_features:
    print(feature)
    
# Finalmente, transformamos el array y comprobamos que las dimensiones sean las adecuadas
my_data_transformed = X_new.transform(X)
print(my_data_transformed.shape)
transformed_dataframe = pd.DataFrame(my_data_transformed, columns=selected_features)
transformed_dataframe['class'] = class_data
print(transformed_dataframe.head(10))

# Creamos una partición de entreno y de test (80-20). Para cada una, tendremos la clase y los atributos separados. Si es conveniente unirlos posteriormente, se hará
# Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

X_train, X_test, Y_train, Y_test = train_test_split(my_data_transformed, Y, test_size = 0.2, random_state = 42)

# Miramos la medida de los ejemplos
print("Ejemplos X_train : ", X_train.shape[0])
print("Ejemplos de X_test: ", X_test.shape[0])

'''
    Se aplicará una clasificación de vecinos más cercanos (K-Nearest Neighbors) 
    Documentación: https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
'''
# Aplicamos clasificador al conjunto de datos de entrenamiento. Decidimos utilizar 5 vecinos próximos para la decisión. Dejamos los pesos de los atributos como uniformes.

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(X_train,Y_train)

# Predecimos sobre el conjunto de test
KNN_pred = KNN.predict(X_test)

# Calculamos las métricas. Para ello, usaremos las metricas de sklearn
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

# Cálculo de la exactitud (accuracy)
knn_accuracy = accuracy_score(Y_test,KNN_pred)
# Cálculo de la precisión
knn_precision = precision_score(Y_test,KNN_pred, average = 'weighted')
# Cálculo del recall
knn_recall = recall_score(Y_test,KNN_pred, average = 'weighted')
# Cálculo de la matriz de confusión
knn_conf_matrix = confusion_matrix(Y_test,KNN_pred)

# Cálculo del fall-out
# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix 
TN = knn_conf_matrix[0][0] 
FN = knn_conf_matrix[1][0] 
TP = knn_conf_matrix[1][1] 
FP = knn_conf_matrix[0][1] 
FPR = FP/(FP+TN)
#Exponemos las métricas calculadas
print("Exactitud: ", knn_accuracy)
print("Precisión: ", knn_precision)
print("Recall: ", knn_recall)
print("Fall-out: ", FPR)
print("Matriz de Confusión: ")
print(knn_conf_matrix)


'''
    Repetimos el mismo proceso usando un árbol de decisión
    Documentación: https://scikit-learn.org/stable/modules/tree.html#classification
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
'''
# Aplicamos el DecisionTreeClassifier sobre el conjunto de entrenamiento
dtc = DecisionTreeClassifier(random_state = 0)
dtc.fit(X_train,Y_train)

#Predecimos sobre el conjunto de test
dtc_pred = dtc.predict(X_test)

# Calculamos las métricas. Para ello, usaremos las metricas de sklearn
dtc_accuracy = accuracy_score(Y_test,dtc_pred)
dtc_precision = precision_score(Y_test,dtc_pred, average ='weighted')
dtc_recall = recall_score(Y_test,dtc_pred, average = 'weighted')
dtc_conf_matrix = confusion_matrix(Y_test,dtc_pred)
TN = dtc_conf_matrix[0][0] 
FN = dtc_conf_matrix[1][0] 
TP = dtc_conf_matrix[1][1] 
FP = dtc_conf_matrix[0][1] 
FPR = FP/(FP+TN)
#Exponemos las métricas calculadas
print("Exactitud: ", dtc_accuracy)
print("Precisión: ", dtc_precision)
print("Recall: ", dtc_recall)
print("Fall-out: ", FPR)
print("Matriz de Confusión: ")
print(dtc_conf_matrix)