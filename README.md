# **Air Pollution in Seoul**
---
[<img src="https://i.guim.co.uk/img/media/1d1abc561aad84be09d26b81d6959cb34c9be236/0_10_5000_3003/master/5000.jpg?width=1200&quality=85&auto=format&fit=max&s=cc0522633035a7c9ffd7081935b3a462" width="500"/>](image.png)

Es de conocimiento común que la contaminación del aire puede causar varios problemas en el medio ambiente y en nuestra salud. La foto de arriba fue tomada el 11 de diciembre de 2019 y muestra cómo puede impactar severamente los paisajes de Seúl. En esta ocasión, un smog de polvo ultrafino, proveniente desde China, duró dos días e hizo que el gobierno local dictara medidas de emergencia para la reducción de emisiones. Según The Korea Times, el Centro de Pronóstico de la Calidad del Aire, con el Ministerio de Medio Ambiente, informó que el 11 de diciembre a las 10 p.m la concentración de partículas PM2.5 era de aproximadamente 118 microgramos por metro cúbico en Seúl.

# **Objetivo Investigacion del dataset**
El proposito de trabajo de este dataset es para lograr dar una prediccion de como sera la contanimacion dentro una ciudad, en este caso el conjunto de datos nos proporciona la informacion de medicion de la contaminacion del aire en Seul, Corea del Sur.

Estos datos proporcionan valores promedio para seis contaminantes $(SO_2, NO_2, CO, O3, PM_{10}, PM_{2.5}).$
- Los datos se midieron cada hora entre 2017 y 2019.
- Se midieron los datos de 25 distritos de Seúl.


- **Measurement** date = Hora y fecha de cuando se midio
- **Station code** = Codigo de Estacion
- **Address** = Direccion donde se obtuvieron los datos
- **Latitude** = Latitud
- **Longitude** = Longitud 
- **$SO_{2}$** = El dióxido de azufre, u óxido de azufre 
- **$NO_2$** = Dióxido de nitrógeno
- **$CO$** = Monóxido de carbono
- **$O3$** = Ozono
- **$PM_{10}$** = Pequeñas partículas sólidas o líquidas de polvo, cenizas, hollín, partículas metálicas, cemento o polen, dispersas en la atmósfera
- **$PM_{2.5}$** = Pequeñas partículas que incluye sustancias químicas orgánicas, polvo, hollín y metales. 
- **class** = Interpretacion si la contaminacion es:
  - Good : Es normal
  - Bad: Es mala 


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
```

## Lectura e Importacion del Conjunto de Datos


```python
df = pd.read_csv('https://gitlab.com/sergiopaucara/datos-air-pollution-in-seoul/-/raw/main/Measurement_summary.csv')
df = df[:-1]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Measurement date</th>
      <th>Station code</th>
      <th>Address</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SO2</th>
      <th>NO2</th>
      <th>O3</th>
      <th>CO</th>
      <th>PM10</th>
      <th>PM2.5</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-01-01 00:00</td>
      <td>101</td>
      <td>19, Jong-ro 35ga-gil, Jongno-gu, Seoul, Republ...</td>
      <td>37.572016</td>
      <td>127.005007</td>
      <td>0.004</td>
      <td>0.059</td>
      <td>0.002</td>
      <td>1.2</td>
      <td>73.0</td>
      <td>57.0</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2017-01-01 01:00</td>
      <td>101</td>
      <td>19, Jong-ro 35ga-gil, Jongno-gu, Seoul, Republ...</td>
      <td>37.572016</td>
      <td>127.005007</td>
      <td>0.004</td>
      <td>0.058</td>
      <td>0.002</td>
      <td>1.2</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2017-01-01 02:00</td>
      <td>101</td>
      <td>19, Jong-ro 35ga-gil, Jongno-gu, Seoul, Republ...</td>
      <td>37.572016</td>
      <td>127.005007</td>
      <td>0.004</td>
      <td>0.056</td>
      <td>0.002</td>
      <td>1.2</td>
      <td>70.0</td>
      <td>59.0</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2017-01-01 03:00</td>
      <td>101</td>
      <td>19, Jong-ro 35ga-gil, Jongno-gu, Seoul, Republ...</td>
      <td>37.572016</td>
      <td>127.005007</td>
      <td>0.004</td>
      <td>0.056</td>
      <td>0.002</td>
      <td>1.2</td>
      <td>70.0</td>
      <td>58.0</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2017-01-01 04:00</td>
      <td>101</td>
      <td>19, Jong-ro 35ga-gil, Jongno-gu, Seoul, Republ...</td>
      <td>37.572016</td>
      <td>127.005007</td>
      <td>0.003</td>
      <td>0.051</td>
      <td>0.002</td>
      <td>1.2</td>
      <td>69.0</td>
      <td>61.0</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.iloc[:, [2, 4, 5, 6, 7, 8, 9, 10, 11]].values
y = df.iloc[:, -1].values
```

## **Preprocesamiento de datos**

### **Prepocesamiento 1 - Ejemplo**

Transformador de imputación para completar valores perdidos.


```python
from sklearn.impute import SimpleImputer
simple = SimpleImputer()
simple.fit(X)
X_1 = simple.transform(X)
X_1
```




    array([[101.       ,  37.5720164, 127.0050075, ...,   1.2      ,
             73.       ,  57.       ],
           [101.       ,  37.5720164, 127.0050075, ...,   1.2      ,
             71.       ,  59.       ],
           [101.       ,  37.5720164, 127.0050075, ...,   1.2      ,
             70.       ,  59.       ],
           ...,
           [125.       ,  37.5449625, 127.1367917, ...,   0.4      ,
             25.       ,  19.       ],
           [125.       ,  37.5449625, 127.1367917, ...,   0.4      ,
             24.       ,  17.       ],
           [125.       ,  37.5449625, 127.1367917, ...,   0.5      ,
             25.       ,  18.       ]])



### **Prepocesamiento 2 - Ejemplo**

Transforme las características escalando cada característica a un rango determinado.

Este estimador escala y traduce cada característica individualmente de modo que esté en el rango dado en el conjunto de entrenamiento, p. Ej. entre cero y uno.


```python
from sklearn.preprocessing import MinMaxScaler
escala=MinMaxScaler()
escala.fit(X)
X_2 =escala.transform(X)
X_2
```




    array([[0.        , 0.57969677, 0.56310927, ..., 0.03026135, 0.02063005,
            0.00926962],
           [0.        , 0.57969677, 0.56310927, ..., 0.03026135, 0.02007248,
            0.00958926],
           [0.        , 0.57969677, 0.56310927, ..., 0.03026135, 0.0197937 ,
            0.00958926],
           ...,
           [1.        , 0.44863272, 1.        , ..., 0.01925722, 0.0072484 ,
            0.00319642],
           [1.        , 0.44863272, 1.        , ..., 0.01925722, 0.00696961,
            0.00287678],
           [1.        , 0.44863272, 1.        , ..., 0.02063274, 0.0072484 ,
            0.0030366 ]])



### **Prepocesamiento 3 - Valido**

Estandarice las características eliminando la media y escalando a la varianza de la unidad.

La puntuación estándar de una muestra $x$ se calcula como:

$ z = (x - u) / s $

Donde $u$ es la media de las muestras de entrenamiento o cero si `with_mean = False`, y $s$ es la desviación estándar de las muestras de entrenamiento o uno si `with_std = False`.


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X
```




    array([[-1.66408315,  0.34788437,  0.19885863, ...,  1.70434583,
             0.4117658 ,  0.7191414 ],
           [-1.66408315,  0.34788437,  0.19885863, ...,  1.70434583,
             0.38365116,  0.76467397],
           [-1.66408315,  0.34788437,  0.19885863, ...,  1.70434583,
             0.36959385,  0.76467397],
           ...,
           [ 1.66402704, -0.1599516 ,  1.87145932, ..., -0.26941053,
            -0.26298534, -0.14597758],
           [ 1.66402704, -0.1599516 ,  1.87145932, ..., -0.26941053,
            -0.27704265, -0.19151016],
           [ 1.66402704, -0.1599516 ,  1.87145932, ..., -0.02269099,
            -0.26298534, -0.16874387]])



### **Prepocesamiento 4 - Valido**
Codifique las etiquetas de destino con un valor entre 0 y n_classes-1.

Este transformador debe usarse para codificar valores objetivo, es decir, $y$, y no la entrada $X$.


```python
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y[:] = le.fit_transform(y[:])
y = y.astype('int')
```

# **Clasificador**
El modelo de regresión logística general tiene múltiples variables explicativas que pueden ser cuantitativas, categóricas o ambas. Para las variables explicativas de $ p $, el modelo para las probabilidades logarítmicas es

$$ logit [P (Y = 1)] = α + β_1x_1 + β_2x_2 + ··· + β_px_p $$

El parámetro $ β_j $ se refiere al efecto de $ x_j $ en las probabilidades de registro de que $ Y = 1 $, ajustando los de los otros $x's$. Por ejemplo, $ \ exp (β_1) $ es el efecto multiplicativo sobre las probabilidades de un aumento de 1 unidad en $ x_1 $, a un valor fijo para $ β_2x_2 + \cdots + β_px_p $, como cuando podemos mantener constante $ x_2, ..., x_p $.

## **Primera Ejecucion**




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


```python
# Crear un objeto de regresión logística, realizar una regresión logística
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
```




    LogisticRegression(max_iter=1000)




```python
# Predecir la respuesta para el conjunto de datos de prueba
y_pred = log_reg.predict(X_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[  6291   3293]
     [  1290 118628]]
    Exactitud del modelo: 0.9646105851647079
    


```python
from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits = 3, test_size=0.2, random_state=0)
iter = cv.split(X, y)
split = []
for index, (train_index, test_index) in enumerate(iter):
  print('SPLIT: ', index + 1)
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
  log_reg.fit(X_train, y_train)
  accuracy = accuracy_score(y_test, y_pred)
  split.append(accuracy)
  print('Exactitud del Modelo', accuracy)
```

    SPLIT:  1
    TRAIN: [112686 198008  32508 ... 420908 427172 413699] TEST: [623950 594476 592772 ... 516776 410002  81049]
    Exactitud del Modelo 0.8759092523667588
    SPLIT:  2
    TRAIN: [355405  20765 501751 ... 578377 555229 151640] TEST: [309248 135555 558425 ... 534456  49367 239487]
    Exactitud del Modelo 0.8762335716822907
    SPLIT:  3
    TRAIN: [246161 414516 528472 ... 582566 567516 245642] TEST: [376161 526223 472326 ... 150167 173992 372735]
    Exactitud del Modelo 0.8760173588052694
    


```python
split = np.array(split)
print('Primera ejecucion 80(train) - 20(test)')
print('Mediana de la Confiabilidad: ', np.median(split))
```

    Primera ejecucion 80(train) - 20(test)
    Mediana de la Confiabilidad:  0.881492177726985
    


```python
print(len(X), len(y))
```

    647510 647510
    


```python
cv = StratifiedShuffleSplit(n_splits = 3, test_size=0.5)
iter = cv.split(X, y)
split_2 = []
for index, (train_index, test_index) in enumerate(iter):
  print('SPLIT: ', index + 1)
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
  log_reg.fit(X_train, y_train)
  accuracy = accuracy_score(y_test, y_pred)
  split_2.append(accuracy)
  print('Exactitud del Modelo', accuracy)

```

    SPLIT:  1
    TRAIN: [530690 163798 453260 ... 477953  31114 588966] TEST: [620518  67016 284161 ... 142866 603403 563474]
    323755 323755
    323755 323755 323755 323755
    Exactitud del Modelo 0.8818458402186838
    SPLIT:  2
    TRAIN: [317591 232239 272181 ... 494794 266715 184856] TEST: [428053 188433 430823 ... 123683 537227 205443]
    323755 323755
    323755 323755 323755 323755
    Exactitud del Modelo 0.8819910117218267
    SPLIT:  3
    TRAIN: [579512 460042 379676 ...  36352 276171 276612] TEST: [426379 330974 331196 ... 255876 241318 204635]
    323755 323755
    323755 323755 323755 323755
    Exactitud del Modelo 0.8819848342110547
    


```python
split_2 = np.array(split_2)
print('Segunda ejecucion 50(train) - 50(test)')
print('Mediana de la Confiabilidad: ', np.median(split_2))
```

    Segunda ejecucion 50(train) - 50(test)
    Mediana de la Confiabilidad:  0.8819848342110547
    


```python
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline 
```


```python
pca=PCA(n_components=2)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[   487   9057]
     [  6345 113613]]
    Exactitud del modelo: 0.8810674738614075
    


```python
pca=PCA(n_components=3)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[   491   9158]
     [  6341 113512]]
    Exactitud del modelo: 0.8803184506802983
    


```python
pca=PCA(n_components=5)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[   499   9043]
     [  6333 113627]]
    Exactitud del modelo: 0.8812682429614986
    


```python
pca=PCA(n_components=7)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[   494   9171]
     [  6338 113499]]
    Exactitud del modelo: 0.880241231795648
    


```python
pca=PCA(n_components=8)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[   526   9011]
     [  6306 113659]]
    Exactitud del modelo: 0.8817238343809362
    
