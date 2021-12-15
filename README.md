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
cv = StratifiedShuffleSplit(n_splits = 100, test_size=0.2, random_state=0)
iter = cv.split(X, y)
split = []
for index, (train_index, test_index) in enumerate(iter):
  print('SPLIT: ', index + 1)
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
  log_reg.fit(X_train, y_train)
  y_pred = log_reg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  split.append(accuracy)
  print('Exactitud del Modelo', accuracy)
```

    SPLIT:  1
    TRAIN: [112686 198008  32508 ... 420908 427172 413699] TEST: [623950 594476 592772 ... 516776 410002  81049]
    Exactitud del Modelo 0.9651356735803308
    SPLIT:  2
    TRAIN: [355405  20765 501751 ... 578377 555229 151640] TEST: [309248 135555 558425 ... 534456  49367 239487]
    Exactitud del Modelo 0.9637920649874133
    SPLIT:  3
    TRAIN: [246161 414516 528472 ... 582566 567516 245642] TEST: [376161 526223 472326 ... 150167 173992 372735]
    Exactitud del Modelo 0.964811354264799
    SPLIT:  4
    TRAIN: [ 38546 274323 381864 ... 299471  77379  45302] TEST: [175036 507293 425540 ... 540222 353467 184156]
    Exactitud del Modelo 0.9646800821608933
    SPLIT:  5
    TRAIN: [115220 143626 634247 ... 229104 133437 506096] TEST: [624214 578946 398971 ... 190585 134388 413425]
    Exactitud del Modelo 0.9643248752915012
    SPLIT:  6
    TRAIN: [443149 178789 626826 ... 329673 112308 312380] TEST: [186422  26343 403708 ...   6383 364244  88851]
    Exactitud del Modelo 0.9648499637071242
    SPLIT:  7
    TRAIN: [391626   2412 389556 ... 280531 232544 554987] TEST: [ 81009 505050 153142 ...  50772 331109 425104]
    Exactitud del Modelo 0.9647264134916835
    SPLIT:  8
    TRAIN: [422884 637088 112588 ... 181513 122467 622748] TEST: [ 14204 510835 481944 ... 575606 340731 503129]
    Exactitud del Modelo 0.9646028632762428
    SPLIT:  9
    TRAIN: [473679 514029 210576 ...  31898 224444 475557] TEST: [405523 634723 221257 ... 384238 386135 114120]
    Exactitud del Modelo 0.9645102006146623
    SPLIT:  10
    TRAIN: [200086  59494  54102 ...  83739  28149 137274] TEST: [466566  20085 267410 ... 470020  21292 596530]
    Exactitud del Modelo 0.9650661765841454
    SPLIT:  11
    TRAIN: [382892 354781 539962 ...  90786 519426 290486] TEST: [   477  83728 526700 ... 490820 569992 276680]
    Exactitud del Modelo 0.9650352890302852
    SPLIT:  12
    TRAIN: [ 68270 552311 559889 ... 217505 309626  66835] TEST: [219754 182574 590468 ... 280774 358238 283175]
    Exactitud del Modelo 0.9639773903105744
    SPLIT:  13
    TRAIN: [418474 407578 364007 ... 560714 340340 236482] TEST: [373191 239009 624381 ... 612536 172194 542668]
    Exactitud del Modelo 0.9642090469645257
    SPLIT:  14
    TRAIN: [544363 403036 524590 ...  10640 481845 299738] TEST: [443322 574826  11596 ...  22087  68092 169385]
    Exactitud del Modelo 0.9647495791570787
    SPLIT:  15
    TRAIN: [191290 173043 623316 ... 136509  30647 411894] TEST: [588707  75084 323327 ...  61354  74095 555966]
    Exactitud del Modelo 0.9639696684221093
    SPLIT:  16
    TRAIN: [350454 235643 621050 ... 534176 183464 298707] TEST: [277478 531820 567481 ... 262084   4984 204090]
    Exactitud del Modelo 0.9650584546956804
    SPLIT:  17
    TRAIN: [612338 594430 153220 ... 384850 629798 127195] TEST: [104398  47310 232947 ... 631730  69640 545237]
    Exactitud del Modelo 0.9641549937452704
    SPLIT:  18
    TRAIN: [568719 128325 410640 ... 181857 598818  77590] TEST: [496146 430014 569276 ... 479680 160863 530783]
    Exactitud del Modelo 0.9648731293725193
    SPLIT:  19
    TRAIN: [ 23698 146283 573221 ... 125562 307876 504032] TEST: [448689 535058 523182 ... 460962  25520 583878]
    Exactitud del Modelo 0.9635835739988572
    SPLIT:  20
    TRAIN: [  5439 591281 412414 ... 188861  42457 557256] TEST: [105899 643401 319305 ... 167728 420957 579773]
    Exactitud del Modelo 0.9648499637071242
    SPLIT:  21
    TRAIN: [635869 118917 144849 ... 228964 295263 189909] TEST: [230841 186295 226629 ... 526102  55178 339527]
    Exactitud del Modelo 0.9647804667109389
    SPLIT:  22
    TRAIN: [322575 367007  53188 ... 313676 622560  87145] TEST: [386923   6933 571550 ... 304614 378408 527683]
    Exactitud del Modelo 0.9649503482571697
    SPLIT:  23
    TRAIN: [262090 222150 147148 ... 632769 269045 112918] TEST: [357230  73603 110123 ... 220206 271827 277033]
    Exactitud del Modelo 0.9640623310836898
    SPLIT:  24
    TRAIN: [433672 487645 424453 ... 238354 604262 566968] TEST: [575878 302078 498138 ... 111166 380836 363478]
    Exactitud del Modelo 0.9650507328072153
    SPLIT:  25
    TRAIN: [462417 398657 119712 ... 541606 547874  54728] TEST: [444603 383928 288212 ... 605435 176099 143150]
    Exactitud del Modelo 0.964641472718568
    SPLIT:  26
    TRAIN: [408260 402999    442 ... 319982 304240 313369] TEST: [189842 328398 111580 ... 380229 423683 387650]
    Exactitud del Modelo 0.9647804667109389
    SPLIT:  27
    TRAIN: [555505 135290 218066 ...  29270 330888  11010] TEST: [573939 562825 221289 ... 451676 172702  80053]
    Exactitud del Modelo 0.9643557628453615
    SPLIT:  28
    TRAIN: [170454 502634 422062 ... 490055 279346 559997] TEST: [347662  95526 508464 ... 483587 442071  40925]
    Exactitud del Modelo 0.9643943722876867
    SPLIT:  29
    TRAIN: [393446 503064 257687 ... 275995 406093 642747] TEST: [482577 253662 496682 ... 230535 434435 379913]
    Exactitud del Modelo 0.9645642538339176
    SPLIT:  30
    TRAIN: [ 14873 299445 148403 ... 262439 218968 331331] TEST: [239520 438526 145860 ... 518487 458306 499154]
    Exactitud del Modelo 0.9646491946070331
    SPLIT:  31
    TRAIN: [619714 640090 466112 ... 102019  39069 568414] TEST: [ 31703 622771 361219 ... 201458  19508  94260]
    Exactitud del Modelo 0.9644252598415468
    SPLIT:  32
    TRAIN: [ 56779 272384 590743 ... 645495  43571 380355] TEST: [553345 606257 266815 ... 596057 182933 297753]
    Exactitud del Modelo 0.9642862658491761
    SPLIT:  33
    TRAIN: [300596 567006 353670 ...  18798 209527 463652] TEST: [   346 268412 490091 ... 307631   7665 213392]
    Exactitud del Modelo 0.9644715911723372
    SPLIT:  34
    TRAIN: [311374 570384 177988 ... 357050 603135 280971] TEST: [462299 357623  79817 ... 581326 322836 574992]
    Exactitud del Modelo 0.964811354264799
    SPLIT:  35
    TRAIN: [266913 191215 163905 ... 321189  88265 611852] TEST: [450180 455481 217768 ... 235143 573235 506000]
    Exactitud del Modelo 0.9645642538339176
    SPLIT:  36
    TRAIN: [332491 438357 570751 ... 477916 360139 599114] TEST: [104985 346351 521839 ... 137941 283211 536046]
    Exactitud del Modelo 0.9637997868758784
    SPLIT:  37
    TRAIN: [236273 449265 302281 ... 155285 491901  29083] TEST: [101679 537128 531008 ... 396599 637000 584327]
    Exactitud del Modelo 0.9638538400951336
    SPLIT:  38
    TRAIN: [170488  84030 149797 ... 554723 492482 162426] TEST: [ 42530 361232 645201 ...  51067 588940 211880]
    Exactitud del Modelo 0.9645410881685225
    SPLIT:  39
    TRAIN: [519073  81228 104483 ...  31669 555968 191936] TEST: [274108 362729 100935 ...  95859 531305 306707]
    Exactitud del Modelo 0.963923337091319
    SPLIT:  40
    TRAIN: [639245 446392 501883 ...   2108 457409 296188] TEST: [526816 328890 486353 ... 175452 607266 452287]
    Exactitud del Modelo 0.9641549937452704
    SPLIT:  41
    TRAIN: [389876 506094 507104 ... 366566 307037 259877] TEST: [240524 359310 252698 ... 215873  18490 509803]
    Exactitud del Modelo 0.9646878040493583
    SPLIT:  42
    TRAIN: [199535 631646  99463 ... 501615 610017 264965] TEST: [281938 102308  56086 ... 249492 364968 533397]
    Exactitud del Modelo 0.9650430109187503
    SPLIT:  43
    TRAIN: [393791  66589  19230 ...  14586 384529 163933] TEST: [ 11775 368531 559292 ... 437339 403304 191410]
    Exactitud del Modelo 0.9645256443915924
    SPLIT:  44
    TRAIN: [364882 574359 183710 ... 114148 199653 216445] TEST: [143026 407829 476764 ... 308573  13826 405649]
    Exactitud del Modelo 0.9654059396766073
    SPLIT:  45
    TRAIN: [227284 426530 177892 ... 196133 522373  18910] TEST: [237308 293082  55441 ... 577387 555036 206497]
    Exactitud del Modelo 0.9646646383839632
    SPLIT:  46
    TRAIN: [497181 477295 624731 ... 489055 217357  86156] TEST: [114173 598032 537090 ...  99380 375770 160797]
    Exactitud del Modelo 0.9645488100569876
    SPLIT:  47
    TRAIN: [186941 266136 589796 ... 432460 479176 625645] TEST: [ 88416 348349 598481 ... 424512 580835   9376]
    Exactitud del Modelo 0.9645179225031274
    SPLIT:  48
    TRAIN: [497870 471173  25887 ...   5520 485116 603757] TEST: [516203 561389 240758 ... 637187 315858 635173]
    Exactitud del Modelo 0.9648499637071242
    SPLIT:  49
    TRAIN: [286867 635086 479820 ... 545672 553415 567268] TEST: [225827 218817 250054 ... 168795 362791  95182]
    Exactitud del Modelo 0.9643866503992217
    SPLIT:  50
    TRAIN: [544035 575369 347445 ...  57748 505732 524888] TEST: [156364 565480 383507 ... 204215 611168 624950]
    Exactitud del Modelo 0.9652592237957714
    SPLIT:  51
    TRAIN: [ 41004 455307 374797 ... 509492 229563 230422] TEST: [481338 149966 637111 ... 401666 253963 622434]
    Exactitud del Modelo 0.9643789285107566
    SPLIT:  52
    TRAIN: [149401 643784 273631 ... 497328 237999 262867] TEST: [438556 143047 559215 ... 466034 306025 144570]
    Exactitud del Modelo 0.9644407036184769
    SPLIT:  53
    TRAIN: [  8103 129351 466244 ... 560598 288895 226554] TEST: [622466 228147 458305 ... 647220 224790 161148]
    Exactitud del Modelo 0.9650738984726105
    SPLIT:  54
    TRAIN: [435573  40157 175844 ... 604784 115670 197717] TEST: [607360 372442 305962 ... 439636 203504 483054]
    Exactitud del Modelo 0.9645796976108477
    SPLIT:  55
    TRAIN: [159208 608866 284200 ... 438575 460066 207348] TEST: [254907 119659 282951 ... 419360  93064 207893]
    Exactitud del Modelo 0.9636144615527173
    SPLIT:  56
    TRAIN: [404377 512944 555164 ... 174189 323250 317565] TEST: [ 67318 222731 271866 ... 234643 174311 126629]
    Exactitud del Modelo 0.964270822072246
    SPLIT:  57
    TRAIN: [157914 344336 490867 ... 136527 421859 384263] TEST: [143411 430245  83165 ...  11389 365119 104896]
    Exactitud del Modelo 0.9642090469645257
    SPLIT:  58
    TRAIN: [135324 224156 537678 ... 100596 165485 112580] TEST: [426620 452551 440200 ... 499973 350380 168543]
    Exactitud del Modelo 0.9642167688529907
    SPLIT:  59
    TRAIN: [516338 352768  42009 ... 412533 323718 330391] TEST: [ 17723 155962 195422 ... 240566 385779 227042]
    Exactitud del Modelo 0.964456147395407
    SPLIT:  60
    TRAIN: [124614 632368 607202 ... 190738 418285 391898] TEST: [291437 294849 459586 ... 208487  22508 119687]
    Exactitud del Modelo 0.9644870349492672
    SPLIT:  61
    TRAIN: [547999 482621 410402 ...  48719 147501  91811] TEST: [243031 504945 195962 ...  51391  75951 621171]
    Exactitud del Modelo 0.9652128924649812
    SPLIT:  62
    TRAIN: [ 90723 502554 530789 ... 215414 233271 187002] TEST: [ 66331 420901 162717 ... 139058 280779 273627]
    Exactitud del Modelo 0.9641549937452704
    SPLIT:  63
    TRAIN: [160592 408033 457886 ... 243283  78165 322415] TEST: [408096 336980 547226 ... 262479 208184 567161]
    Exactitud del Modelo 0.9644175379530818
    SPLIT:  64
    TRAIN: [265884 206738 156701 ... 165719 428827 377130] TEST: [115226 243747 322252 ...  83986 255146 551760]
    Exactitud del Modelo 0.9647495791570787
    SPLIT:  65
    TRAIN: [442944 314637 450557 ...  63440 382647 218530] TEST: [476092 545807 399419 ... 547956 147354  69202]
    Exactitud del Modelo 0.9644715911723372
    SPLIT:  66
    TRAIN: [463063 585231 149412 ... 418911 109157 342232] TEST: [547333  83403  53168 ... 290782 507542 195509]
    Exactitud del Modelo 0.9651742830226561
    SPLIT:  67
    TRAIN: [532438 569855 398048 ... 632344 450808 558210] TEST: [403208 319749  10976 ... 580875 323246 440353]
    Exactitud del Modelo 0.9642631001837809
    SPLIT:  68
    TRAIN: [521289 434047 533895 ... 359653  29855 316016] TEST: [402567 204646  25899 ... 484772 274478 597783]
    Exactitud del Modelo 0.9649349044802397
    SPLIT:  69
    TRAIN: [413345 472815 355117 ... 321894  49282 604575] TEST: [294046 429294 643647 ... 441305 198084 292908]
    Exactitud del Modelo 0.9646028632762428
    SPLIT:  70
    TRAIN: [443819  64156 363859 ... 201471 589777 623269] TEST: [570002 488801 217355 ... 388109 632798 215643]
    Exactitud del Modelo 0.9641781594106654
    SPLIT:  71
    TRAIN: [641534 370664  96656 ... 465642 282437 207087] TEST: [412921 140988 380693 ... 474818  30773 453587]
    Exactitud del Modelo 0.9643171534030363
    SPLIT:  72
    TRAIN: [185885 485070 282973 ...  23094 153063 612465] TEST: [452541 495248 231278 ...  37939 578759 385326]
    Exactitud del Modelo 0.9647109697147535
    SPLIT:  73
    TRAIN: [299811 244310 558722 ... 606008 267730 330478] TEST: [ 70300 520769 388718 ...  55937 629246 568937]
    Exactitud del Modelo 0.9641472718568053
    SPLIT:  74
    TRAIN: [367548 245094 287187 ... 641992 420171 249106] TEST: [474349 229437 151527 ... 564560  98581 379389]
    Exactitud del Modelo 0.964641472718568
    SPLIT:  75
    TRAIN: [247888  78715 366574 ... 380773  30292   3126] TEST: [525967 511854 163900 ...  10423 500002 408823]
    Exactitud del Modelo 0.9649117388148446
    SPLIT:  76
    TRAIN: [324910  94730   2903 ... 139584 110718 376074] TEST: [439318 622418  13530 ... 614976 430245 415379]
    Exactitud del Modelo 0.964795910487869
    SPLIT:  77
    TRAIN: [229533   2223 647227 ... 142544 436962 358859] TEST: [411536  44117  40977 ... 424729 452289 159110]
    Exactitud del Modelo 0.9647727448224738
    SPLIT:  78
    TRAIN: [504789 533570  35668 ...  64018 102394 331960] TEST: [332979 294821  33938 ... 193005 324505 626271]
    Exactitud del Modelo 0.9648808512609844
    SPLIT:  79
    TRAIN: [118196 393794 185441 ... 602139 515856  59652] TEST: [173259 449746 342770 ... 478352  72953 487665]
    Exactitud del Modelo 0.9644175379530818
    SPLIT:  80
    TRAIN: [629571 581848 380451 ... 296618 519979 207475] TEST: [408118 104494 168496 ... 224190 361424  91605]
    Exactitud del Modelo 0.9643248752915012
    SPLIT:  81
    TRAIN: [553460 223289 306288 ... 502002 503575  51412] TEST: [144734 607886 217297 ... 327327  83124 449712]
    Exactitud del Modelo 0.9644407036184769
    SPLIT:  82
    TRAIN: [209467 348946 467420 ... 348899 107984 229345] TEST: [423217 136658 109164 ... 170952 510238 263017]
    Exactitud del Modelo 0.9644252598415468
    SPLIT:  83
    TRAIN: [ 31189 384487 528522 ...  55094 298471 116938] TEST: [389779 248692 423949 ... 471450 346582 605825]
    Exactitud del Modelo 0.9646028632762428
    SPLIT:  84
    TRAIN: [492722 386366 484196 ... 257677 434613  30521] TEST: [ 34066 299652 451234 ... 456870 132403 239183]
    Exactitud del Modelo 0.964811354264799
    SPLIT:  85
    TRAIN: [375969  86371 266881 ... 389898   9953 514526] TEST: [135371  79288 339857 ... 131677 515022 182548]
    Exactitud del Modelo 0.9647186916032184
    SPLIT:  86
    TRAIN: [ 93596 266449 473898 ... 596503 454205  86068] TEST: [ 87589 364126 517712 ... 319839 342237 218668]
    Exactitud del Modelo 0.9646878040493583
    SPLIT:  87
    TRAIN: [110948 370344 322691 ... 107048 502459 165703] TEST: [367894 163086 602517 ... 162659  16663 275338]
    Exactitud del Modelo 0.964641472718568
    SPLIT:  88
    TRAIN: [447529 548499 168676 ...  66907 151244 561958] TEST: [387763 416011  55125 ...  65023 398685 271291]
    Exactitud del Modelo 0.9641163843029451
    SPLIT:  89
    TRAIN: [214301 586467 406771 ... 199792  79640 625975] TEST: [554149 199515 105997 ... 639466 124011 315887]
    Exactitud del Modelo 0.9642862658491761
    SPLIT:  90
    TRAIN: [ 57872 472007 633496 ... 377206 488046  46352] TEST: [116586 466198 322851 ... 386222 460867 617629]
    Exactitud del Modelo 0.9650584546956804
    SPLIT:  91
    TRAIN: [402901 409914 126717 ...  56582 300021 161153] TEST: [360556 641001 428081 ... 136334 471712 179725]
    Exactitud del Modelo 0.9651511173572609
    SPLIT:  92
    TRAIN: [160516 214317 414486 ... 552442 636320  15358] TEST: [356607 616738 333499 ... 163202 307960 470262]
    Exactitud del Modelo 0.9642399345183859
    SPLIT:  93
    TRAIN: [234378 540285 307587 ... 629624 535072 248919] TEST: [246746 257296 231907 ...  40944 122725 293640]
    Exactitud del Modelo 0.964819076153264
    SPLIT:  94
    TRAIN: [302788 351120 146277 ... 135178 220897 287761] TEST: [114374 625771 414271 ... 198795 524132 641933]
    Exactitud del Modelo 0.9643634847338265
    SPLIT:  95
    TRAIN: [451867 224162  12712 ... 444142 562273 133962] TEST: [ 36865 534880 266225 ...  13832 512721  34093]
    Exactitud del Modelo 0.9638461182066687
    SPLIT:  96
    TRAIN: [150505 453998 296523 ... 216325 543431 459251] TEST: [591582 359783 484035 ... 605345 487438 313027]
    Exactitud del Modelo 0.9651356735803308
    SPLIT:  97
    TRAIN: [591940 164899 159445 ... 110194 400218  74572] TEST: [554359  33208 364386 ... 473586   2325 111518]
    Exactitud del Modelo 0.9644020941761517
    SPLIT:  98
    TRAIN: [ 81146 399600 423180 ... 255425 228508 589671] TEST: [ 64711 301292 503263 ... 290639 328378 553010]
    Exactitud del Modelo 0.9642167688529907
    SPLIT:  99
    TRAIN: [480014 173130 279256 ... 477961 551645 622970] TEST: [377400 116524 620138 ... 432844  61066 324324]
    Exactitud del Modelo 0.9644252598415468
    SPLIT:  100
    TRAIN: [512409  10361 449808 ... 598958 193672 443302] TEST: [351106 572583 314495 ... 111382 394521  65705]
    Exactitud del Modelo 0.9651047860264707
    


```python
split = np.array(split)
print('Primera ejecucion 80(train) - 20(test)')
print('Mediana de la Confiabilidad: ', np.median(split))
```

    Primera ejecucion 80(train) - 20(test)
    Mediana de la Confiabilidad:  0.9645565319454525
    


```python
cv = StratifiedShuffleSplit(n_splits = 100, test_size=0.5, random_state=0)
iter = cv.split(X, y)
split_2 = []
for index, (train_index, test_index) in enumerate(iter):
  print('SPLIT: ', index + 1)
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
  log_reg.fit(X_train, y_train)
  y_pred = log_reg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  split_2.append(accuracy)
  print('Exactitud del Modelo', accuracy)

```

    SPLIT:  1
    TRAIN: [515898 263945 402627 ... 408913 139508  37600] TEST: [170239 631015 291453 ...  76455 288151 589059]
    Exactitud del Modelo 0.9648777625055984
    SPLIT:  2
    TRAIN: [404763 174232 624126 ... 267864 270561  58844] TEST: [  6027 142932  74813 ... 356535 158453 185879]
    Exactitud del Modelo 0.964627573319331
    SPLIT:  3
    TRAIN: [242487  21856 364320 ... 603612 626892 421699] TEST: [401763 316095 572419 ... 622212 447926 541270]
    Exactitud del Modelo 0.964979691433337
    SPLIT:  4
    TRAIN: [342340  95864 445978 ... 623319 313287 183261] TEST: [445709 186780 290310 ... 403172 380401  73855]
    Exactitud del Modelo 0.9643619403561335
    SPLIT:  5
    TRAIN: [250791  25073 558351 ... 180686 586396  17415] TEST: [  2411 379535 185373 ... 417283 551633 300691]
    Exactitud del Modelo 0.9648623187286682
    SPLIT:  6
    TRAIN: [313203  91195 187763 ... 538819 398612 411591] TEST: [413673 581302 126284 ...  24438 407171 258140]
    Exactitud del Modelo 0.9651094191595496
    SPLIT:  7
    TRAIN: [378704 623658 109284 ... 598291 324837 373699] TEST: [130832  30311 149649 ...  66587 267653 606158]
    Exactitud del Modelo 0.9646461058516471
    SPLIT:  8
    TRAIN: [530916 223379 278086 ... 106312 640408 621853] TEST: [470077   2660 529350 ...  81316 203317 576327]
    Exactitud del Modelo 0.9645071118592763
    SPLIT:  9
    TRAIN: [205540 254467 166806 ... 418037  49760  95174] TEST: [449434 642987 541014 ... 326572  35543 211336]
    Exactitud del Modelo 0.9645750644777686
    SPLIT:  10
    TRAIN: [523417 505997  14809 ... 424112 259840 377119] TEST: [427826 158113 586858 ... 314047  70002 440920]
    Exactitud del Modelo 0.9644793130608021
    SPLIT:  11
    TRAIN: [619552  54713 455028 ...  96330 570960 616551] TEST: [439245 102278 491632 ...  26716 631415  33842]
    Exactitud del Modelo 0.9645379994131364
    SPLIT:  12
    TRAIN: [622558 187924  69207 ... 624666 102025 404915] TEST: [495186 615567 146910 ... 302176 336967 503647]
    Exactitud del Modelo 0.9645657982116106
    SPLIT:  13
    TRAIN: [502747 363398 616355 ... 342941  59468  64278] TEST: [145657 290020 645385 ... 308659 144532 367051]
    Exactitud del Modelo 0.9643094315145712
    SPLIT:  14
    TRAIN: [ 97148 640231 247296 ... 602165 126821 178042] TEST: [245626  85562 536897 ... 579116 183584 469440]
    Exactitud del Modelo 0.9643156090253432
    SPLIT:  15
    TRAIN: [124436 493378 393727 ... 362963 275979 212869] TEST: [558111 287019 173837 ... 468288 315170  39865]
    Exactitud del Modelo 0.9648036323763339
    SPLIT:  16
    TRAIN: [368568 280340 500893 ... 353967 515888 541267] TEST: [598671 493736 418660 ... 280640 461767  37288]
    Exactitud del Modelo 0.9647078809593674
    SPLIT:  17
    TRAIN: [391581 479296 385995 ... 581382 499085 253077] TEST: [ 31339 190766 180366 ... 185698 416820 110190]
    Exactitud del Modelo 0.9647665673117017
    SPLIT:  18
    TRAIN: [ 11568  45910 268887 ...  43330  80294 146245] TEST: [261032 586061 646453 ... 261807 157549 366201]
    Exactitud del Modelo 0.9646893484270513
    SPLIT:  19
    TRAIN: [199646 334941 621248 ... 187099 208345 128748] TEST: [316749  46694 310697 ... 239285 534713 322746]
    Exactitud del Modelo 0.9641920588099026
    SPLIT:  20
    TRAIN: [ 56910  73065 274376 ...  71138 333792 233013] TEST: [460293 104418  18502 ... 564537 541636 272036]
    Exactitud del Modelo 0.96444224799617
    SPLIT:  21
    TRAIN: [121599 309553 501408 ... 230630 176230 446603] TEST: [ 30291 294421 295909 ... 421886 432039 342620]
    Exactitud del Modelo 0.9643650291115196
    SPLIT:  22
    TRAIN: [103391 264789 553531 ... 558434 252651 177993] TEST: [430344 418061 638489 ... 334794  31173  89319]
    Exactitud del Modelo 0.9643248752915012
    SPLIT:  23
    TRAIN: [158743 574316 484102 ... 504335 366209 317927] TEST: [419835 180869 512624 ... 243443 625821 525049]
    Exactitud del Modelo 0.9646739046501213
    SPLIT:  24
    TRAIN: [439876 211944 230739 ... 482954 287782 352131] TEST: [283866 540093 589817 ... 141643 489695   1214]
    Exactitud del Modelo 0.9646677271393492
    SPLIT:  25
    TRAIN: [238901  80091 581292 ... 526033 176249 167836] TEST: [565927 217708 362417 ... 219869  20930 135167]
    Exactitud del Modelo 0.964278543960711
    SPLIT:  26
    TRAIN: [382616 555025  88426 ... 199541 278643 631977] TEST: [545307 502863  14916 ...  91357 174769  49279]
    Exactitud del Modelo 0.9643928279099937
    SPLIT:  27
    TRAIN: [549700 126662 340579 ...  31247  55893 294901] TEST: [447623 223051 298208 ... 541389 641812 449149]
    Exactitud del Modelo 0.9643588516007475
    SPLIT:  28
    TRAIN: [606716 526597 303782 ... 499731 498133 458988] TEST: [165316 162846 363088 ... 470931 272746 209886]
    Exactitud del Modelo 0.9643526740899755
    SPLIT:  29
    TRAIN: [162938 590864 459737 ... 578978 615245 437564] TEST: [129836 235946 503963 ...  18630  40275 156570]
    Exactitud del Modelo 0.9648839400163705
    SPLIT:  30
    TRAIN: [405148 302457  34774 ... 433046 487889 383928] TEST: [289922 555766   2376 ... 244079 470176 404265]
    Exactitud del Modelo 0.9643650291115196
    SPLIT:  31
    TRAIN: [473849   7803 338070 ... 150943 535116 382565] TEST: [216861 422276 542724 ... 555326 438172   7296]
    Exactitud del Modelo 0.9644700467946441
    SPLIT:  32
    TRAIN: [431314 389500 524489 ...  95194 172993 529545] TEST: [511394  82010 646378 ... 534560 426976 201538]
    Exactitud del Modelo 0.9645657982116106
    SPLIT:  33
    TRAIN: [643540 516136 455290 ... 420044 236015 123249] TEST: [157173 620594 457257 ... 207360 158264 109277]
    Exactitud del Modelo 0.9646461058516471
    SPLIT:  34
    TRAIN: [516866 400599 294001 ... 376351  21172 377516] TEST: [220932 577071 189748 ...  48582  63360 572797]
    Exactitud del Modelo 0.9640190885082855
    SPLIT:  35
    TRAIN: [576972 544712 263466 ... 178122    350 375066] TEST: [ 77808 511643 447622 ... 247418 335610 366037]
    Exactitud del Modelo 0.9645874194993127
    SPLIT:  36
    TRAIN: [630488 443929  23608 ... 249383 547596 126470] TEST: [605694 226426 187769 ... 103749 484127 483802]
    Exactitud del Modelo 0.9643032540037991
    SPLIT:  37
    TRAIN: [421989 247433 329637 ...  54584   7508 233017] TEST: [ 27823 645845 385645 ... 295265 353233 586048]
    Exactitud del Modelo 0.9650322002748992
    SPLIT:  38
    TRAIN: [615215 403798 414525 ... 158923 491278 252570] TEST: [383400 501019 520161 ... 575045 506947 308505]
    Exactitud del Modelo 0.9640592423283038
    SPLIT:  39
    TRAIN: [  5004 149420 504881 ... 452497 618569 631307] TEST: [118734 123466 574637 ... 586835 218639 570331]
    Exactitud del Modelo 0.9643866503992217
    SPLIT:  40
    TRAIN: [178698  59070 409286 ... 483026 186254 344620] TEST: [124083  69438  17412 ... 540739 128211 485166]
    Exactitud del Modelo 0.9640592423283038
    SPLIT:  41
    TRAIN: [611387 572174 637556 ... 470161 639031 154554] TEST: [510589 538019 199842 ... 563039 362470 569691]
    Exactitud del Modelo 0.9648654074840542
    SPLIT:  42
    TRAIN: [452469 575868  12468 ... 126841 187626 603017] TEST: [456152 266788 447582 ... 413004 206801 270318]
    Exactitud del Modelo 0.9643835616438357
    SPLIT:  43
    TRAIN: [176741 117231 615178 ... 310401 222838 158189] TEST: [645484 446851 493858 ... 500197 280293 574251]
    Exactitud del Modelo 0.9644978455931182
    SPLIT:  44
    TRAIN: [550911 425858 182163 ... 295199  44434 615181] TEST: [ 19872 496368 327221 ... 289597 147174 140002]
    Exactitud del Modelo 0.9643464965792035
    SPLIT:  45
    TRAIN: [529019  40503 345628 ... 353302 504996 644896] TEST: [276137 212329 236115 ... 264545 515618 161015]
    Exactitud del Modelo 0.9641024849037081
    SPLIT:  46
    TRAIN: [254207  89867 456965 ... 376344 259425 500386] TEST: [142840   8512 209814 ... 495484 414823 597821]
    Exactitud del Modelo 0.9645688869669966
    SPLIT:  47
    TRAIN: [213547 480102  50735 ...  86299  13281 411712] TEST: [592980 379901 409115 ... 112513 318212 420339]
    Exactitud del Modelo 0.9646769934055073
    SPLIT:  48
    TRAIN: [173842 333829 484979 ...  42436 501449 459588] TEST: [464352 419903 133732 ... 571598 192140 313660]
    Exactitud del Modelo 0.9647078809593674
    SPLIT:  49
    TRAIN: [467626 300151 483463 ... 163280  51495 228400] TEST: [416332 508951 304733 ...  77270  18885 406646]
    Exactitud del Modelo 0.9646491946070331
    SPLIT:  50
    TRAIN: [328768 519853 330430 ... 110860  56576 381462] TEST: [561868 465523 605983 ... 584081 353498 469844]
    Exactitud del Modelo 0.9645534431900665
    SPLIT:  51
    TRAIN: [279914 641309 228000 ... 161972  78088 214195] TEST: [ 53214 167410 294973 ... 211513 634458 285940]
    Exactitud del Modelo 0.9645040231038903
    SPLIT:  52
    TRAIN: [389999 578644 213867 ... 223720 558681 427136] TEST: [512647  37579 313665 ... 585786 645054 251571]
    Exactitud del Modelo 0.9645132893700483
    SPLIT:  53
    TRAIN: [ 72943 620566  54064 ... 139510 340357  25119] TEST: [270624  20503 247188 ... 285335 613762 635595]
    Exactitud del Modelo 0.9646430170962611
    SPLIT:  54
    TRAIN: [263938 578747 647075 ...  14385 373551 398123] TEST: [587556  50228 497912 ...  44144 515863 361165]
    Exactitud del Modelo 0.9647140584701395
    SPLIT:  55
    TRAIN: [285181 505233 338044 ...  65375 117455 553665] TEST: [180996 257900 267828 ... 474574  48639 540157]
    Exactitud del Modelo 0.9643032540037991
    SPLIT:  56
    TRAIN: [139697 579338 151494 ... 627360 103694 595147] TEST: [487319 178289 457507 ... 211108 307610 374337]
    Exactitud del Modelo 0.9644947568377322
    SPLIT:  57
    TRAIN: [630493  16179 260938 ... 151122 436808 277548] TEST: [247428 107478 266648 ...  19431 466926 510418]
    Exactitud del Modelo 0.9650692653395314
    SPLIT:  58
    TRAIN: [330016   3827 188309 ... 539811 266169 603137] TEST: [125049  33665 302109 ... 522767 535377 592245]
    Exactitud del Modelo 0.9645472656792945
    SPLIT:  59
    TRAIN: [639950 106778 215106 ... 448035 154256 602633] TEST: [267494 318124 594444 ...  23553 230451 282265]
    Exactitud del Modelo 0.9643464965792035
    SPLIT:  60
    TRAIN: [531506 510073  59203 ... 499396 454447 501587] TEST: [624156 304297 554977 ... 238731 633001 291622]
    Exactitud del Modelo 0.9645627094562246
    SPLIT:  61
    TRAIN: [373531  35577 480285 ... 595096 411832 209147] TEST: [539925 576152   8075 ...  36167 260121 482438]
    Exactitud del Modelo 0.9649611589010209
    SPLIT:  62
    TRAIN: [497820  96614 284935 ... 209798  85358 370162] TEST: [275451 289480 331722 ...  68511 460641 559672]
    Exactitud del Modelo 0.9642198576083767
    SPLIT:  63
    TRAIN: [261403 260728 508586 ...  30127 162467 167754] TEST: [477744 515604 510991 ... 406673 529769 505241]
    Exactitud del Modelo 0.9645040231038903
    SPLIT:  64
    TRAIN: [488678 334551 370102 ... 207777  18396  97980] TEST: [415140 138141  44949 ... 231372  17833  55016]
    Exactitud del Modelo 0.9645688869669966
    SPLIT:  65
    TRAIN: [321283 643840  96289 ... 281252 631034 565726] TEST: [233609 361133 154423 ... 168532 375216 219451]
    Exactitud del Modelo 0.9641210174360242
    SPLIT:  66
    TRAIN: [236988 249566   3319 ...  53471 377965  71082] TEST: [251198 273777 605559 ... 433506 588463 201284]
    Exactitud del Modelo 0.9647542122901577
    SPLIT:  67
    TRAIN: [ 17317 334453 294701 ...  18524 584911 442783] TEST: [374857 329891 419221 ... 175813 310409 565451]
    Exactitud del Modelo 0.9643464965792035
    SPLIT:  68
    TRAIN: [239344 517132 338916 ... 343694 499291 466075] TEST: [537647 158130 485250 ... 586253 159204  36482]
    Exactitud del Modelo 0.9647047922039814
    SPLIT:  69
    TRAIN: [244003 279852 544690 ... 478873 311658 474254] TEST: [488885 463265 108028 ... 608095 367810  16393]
    Exactitud del Modelo 0.9643186977807292
    SPLIT:  70
    TRAIN: [548558 603165 355413 ... 329565 509180 361840] TEST: [365981  18250 620347 ...  96944 352945 402127]
    Exactitud del Modelo 0.9643156090253432
    SPLIT:  71
    TRAIN: [613186 480348  58685 ...  42603 384116 196506] TEST: [295219 118335 157782 ...  60506 570421 161670]
    Exactitud del Modelo 0.9645040231038903
    SPLIT:  72
    TRAIN: [ 17097 101505 101370 ... 365460 422065 386628] TEST: [592939 456502 147986 ... 557278 371679 399059]
    Exactitud del Modelo 0.964287810226869
    SPLIT:  73
    TRAIN: [643539 481155 497361 ... 619739 596826 432741] TEST: [ 54073 420744 156908 ... 509038 304777 577350]
    Exactitud del Modelo 0.9642661889391669
    SPLIT:  74
    TRAIN: [282592 601249  19691 ... 497051 341497 351711] TEST: [322604 387605 386967 ... 129376 209527 218064]
    Exactitud del Modelo 0.9647078809593674
    SPLIT:  75
    TRAIN: [205001 293669 101159 ... 507726  33885 300401] TEST: [ 13244 430721 486850 ... 394700 414215  98300]
    Exactitud del Modelo 0.9650352890302852
    SPLIT:  76
    TRAIN: [482658 629558 149043 ... 567162  91498   9889] TEST: [196464 104666 220244 ... 561829 616968 201042]
    Exactitud del Modelo 0.9647171472255255
    SPLIT:  77
    TRAIN: [628561 321259 350300 ... 502405 161921 424030] TEST: [100771 429288 170481 ... 501047 298703 555378]
    Exactitud del Modelo 0.9647078809593674
    SPLIT:  78
    TRAIN: [574539 523735 409897 ... 145581 473738 447773] TEST: [192670 328537 389666 ... 196642 471809 503345]
    Exactitud del Modelo 0.9645843307439267
    SPLIT:  79
    TRAIN: [623107 234929 498845 ... 196648 385919 618161] TEST: [149864  44651 388731 ... 498599 621275 204468]
    Exactitud del Modelo 0.9642229463637627
    SPLIT:  80
    TRAIN: [419784 567585 591512 ... 531416 156222 569144] TEST: [ 29046 270420 350725 ... 362932 115956  66128]
    Exactitud del Modelo 0.9644206267084678
    SPLIT:  81
    TRAIN: [ 85850 540089 595857 ... 562364 453363 625110] TEST: [444057 472000  70911 ...  63632 515499 439226]
    Exactitud del Modelo 0.9640993961483221
    SPLIT:  82
    TRAIN: [566558 458660 139691 ... 171883 357956 178784] TEST: [209336 591434 199790 ... 515081  93586  35672]
    Exactitud del Modelo 0.9642507451622369
    SPLIT:  83
    TRAIN: [337083 165205  17639 ... 393769 487146 215540] TEST: [ 38148 304923 428825 ... 619762 646099   1090]
    Exactitud del Modelo 0.9645719757223826
    SPLIT:  84
    TRAIN: [ 50762 566191 242939 ... 641418 171340 200252] TEST: [112478  19261 201139 ... 517731  32141 293375]
    Exactitud del Modelo 0.9645688869669966
    SPLIT:  85
    TRAIN: [ 95761 552057 607702 ...   4618 347799 517168] TEST: [185504 631797  82177 ... 492720  45080 226913]
    Exactitud del Modelo 0.9644206267084678
    SPLIT:  86
    TRAIN: [423764 418665 221212 ... 371342 586040 459249] TEST: [339487 409553 415011 ...  30108 100649  35694]
    Exactitud del Modelo 0.9645163781254343
    SPLIT:  87
    TRAIN: [297985 377003 250637 ... 335266 252060 572737] TEST: [308818  19634 232315 ...  46719 172927  57142]
    Exactitud del Modelo 0.9646368395854891
    SPLIT:  88
    TRAIN: [342767 107518 540065 ... 402053 249381 472550] TEST: [204879  54612 550406 ... 262281 301218 333635]
    Exactitud del Modelo 0.9645287331469784
    SPLIT:  89
    TRAIN: [ 65565 409905  19383 ... 287989 400659 617083] TEST: [504059 634545 374547 ... 356235 282381   6159]
    Exactitud del Modelo 0.9644947568377322
    SPLIT:  90
    TRAIN: [ 36514 267817 564475 ... 404974 462209 589716] TEST: [510550 643440 249988 ... 225666 478540 411833]
    Exactitud del Modelo 0.9644206267084678
    SPLIT:  91
    TRAIN: [338513 259648  87441 ... 579315  19784 500485] TEST: [ 11275 336905  40185 ... 545882 465767 527859]
    Exactitud del Modelo 0.9644824018161882
    SPLIT:  92
    TRAIN: [390129 477061  96754 ... 185610 622382 510084] TEST: [ 74885 346977 306644 ... 374999 439608 591521]
    Exactitud del Modelo 0.9647480347793856
    SPLIT:  93
    TRAIN: [ 22263 240315   9646 ... 536708 614889 100094] TEST: [ 23064 570345 515780 ... 256873 574814  52988]
    Exactitud del Modelo 0.9642353013853068
    SPLIT:  94
    TRAIN: [229770 111170 613328 ... 591110 485482 243469] TEST: [627199 235109 138233 ... 292258 611188 261491]
    Exactitud del Modelo 0.9648499637071242
    SPLIT:  95
    TRAIN: [352022 508567 183379 ... 515041 457867  88383] TEST: [200213 125703 310201 ... 468831  73029   5528]
    Exactitud del Modelo 0.9643990054207657
    SPLIT:  96
    TRAIN: [ 42097 569327 401482 ...  83684  69866 190079] TEST: [322193 264452 184574 ...  77986  79419 612859]
    Exactitud del Modelo 0.96444224799617
    SPLIT:  97
    TRAIN: [238476 406650 428505 ... 631725 395328 343804] TEST: [552001 431040 596756 ... 169742 293385  28474]
    Exactitud del Modelo 0.9643959166653797
    SPLIT:  98
    TRAIN: [463410 123629 567257 ...  58536  99773 589709] TEST: [532032 189933 307960 ... 535339 268380 333117]
    Exactitud del Modelo 0.9644576917731
    SPLIT:  99
    TRAIN: [ 47046 238529 264050 ...   4368 240318  88043] TEST: [183741 168972 406613 ...  27094 184707  96072]
    Exactitud del Modelo 0.9644020941761517
    SPLIT:  100
    TRAIN: [532912 627928 225083 ... 104734 107306  60178] TEST: [291618  64469  36754 ...  43146 258719  67842]
    Exactitud del Modelo 0.9640468873067597
    


```python
split_2 = np.array(split_2)
print('Segunda ejecucion 50(train) - 50(test)')
print('Mediana de la Confiabilidad: ', np.median(split_2))
```

    Segunda ejecucion 50(train) - 50(test)
    Mediana de la Confiabilidad:  0.9645040231038903
    

# **PCA - Principal Component Analysis**

PCA siempre se puede utilizar para simplificar los datos con grandes dimensiones (mayores de 2) en datos bidimensionales eliminando las características menos influntiales de los datos. Sin embargo, debemos saber que la eliminación de datos hace que la variable independiente sea menos interpretable. Antes de comenzar a tratar con el PCA, primero debemos aprender cómo el PCA utiliza los vectores propios para obtener una matriz de covarianza de diagonalización.

## **Autovectores**

Los autovectores y autovalores son las principales herramientas utilizadas por PCA para obtener una matriz de covarianza de diagnóstico. El autovector es un vector cuya dirección no se verá afectada por la transformación lineal, por lo tanto, los autovectores representan la dirección de mayor varianza de los datos, mientras que el autovalor decide la magnitud de esta varianza en esas direcciones.

Aquí usamos una matriz simple (2x2) $ A $ para explicarlo.
$$
A = \begin {bmatrix}
1 & 4 \\
3 & 2
\end {bmatrix}
$$

En general, el vector propio $v$ de una matriz $A$ es el vector donde se cumple lo siguiente:

$$
Av = \lambda v
$$

para el cual $\lambda$ representa el autovalor tal que la transformación lineal en $v$ puede definirse mediante $\lambda$

Además, podemos resolver la ecuación de la siguiente manera:

$$
Av - \lambda v = 0 \\
v(A-\lambda I) = 0
$$
Mientras que $I$ es la matriz identidad de A

$$
I = A^TA = AA^T
$$

En este caso, si $v$ es un vector sin cero entonces $Det(A - \lambda I) = 0$, ya que no puede ser invertible, y podemos resolver $v$ para $A$ depende de esta relación.

$$
I = \begin{bmatrix} 
1 & 0 \\
0 & 1 
\end{bmatrix} \\
$$

$$
(A - \lambda I) = \begin{bmatrix}
1-\lambda & 4 \\
3 & 2 - \lambda 
\end{bmatrix} \\
$$

Para resolver el $\lambda$ podemos usar la función resolver en sympy o calculando.

En este caso, $\lambda_1 = -2 $ y $ \lambda_2 = 5 $, y podemos calcular los vectores propios en dos casos.

Por $ \lambda_1 = -2 $

Con base en la matriz, podemos inferir que el vector propio puede ser
$$
v_1 = \begin {bmatrix}
-4 \\
3 \end {bmatrix}
$$

Por $ \lambda = 5 $

Con base en la matriz, podemos inferir que el vector propio puede ser
$$
v_2 = \begin {bmatrix}
1 \\
1 \end {bmatrix}
$$
Con todo, la matriz de covarianza $ A'$ ahora puede ser:
$$
A' = v * A \\
$$

De tal manera que podamos obtener la matriz $V$
$$
V = \begin {bmatrix}
-4 & 1 \\
3 & 1
\end {bmatrix}
$$
donde $ A' = V ^ {- 1} A V $ 
# Ejecutando PCA



```python
cov_mat = np.dot(X.T, X)
cov_mat
```




    array([[ 6.47510000e+05, -3.17134965e+05,  3.04718613e+04,
             1.78240818e+04,  1.39381385e+04,  1.22972438e+04,
            -2.25013534e+04,  1.95184392e+04,  1.12813560e+04],
           [-3.17134965e+05,  6.47510000e+05,  1.99162748e+05,
            -4.40090867e+03, -1.35728720e+04,  2.41555797e+02,
             3.65483233e+04, -1.41639443e+04, -1.49420706e+04],
           [ 3.04718613e+04,  1.99162748e+05,  6.47510000e+05,
             9.84785032e+03,  5.55641763e+03,  4.79069353e+03,
             2.29083675e+04, -1.47346880e+04, -1.22591124e+04],
           [ 1.78240818e+04, -4.40090867e+03,  9.84785032e+03,
             6.47510000e+05,  4.61300409e+05,  5.21602189e+05,
             1.97440402e+05,  3.14516841e+04,  3.07768689e+04],
           [ 1.39381385e+04, -1.35728720e+04,  5.55641763e+03,
             4.61300409e+05,  6.47510000e+05,  5.08816925e+05,
             1.59122795e+05,  3.59576893e+04,  3.74545357e+04],
           [ 1.22972438e+04,  2.41555797e+02,  4.79069353e+03,
             5.21602189e+05,  5.08816925e+05,  6.47510000e+05,
             1.22378247e+05,  2.49954178e+04,  2.19300755e+04],
           [-2.25013534e+04,  3.65483233e+04,  2.29083675e+04,
             1.97440402e+05,  1.59122795e+05,  1.22378247e+05,
             6.47510000e+05,  9.78816458e+04,  1.18408099e+05],
           [ 1.95184392e+04, -1.41639443e+04, -1.47346880e+04,
             3.14516841e+04,  3.59576893e+04,  2.49954178e+04,
             9.78816458e+04,  6.47510000e+05,  1.48269137e+05],
           [ 1.12813560e+04, -1.49420706e+04, -1.22591124e+04,
             3.07768689e+04,  3.74545357e+04,  2.19300755e+04,
             1.18408099e+05,  1.48269137e+05,  6.47510000e+05]])




```python
autovalores, autovectores = np.linalg.eig(cov_mat)
print(autovalores)
print(autovectores)
```

    [1727870.83882243 1016466.95409743  107188.72579891  259164.27252604
      185141.75603897  845714.30088854  674470.3096866   519889.80291888
      491683.0392222 ]
    [[-0.02173509 -0.59545712 -0.00404949  0.59602315  0.04662498 -0.04094427
       0.53235712  0.04129027 -0.02718651]
     [ 0.00635899  0.71310549 -0.02703131  0.69058172  0.05252777  0.03882337
       0.03118996 -0.08799049  0.02871867]
     [-0.01321286  0.34628096  0.01115557 -0.40156531 -0.0235633  -0.0182809
       0.84393135 -0.05765174  0.04585307]
     [-0.5552525   0.00459065 -0.51165811  0.03375465 -0.64649305 -0.10243596
      -0.00919118 -0.01388406  0.00406858]
     [-0.54468574 -0.0073891  -0.33366724 -0.0650103   0.74901775 -0.11207502
      -0.02366649 -0.0963212   0.0653942 ]
     [-0.55820865  0.0031062   0.78345968  0.02733587 -0.10871726 -0.16434567
      -0.02994288 -0.15826454  0.09520479]
     [-0.26152009  0.09070118  0.11031143 -0.0029807   0.05499375  0.40123995
       0.04085239  0.7357603  -0.45185517]
     [-0.08303581 -0.07237009 -0.00364312 -0.02054758 -0.01405272  0.61368815
       0.01299614 -0.63448148 -0.45594504]
     [-0.08625584 -0.0587488  -0.00310891  0.00738527 -0.0221274   0.63938412
       0.00489814  0.09381306  0.75559099]]
    


```python
from random import shuffle
components = 2
v = [i for i in range(9)]
shuffle(v)
pca = np.delete(autovectores, obj = v[:9 - components], axis = 1)

var_acc = (np.sum(autovalores) - autovalores[2]) / (np.sum(autovalores))
print(var_acc)
```

    0.9816066803260156
    


```python
pca
```




    array([[-0.59545712,  0.04129027],
           [ 0.71310549, -0.08799049],
           [ 0.34628096, -0.05765174],
           [ 0.00459065, -0.01388406],
           [-0.0073891 , -0.0963212 ],
           [ 0.0031062 , -0.15826454],
           [ 0.09070118,  0.7357603 ],
           [-0.07237009, -0.63448148],
           [-0.0587488 ,  0.09381306]])




```python
data_pca = np.dot(X, pca)
print(len(data_pca[0]))
data_pca
```

    2
    




    array([[ 1.38786405,  0.94334064],
           [ 1.38728789,  0.96628687],
           [ 1.38843355,  0.9768789 ],
           ...,
           [-0.45366914, -0.06834017],
           [-0.44984849, -0.06201969],
           [-0.43126047,  0.11603394]])




```python
X_train, X_test, y_train, y_test = train_test_split(data_pca, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[     3   9571]
     [     0 119928]]
    Exactitud del modelo: 0.9260938055010733
    

# **PCA con Python**


```python
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline 
```

### **PCA - 2 componentes**


```python
pca=PCA(n_components=2)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))

```

    Confusion Matrix:
    [[  4737   4856]
     [  1291 118618]]
    Exactitud del modelo: 0.9525335516053806
    

### **PCA - 3 componentes**


```python
pca=PCA(n_components=3)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[  5021   4575]
     [  1289 118617]]
    Exactitud del modelo: 0.9547188460409878
    

### **PCA - 5 componentes**


```python
pca=PCA(n_components=5)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[  5233   4360]
     [  1381 118528]]
    Exactitud del modelo: 0.9556686383221881
    

### **PCA - 7 componentes**


```python
pca=PCA(n_components=7)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[  5364   4346]
     [  1372 118420]]
    Exactitud del modelo: 0.955846241756884
    

### **PCA - 8 componentes**


```python
pca=PCA(n_components=8)
pca.fit(X)
transformada=pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformada, y, test_size=0.2)
log_reg = LogisticRegression(solver = 'lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
```

    Confusion Matrix:
    [[  6354   3407]
     [  1362 118379]]
    Exactitud del modelo: 0.9631743139102099
    
