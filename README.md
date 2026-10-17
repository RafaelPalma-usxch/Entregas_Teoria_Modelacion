# Entregas_Teoria_Modelacion
"Guía técnica paso a paso para ejecutar con éxito el programa de Estanques Acoplados (Tarea 1)"

---------------------------------------------
### PASO 0 (Antes de ejecutar el programa) ###
Es necesario instalar todas las librerías que se utilizarán a lo largo de la ejecución del programa en Python (pandas, numpy, matplotlib, scikit-learn).

Para ello, basta con colocar la siguiente línea de código en la terminal de tu computador o PC (con el comando pip):
pip install numpy matplotlib pandas scikit-learn

También se vuelve necesario tener instalada alguna versión de Python, descargando cualquiera de sus versiones desde el sitio web:
https://www.python.org/downloads/
---------------------------------------------

### PASO 1 ###
Encuentre el archivo 'tarea1_estanques_acoplados.py' en el repositorio de GitHub,
y ejecute el programa mediante la tecla F5.
---------------------------------------------

---------------------------------------------
### PASO 2 ###
Luego, se comenzará a ejecutar el programa; luego de unos segundos,
se le pedirá que confirme la ruta del archivo CSV llamado 'resultado_estanques.csv'. Si el archivo CSV
tiene la extensión corecta, presione Enter y se cargará el archivo correctamente.
PD: Si la ruta del archivo no coincide con la del CSV original, procure cambiarlo hasta que coincida con el nombre de 'resultado_estanques.csv',
sino, el programa le pedirá que ingrese la dirección nuevamente hasta que esta sea la correcta.
---------------------------------------------

---------------------------------------------
### PASO 3 ###
Si se cumple lo estipulado en el paso 2, el programa comienza a revisar y cargar el archivo CSV,
entregándole un resumen de las filas y columnas del archivo, cálculos de estadísticos descrptivos
y las primeras cinco filas del CSV original.
---------------------------------------------

---------------------------------------------
### PASO 4 ###
Luego, se le pedirá que ingrese el número de la columna que representa a la variable de entrada (X); 
en nuestro caso, será la columna 1. Y enseguida, indique el número de columnas que representen la 
salida del sistema (ya sea una o más salidas), separadas por coma. (no puntos)
---------------------------------------------

---------------------------------------------
### PASO 5 ###
Luego de indicar al programa las variables de entrada y salida, se mostrará un resumen de los datos 
atípicos, límites inferior y superior para cada variable de entrada y salida escogidos anteriormente.
---------------------------------------------

---------------------------------------------
### PASO 6 ###
Después, se le pedirá que ingrese una nueva entrada acotado entre [3, 3.5] (sin que coincida con un dato
ya existente en el CSV y que éste no se encuentre fuera del rango establecido). Presione enter cuando 
haya ingresado el número y se creará un gráfico exploratorio entre la variable de entrada (X) y la primera 
salida (y) impuesta en el paso 4; el cual será guardado en una carpeta llamada 'resultados_modelacion' con
el propósito de no saturar la terminal de Python o del CMD en caso contrario.
---------------------------------------------

---------------------------------------------
### PASO 7 ###
Una vez finalizado el paso 6, se procede con la etapa de entrenamiento de cada uno de los modelos
de la entrada respecto a la primera salida (estanque 1), realizando la comparación de desempeño entre cada modelo de
regresión (Lineal, Polinomial y Árbol), calculando las métricas de error para los datos de entrenamiento
y prueba; se realiza el mismo procedimiento pero para la predicción de la nueva entrada impuesta en el 
paso 6, seleccionando aquel modelo de mejor representa a la predicción (es decir, dado el RMSE más pequeño).
Finalmente, se obtiene el gráfico comparativo entre los modelos de regresión con respecto a la primera salida, 
guardando el gráfico en la carpeta 'resultados_modelacion'.
---------------------------------------------

---------------------------------------------
### PASO 8 ###
Luego, se generará el gráfico exploratorio pero con la segunda salida impuesta en el paso 4, y se repite el 
mismo procedimiento descrito en el paso 7; otorgándonos el último gráfico de comparación entre modelos de
regresión para la segunda salida (estanque 2) y guardándolos en la misma carpeta.
---------------------------------------------

-----------------------------------------------
### PASO 9 ###
Por último, el programa nos muestra un resumen final de la o las salidas simuladas, indicando el mejor modelo
que representa la tendencia de los datos, cálculo del RMSE, la entrada nueva y el valor de la nueva predicción 
obtenido para esa nueva entrada. Además, se nos generan dos archivos .csv de las métricas y predicciones anteriores
y guardadas en la carpeta 'resultados_modelacion' para realizar un análisis exhausto finalizado el programa.
-----------------------------------------------



















