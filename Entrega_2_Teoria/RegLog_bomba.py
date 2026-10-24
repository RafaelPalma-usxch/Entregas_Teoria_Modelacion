
""" Tarea 2: MODELACIÓN Y SIMULACIÓN DE SISTEMAS MECATRÓNICOS 
      Regresión logística aplicada al estado de una bomba centrífuga."""

""" Nombres de los estudiantes:   Rafael Palma Arenas
                                                    Matías Vásquez Pino

Profesores de teoría:    Mohammad Hosein Sabzalian
                                    Bastian Caris Meneses

Profesor de laboratorio:   Aaron Aguilar Cuevas """

# ===========================================================
# 0. PREÁMBULO (IDENTIFICAR LAS LIBRERÍAS A UTILIZAR)
# ===========================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CARGA Y LECTURA DEL ARCHIVO CSV
# ============================================================

# Solicitar la ruta del archivo CSV
ruta_csv = input("Ingrese la ruta del archivo CSV: ").strip().strip('"')

try:
    # Leer el archivo y detectar automáticamente el separador
    df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")

    # Limpiar espacios en los nombres de las columnas
    df.columns = df.columns.astype(str).str.strip()

    # Verificar que existan datos y al menos tres columnas
    if df.empty:
        raise ValueError("El archivo CSV no contiene datos.")

    if len(df.columns) < 3:
        raise ValueError(
            "El archivo debe contener dos variables de entrada "
            "y una variable objetivo."
        )

    # Mostrar las columnas disponibles
    print("\nColumnas disponibles:")
    for i, columna in enumerate(df.columns, start=1):
        print(f"{i}. {columna}")

    # Seleccionar las dos variables de entrada X
    indices_X = input(
        "\nIngrese los números de las dos variables de entrada, separados por coma: "
    ).split(",")

    indices_X = [int(i.strip()) - 1 for i in indices_X]

    if len(indices_X) != 2 or any(
        i < 0 or i >= len(df.columns) for i in indices_X
    ):
        raise ValueError("Debe seleccionar exactamente dos columnas válidas.")

    if indices_X[0] == indices_X[1]:
        raise ValueError("Las dos variables de entrada deben ser diferentes.")

    # Seleccionar la variable objetivo y
    indice_y = int(
        input("Ingrese el número de la variable objetivo y: ").strip()
    ) - 1

    if indice_y < 0 or indice_y >= len(df.columns):
        raise ValueError("La columna objetivo seleccionada no es válida.")

    if indice_y in indices_X:
        raise ValueError("La variable objetivo no puede ser una variable de entrada.")

    # Construir X e y para la regresión logística
    X = df.iloc[:, indices_X].copy()
    y = df.iloc[:, indice_y].copy()

    # Verificar que las variables de entrada sean numéricas
    X = X.apply(pd.to_numeric, errors="raise")

    # Comprobar datos faltantes
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("El archivo contiene valores faltantes en X o y.")

    # Mostrar el resultado
    print("\nArchivo CSV cargado correctamente.")
    print(f"Filas: {len(df)}")
    print(f"Variables de entrada X: {list(X.columns)}")
    print(f"Variable objetivo y: {y.name}")

    print("\nPrimeras cinco filas de X:")
    print(X.head())

    print("\nPrimeros cinco valores de y:")
    print(y.head())

except FileNotFoundError:
    print("Error: no se encontró el archivo. Verifique la ruta.")
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    print(f"Error al cargar o validar los datos: {error}")

N_MUESTRAS = 260
CARPETA_SALIDA = Path(__file__).parent
CARPETA_DATA = CARPETA_SALIDA.parent / "Resultados_RegLog"
CARPETA_DATA.mkdir(exist_ok=True)

# La semilla hace que el ejemplo entregue siempre los mismos resultados.
rng = np.random.default_rng(12)

temperatura = np.clip(rng.normal(65, 12, N_MUESTRAS), 35, 100)
vibracion = np.clip(rng.normal(4.5, 2.0, N_MUESTRAS), 0.5, 10)

# Esta ecuacion se usa para fabricar el conjunto de ejemplo, no para entrenar.
z = -8.5 + 0.07 * temperatura + 0.85 * vibracion
probabilidad_falla = 1 / (1 + np.exp(-z))

# Cada medicion recibe una clase conocida: 0 normal o 1 falla.
estado = rng.binomial(1, probabilidad_falla)

datos = pd.DataFrame(
    {
        "temperatura_C": temperatura,
        "vibracion_mm_s": vibracion,
        "estado": estado,
    }
)
datos["etiqueta"] = datos["estado"].map({0: "normal", 1: "falla"})


# X contiene las mediciones; y contiene la respuesta que el modelo debe aprender.
X = datos[["temperatura_C", "vibracion_mm_s"]]
y = datos["estado"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=10,
    stratify=y,  # conserva una proporcion similar de normales y fallas
)

datos["conjunto"] = "entrenamiento"
datos.loc[X_test.index, "conjunto"] = "prueba"

ruta_datos = CARPETA_DATA / "datos_reglog_clasificacion.csv"
datos.to_csv(ruta_datos, index=False)


modelo = LogisticRegression(max_iter=1000)
# fit ajusta beta_0, beta_1 y beta_2 usando los ejemplos de entrenamiento.
modelo.fit(X_train, y_train)


# predict entrega la clase; predict_proba conserva la probabilidad de falla.
clase_predicha = modelo.predict(X_test)
probabilidad_predicha = modelo.predict_proba(X_test)[:, 1]


# Todas las metricas comparan la clase real con la clase predicha en prueba.
accuracy = accuracy_score(y_test, clase_predicha)
precision = precision_score(y_test, clase_predicha)
recall = recall_score(y_test, clase_predicha)
f1 = f1_score(y_test, clase_predicha)
matriz = confusion_matrix(y_test, clase_predicha)

print("\nRESULTADOS SOBRE LOS DATOS DE PRUEBA")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

print("\nMATRIZ DE CONFUSION")
print(matriz)


resultados_prueba = X_test.copy()
resultados_prueba["estado_real"] = y_test
resultados_prueba["probabilidad_falla"] = probabilidad_predicha
resultados_prueba["estado_predicho"] = clase_predicha
resultados_prueba["acierto"] = y_test.to_numpy() == clase_predicha
resultados_prueba = resultados_prueba.sort_values("temperatura_C")

ruta_predicciones = CARPETA_DATA / "predicciones_reglog_clasificacion.csv"
resultados_prueba.to_csv(ruta_predicciones, index=False)


plt.style.use("dark_background")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.2),
    facecolor="#030712",
    gridspec_kw={"width_ratios": [1.45, 1]},
)

ax = axes[0]
ax.set_facecolor("#07111F")

temperaturas_malla = np.linspace(35, 100, 180)
vibraciones_malla = np.linspace(0.5, 10, 180)

# La malla permite consultar qué clase elegiria el modelo en toda la figura.
tt, vv = np.meshgrid(temperaturas_malla, vibraciones_malla)

malla = pd.DataFrame(
    {
        "temperatura_C": tt.ravel(),
        "vibracion_mm_s": vv.ravel(),
    }
)
probabilidades_malla = modelo.predict_proba(malla)[:, 1].reshape(tt.shape)

# El cambio de color ocurre cuando la probabilidad de falla cruza 0.5.
ax.contourf(
    tt,
    vv,
    probabilidades_malla,
    levels=[0, 0.5, 1],
    colors=["#082F49", "#4A044E"],
    alpha=0.52,
)
ax.contour(
    tt,
    vv,
    probabilidades_malla,
    levels=[0.5],
    colors=["#FFFFFF"],
    linewidths=2,
)

for clase, etiqueta, color, marcador in [
    (0, "Normal", "#00E5FF", "o"),
    (1, "Falla", "#FF2BD6", "^"),
]:
    seleccion = y_test == clase
    ax.scatter(
        X_test.loc[seleccion, "temperatura_C"],
        X_test.loc[seleccion, "vibracion_mm_s"],
        color=color,
        marker=marcador,
        s=42,
        alpha=0.82,
        edgecolors="#FFFFFF",
        linewidths=0.45,
        label=etiqueta,
    )

ax.set_title("FRONTERA DE CLASIFICACION", fontweight="bold", color="#F8FAFC")
ax.set_xlabel("Temperatura [°C]")
ax.set_ylabel("Vibracion [mm/s]")
ax.grid(color="#334155", alpha=0.28)
ax.legend(frameon=False, labelcolor="#E2E8F0")


ax_matriz = axes[1]
ax_matriz.set_facecolor("#07111F")
imagen = ax_matriz.imshow(matriz, cmap="magma")

for fila in range(2):
    for columna in range(2):
        ax_matriz.text(
            columna,
            fila,
            str(matriz[fila, columna]),
            ha="center",
            va="center",
            color="#FFFFFF",
            fontsize=20,
            fontweight="bold",
        )

ax_matriz.set_xticks([0, 1], labels=["Normal", "Falla"])
ax_matriz.set_yticks([0, 1], labels=["Normal", "Falla"])
ax_matriz.set_xlabel("Clase predicha")
ax_matriz.set_ylabel("Clase real")
ax_matriz.set_title("MATRIZ DE CONFUSION", fontweight="bold", color="#F8FAFC")

fig.suptitle(
    "REGRESIÓN LOGÍSTICA PARA CLASIFICAR EL ESTADO DE UNA BOMBA CENTRIFUGA",
    fontsize=17,
    fontweight="bold",
    color="#F8FAFC",
)
fig.tight_layout()

ruta_figura = CARPETA_SALIDA / "regresion_logistica_bomba.png"
fig.savefig(
    ruta_figura,
    dpi=200,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

print(f"\nDatos guardados en: {ruta_datos}")
print(f"Predicciones guardadas en: {ruta_predicciones}")
print(f"Figura guardada en: {ruta_figura}")

plt.show()
