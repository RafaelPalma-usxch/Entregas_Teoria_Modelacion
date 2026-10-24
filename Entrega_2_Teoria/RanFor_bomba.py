"""
Tarea 2: MODELACIÓN Y SIMULACIÓN DE SISTEMAS MECATRÓNICOS
Clasificación del estado de una bomba centrífuga mediante Random Forest. """

""" Nombres de los estudiantes:   Rafael Palma Arenas
                                                    Matías Vásquez Pino

Profesores de teoría:    Mohammad Hosein Sabzalian
                                    Bastian Caris Meneses

Profesor de laboratorio:   Aaron Aguilar Cuevas """

### IMPORTACIÓN DE LAS LIBRERÍAS NECESARIAS.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_DATA = CARPETA_CODIGO.parent / "Resultados_RanFor"
CARPETA_DATA.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGA Y LECTURA DEL ARCHIVO CSV
# ============================================================

def cargar_csv() -> tuple[pd.DataFrame, Path]:
    """Solicita la ruta del CSV, lo lee y comprueba su estructura básica."""

    texto_ruta = input("Ingrese la ruta del archivo CSV: ").strip().strip('"').strip("'")
    if not texto_ruta:
        raise ValueError("No se indicó una ruta para el archivo CSV.")

    ruta = Path(texto_ruta).expanduser()
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}\nVerifique la ruta e inténtelo nuevamente."
        )

    df = pd.read_csv(
        ruta,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("El archivo CSV no contiene filas de datos.")
    if len(df.columns) < 3:
        raise ValueError(
            "El CSV debe tener al menos tres columnas: dos variables de entrada "
            "y una variable objetivo."
        )
    if df.columns.duplicated().any():
        raise ValueError("El CSV contiene nombres de columnas duplicados.")

    print("\nArchivo CSV cargado correctamente.")
    print(f"Ruta: {ruta}")
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    print("\nColumnas disponibles:")
    for i, columna in enumerate(df.columns, start=1):
        print(f"{i}. {columna}")

    print("\nPrimeras cinco filas:")
    print(df.head().to_string(index=False))
    return df, ruta


# ============================================================
# 2. SELECCIÓN DE LAS DOS VARIABLES X Y LA VARIABLE OBJETIVO y
# ============================================================

def seleccionar_variables(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Permite seleccionar dos entradas numéricas y una salida binaria."""

    while True:
        texto = input(
            "\nIngrese los números de las DOS variables de entrada, separados por coma: "
        ).strip()
        try:
            indices_x = [int(parte.strip()) - 1 for parte in texto.split(",")]
        except ValueError:
            print("Entrada no válida. Ejemplo: 1,2")
            continue

        if len(indices_x) != 2:
            print("Debe seleccionar exactamente dos columnas.")
            continue
        if any(i < 0 or i >= len(df.columns) for i in indices_x):
            print("Uno o ambos números están fuera de rango.")
            continue
        if indices_x[0] == indices_x[1]:
            print("Las dos variables de entrada deben ser diferentes.")
            continue
        break

    nombres_x = [df.columns[i] for i in indices_x]

    while True:
        texto_y = input(
            "\nIngrese el número de la columna correspondiente a la variable objetivo y: "
        ).strip()
        try:
            indice_y = int(texto_y) - 1
        except ValueError:
            print("Ingrese un número de columna válido.")
            continue

        if indice_y < 0 or indice_y >= len(df.columns):
            print("El número está fuera de rango.")
            continue
        if indice_y in indices_x:
            print("La variable objetivo no puede ser una de las variables de entrada.")
            continue
        break

    nombre_y = df.columns[indice_y]
    X = df[nombres_x].copy()
    y_original = df[nombre_y].copy()

    # Convertir las entradas a números; los valores no convertibles se tratarán como faltantes.
    for columna in X.columns:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")

    # Quitar filas con valores faltantes en las variables elegidas.
    datos_seleccionados = pd.concat(
        [X, y_original.rename("__objetivo_original__")],
        axis=1
    )
    filas_originales = len(datos_seleccionados)
    datos_seleccionados = datos_seleccionados.dropna()
    eliminadas = filas_originales - len(datos_seleccionados)

    if eliminadas:
        print(
            f"\nAviso: se eliminaron {eliminadas} filas con valores faltantes "
            "o entradas no numéricas."
        )

    X = datos_seleccionados[nombres_x].copy()
    y_original = datos_seleccionados["__objetivo_original__"].copy()

    if len(X) < 4:
        raise ValueError("Hay muy pocas filas válidas para entrenar y evaluar el modelo.")

    clases = list(pd.unique(y_original))
    if len(clases) != 2:
        raise ValueError(
            "Este programa requiere una variable objetivo binaria con exactamente "
            f"dos clases. Se encontraron: {clases}"
        )

    # Codificar las clases como 0 y 1 para calcular las métricas de forma consistente.
    # La primera clase encontrada será 0 y la segunda será 1 (clase positiva).
    mapa_clases = {clases[0]: 0, clases[1]: 1}
    y = y_original.map(mapa_clases).astype(int)

    print("\nSelección de variables:")
    print(f"X: {nombres_x}")
    print(f"y: {nombre_y}")
    print(f"Clase 0: {clases[0]}")
    print(f"Clase 1 (clase positiva para las métricas): {clases[1]}")
    print(f"Filas válidas: {len(X)}")

    return X, y, clases


# ============================================================
# 3. DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# ============================================================

def dividir_datos(X: pd.DataFrame, y: pd.Series):
    """Divide los datos en 75 % entrenamiento y 25 % prueba."""

    conteos = y.value_counts()
    estratificar = y if len(conteos) == 2 and conteos.min() >= 2 else None

    if estratificar is None:
        print(
            "\nAviso: no se aplicará estratificación porque alguna clase "
            "tiene menos de dos observaciones."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=estratificar
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "El conjunto de entrenamiento quedó con una sola clase. "
            "Se necesitan más observaciones de ambas clases."
        )
    if y_test.nunique() < 2:
        print(
            "\nAviso: el conjunto de prueba contiene una sola clase; "
            "algunas métricas podrían ser menos representativas."
        )

    print("\nDivisión de datos:")
    print(f"Entrenamiento: {len(X_train)} filas")
    print(f"Prueba: {len(X_test)} filas")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. ENTRENAMIENTO DEL MODELO RANDOM FOREST
# ============================================================

def entrenar_modelo(X_train, y_train) -> RandomForestClassifier:
    """Entrena un bosque aleatorio de árboles de decisión."""

    modelo = RandomForestClassifier(
        n_estimators=100,  # Número de árboles del bosque
        max_depth=4,       # Profundidad máxima de cada árbol
        random_state=10,
        n_jobs=-1           # Usa los procesadores disponibles
    )
    modelo.fit(X_train, y_train)
    return modelo


# ============================================================
# 5. PREDICCIONES Y MÉTRICAS DE EVALUACIÓN
# ============================================================

def evaluar_modelo(modelo, X_test, y_test):
    """Calcula métricas, predicciones y matriz de confusión."""

    clase_predicha = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)

    # Obtener la columna de probabilidad correspondiente a la clase 1.
    indice_clase_positiva = list(modelo.classes_).index(1)
    probabilidad_positiva = probabilidades[:, indice_clase_positiva]

    accuracy = accuracy_score(y_test, clase_predicha)
    precision = precision_score(
        y_test, clase_predicha, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, clase_predicha, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, clase_predicha, pos_label=1, zero_division=0
    )
    matriz = confusion_matrix(y_test, clase_predicha, labels=[0, 1])

    print("\nRESULTADOS SOBRE LOS DATOS DE PRUEBA")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\nMATRIZ DE CONFUSION")
    print(matriz)
    print("Filas = clase real (0, 1); columnas = clase predicha (0, 1).")

    return clase_predicha, probabilidad_positiva, matriz


# ============================================================
# 6. IMPORTANCIA DE LAS VARIABLES
# ============================================================

def mostrar_importancia(modelo, nombres_x: list[str]) -> None:
    """Muestra la importancia relativa estimada para cada entrada."""

    print("\nIMPORTANCIA DE LAS VARIABLES")
    importancias = sorted(
        zip(nombres_x, modelo.feature_importances_),
        key=lambda elemento: elemento[1],
        reverse=True
    )
    for variable, importancia in importancias:
        print(f"{variable:<25}: {importancia:.3f}")


# ============================================================
# 7. GUARDAR LAS PREDICCIONES
# ============================================================

def guardar_predicciones(
    X_test,
    y_test,
    clase_predicha,
    probabilidad_positiva,
    clases
) -> Path:
    """Guarda las predicciones del conjunto de prueba en un CSV."""

    resultados = X_test.copy()
    resultados["clase_real_codificada"] = y_test
    resultados["estado_real"] = y_test.map({0: clases[0], 1: clases[1]})
    resultados["probabilidad_clase_positiva"] = probabilidad_positiva
    resultados["clase_predicha_codificada"] = clase_predicha
    resultados["estado_predicho"] = pd.Series(
        clase_predicha, index=X_test.index
    ).map({0: clases[0], 1: clases[1]})
    resultados["acierto"] = y_test.to_numpy() == clase_predicha

    ruta_predicciones = CARPETA_DATA / "predicciones_ranfor_clasificacion.csv"
    resultados.to_csv(
        ruta_predicciones,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nPredicciones guardadas en: {ruta_predicciones}")
    return ruta_predicciones


# ============================================================
# 8. GRÁFICA DE REGIONES DE CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================

def graficar_resultados(
    modelo,
    X,
    X_test,
    y_test,
    matriz,
    clases
) -> Path:
    """Grafica las regiones de clasificación y la matriz de confusión."""

    col_x1, col_x2 = list(X.columns)

    x1_min, x1_max = X[col_x1].min(), X[col_x1].max()
    x2_min, x2_max = X[col_x2].min(), X[col_x2].max()

    margen_x1 = (x1_max - x1_min) * 0.05 or 1
    margen_x2 = (x2_max - x2_min) * 0.05 or 1
    x1_min, x1_max = x1_min - margen_x1, x1_max + margen_x1
    x2_min, x2_max = x2_min - margen_x2, x2_max + margen_x2

    x1_malla = np.linspace(x1_min, x1_max, 140)
    x2_malla = np.linspace(x2_min, x2_max, 140)
    tt, vv = np.meshgrid(x1_malla, x2_malla)

    malla = pd.DataFrame({
        col_x1: tt.ravel(),
        col_x2: vv.ravel()
    })

    probabilidades = modelo.predict_proba(malla)
    indice_clase_positiva = list(modelo.classes_).index(1)
    probabilidades_malla = probabilidades[:, indice_clase_positiva].reshape(tt.shape)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.2),
        facecolor="#030712",
        gridspec_kw={"width_ratios": [1.45, 1]}
    )

    # Regiones de probabilidad del bosque.
    ax = axes[0]
    ax.set_facecolor("#07111F")
    ax.contourf(
        tt,
        vv,
        probabilidades_malla,
        levels=[0, 0.5, 1],
        colors=["#082F49", "#4A044E"],
        alpha=0.58
    )
    # Dibujar el umbral de probabilidad 0.5 solo si hay valores a ambos lados.
    if np.nanmin(probabilidades_malla) <= 0.5 <= np.nanmax(probabilidades_malla):
        ax.contour(
            tt,
            vv,
            probabilidades_malla,
            levels=[0.5],
            colors=["#B6FF00"],
            linewidths=2
        )

    configuracion_clases = [
        (0, str(clases[0]), "#00E5FF", "o"),
        (1, str(clases[1]), "#FF2BD6", "^")
    ]

    for clase, etiqueta, color, marcador in [
        (0, "Normal", "#00E5FF", "o"),
        (1, "Falla", "#FF2BD6", "^"),
] :
        seleccion = y_test == clase
        ax.scatter(
            X_test.loc[seleccion, "temperatura_C"],
            X_test.loc[seleccion, "vibracion_mm_s"],
            color=color,
            marker=marcador,
            s=42,
            alpha=0.84,
            edgecolors="#FFFFFF",
            linewidths=0.45,
            label=etiqueta
        )

    ax.set_title("REGIONES DECIDIDAS POR EL BOSQUE", fontweight="bold")
    ax.set_xlabel("Temperatura [°C]")
    ax.set_ylabel("Vibracion [mm/s]")
    ax.grid(color="#334155", alpha=0.28)
    ax.legend(frameon=False, labelcolor="#E2E8F0")

    # Matriz de confusión.
    ax_matriz = axes[1]
    ax_matriz.set_facecolor("#07111F")
    ax_matriz.imshow(matriz, cmap="magma")

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
                fontweight="bold"
            )

    ax_matriz.set_xticks([0, 1], labels=["Normal", "Falla"])
    ax_matriz.set_yticks([0, 1], labels=["Normal", "Falla"])
    ax_matriz.set_xlabel("Clase predicha")
    ax_matriz.set_ylabel("Clase real")
    ax_matriz.set_title("MATRIZ DE CONFUSION", fontweight="bold", color="#F8FAFC")

    fig.suptitle(
        "RANDOM FOREST PARA CLASIFICAR EL ESTADO DE UNA BOMBA CENTRIFUGA",
        fontsize=17,
        fontweight="bold",
        color="#F8FAFC"
    )
    fig.tight_layout()

    ruta_figura = CARPETA_CODIGO / "random_forest_bomba.png"
    fig.savefig(
        ruta_figura,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    print(f"Figura guardada en: {ruta_figura}")
    return ruta_figura


# ============================================================
# 9. EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Ejecuta los bloques del programa en orden."""

    df, ruta_csv = cargar_csv()
    X, y, clases = seleccionar_variables(df)

    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    modelo = entrenar_modelo(X_train, y_train)

    clase_predicha, probabilidad_positiva, matriz = evaluar_modelo(
        modelo,
        X_test,
        y_test
    )

    mostrar_importancia(modelo, list(X.columns))

    guardar_predicciones(
        X_test,
        y_test,
        clase_predicha,
        probabilidad_positiva,
        clases
    )

    graficar_resultados(
        modelo,
        X,
        X_test,
        y_test,
        matriz,
        clases
    )

    print("\nPROGRAMA FINALIZADO CORRECTAMENTE.")
    print(f"Archivo analizado: {ruta_csv}")
    print(f"Resultados guardados en: {CARPETA_DATA}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"\nERROR: {error}")
