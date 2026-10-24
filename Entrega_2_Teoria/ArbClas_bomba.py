

""" Tarea 2: MODELACIÓN Y SIMULACIÓN DE SISTEMAS MECATRÓNICOS
     Árbol de clasificación aplicado al estado de una bomba centrífuga. """

""" Nombres de los estudiantes:   Rafael Palma Arenas
                                                    Matías Vásquez Pino

Profesores de teoría:    Mohammad Hosein Sabzalian
                                    Bastian Caris Meneses

Profesor de laboratorio:   Aaron Aguilar Cuevas """

# ===========================================================
# IMPLEMENTACIÓN DE LAS LIBRERÍAS
# ===========================================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_DATA = CARPETA_CODIGO.parent / "Resultados_ArbClas"
CARPETA_DATA.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. CARGA Y LECTURA DEL ARCHIVO CSV
# ============================================================

def cargar_csv() -> tuple[pd.DataFrame, Path]:
    """Solicita la ruta y carga un CSV detectando el separador."""

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

    # Limpiar espacios en los nombres de columnas.
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("El archivo CSV no contiene filas de datos.")

    if len(df.columns) < 3:
        raise ValueError(
            "El CSV debe contener al menos tres columnas: "
            "dos variables de entrada y una variable objetivo."
        )

    if df.columns.duplicated().any():
        raise ValueError("El CSV contiene nombres de columnas duplicados.")

    print("\nArchivo CSV cargado correctamente.")
    print(f"Ruta: {ruta}")
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print("\nPrimeras cinco filas:")
    print(df.head().to_string(index=False))

    return df, ruta


def seleccionar_columna(df: pd.DataFrame, mensaje: str) -> str:
    """Permite seleccionar una columna mediante su número o nombre exacto."""

    print(f"\n{mensaje}")
    for i, nombre in enumerate(df.columns, start=1):
        print(f"{i}. {nombre}")

    while True:
        eleccion = input("Ingrese el número o el nombre exacto de la columna: ").strip()

        # Primero se intenta interpretar la elección como número.
        try:
            indice = int(eleccion)
            if 1 <= indice <= len(df.columns):
                return str(df.columns[indice - 1])
            print("Número fuera de rango. Inténtelo nuevamente.")
            continue
        except ValueError:
            pass

        if eleccion in df.columns:
            return eleccion

        print("Selección no válida. Ingrese un número o nombre de columna existente.")


def seleccionar_variables(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Selecciona dos variables numéricas X y una variable objetivo y."""

    print("\nCOLUMNAS DISPONIBLES")
    for i, nombre in enumerate(df.columns, start=1):
        print(f"{i}. {nombre}")

    while True:
        texto = input(
            "\nIngrese los números de las DOS variables de entrada, separados por coma: "
        ).strip()

        try:
            indices = [int(parte.strip()) - 1 for parte in texto.split(",")]
        except ValueError:
            print("Entrada no válida. Ejemplo de formato: 1,2")
            continue

        if len(indices) != 2:
            print("Debe seleccionar exactamente dos columnas.")
            continue

        if any(i < 0 or i >= len(df.columns) for i in indices):
            print("Uno o ambos números están fuera de rango.")
            continue

        if indices[0] == indices[1]:
            print("Las dos variables de entrada deben ser diferentes.")
            continue

        nombres_X = [str(df.columns[i]) for i in indices]
        break

    while True:
        y_nombre = seleccionar_columna(
            df,
            "Seleccione la columna correspondiente a la variable objetivo y:"
        )
        if y_nombre in nombres_X:
            print("La variable objetivo no puede ser una de las variables de entrada.")
            continue
        break

    X = df[nombres_X].copy()
    y_original = df[y_nombre].copy()

    # Convertir las dos variables predictoras a valores numéricos.
    for columna in X.columns:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")

    # Eliminar filas con datos faltantes o no numéricos en X o y.
    datos = pd.concat([X, y_original.rename("__objetivo__")], axis=1)
    filas_iniciales = len(datos)
    datos = datos.dropna()
    filas_eliminadas = filas_iniciales - len(datos)

    if filas_eliminadas:
        print(
            f"\nAviso: se eliminaron {filas_eliminadas} filas con valores "
            "faltantes o valores no numéricos en las columnas seleccionadas."
        )

    X = datos[nombres_X].copy()
    y_original = datos["__objetivo__"].copy()

    if len(X) < 4:
        raise ValueError(
            "No hay suficientes filas válidas para dividir los datos "
            "en entrenamiento y prueba."
        )

    clases = list(pd.unique(y_original))
    if len(clases) != 2:
        raise ValueError(
            "La variable objetivo debe contener exactamente dos clases "
            "para este programa de clasificación binaria. "
            f"Clases detectadas: {clases}"
        )

    # Codificar las dos clases como 0 y 1, conservando sus nombres originales.
    # La segunda clase detectada se considera la clase positiva (1).
    y = y_original.map({clases[0]: 0, clases[1]: 1}).astype(int)

    print("\nSELECCIÓN DE VARIABLES")
    print(f"Variables de entrada X: {nombres_X}")
    print(f"Variable objetivo y: {y_nombre}")
    print(f"Clase 0: {clases[0]}")
    print(f"Clase 1 (positiva): {clases[1]}")
    print(f"Filas válidas: {len(X)}")

    return X, y, clases


# ============================================================
# 2. DIVISIÓN DE DATOS
# ============================================================

def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series
):
    """Divide los datos, manteniendo proporciones de clases cuando sea posible."""

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
        random_state=10,
        stratify=estratificar
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "El conjunto de entrenamiento contiene una sola clase. "
            "Agregue más observaciones de cada clase."
        )

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. ENTRENAMIENTO DEL ÁRBOL DE CLASIFICACIÓN
# ============================================================

def entrenar_modelo(X_train, y_train) -> DecisionTreeClassifier:
    """Crea y entrena el árbol de clasificación."""

    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=10
    )
    modelo.fit(X_train, y_train)
    return modelo

# ============================================================
# 4. EVALUACIÓN DEL MODELO
# ============================================================

def evaluar_modelo(modelo, X_test, y_test):
    """Calcula métricas y muestra la matriz de confusión."""

    clase_predicha = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)

    # Localizar la columna asociada a la clase positiva codificada como 1.
    indice_positivo = list(modelo.classes_).index(1)
    probabilidad_positiva = probabilidades[:, indice_positivo]

    accuracy = accuracy_score(y_test, clase_predicha)
    precision = precision_score(y_test, clase_predicha, pos_label=1, zero_division=0)
    recall = recall_score(y_test, clase_predicha, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, clase_predicha, pos_label=1, zero_division=0)
    matriz = confusion_matrix(y_test, clase_predicha, labels=[0, 1])

    print("\nRESULTADOS SOBRE LOS DATOS DE PRUEBA")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\nMATRIZ DE CONFUSIÓN")
    print(matriz)
    print("\nFilas = clase real (0, 1); columnas = clase predicha (0, 1).")

    return clase_predicha, probabilidad_positiva, matriz


# ============================================================
# 5. MOSTRAR Y GUARDAR LAS REGLAS DEL ÁRBOL
# ============================================================

def mostrar_reglas(modelo, nombres_X: list[str]) -> None:
    """Imprime las reglas aprendidas por el árbol."""

    reglas = export_text(
        modelo,
        feature_names=nombres_X,
        decimals=2
    )

    print("\nREGLAS APRENDIDAS POR EL ÁRBOL")
    print(reglas)


# ============================================================
# 6. GUARDAR LAS PREDICCIONES
# ============================================================

def guardar_predicciones(
    X_test,
    y_test,
    clase_predicha,
    probabilidad_positiva,
    clases
) -> Path:
    """Guarda datos de prueba, clases reales, predicciones y aciertos."""

    resultados = X_test.copy()
    resultados["clase_real_codificada"] = y_test
    resultados["clase_real"] = y_test.map({0: clases[0], 1: clases[1]})
    resultados["probabilidad_clase_positiva"] = probabilidad_positiva
    resultados["clase_predicha_codificada"] = clase_predicha
    resultados["clase_predicha"] = pd.Series(
        clase_predicha, index=X_test.index
    ).map({0: clases[0], 1: clases[1]})
    resultados["acierto"] = y_test.to_numpy() == clase_predicha

    ruta_predicciones = CARPETA_DATA / "predicciones_arbol_clasificacion.csv"
    resultados.to_csv(ruta_predicciones, index=False, encoding="utf-8-sig")

    print(f"\nPredicciones guardadas en: {ruta_predicciones}")
    return ruta_predicciones


# ============================================================
# 7. GRAFICAR LAS REGIONES DE CLASIFICACIÓN Y LA MATRIZ
# ============================================================

def graficar_resultados(
    modelo,
    X,
    X_test,
    y_test,
    matriz,
    clases
) -> Path:
    """Genera una gráfica de regiones y una matriz de confusión."""

    col_x1, col_x2 = list(X.columns)

    x1_min, x1_max = X[col_x1].min(), X[col_x1].max()
    x2_min, x2_max = X[col_x2].min(), X[col_x2].max()

    # Añadir un pequeño margen a los límites.
    margen_x1 = (x1_max - x1_min) * 0.05 or 1
    margen_x2 = (x2_max - x2_min) * 0.05 or 1
    x1_min, x1_max = x1_min - margen_x1, x1_max + margen_x1
    x2_min, x2_max = x2_min - margen_x2, x2_max + margen_x2

    x1_malla = np.linspace(x1_min, x1_max, 100)
    x2_malla = np.linspace(x2_min, x2_max, 100)
    tt, vv = np.meshgrid(x1_malla, x2_malla)

    malla = pd.DataFrame({
        col_x1: tt.ravel(),
        col_x2: vv.ravel()
    })
    clases_malla = modelo.predict(malla).reshape(tt.shape)

    plt.style.use("dark_background")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        facecolor="#030712",
        gridspec_kw={"width_ratios": [1.45, 1]}
    )

    # Gráfico de regiones de clasificación.
    ax = axes[0]
    ax.set_facecolor("#07111F")
    ax.contourf(
        tt,
        vv,
        clases_malla,
        levels=[-0.5, 0.5, 1.5],
        colors=["#082F49", "#4A044E"],
        alpha=0.58
    )

    configuracion_clases = [
        (0, str(clases[0]), "#00E5FF", "o"),
        (1, str(clases[1]), "#FF2BD6", "^"),    
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

    ax.set_title("REGIONES APRENDIDAS POR EL ÁRBOL", fontweight="bold")
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
        "ÁRBOL DE CLASIFICACIÓN DEL ESTADO DE UNA BOMBA CENTRIFUGA",
        fontsize=17,
        fontweight="bold",
        color="#F8FAFC"
    )
    fig.tight_layout()

    ruta_figura = CARPETA_CODIGO / "arbol_clasificacion_bomba.png"
    fig.savefig(
        ruta_figura,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    print(f"Figura guardada en: {ruta_figura}")
    return ruta_figura


# ============================================================
# 8. GRAFICAR LA ESTRUCTURA DEL ÁRBOL
# ============================================================

def graficar_arbol(modelo, nombres_X: list[str], clases) -> Path:
    """Guarda una visualización de la estructura del árbol."""

    fig_arbol, ax_arbol = plt.subplots(
        figsize=(18, 8),
        facecolor="#030712"
    )
    ax_arbol.set_facecolor("#030712")

    nodos = plot_tree(
        modelo,
        feature_names=["Temperatura [°C]", "Vibracion [mm/s]"],
        class_names=["Normal", "Falla"],
        filled=False,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=8,
        ax=ax_arbol
    )

    for nodo in nodos:
        nodo.set_color("#F8FAFC")
        caja = nodo.get_bbox_patch()
        if caja is not None:
            caja.set_facecolor("#07111F")
            caja.set_edgecolor("#94A3B8")
            caja.set_linewidth(1.3)

    ax_arbol.set_title(
        "ESTRUCTURA DEL ÁRBOL DE CLASIFICACIÓN",
        fontsize=17,
        fontweight="bold",
        color="#F8FAFC",
        pad=18
    )

    fig_arbol.tight_layout()
    ruta_estructura = CARPETA_CODIGO / "estructura_arbol_clasificacion.png"
    fig_arbol.savefig(
        ruta_estructura,
        dpi=160,
        bbox_inches="tight",
        facecolor=fig_arbol.get_facecolor()
    )
    plt.close(fig_arbol)

    print(f"Estructura del árbol guardada en: {ruta_estructura}")
    return ruta_estructura


# ============================================================
# 9. EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Coordina la ejecución completa del programa."""

    df, ruta_csv = cargar_csv()
    X, y, clases = seleccionar_variables(df)

    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    modelo = entrenar_modelo(X_train, y_train)

    clase_predicha, probabilidad_positiva, matriz = evaluar_modelo(
        modelo,
        X_test,
        y_test
    )

    mostrar_reglas(modelo, list(X.columns))

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

    graficar_arbol(modelo, list(X.columns), clases)

    print("\nPROGRAMA FINALIZADO CORRECTAMENTE.")
    print(f"Archivo analizado: {ruta_csv}")
    print(f"Resultados guardados en: {CARPETA_DATA}")

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"\nERROR: {error}")
    except Exception as error:
        print(f"\nSe produjo un error inesperado: {error}")
        raise
