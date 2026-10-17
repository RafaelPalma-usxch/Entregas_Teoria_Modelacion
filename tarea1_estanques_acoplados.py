#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tarea 1 --- Modelación y Simulación de Sistemas Mecatrónicos

Nombres de los estudiantes  --- Rafael Palma Arenas
                                            --- Matías Vásquez Pino

Profesores de teoría             --- Mohammad Hosein Sabzalian
                                            --- Bastian Caris Meneses

Profesor de laboratorio        --- Aaron Aguilar Cuevas
    

*** Modelo de un sistema de estanques acoplados que nos permite una variable de entrada y una o más variables de salida.
Para cada salida seleccionada, el programa ajusta tres modelos independientes:
  1. Regresión lineal.
  2. Árbol de regresión.
  3. Regresión polinomial.

*** Dependencias necesarias para correr el programa:
        pip install pandas numpy matplotlib scikit-learn

*** Cada salida se evalúa por separado, para que sus métricas tengan las unidades correctas y una salida de gran magnitud no domine el ajuste de otra salida.
        Los datos atípicos se informan, pero no se eliminan automáticamente.
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# 1. CONFIGURACIÓN DEL PROGRAMA
# =============================================================================
RUTA_CSV_PREDETERMINADA = "resultado_estanques.csv"

# Puedes indicar los nombres directamente o dejar None para escoger por consola.
# Ejemplo:
# COLUMNA_ENTRADA = "voltaje_V"
# COLUMNAS_SALIDA = ["velocidad_rad_s", "corriente_A"]
COLUMNA_ENTRADA = None
COLUMNAS_SALIDA = None      # None = selección por consola; también acepta una lista.

# None = pedir la nueva entrada por consola. Debe estar dentro del rango común
# de datos válidos y no coincidir con un valor registrado en la entrada.
NUEVA_ENTRADA = None

## Definimos los parámetros de los modelos a utlizar y se distribuyen los datos para entrenamiento y prueba
PROPORCION_PRUEBA = 0.20
SEMILLA_ALEATORIA = 33
GRADO_POLINOMIAL = 2
PROFUNDIDAD_MAXIMA_ARBOL = 3
MINIMO_MUESTRAS_POR_HOJA = 2
CARPETA_RESULTADOS = Path("resultados_modelacion")

# False mantiene la terminal limpia y guarda los PNG sin abrir ventanas emergentes.
# Cambia a True si deseas que cada gráfico también se abra durante la ejecución.
MOSTRAR_GRAFICOS = False

# Puedes definir etiquetas de ejes distintas de los nombres del CSV. Ejemplo:
# ETIQUETA_EJE_ENTRADA = "Voltaje aplicado (V)"
# ETIQUETAS_EJE_SALIDA = {"velocidad_rad_s": "Velocidad angular (rad/s)"}
ETIQUETA_EJE_ENTRADA = None
ETIQUETAS_EJE_SALIDA = {}


# =============================================================================
# COMANDOS UTILIZADOS PARA MOSTRAR RESULTADOS ORDENADOS EN LA TERMINAL
# =============================================================================
ANCHO_TERMINAL = 96


def imprimir_seccion(titulo: str, nivel: int = 1) -> None:
    """Imprime un título uniforme para distinguir las etapas del análisis."""
    if nivel == 1:
        print("\n" + "=" * ANCHO_TERMINAL)
        print(titulo.center(ANCHO_TERMINAL))
        print("=" * ANCHO_TERMINAL)
    else:
        print("\n" + "-" * ANCHO_TERMINAL)
        print(titulo)
        print("-" * ANCHO_TERMINAL)


def imprimir_tabla(
    tabla: pd.DataFrame,
    columnas: list[str] | None = None,
    decimales: int = 5,
) -> None:
    """Imprime una tabla legible sin requerir paquetes extra como tabulate."""
    vista = tabla.copy()
    if columnas is not None:
        vista = vista[columnas]
    print(vista.to_string(
        index=False,
        max_colwidth=32,
        float_format=lambda valor: f"{valor:.{decimales}g}",
    ))


def etiqueta_modelo_corta(nombre: str) -> str:
    """Acorta los nombres solo para la consola; los CSV conservan nombres completos."""
    if nombre == "Regresión lineal":
        return "Lineal"
    if nombre == "Árbol de regresión":
        return "Árbol"
    prefijo = "Regresión polinomial"
    if nombre.startswith(prefijo):
        detalle = nombre.replace(prefijo, "Polinomial", 1)
        return detalle.strip()
    return nombre


# =============================================================================
# 2. CARGA Y REVISIÓN DEL CSV
# =============================================================================
def cargar_csv() -> tuple[pd.DataFrame, Path]:
    """Pide la ruta del CSV y detecta separadores comunes automáticamente."""
    ruta_default = Path(RUTA_CSV_PREDETERMINADA)
    if ruta_default.exists():
        texto = input(
            f"Ruta del CSV [Enter para usar '{RUTA_CSV_PREDETERMINADA}']: "
        ).strip()
        ruta = Path(texto) if texto else ruta_default
    else:
        texto = input(
            f"No se encontró '{RUTA_CSV_PREDETERMINADA}'. "
            "Escribe la ruta del CSV: "
        ).strip()
        if not texto:
            raise FileNotFoundError(
                "No se indicó una ruta. Copia el CSV junto al programa o escribe "
                "su ruta completa."
            )
        ruta = Path(texto)

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo '{ruta}'. Verifica la ruta.")

    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    if df.empty:
        raise ValueError("El CSV no contiene filas de datos.")
    if len(df.columns) < 2:
        raise ValueError("El CSV debe contener al menos dos columnas.")
    return df, ruta


def mostrar_revision_inicial(df: pd.DataFrame) -> None:
    """Muestra el diagnóstico inicial del CSV en bloques compactos."""
    imprimir_seccion("ETAPA 1 | REVISIÓN INICIAL DEL CSV")
    print(f"Resumen del archivo: {df.shape[0]} filas y {df.shape[1]} columnas")
    print(f"Filas duplicadas completas: {int(df.duplicated().sum())}")

    imprimir_seccion("Columnas y calidad de datos", nivel=2)
    calidad = pd.DataFrame({
        "Columna": df.columns.astype(str),
        "Tipo": [str(tipo) for tipo in df.dtypes],
        "Faltantes": [int(valor) for valor in df.isna().sum()],
        "Valores únicos": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    imprimir_tabla(calidad, decimales=4)

    imprimir_seccion("Primeras cinco filas", nivel=2)
    print(df.head().to_string(index=False, max_cols=8))

    numericas = df.select_dtypes(include=[np.number])
    if not numericas.empty:
        imprimir_seccion("Estadísticos descriptivos de columnas numéricas", nivel=2)
        print(numericas.describe().round(5).to_string())
    else:
        print("\nAVISO: no se detectaron columnas numéricas automáticamente.")
        print("Verifica que las columnas de entrada y salida contengan datos numéricos.")

    if len(df) < 50:
        print("\nADVERTENCIA: el CSV tiene menos de 50 filas; el enunciado solicita al menos 50 simulaciones.")


def elegir_columna(df: pd.DataFrame, rol: str, configurada: str | None) -> str:
    """Valida una columna indicada en la configuración o seleccionada por consola."""
    if configurada is not None:
        if configurada not in df.columns:
            raise ValueError(
                f"La columna configurada para {rol} ('{configurada}') no existe. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        return configurada

    print(f"\nEscoge la columna correspondiente a {rol}:")
    for i, nombre in enumerate(df.columns, start=1):
        print(f"  {i}. {nombre}")

    while True:
        eleccion = input("Escribe el número de la columna o su nombre exacto: ").strip()
        if eleccion in df.columns:
            return eleccion
        try:
            indice = int(eleccion)
            if 1 <= indice <= len(df.columns):
                return str(df.columns[indice - 1])
        except ValueError:
            pass
        print("Selección no válida. Inténtalo de nuevo.")


def elegir_columnas_salida(
    df: pd.DataFrame, configuradas: str | list[str] | None
) -> list[str]:
    """Permite seleccionar una o varias salidas; además acepta índices separados por comas."""
    if configuradas is not None:
        salidas = [configuradas] if isinstance(configuradas, str) else list(configuradas)
        if not salidas:
            raise ValueError("Debes configurar al menos una columna de salida.")
        inexistentes = [nombre for nombre in salidas if nombre not in df.columns]
        if inexistentes:
            raise ValueError(
                f"Columnas de salida inexistentes: {inexistentes}. "
                f"Disponibles: {list(df.columns)}"
            )
        if len(set(salidas)) != len(salidas):
            raise ValueError("Hay nombres de salida repetidos en la configuración.")
        return salidas

    print("\nColumnas disponibles para salida:")
    for i, nombre in enumerate(df.columns, start=1):
        print(f"  {i}. {nombre}")
    print(
        "Selecciona una o varias columnas de salida con sus números separados por "
        "comas; por ejemplo: 2,3"
    )

    while True:
        texto = input("Columnas de salida: ").strip()
        partes = [parte.strip() for parte in texto.split(",") if parte.strip()]
        if not partes:
            print("Debes seleccionar al menos una columna.")
            continue

        # Se priorizan índices para que la selección de nombres con espacios sea simple.
        try:
            indices = [int(parte) for parte in partes]
            if all(1 <= indice <= len(df.columns) for indice in indices):
                salidas = [str(df.columns[indice - 1]) for indice in indices]
            else:
                print("Uno o más números están fuera de rango.")
                continue
        except ValueError:
            # También se aceptan nombres exactos separados por comas.
            salidas = partes
            if any(nombre not in df.columns for nombre in salidas):
                print("No reconozco todas las columnas. Usa los números indicados arriba.")
                continue

        if len(set(salidas)) != len(salidas):
            print("No repitas una misma columna de salida.")
            continue
        return salidas


# =============================================================================
# 3. LIMPIEZA Y PREPARACIÓN POR CADA SALIDA
# =============================================================================
def preparar_datos_salida(
    df: pd.DataFrame, col_entrada: str, col_salida: str
) -> pd.DataFrame:
    """Prepara entrada y una salida; solo excluye filas inválidas para ese análisis."""
    entrada = pd.to_numeric(df[col_entrada], errors="coerce")
    salida = pd.to_numeric(df[col_salida], errors="coerce")
    mascara = (
        entrada.notna()
        & salida.notna()
        & np.isfinite(entrada)
        & np.isfinite(salida)
    )
    datos = pd.DataFrame(
        {
            "x": entrada[mascara].astype(float).to_numpy(),
            "y": salida[mascara].astype(float).to_numpy(),
        }
    )
    excluidas = int((~mascara).sum())

    imprimir_seccion(f"PREPARACIÓN DE DATOS | SALIDA: {col_salida}", nivel=1)
    print(f"Entrada (X)       : {col_entrada}")
    print(f"Salida (y)        : {col_salida}")
    print(f"Filas originales  : {len(df)}")
    print(f"Filas excluidas   : {excluidas} (faltantes, no numéricas o infinitas)")
    print(f"Filas utilizadas  : {len(datos)}")

    if len(datos) < 10:
        raise ValueError(
            f"La salida '{col_salida}' tiene menos de 10 filas válidas. "
            "Revisa la columna antes de ajustar los modelos."
        )
    if datos["x"].nunique() < 2:
        raise ValueError(
            f"La entrada tiene menos de dos valores distintos para la salida '{col_salida}'."
        )
    if len(datos) < 50:
        print("ADVERTENCIA: quedan menos de 50 filas válidas para esta salida.")

    print("\nRevisión de posibles valores atípicos (regla IQR; no se eliminan automáticamente):")
    filas_atipicos = []
    for variable, nombre in (("x", col_entrada), ("y", col_salida)):
        q1 = datos[variable].quantile(0.25)
        q3 = datos[variable].quantile(0.75)
        iqr = q3 - q1
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cantidad = int(((datos[variable] < inferior) | (datos[variable] > superior)).sum())
        filas_atipicos.append({
            "Variable": nombre,
            "Posibles atípicos": cantidad,
            "Límite inferior": inferior,
            "Límite superior": superior,
        })
    imprimir_tabla(pd.DataFrame(filas_atipicos), decimales=5)
    return datos


def obtener_nueva_entrada(
    valores_entrada: np.ndarray, minimo: float, maximo: float
) -> float:
    """Solicita un valor no registrado, dentro del rango común a todas las salidas."""
    configurado = NUEVA_ENTRADA
    imprimir_seccion("ETAPA 3 | SELECCIÓN DE UNA ENTRADA NUEVA")
    print(f"Rango permitido común a las salidas: [{minimo:.6g}, {maximo:.6g}]")
    print("El valor no debe coincidir con ninguna entrada registrada en el CSV.")

    while True:
        if configurado is not None:
            try:
                candidato = float(configurado)
            except (TypeError, ValueError) as error:
                raise ValueError("NUEVA_ENTRADA debe ser un número.") from error
            configurado = None
        else:
            texto = input("Introduce la nueva entrada numérica: ").strip()
            try:
                candidato = float(texto.replace(",", "."))
            except ValueError:
                print("No se pudo interpretar el valor. Introduce un número.")
                continue

        if not np.isfinite(candidato):
            print("La entrada debe ser finita.")
            continue
        if candidato < minimo or candidato > maximo:
            print(f"Fuera del rango permitido: [{minimo:.6g}, {maximo:.6g}].")
            continue
        registrada = np.any(np.isclose(valores_entrada, candidato, rtol=1e-9, atol=1e-12))
        if registrada:
            print("Ese valor ya aparece en la entrada del CSV. Escoge otro.")
            continue
        return candidato


# =============================================================================
# 4. DEFINICIÓN Y EVALUACIÓN DE LOS MODELOS
# =============================================================================
def rmse(y_real: np.ndarray, y_predicho: np.ndarray) -> float:
    """Calcula RMSE; sus unidades son las mismas que las de la salida."""
    return float(np.sqrt(mean_squared_error(y_real, y_predicho)))


def construir_modelos() -> dict[str, object]:
    """Crea los tres modelos con complejidad controlada y semilla fija en el árbol."""
    return {
        "Regresión lineal": LinearRegression(),
        "Árbol de regresión": DecisionTreeRegressor(
            max_depth=PROFUNDIDAD_MAXIMA_ARBOL,
            min_samples_leaf=MINIMO_MUESTRAS_POR_HOJA,
            random_state=SEMILLA_ALEATORIA,
        ),
        f"Regresión polinomial (grado {GRADO_POLINOMIAL})": make_pipeline(
            PolynomialFeatures(degree=GRADO_POLINOMIAL, include_bias=False),
            LinearRegression(),
        ),
    }


def nombre_archivo_seguro(texto: str) -> str:
    """Convierte un nombre de variable en un fragmento seguro para nombres de archivo."""
    limpio = re.sub(r"[^A-Za-z0-9_-]+", "_", texto.strip()).strip("_")
    return limpio or "salida"


def guardar_grafico_exploratorio(
    datos: pd.DataFrame, etiqueta_x: str, etiqueta_y: str, salida: str
) -> None:
    """Guarda un gráfico de dispersión específico para cada variable de salida."""
    plt.figure(figsize=(9, 6))
    plt.scatter(datos["x"], datos["y"], alpha=0.8, label="Datos del CSV")
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.title(f"Relación exploratoria: {etiqueta_x} vs. {etiqueta_y}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    destino = CARPETA_RESULTADOS / f"01_relacion_entrada_salida_{nombre_archivo_seguro(salida)}.png"
    plt.savefig(destino, dpi=200, bbox_inches="tight")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close()
    print(f"Gráfico exploratorio guardado en: {destino}")


def analizar_una_salida(
    datos: pd.DataFrame,
    col_salida: str,
    entrada_nueva: float,
    etiqueta_x: str,
    etiqueta_y: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Ajusta y evalúa tres modelos para una salida y devuelve métricas/predicciones."""
    imprimir_seccion(f"ETAPA 4 | MODELOS Y MÉTRICAS | SALIDA: {col_salida}")
    n_train_pct = int((1 - PROPORCION_PRUEBA) * 100)
    n_test_pct = int(PROPORCION_PRUEBA * 100)
    print(f"Partición: {n_train_pct}% entrenamiento / {n_test_pct}% prueba | semilla: {SEMILLA_ALEATORIA}")
    print(f"Muestras válidas: {len(datos)} | Entrada: {etiqueta_x} | Salida: {etiqueta_y}")

    X = datos[["x"]]
    y = datos["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA_ALEATORIA, shuffle=True
    )

    modelos = construir_modelos()
    filas_metricas: list[dict[str, object]] = []
    for nombre, modelo in modelos.items():
        # Cada modelo aprende únicamente con entrenamiento. El conjunto de prueba
        # se reserva para evaluar el desempeño con observaciones no usadas en el ajuste.
        modelo.fit(X_train, y_train)
        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)
        filas_metricas.append(
            {
                "Salida": col_salida,
                "Modelo": nombre,
                "Filas_validas": len(datos),
                "Muestras_entrenamiento": len(X_train),
                "Muestras_prueba": len(X_test),
                "RMSE_entrenamiento": rmse(y_train.to_numpy(), pred_train),
                "RMSE_prueba": rmse(y_test.to_numpy(), pred_test),
                "MAE_prueba": float(mean_absolute_error(y_test, pred_test)),
                "R2_prueba": float(r2_score(y_test, pred_test)),
            }
        )

    metricas_locales = pd.DataFrame(filas_metricas).sort_values("RMSE_prueba").reset_index(drop=True)
    imprimir_seccion("Comparación de desempeño (ordenada por RMSE de prueba)", nivel=2)
    tabla_consola = metricas_locales[[
        "Modelo", "RMSE_entrenamiento", "RMSE_prueba", "MAE_prueba", "R2_prueba"
    ]].copy()
    tabla_consola["Modelo"] = tabla_consola["Modelo"].map(etiqueta_modelo_corta)
    tabla_consola = tabla_consola.rename(columns={
        "Modelo": "Modelo",
        "RMSE_entrenamiento": "RMSE train",
        "RMSE_prueba": "RMSE prueba",
        "MAE_prueba": "MAE prueba",
        "R2_prueba": "R² prueba",
    })
    tabla_consola.insert(0, "#", range(1, len(tabla_consola) + 1))
    imprimir_tabla(tabla_consola, decimales=6)
    print("\nCriterio inicial: el RMSE de prueba más bajo identifica al mejor candidato de esta partición.")
    print("Nota: RMSE y MAE conservan las unidades de la salida; R² es adimensional.")

    mejor_nombre = str(metricas_locales.iloc[0]["Modelo"])
    mejor_rmse = float(metricas_locales.iloc[0]["RMSE_prueba"])
    mejor_modelo = modelos[mejor_nombre]

    X_nueva = pd.DataFrame({"x": [entrada_nueva]})
    predicciones = {
        nombre: float(modelo.predict(X_nueva)[0]) for nombre, modelo in modelos.items()
    }
    prediccion_ganadora = predicciones[mejor_nombre]

    imprimir_seccion(f"Predicción para entrada nueva: {entrada_nueva:.7g} {etiqueta_x}", nivel=2)
    filas_prediccion_consola = []
    rmse_lookup = {str(fila["Modelo"]): float(fila["RMSE_prueba"]) for fila in filas_metricas}
    for nombre, valor in predicciones.items():
        filas_prediccion_consola.append({
            "Modelo": etiqueta_modelo_corta(nombre),
            "Predicción": valor,
            "RMSE prueba": rmse_lookup[nombre],
            "Selección": "MEJOR" if nombre == mejor_nombre else "",
        })
    imprimir_tabla(pd.DataFrame(filas_prediccion_consola), decimales=7)
    print(f"\nModelo recomendado por RMSE: {etiqueta_modelo_corta(mejor_nombre)}")
    print(f"Predicción principal para {col_salida}: {prediccion_ganadora:.8g} ({etiqueta_y})")

    # La curva se dibuja solo dentro del rango observado, sin extrapolarla. para evitar sobreajustes
    x_grid = np.linspace(float(datos["x"].min()), float(datos["x"].max()), 500)
    X_grid = pd.DataFrame({"x": x_grid})
    rmse_por_modelo = {str(fila["Modelo"]): float(fila["RMSE_prueba"]) for fila in filas_metricas}

    plt.figure(figsize=(10, 7))
    plt.scatter(datos["x"], datos["y"], alpha=0.75, label="Datos del CSV", zorder=3)
    for nombre, modelo in modelos.items():
        plt.plot(
            x_grid,
            modelo.predict(X_grid),
            linewidth=2,
            label=f"{nombre} (RMSE prueba={rmse_por_modelo[nombre]:.4g})",
        )
    plt.scatter(
        [entrada_nueva], [prediccion_ganadora], marker="*", s=240,
        edgecolors="black", linewidths=0.8,
        label=f"Predicción nueva: {mejor_nombre}", zorder=5,
    )
    plt.annotate(
        f"({entrada_nueva:.4g}, {prediccion_ganadora:.4g})",
        (entrada_nueva, prediccion_ganadora), textcoords="offset points", xytext=(8, 8),
    )
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.title(f"Comparación de modelos — salida: {col_salida}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    destino = CARPETA_RESULTADOS / f"02_comparacion_modelos_{nombre_archivo_seguro(col_salida)}.png"
    plt.savefig(destino, dpi=220, bbox_inches="tight")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close()
    print(f"Gráfico comparativo guardado en: {destino}")

    # Se añade una fila de predicciones por variable de salida. De este modo, el
    # CSV final incluye el ganador y los tres valores para la misma entrada.
    fila_pred: dict[str, object] = {
        "Salida": col_salida,
        "Entrada_nueva": entrada_nueva,
        "Modelo_mejor_RMSE": mejor_nombre,
        "RMSE_prueba_modelo_elegido": mejor_rmse,
        "Prediccion_modelo_elegido": prediccion_ganadora,
    }
    for nombre, valor in predicciones.items():
        fila_pred[f"Prediccion_{nombre_archivo_seguro(nombre)}"] = valor

    print("\nInterpretación rápida:")
    print("  • Si RMSE train es mucho menor que RMSE prueba, revisa posible sobreajuste.")
    print("  • El árbol genera tramos constantes; el polinomio depende del grado seleccionado.")
    print("  • Evita extrapolar fuera del rango simulado y valida con Simulink o teoría cuando sea posible.")
    return filas_metricas, fila_pred


# =============================================================================
# 5. FLUJO PRINCIPAL DEL PROGRAMA
# =============================================================================
def main() -> None:
    """Ejecuta el análisis para cada salida elegida y guarda resultados consolidados."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    imprimir_seccion("TAREA 1 | MODELACIÓN Y SIMULACIÓN DE SISTEMAS MECATRÓNICOS")
    print("Comparación de regresión lineal, árbol de regresión y regresión polinomial para un sistema de estanques acoplados.")

    df, ruta_csv = cargar_csv()
    print(f"\nArchivo cargado: {ruta_csv.resolve()}")
    mostrar_revision_inicial(df)

    col_x = elegir_columna(df, "la entrada (X)", COLUMNA_ENTRADA)
    columnas_salida = elegir_columnas_salida(df, COLUMNAS_SALIDA)
    if col_x in columnas_salida:
        raise ValueError("La columna de entrada no puede ser también una salida.")

    # Preparar cada salida por separado permite aprovechar todas las filas válidas
    # disponibles para esa variable si hubiera celdas faltantes en otras salidas.
    datos_por_salida: dict[str, pd.DataFrame] = {}
    for col_y in columnas_salida:
        datos_por_salida[col_y] = preparar_datos_salida(df, col_x, col_y)

    # Se limita la entrada nueva a la intersección de rangos válidos de todas las
    # salidas para que sea interpolación para cada modelo, no extrapolación.
    minimo_comun = max(float(datos["x"].min()) for datos in datos_por_salida.values())
    maximo_comun = min(float(datos["x"].max()) for datos in datos_por_salida.values())
    if minimo_comun >= maximo_comun:
        raise ValueError(
            "Las salidas seleccionadas no comparten un rango de entrada válido. "
            "No es posible definir una entrada nueva dentro del rango de todas ellas."
        )

    entrada_numerica = pd.to_numeric(df[col_x], errors="coerce").to_numpy(dtype=float)
    valores_entrada_validos = entrada_numerica[np.isfinite(entrada_numerica)]
    if valores_entrada_validos.size == 0:
        raise ValueError("La columna de entrada no contiene valores numéricos válidos.")

    entrada_nueva = obtener_nueva_entrada(
        valores_entrada_validos, minimo_comun, maximo_comun
    )
    etiqueta_x = ETIQUETA_EJE_ENTRADA or col_x

    metricas_totales: list[dict[str, object]] = []
    predicciones_totales: list[dict[str, object]] = []
    for col_y, datos in datos_por_salida.items():
        etiqueta_y = ETIQUETAS_EJE_SALIDA.get(col_y, col_y)
        guardar_grafico_exploratorio(datos, etiqueta_x, etiqueta_y, col_y)
        metricas, prediccion = analizar_una_salida(
            datos, col_y, entrada_nueva, etiqueta_x, etiqueta_y
        )
        metricas_totales.extend(metricas)
        predicciones_totales.append(prediccion)

    metricas_df = pd.DataFrame(metricas_totales).sort_values(
        ["Salida", "RMSE_prueba"], ascending=[True, True]
    )
    ruta_metricas = CARPETA_RESULTADOS / "metricas_modelos.csv"
    metricas_df.to_csv(ruta_metricas, index=False, encoding="utf-8-sig")

    predicciones_df = pd.DataFrame(predicciones_totales)
    ruta_predicciones = CARPETA_RESULTADOS / "predicciones_entradas_nuevas.csv"
    predicciones_df.to_csv(ruta_predicciones, index=False, encoding="utf-8-sig")
    
## Se imprime una sección final con los resultados obtenidos de la simulación completa ##
    imprimir_seccion("RESUMEN FINAL")
    resumen_filas = []
    for col_y in columnas_salida:
        subset = metricas_df[metricas_df["Salida"] == col_y].sort_values("RMSE_prueba")
        mejor = subset.iloc[0]
        prediccion_fila = predicciones_df[predicciones_df["Salida"] == col_y].iloc[0]
        resumen_filas.append({
            "Salida": col_y,
            "Mejor modelo": etiqueta_modelo_corta(str(mejor["Modelo"])),
            "RMSE prueba": float(mejor["RMSE_prueba"]),
            "Entrada nueva": float(entrada_nueva),
            "Predicción elegida": float(prediccion_fila["Prediccion_modelo_elegido"]),
        })
    imprimir_tabla(pd.DataFrame(resumen_filas), decimales=7)

    print("\nARCHIVOS GENERADOS")
    print(f"  1. Métricas       : {ruta_metricas.resolve()}")
    print(f"  2. Predicciones   : {ruta_predicciones.resolve()}")
    print(f"  3. Gráficos       : {CARPETA_RESULTADOS.resolve()}")
    print(f"\nListo: se analizaron {len(columnas_salida)} salida(s), con 3 modelos por salida.")
    print("Se generó un gráfico exploratorio y otro comparativo para cada salida seleccionada.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)

"" FIN DEL PROGRAMA ""
