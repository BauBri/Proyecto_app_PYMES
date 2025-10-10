from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Dict, List, Union, Iterable

import pandas as pd


def _leer_tabla(archivo: Union[str, BytesIO, "UploadedFile"], sheet: Union[int, str, None] = 0) -> pd.DataFrame:
    # Detecta el tipo de archivo por la extensión
    nombre = getattr(archivo, "name", None)
    if nombre:
        nombre_l = nombre.lower()
        if nombre_l.endswith(".csv"):
            try:
                return pd.read_csv(archivo)
            except Exception:
                return pd.read_csv(archivo, sep=";", engine="python")
        else:
            return pd.read_excel(archivo, sheet_name=sheet)
    try:
        return pd.read_excel(archivo, sheet_name=sheet)
    except Exception:
        try:
            return pd.read_csv(archivo)
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo: {e}")


# Convierte una serie a flotantes, aceptando comas decimales y vacíos.
def _serie_a_flotantes(s: pd.Series) -> List[float]:
    # Reemplaza coma por punto y convierte a número.
    as_str = s.astype(str).str.strip().replace({"": None, "nan": None})
    normalizada = as_str.str.replace(",", ".", regex=False)
    nums = pd.to_numeric(normalizada, errors="coerce").dropna()
    return [float(x) for x in nums.tolist()]


# Elimina columnas que son índices exportados 
def _descartar_indices_exportados(df: pd.DataFrame) -> pd.DataFrame:
    cols_validas = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    return df[cols_validas]


# Interpreta de la forma que cada columna es un conjunto y cada fila es un dato.
def _parsear_formato_ancho(df: pd.DataFrame) -> "OrderedDict[str, List[float]]":
    if df is None or df.empty:
        raise ValueError("El archivo está vacío.")

    df = _descartar_indices_exportados(df)
    if df.empty:
        raise ValueError("No hay columnas válidas para procesar.")

    df = df.dropna(how="all")

    conjuntos: "OrderedDict[str, List[float]]" = OrderedDict()
    for col in df.columns:
        valores = _serie_a_flotantes(df[col])
        if valores:
            conjuntos[str(col)] = valores

    if not conjuntos:
        raise ValueError("No se encontraron valores numéricos válidos en ninguna columna.")

    return conjuntos


def leer_conjuntos_multi_atributo(
    archivo: Union[str, BytesIO, "UploadedFile"],
    sheet: Union[int, str, None] = 0,
) -> "OrderedDict[str, List[float]]":
    df = _leer_tabla(archivo, sheet=sheet)
    return _parsear_formato_ancho(df)



def a_lista_de_conjuntos(conjuntos: "Dict[str, List[float]]") -> List[dict]:
    # Para que sigan ordenados los datos
    if isinstance(conjuntos, OrderedDict):
        items = conjuntos.items()
    else:
        items = conjuntos.items()
    return [{"nombre": k, "valores": v} for k, v in items]
