from __future__ import annotations
import numpy as np
import skfuzzy as fuzz
import plotly.graph_objs as go
import plotly.subplots as sp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_EPS = 1e-9


# utilidades numéricas
def _pad_range(a: float, b: float, pad: float = 0.05) -> Tuple[float, float]:
    """Amplía un poco el dominio para dibujar curvas."""
    span = max(b - a, _EPS)
    return a - pad * span, b + pad * span


def _asegura_creciente(vals: Sequence[float]) -> np.ndarray:
    """Fuerza parámetros estrictamente crecientes."""
    arr = np.asarray(vals, dtype=float).copy()
    for i in range(1, arr.size):
        if not (arr[i] > arr[i - 1]):
            arr[i] = arr[i - 1] + _EPS
    return arr


# wrappers de funciones de membresía
def _trapmf(x: np.ndarray, abcd: Sequence[float]) -> np.ndarray:
    a, b, c, d = _asegura_creciente(abcd)
    return fuzz.trapmf(x, [a, b, c, d])


def _trimf(x: np.ndarray, abc: Sequence[float]) -> np.ndarray:
    a, b, c = _asegura_creciente(abc)
    return fuzz.trimf(x, [a, b, c])


def _smf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    a, b = _asegura_creciente([a, b])
    return fuzz.smf(x, a, b)  # S/Γ creciente


def _zmf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    a, b = _asegura_creciente([a, b])
    return fuzz.zmf(x, a, b)  # L/Z decreciente


def _pimf(x: np.ndarray, abcd: Sequence[float]) -> np.ndarray:
    a, b, c, d = _asegura_creciente(abcd)
    return fuzz.pimf(x, a, b, c, d)


def _gaussmf(x: np.ndarray, media: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), _EPS)
    return fuzz.gaussmf(x, media, sigma)


# parámetros sugeridos desde datos
def _estadisticos_basicos(arr: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Retorna (min, Q1, mediana, Q3, max)."""
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if np.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    try:
        Q1, Q2, Q3 = np.percentile(arr, [25, 50, 75])
    except Exception:
        Q1 = lo + 0.25 * (hi - lo)
        Q2 = lo + 0.50 * (hi - lo)
        Q3 = lo + 0.75 * (hi - lo)
    vals = _asegura_creciente([lo, Q1, Q2, Q3, hi])
    return vals[0], vals[1], vals[2], vals[3], vals[4]


def estimar_parametros(
    datos: Sequence[float],
    tipo: str = "trap",
) -> Dict[str, Tuple[float, ...]]:
    """Propone parámetros (tri, trap, smf, zmf, pimf, gauss) usando cuartiles/medias."""
    arr = np.asarray(datos, dtype=float)
    lo, Q1, Q2, Q3, hi = _estadisticos_basicos(arr)
    tipo = tipo.lower()

    if tipo in ("tri", "triangular"):
        return {"tri": (lo, Q2, hi)}
    if tipo in ("trap", "trapezoidal"):
        return {"trap": (lo, Q1, Q3, hi)}
    if tipo in ("gamma", "s", "smf"):
        return {"smf": (Q1, Q3)}
    if tipo in ("l", "z", "zmf"):
        return {"zmf": (Q1, Q3)}
    if tipo in ("pi", "pico", "pimf"):
        return {"pimf": (lo, Q1, Q3, hi)}
    if tipo in ("gauss", "gaussiana"):
        media = float(np.nanmean(arr))
        sigma = float(np.nanstd(arr, ddof=0))
        if sigma < _EPS:
            sigma = (hi - lo) / 6.0 if hi > lo else 1.0
        return {"gauss": (media, sigma)}

    raise ValueError("Tipo de membresía no soportado en estimación.")


# construcción de µ(x) para un conjunto
def construir_unverso(
    datos: Sequence[float],
    dominio: Optional[Tuple[float, float]] = None,
    n: int = 2001,
    pad: float = 0.08,
) -> np.ndarray:
    """Malla x para graficar y desdifuzificar."""
    arr = np.asarray(datos, dtype=float)
    lo = float(np.nanmin(arr)) if dominio is None else float(dominio[0])
    hi = float(np.nanmax(arr)) if dominio is None else float(dominio[1])
    lo, hi = _pad_range(lo, hi, pad)
    return np.linspace(lo, hi, int(n))

construir_universo = construir_unverso


def construir_membresia(
    x: np.ndarray,
    tipo: str,
    parametros: Sequence[float],
    invertir: bool = False,
) -> np.ndarray:
    """Retorna µ(x) según el tipo; si invertir=True usa 1-µ."""
    t = tipo.lower()
    if t in ("tri", "triangular"):
        mu = _trimf(x, parametros)
    elif t in ("trap", "trapezoidal"):
        mu = _trapmf(x, parametros)
    elif t in ("gamma", "s", "smf"):
        a, b = parametros
        mu = _smf(x, a, b)
    elif t in ("l", "z", "zmf"):
        a, b = parametros
        mu = _zmf(x, a, b)
    elif t in ("pi", "pico", "pimf"):
        mu = _pimf(x, parametros)
    elif t in ("gauss", "gaussiana"):
        media, sigma = parametros
        mu = _gaussmf(x, media, sigma)
    else:
        raise ValueError("Tipo de membresía no soportado.")
    return 1.0 - mu if invertir else mu


# fuzificación desde lista de datos
def fuzificar_lista(
    lista: Sequence[float],
    forma: str = "trap",
    dominio: Optional[Tuple[float, float]] = None,
) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crea L, M, H con cuartiles y devuelve curva combinada."""
    arr = np.asarray(lista, dtype=float)
    lo, Q1, Q2, Q3, hi = _estadisticos_basicos(arr)
    x = construir_universo(arr, dominio)

    if forma == "tri":
        L = _trimf(x, [lo, Q1, Q2])
        M = _trimf(x, [Q1, Q2, Q3])
        H = _trimf(x, [Q2, Q3, hi])
        titulo = "Funciones de membresía (triangulares L–M–H)"
    else:
        L = _trapmf(x, [lo, lo, Q1, Q2])
        M = _trimf(x, [Q1, Q2, Q3])
        H = _trapmf(x, [Q2, Q3, hi, hi])
        titulo = "Funciones de membresía (L/H trapezoidales + M triangular)"

    comb = np.fmax(L, np.fmax(M, H))

    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("Bajo (L)", "Medio (M)", "Alto (H)"))
    fig.add_trace(go.Scatter(x=x, y=L, mode='lines', name='Bajo (L)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=M, mode='lines', name='Medio (M)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=H, mode='lines', name='Alto (H)'), row=3, col=1)
    fig.update_layout(height=720, title_text=titulo, template="plotly_dark", showlegend=True)
    fig.update_xaxes(title_text="Valor", row=3, col=1)
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="μ", row=r, col=1)

    return fig, x, L, M, H, comb


# fuzificación desde (min, prom, max)
def fuzificar_resumen(
    minimo: float,
    promedio: float,
    maximo: float,
    forma: str = "trap",
) -> Tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crea L, M, H con solo min–prom–max y devuelve la curva combinada."""
    minimo, promedio, maximo = float(minimo), float(promedio), float(maximo)
    if minimo > maximo:
        minimo, maximo = maximo, minimo
    promedio = float(np.clip(promedio, minimo, maximo))

    Q1 = (2 * minimo + promedio) / 3.0
    Q2 = promedio
    Q3 = (2 * maximo + promedio) / 3.0

    lo, hi = _pad_range(minimo, maximo, pad=0.08)
    x = np.linspace(lo, hi, 1000)

    if forma == "tri":
        L = _trimf(x, [lo, Q1, Q2])
        M = _trimf(x, [Q1, Q2, Q3])
        H = _trimf(x, [Q2, Q3, hi])
        titulo = "Funciones de membresía (min, prom, max) – Triangular"
    else:
        L = _trapmf(x, [lo, lo, Q1, Q2])
        M = _trimf(x, [Q1, Q2, Q3])
        H = _trapmf(x, [Q2, Q3, hi, hi])
        titulo = "Funciones de membresía (min, prom, max) – Trap+Tri"

    comb = np.fmax(L, np.fmax(M, H))

    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("Bajo (L)", "Medio (M)", "Alto (H)"))
    fig.add_trace(go.Scatter(x=x, y=L, mode='lines', name='Bajo (L)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=M, mode='lines', name='Medio (M)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=H, mode='lines', name='Alto (H)'), row=3, col=1)
    fig.update_layout(height=720, title_text=titulo, template="plotly_dark", showlegend=True)
    fig.update_xaxes(title_text="Valor", row=3, col=1)
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="μ", row=r, col=1)

    return fig, x, L, M, H, comb


# desdifuzificación
def desdifuzificar(x: np.ndarray, mu: np.ndarray, metodo: str = "centroid") -> float:
    """Retorna el valor crisp según el método."""
    metodo = metodo.lower()
    if metodo not in ["centroid", "bisector", "mom", "som", "lom"]:
        raise ValueError("Método de desdifusificación no válido.")
    return float(fuzz.defuzz(x, mu, metodo))


# membresía por tramos 
def membresia_por_tramos(
    puntos: Sequence[Tuple[float, float]],
    n: int = 4001,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpola linealmente una µ(x) a partir de puntos (x,µ)."""
    pts = np.asarray(puntos, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    xlo, xhi = float(pts[0, 0]), float(pts[-1, 0])
    x = np.linspace(xlo, xhi, int(n))
    y = np.interp(x, pts[:, 0], pts[:, 1], left=0.0, right=0.0)
    y = np.clip(y, 0.0, 1.0)
    return x, y


# clase de trabajo para un conjunto
@dataclass
class ConjuntoDifuso:
    nombre: str
    datos: Sequence[float]
    tipo_membresia: str = "trap"
    parametros: Optional[Sequence[float]] = None
    invertir: bool = False
    dominio: Optional[Tuple[float, float]] = None
    escala: float = 1.0          # multiplicador a los datos
    desplazamiento: float = 0.0  # suma a los datos
    x: np.ndarray = field(init=False, default=None)
    mu: np.ndarray = field(init=False, default=None)
    parametros_usados: Tuple[float, ...] = field(init=False, default=None)

    def _prepara_datos(self) -> np.ndarray:
        """Aplica escala y desplazamiento a los datos crudos."""
        arr = np.asarray(self.datos, dtype=float) * self.escala + self.desplazamiento
        return arr

    def construir(self, n: int = 4001) -> Tuple[np.ndarray, np.ndarray]:
        """Construye µ(x) con parámetros manuales o sugeridos."""
        arr = self._prepara_datos()
        self.x = construir_universo(arr, self.dominio, n=n)
        if self.parametros is None:
            sugeridos = estimar_parametros(arr, self.tipo_membresia)
            self.parametros_usados = tuple(next(iter(sugeridos.values())))
            clave = next(iter(sugeridos.keys()))
            tipo_real = {
                "tri": "tri",
                "trap": "trap",
                "smf": "gamma",
                "zmf": "l",
                "pimf": "pi",
                "gauss": "gauss",
            }[clave]
            mu = construir_membresia(self.x, tipo_real, self.parametros_usados, invertir=self.invertir)
        else:
            self.parametros_usados = tuple(self.parametros)
            mu = construir_membresia(self.x, self.tipo_membresia, self.parametros, invertir=self.invertir)
        self.mu = mu
        return self.x, self.mu

    def desdifuzificar(self, metodo: str = "centroid") -> float:
        """Calcula el crisp del conjunto."""
        if self.x is None or self.mu is None:
            self.construir()
        return desdifuzificar(self.x, self.mu, metodo)

    def figura(self, titulo_extra: str = "") -> go.Figure:
        """Figura Plotly con µ(x) del conjunto."""
        if self.x is None or self.mu is None:
            self.construir()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.mu, mode="lines", name=self.nombre))
        ttl = f"Conjunto {self.nombre} – µ({self.tipo_membresia})"
        if titulo_extra:
            ttl += f" · {titulo_extra}"
        fig.update_layout(height=380, template="plotly_dark", title_text=ttl,
                          xaxis_title="Valor", yaxis_title="μ")
        return fig


# evaluación de varios conjuntos y agregación
def evaluar_multiconjuntos(
    conjuntos: List[ConjuntoDifuso],
    metodo: str = "centroid",
    pesos: Optional[Sequence[float]] = None,
    agregacion: str = "media",
) -> Tuple[Dict[str, float], float]:
    """Retorna (crisp_por_conjunto, crisp_global) con media o ponderación."""
    crisp: Dict[str, float] = {}
    for c in conjuntos:
        crisp[c.nombre] = c.desdifuzificar(metodo=metodo)

    if agregacion == "media":
        global_crisp = float(np.mean(list(crisp.values()))) if crisp else np.nan
    elif agregacion == "ponderada":
        if pesos is None or len(pesos) != len(conjuntos):
            raise ValueError("Pesos inválidos para agregación ponderada.")
        valores = np.array([crisp[c.nombre] for c in conjuntos], dtype=float)
        w = np.asarray(pesos, dtype=float)
        if np.allclose(w.sum(), 0.0):
            raise ValueError("La suma de pesos no puede ser cero.")
        w = w / w.sum()
        global_crisp = float(np.dot(w, valores))
    else:
        raise ValueError("Agregación no soportada.")

    return crisp, global_crisp
