import numpy as np      
import skfuzzy as fuzz     
import plotly.graph_objs as go   
import plotly.subplots as sp     

_EPS = 1e-9  

def _pad_range(a, b, pad=0.05):
    span = max(b - a, _EPS)
    return a - pad*span, b + pad*span

def _safe_trapmf(x, abcd):
    a,b,c,d = abcd
    b = max(b, a + _EPS)
    c = max(c, b + _EPS)
    d = max(d, c + _EPS)
    return fuzz.trapmf(x, [a,b,c,d])

def _safe_trimf(x, abc):
    a,b,c = abc
    b = max(b, a + _EPS)
    c = max(c, b + _EPS)
    return fuzz.trimf(x, [a,b,c])

# Fuzificación
def fuzificar_lista(lista, forma="trap"):
    """Genera L, M, H a partir de datos crudos usando Q1–Q2–Q3.
       forma: 'trap' (L/H trapecio + M triángulo) | 'tri' (todo triangular)
    """
    arr = np.asarray(lista, dtype=float)
    if arr.size == 0:
        raise ValueError("La lista no puede estar vacía.")
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError("Datos no finitos.")

    if np.isclose(min_val, max_val):
        min_val -= 0.5
        max_val += 0.5

    Q1, Q2, Q3 = np.percentile(arr, [25, 50, 75])
    if not (min_val <= Q1 <= Q2 <= Q3 <= max_val):
        Q1 = min_val + (max_val - min_val) * 0.25
        Q2 = min_val + (max_val - min_val) * 0.50
        Q3 = min_val + (max_val - min_val) * 0.75

    lo, hi = _pad_range(min_val, max_val, pad=0.08)
    x = np.linspace(lo, hi, 1000)

    if forma == "tri":
        L = _safe_trimf(x, [lo, Q1, Q2])
        M = _safe_trimf(x, [Q1, Q2, Q3])
        H = _safe_trimf(x, [Q2, Q3, hi])
        titulo = "Funciones de membresía (triangulares L–M–H)"
    else:  
        L = _safe_trapmf(x, [lo, lo, Q1, Q2])
        M = _safe_trimf(x, [Q1, Q2, Q3])
        H = _safe_trapmf(x, [Q2, Q3, hi, hi])
        titulo = "Funciones de membresía (L/H trapezoidales + M triangular)"

    comb = np.fmax(L, np.fmax(M, H))

    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("Bajo (L)", "Medio (M)", "Alto (H)"))
    fig.add_trace(go.Scatter(x=x, y=L, mode='lines', name='Bajo (L)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=M, mode='lines', name='Medio (M)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=H, mode='lines', name='Alto (H)'), row=3, col=1)
    fig.update_layout(height=720, title_text=titulo, template="plotly_dark", showlegend=True)
    fig.update_xaxes(title_text="Valor", row=3, col=1)
    for r in (1,2,3):
        fig.update_yaxes(title_text="μ", row=r, col=1)

    return fig, x, L, M, H, comb


def fuzificar_resumen(minimo, promedio, maximo, forma="trap"):
    """Genera L, M, H desde solo 3 valores: min, prom, max.
       forma: 'trap' (L/H trapecio + M triángulo) | 'tri' (todo triangular)
    """
    minimo, promedio, maximo = float(minimo), float(promedio), float(maximo)
    if not (np.isfinite(minimo) and np.isfinite(promedio) and np.isfinite(maximo)):
        raise ValueError("Entradas no finitas.")
    if minimo > maximo:
        minimo, maximo = maximo, minimo
    if not (minimo <= promedio <= maximo):
        promedio = float(np.clip(promedio, minimo, maximo))

    # Cuartiles aproximados
    Q1 = (2*minimo + promedio) / 3.0
    Q2 = promedio
    Q3 = (2*maximo + promedio) / 3.0

    lo, hi = _pad_range(minimo, maximo, pad=0.08)
    x = np.linspace(lo, hi, 1000)

    if forma == "tri":
        L = _safe_trimf(x, [lo, Q1, Q2])
        M = _safe_trimf(x, [Q1, Q2, Q3])
        H = _safe_trimf(x, [Q2, Q3, hi])
        titulo = "Funciones de membresía desde (min, prom, max) – Triangular"
    else:  
        L = _safe_trapmf(x, [lo, lo, Q1, Q2])
        M = _safe_trimf(x, [Q1, Q2, Q3])
        H = _safe_trapmf(x, [Q2, Q3, hi, hi])
        titulo = "Funciones de membresía desde (min, prom, max) – Trap+Tri"

    comb = np.fmax(L, np.fmax(M, H))

    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("Bajo (L)", "Medio (M)", "Alto (H)"))
    fig.add_trace(go.Scatter(x=x, y=L, mode='lines', name='Bajo (L)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=M, mode='lines', name='Medio (M)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=H, mode='lines', name='Alto (H)'), row=3, col=1)
    fig.update_layout(height=720, title_text=titulo, template="plotly_dark", showlegend=True)
    fig.update_xaxes(title_text="Valor", row=3, col=1)
    for r in (1,2,3):
        fig.update_yaxes(title_text="μ", row=r, col=1)

    return fig, x, L, M, H, comb


# Desdifusificación
def desdifuzificar(x, membresia, metodo='centroid'):
    metodo = metodo.lower()
    if metodo not in ['centroid', 'bisector', 'mom', 'som', 'lom']:
        raise ValueError("Método de desdifusificación no válido.")
    return fuzz.defuzz(x, membresia, metodo)
