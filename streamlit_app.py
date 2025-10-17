import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from base_difusa import (
    desdifuzificar,
    ConjuntoDifuso, evaluar_multiconjuntos, estimar_parametros,
    construir_membresia
)
# Lectura de la plantilla
from lectura_datos import leer_conjuntos_multi_atributo

st.set_page_config(
    page_title="Estimador de datos con Lógica difusa",
    page_icon="unam_logo.png",  
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      .hero { margin:24px 0 8px; padding:28px 36px; border-radius:18px;
              background:rgba(255,255,255,.03); border:1px solid rgba(212,175,55,.35);
              box-shadow:0 2px 14px rgba(0,0,0,.25) inset, 0 4px 20px rgba(212,175,55,.10); }
      .hero h1 { margin:0 0 6px; font-size:clamp(28px,4vw,48px); line-height:1.1; letter-spacing:.3px;
                 background:linear-gradient(90deg,#fff 0%,#e9d699 40%,#d4af37 80%);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
      .hero p { margin:4px 0 0; font-size:.95rem; color:#e6e6e6; opacity:.85; }
      .kpi { margin-top:8px; padding:18px 22px; border-radius:16px;
             border:1px solid rgba(212,175,55,.40); background:rgba(212,175,55,.06); }
      .kpi-label { font-size:.95rem; opacity:.85; letter-spacing:.2px; }
      .kpi-value { font-size:clamp(36px,6vw,68px); font-weight:700; line-height:1.05; margin-top:4px; }
      .pill { display:inline-block; padding:6px 12px; border-radius:999px; font-weight:600;
              border:1px solid rgba(255,255,255,.25); margin-top:8px; }
    </style>
    <div class="hero">
      <h1>Estimador de datos con Lógica difusa</h1>
      <p></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Utilidades
EPS = 1e-9

def _fix_monotonic(vals):
    arr = list(vals)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i-1]:
            arr[i] = arr[i-1] + EPS
    return tuple(arr)

def _ajusta_por_sens(params, tipo, s):
    if s is None or s == 1.0:
        return params
    if tipo == "gauss":
        m, sigma = params
        sigma = max(sigma * float(s), 1e-9)
        return (float(m), float(sigma))
    if tipo == "tri":
        a, b, c = map(float, params)
        a2 = b - (b - a) * s
        c2 = b + (c - b) * s
        return _fix_monotonic((a2, b, c2))
    if tipo in ("trap", "pi"):
        a, b, c, d = map(float, params)
        mid = (b + c) / 2.0
        a2 = mid - (mid - a) * s
        b2 = mid - (mid - b) * s
        c2 = mid + (c - mid) * s
        d2 = mid + (d - mid) * s
        return _fix_monotonic((a2, b2, c2, d2))
    if tipo in ("gamma", "l"):
        a, b = map(float, params)
        m = (a + b) / 2.0
        a2 = m - (m - a) * s
        b2 = m + (b - m) * s
        return _fix_monotonic((a2, b2))
    return params

# Estado
if "ultimo" not in st.session_state:
    st.session_state.ultimo = {"metodo": "centroid"}
if "num_conjuntos" not in st.session_state:
    st.session_state.num_conjuntos = 1
if "sens" not in st.session_state:
    st.session_state.sens = 1.0

# Membresías
opciones_memb = [
    "Triangular (tri)", "Trapezoidal (trap)", "Gamma/S (smf)",
    "L/Z (zmf)", "Π (pimf)", "Gaussiana (gauss)"
]
map_memb = {
    "Triangular (tri)": "tri",
    "Trapezoidal (trap)": "trap",
    "Gamma/S (smf)": "gamma",
    "L/Z (zmf)": "l",
    "Π (pimf)": "pi",
    "Gaussiana (gauss)": "gauss",
}

def _plantilla_multi_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Conjunto 1": [0.0, 0.5, 1.0],
        "Conjunto 2": [0.1, 0.6, 0.9],
        "Conjunto 3": [0.0, 0.4, 1.0],
    })

def _df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "datos") -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()

# Vaciado y aplicación de datos cargados a la UI
def _limpiar_campos_multi():
    # Elimina claves de nombre, N y datos para evitar residuos cuando se cargan nuevos conjuntos.
    borrar = [k for k in st.session_state.keys() if k.startswith(("nm_", "N_", "d_"))]
    for k in borrar:
        del st.session_state[k]

def _aplicar_sets_a_ui(sets):
    # Actualiza número de conjuntos y rellena nombre, N y cada dato.
    _limpiar_campos_multi()
    nombres = list(sets.keys())
    st.session_state.num_conjuntos = len(nombres)
    for i, nombre in enumerate(nombres):
        vals = list(sets[nombre])
        st.session_state[f"nm_{i}"] = str(nombre)
        st.session_state[f"N_{i}"] = int(len(vals))
        for j, v in enumerate(vals):
            st.session_state[f"d_{i}_{j}"] = float(v)

def _token_archivo(archivo) -> str:
    nombre = getattr(archivo, "name", "")
    size = getattr(archivo, "size", None)
    if size is None:
        try:
            size = len(archivo.getvalue())
        except Exception:
            size = ""
    return f"{nombre}:{size}"

# Barra lateral
with st.sidebar:
    modo = st.selectbox(
        "Modo de trabajo",
        options=["Un solo conjunto (simple)", "Varios conjuntos (multi-atributo)"],
        index=0
    )
    metodo = st.selectbox(
        "Método de desdifuzificación",
        options=[
            "Centroide (promedio ponderado)",
            "Bisector (divide el área en dos partes iguales)",
            "Media de los máximos",
            "Mínimo de los máximimos",
            "Máximo de los máximos",
        ],
        index=0
    )
    mapa_metodos = {
        "Centroide (promedio ponderado)": "centroid",
        "Bisector (divide el área en dos partes iguales)": "bisector",
        "Media de los máximos": "mom",
        "Mínimo de los máximimos": "som",
        "Máximo de los máximos": "lom",
    }
    metodo_key = mapa_metodos[metodo]

    st.session_state.sens = st.slider(
        "Sensibilidad (ancho de las membresías)",
        min_value=0.50, max_value=1.50, value=float(st.session_state.sens), step=0.01,
        help="0.50 = estrecho; 1.00 = normal; 1.50 = ancho"
    )
    s_factor = float(st.session_state.sens)

    # Subir datos (solo visible en multi-atributo)
    if "multi-atributo" in modo:
        with st.expander("Subir datos"):
            st.markdown(
                """
**Instrucciones**
- Descarga la plantilla. Cada columna es un conjunto.
- Escribe solo números. El encabezado de cada columna es el nombre del conjunto.
- Puedes dejar celdas vacías al final de una columna.
- Guarda en .xlsx o .csv y súbelo. La app llenará los campos automáticamente.
                """
            )
            st.download_button(
                "Descargar plantilla",
                data=_df_to_xlsx_bytes(_plantilla_multi_df()),
                file_name="plantilla_multiatributo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            archivo = st.file_uploader(
                "Sube tu archivo (.xlsx o .csv)",
                type=["xlsx", "csv"],
                accept_multiple_files=False
            )
            if archivo is not None:
                token = _token_archivo(archivo)
                if st.session_state.get("last_upload_token") != token:
                    try:
                        sets = leer_conjuntos_multi_atributo(archivo)  # {nombre: [valores]}
                        _aplicar_sets_a_ui(sets)                      # aplica una sola vez
                        st.session_state["last_upload_token"] = token
                        st.success("Datos cargados en la interfaz.")
                    except Exception as e:
                        st.error(f"Archivo inválido: {e}")
                else:
                    st.info("Este archivo ya fue aplicado. Para recargarlo, vuelve a subirlo o limpia todos.")

# Modo simple
if modo.startswith("Un solo"):
    tipo_label_simple = st.selectbox("Función de membresía", options=opciones_memb, index=0, key="tipo_simple_lbl")
    tipo_simple = map_memb[tipo_label_simple]

    N_simple = int(st.number_input("Cantidad de datos (N)", min_value=3, max_value=1000, value=6, step=1, key="N_simple"))
    base_vals = np.linspace(0.0, 1.0, N_simple)
    cols = st.columns(3)
    datos_simple = []
    for j in range(N_simple):
        key = f"ds_{j}"
        if key not in st.session_state:
            st.session_state[key] = float(base_vals[j])
        col = cols[j % 3]
        datos_simple.append(col.number_input(f"Dato #{j+1}", value=float(st.session_state[key]),
                                             step=0.01, format="%.4f", key=key))
    datos_simple = np.array(datos_simple, dtype=float)

    editar_simple = st.checkbox("Editar parámetros manualmente", value=False, key="edit_simple")
    parametros_simple = None
    if editar_simple:
        try:
            sugeridos = estimar_parametros(datos_simple, tipo_simple)
            sug = next(iter(sugeridos.values()))
        except Exception:
            if tipo_simple == "tri":
                sug = (np.min(datos_simple), np.median(datos_simple), np.max(datos_simple))
            elif tipo_simple == "trap":
                lo_, hi_ = float(np.min(datos_simple)), float(np.max(datos_simple))
                m1 = lo_ + (hi_ - lo_) * 0.25
                m2 = lo_ + (hi_ - lo_) * 0.75
                sug = (lo_, m1, m2, hi_)
            elif tipo_simple in ("gamma", "l"):
                lo_, hi_ = float(np.min(datos_simple)), float(np.max(datos_simple))
                sug = (lo_, hi_)
            elif tipo_simple == "pi":
                lo_, hi_ = float(np.min(datos_simple)), float(np.max(datos_simple))
                a = lo_; d = hi_; b = a + (d-a)*0.33; c = a + (d-a)*0.66
                sug = (a, b, c, d)
            else:
                sug = (float(np.mean(datos_simple)), max(1e-3, float(np.std(datos_simple) or 0.1)))
        if tipo_simple == "tri":
            a = st.number_input("a", value=float(sug[0]), format="%.4f", key="a_s")
            b = st.number_input("b", value=float(sug[1]), format="%.4f", key="b_s")
            c = st.number_input("c", value=float(sug[2]), format="%.4f", key="c_s")
            parametros_simple = (a, b, c)
        elif tipo_simple == "trap":
            a = st.number_input("a", value=float(sug[0]), format="%.4f", key="a_s")
            b = st.number_input("b", value=float(sug[1]), format="%.4f", key="b_s")
            c = st.number_input("c", value=float(sug[2]), format="%.4f", key="c_s")
            d = st.number_input("d", value=float(sug[3]), format="%.4f", key="d_s")
            parametros_simple = (a, b, c, d)
        elif tipo_simple in ("gamma", "l"):
            a = st.number_input("a", value=float(sug[0]), format="%.4f", key="a_s")
            b = st.number_input("b", value=float(sug[1]), format="%.4f", key="b_s")
            parametros_simple = (a, b)
        elif tipo_simple == "pi":
            a = st.number_input("a", value=float(sug[0]), format="%.4f", key="a_s")
            b = st.number_input("b", value=float(sug[1]), format="%.4f", key="b_s")
            c = st.number_input("c", value=float(sug[2]), format="%.4f", key="c_s")
            d = st.number_input("d", value=float(sug[3]), format="%.4f", key="d_s")
            parametros_simple = (a, b, c, d)
        elif tipo_simple == "gauss":
            media = st.number_input("media", value=float(sug[0]), format="%.4f", key="m_s")
            sigma = st.number_input("sigma", value=float(sug[1]), min_value=1e-6, format="%.4f", key="sg_s")
            parametros_simple = (media, sigma)

    submit_simple = st.button("Calcular (simple)", use_container_width=True)

    if submit_simple:
        all_vals = np.asarray(datos_simple, dtype=float)
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        span = max(hi - lo, 1e-6)
        x = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 10001)

        if parametros_simple is None:
            ctmp = ConjuntoDifuso(
                nombre="Simple",
                datos=all_vals,
                tipo_membresia=tipo_simple,
                parametros=None,
                invertir=False
            )
            ctmp.construir()
            params_usados = ctmp.parametros_usados
        else:
            params_usados = parametros_simple

        params_usados = _ajusta_por_sens(params_usados, tipo_simple, s_factor)

        mu = construir_membresia(x, tipo_simple, params_usados, invertir=False)
        valor = float(desdifuzificar(x, mu, metodo=metodo_key))

        f = go.Figure()
        f.add_trace(go.Scatter(x=x, y=mu, mode="lines", name="μ(x)"))
        f.add_trace(go.Scatter(x=[valor, valor], y=[0, 1], mode="lines",
                               name=f"z* = {valor:.4f}", line=dict(dash="dot")))
        f.update_layout(template="plotly_dark", title="Función de membresía y valor desdifuzificado",
                        xaxis_title="x", yaxis_title="μ(x)")

        t1, t2 = st.tabs(["Gráfica", "Resultado"])
        with t1:
            st.plotly_chart(f, use_container_width=True)
        with t2:
            st.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Valor desdifuzificado</div>
              <div class="kpi-value">{valor:,.4f}</div>
              <div class="pill">Método: {metodo_key}</div>
              <div class="pill">Membresía: {tipo_label_simple}</div>
              <div class="pill">Sensibilidad: {s_factor:.2f}×</div>
            </div>
            """, unsafe_allow_html=True)

# Modo multi-atributo
else:
    cols_btn = st.columns(3)
    if cols_btn[0].button("Añadir conjunto", use_container_width=True):
        st.session_state.num_conjuntos += 1
    if cols_btn[1].button("Quitar último", use_container_width=True) and st.session_state.num_conjuntos > 1:
        st.session_state.num_conjuntos -= 1
    if cols_btn[2].button("Limpiar todos", use_container_width=True):
        st.session_state.num_conjuntos = 1
        _limpiar_campos_multi()
        st.session_state.pop("last_upload_token", None)  # limpia el token del último archivo aplicado

    conjuntos = []
    tabs_c = st.tabs([f"Conjunto {i+1}" for i in range(st.session_state.num_conjuntos)])

    for i in range(st.session_state.num_conjuntos):
        with tabs_c[i]:
            nombre = st.text_input("Nombre del conjunto", value=st.session_state.get(f"nm_{i}", f"Conjunto {i+1}"), key=f"nm_{i}")
            N_i = int(st.number_input("Cantidad de datos (N)", min_value=3, max_value=1000,
                                      value=st.session_state.get(f"N_{i}", 3), step=1, key=f"N_{i}"))

            base_vals = np.linspace(0.0, 1.0, N_i)
            cols = st.columns(3)
            datos_i = []
            for j in range(N_i):
                key = f"d_{i}_{j}"
                if key not in st.session_state:
                    st.session_state[key] = float(base_vals[j])
                col = cols[j % 3]
                datos_i.append(col.number_input(f"Dato #{j+1}", value=float(st.session_state[key]),
                                                step=0.01, format="%.4f", key=key))
            datos_i = np.array(datos_i, dtype=float)

            tipo_label = st.selectbox("Función de membresía", options=opciones_memb,
                                      index=opciones_memb.index(st.session_state.get(f"tipo_{i}", "Triangular (tri)"))
                                      if st.session_state.get(f"tipo_{i}") in opciones_memb else 0,
                                      key=f"tipo_{i}")
            tipo = map_memb[tipo_label]

            editar = st.checkbox("Editar parámetros manualmente", value=False, key=f"edit_{i}")
            parametros = None
            if editar:
                try:
                    sugeridos = estimar_parametros(datos_i, tipo)
                    sug = next(iter(sugeridos.values()))
                except Exception:
                    if tipo == "tri":
                        sug = (np.min(datos_i), np.median(datos_i), np.max(datos_i))
                    elif tipo == "trap":
                        lo, hi = float(np.min(datos_i)), float(np.max(datos_i))
                        m1 = lo + (hi - lo) * 0.25
                        m2 = lo + (hi - lo) * 0.75
                        sug = (lo, m1, m2, hi)
                    elif tipo in ("gamma", "l"):
                        lo, hi = float(np.min(datos_i)), float(np.max(datos_i))
                        sug = (lo, hi)
                    elif tipo == "pi":
                        lo, hi = float(np.min(datos_i)), float(np.max(datos_i))
                        a = lo; d = hi; b = a + (d-a)*0.33; c = a + (d-a)*0.66
                        sug = (a, b, c, d)
                    else:
                        sug = (float(np.mean(datos_i)), max(1e-3, float(np.std(datos_i) or 0.1)))
                if tipo == "tri":
                    a = st.number_input("a", value=float(sug[0]), format="%.4f", key=f"a_{i}")
                    b = st.number_input("b", value=float(sug[1]), format="%.4f", key=f"b_{i}")
                    c = st.number_input("c", value=float(sug[2]), format="%.4f", key=f"c_{i}")
                    parametros = (a, b, c)
                elif tipo == "trap":
                    a = st.number_input("a", value=float(sug[0]), format="%.4f", key=f"a_{i}")
                    b = st.number_input("b", value=float(sug[1]), format="%.4f", key=f"b_{i}")
                    c = st.number_input("c", value=float(sug[2]), format="%.4f", key=f"c_{i}")
                    d = st.number_input("d", value=float(sug[3]), format="%.4f", key=f"d_{i}")
                    parametros = (a, b, c, d)
                elif tipo in ("gamma", "l"):
                    a = st.number_input("a", value=float(sug[0]), format="%.4f", key=f"a_{i}")
                    b = st.number_input("b", value=float(sug[1]), format="%.4f", key=f"b_{i}")
                    parametros = (a, b)
                elif tipo == "pi":
                    a = st.number_input("a", value=float(sug[0]), format="%.4f", key=f"a_{i}")
                    b = st.number_input("b", value=float(sug[1]), format="%.4f", key=f"b_{i}")
                    c = st.number_input("c", value=float(sug[2]), format="%.4f", key=f"c_{i}")
                    d = st.number_input("d", value=float(sug[3]), format="%.4f", key=f"d_{i}")
                    parametros = (a, b, c, d)
                elif tipo == "gauss":
                    media = st.number_input("media", value=float(sug[0]), format="%.4f", key=f"m_{i}")
                    sigma = st.number_input("sigma", value=float(sug[1]), min_value=1e-6, format="%.4f", key=f"s_{i}")
                    parametros = (media, sigma)

            c = ConjuntoDifuso(
                nombre=nombre,
                datos=datos_i,
                tipo_membresia=tipo,
                parametros=parametros,
                invertir=False
            )
            conjuntos.append(c)

    submit_multi = st.button("Calcular (multi-atributo)", use_container_width=True)

    if submit_multi:
        all_vals = np.concatenate([np.asarray(c.datos, dtype=float) for c in conjuntos])
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        span = max(hi - lo, 1e-6)
        x = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 10001)

        mus = []
        for c in conjuntos:
            if c.parametros is None and getattr(c, "parametros_usados", None) is None:
                c.construir()
            params = c.parametros if c.parametros is not None else c.parametros_usados
            params = _ajusta_por_sens(params, c.tipo_membresia, s_factor)
            mu = construir_membresia(x, c.tipo_membresia, params, invertir=False)
            mus.append(mu)

        mu_union = np.maximum.reduce(mus)

        crisp_por_conjunto, _ = evaluar_multiconjuntos(
            conjuntos=conjuntos, metodo=metodo_key, pesos=None, agregacion="media"
        )

        total = float(desdifuzificar(x, mu_union, metodo=metodo_key))

        tabs_out = st.tabs(["Resultado", "Gráficas", "Tabla"])

        with tabs_out[0]:
            st.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Valor desdifuzificado: </div>
              <div class="kpi-value">{total:,.4f}</div>
              <div class="pill">Método: {metodo_key}</div>
              <div class="pill">Sensibilidad: {s_factor:.2f}×</div>
            </div>
            """, unsafe_allow_html=True)

        with tabs_out[1]:
            f1 = go.Figure()
            for c, mu in zip(conjuntos, mus):
                f1.add_trace(go.Scatter(x=x, y=mu, mode="lines", name=c.nombre))
            f1.update_layout(template="plotly_dark", title="Funciones de membresía individuales",
                             xaxis_title="x", yaxis_title="μ(x)")
            st.plotly_chart(f1, use_container_width=True)

            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=x, y=mu_union, mode="lines", name="Unión (máx)"))
            f2.add_trace(go.Scatter(x=[total, total], y=[0, 1], mode="lines",
                                    name=f"z* = {total:.4f}", line=dict(dash="dot")))
            f2.update_layout(template="plotly_dark", title="Unión de funciones (máx)",
                             xaxis_title="x", yaxis_title="μ(x)")
            st.plotly_chart(f2, use_container_width=True)

        with tabs_out[2]:
            df = pd.DataFrame({
                "conjunto": [c.nombre for c in conjuntos],
                "valor_desdifuzificado": [v for v in crisp_por_conjunto.values()]
            })
            st.dataframe(df, use_container_width=True)

        with st.expander("Detalles avanzados: tabla de la unión"):
            dfu = pd.DataFrame({"x": x, "mu_union": mu_union})
            st.dataframe(dfu.iloc[::200, :], use_container_width=True)
