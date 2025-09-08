import numpy as np 
import pandas as pd
import streamlit as st
import skfuzzy as fuzz
import plotly.graph_objs as go

from base_difusa import (
    fuzificar_resumen, desdifuzificar,
    ConjuntoDifuso, evaluar_multiconjuntos, estimar_parametros,
    construir_membresia  # para mu sobre una malla común
)

# Configuración de la página
st.set_page_config(
    page_title="Estimador de datos con Lógica difusa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
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

# Estado
if "ultimo" not in st.session_state:
    st.session_state.ultimo = {
        "vmin": 100.0, "vavg": 150.0, "vmax": 220.0,
        "metodo": "centroid", "forma": "trap",
        "sensibilidad": 1.00, "umbral_margen": 0.20
    }
if "num_conjuntos" not in st.session_state:
    st.session_state.num_conjuntos = 1

# Utilidades
def q_from_min_prom_max(vmin: float, vavg: float, vmax: float):
    q1 = (2 * vmin + vavg) / 3.0
    q2 = vavg
    q3 = (2 * vmax + vavg) / 3.0
    return float(q1), float(q2), float(q3)

def _marcar_valor_y_cuartiles(fig, x, L, M, H, valor: float, Qs):
    vline_y = np.array([0.0, 1.0])
    for r in (1, 2, 3):
        fig.add_trace(
            go.Scatter(x=[valor, valor], y=vline_y, mode="lines",
                       name="Valor desdifusificado" if r == 1 else None,
                       showlegend=(r == 1), line=dict(width=2, dash="dot")),
            row=r, col=1
        )
    for q in Qs:
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, L, q)], mode="markers",
                                 name="Q", marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, M, q)], mode="markers",
                                 showlegend=False, marker=dict(size=8)), row=2, col=1)
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, H, q)], mode="markers",
                                 showlegend=False, marker=dict(size=8)), row=3, col=1)

# Barra lateral
with st.sidebar:
    modo = st.selectbox(
        "Modo de trabajo",
        options=["Un solo conjunto (simple)", "Varios conjuntos (multi-atributo)"],
        index=0
    )

    metodo = st.selectbox(
        "Método de desdifusificación",
        options=[
            "Centroide (promedio ponderado)",
            "Bisector (divide el área en dos partes iguales)",
            "Media de los máximos",
            "Mínimo de los máximos",
            "Máximo de los máximos",
        ],
        index=0
    )
    mapa_metodos = {
        "Centroide (promedio ponderado)": "centroid",
        "Bisector (divide el área en dos partes iguales)": "bisector",
        "Media de los máximos": "mom",
        "Mínimo de los máximos": "som",
        "Máximo de los máximos": "lom",
    }
    metodo_key = mapa_metodos[metodo]

    if modo.startswith("Un solo"):
        forma_num_dif = st.selectbox(
            "Forma L–M–H",
            options=["Triangular (3 datos)", "Trapezoidal (N datos)"],
            index=0
        )

        if forma_num_dif.startswith("Triangular"):
            vmin = st.number_input("Dato mínimo", value=float(st.session_state.ultimo["vmin"]), step=0.01, format="%.2f", key="vmin_tri")
            vavg = st.number_input("Dato promedio", value=float(st.session_state.ultimo["vavg"]), step=0.01, format="%.2f", key="vavg_tri")
            vmax = st.number_input("Dato máximo", value=float(st.session_state.ultimo["vmax"]), step=0.01, format="%.2f", key="vmax_tri")
            datos_arr = None
        else:
            N = int(st.number_input("Cantidad de datos (N)", min_value=3, max_value=500, value=6, step=1, key="N_trap"))
            base_min = float(st.session_state.ultimo["vmin"])
            base_max = float(st.session_state.ultimo["vmax"])
            base_vals = np.linspace(base_min, base_max, N)
            cols = st.columns(3)
            datos = []
            for i in range(N):
                col = cols[i % 3]
                default_val = float(base_vals[i])
                datos.append(col.number_input(f"Dato #{i+1}", value=default_val, step=0.01, format="%.2f", key=f"dato_{i}"))
            datos_arr = np.array(datos, dtype=float)
            vmin, vmax, vavg = float(np.min(datos_arr)), float(np.max(datos_arr)), float(np.mean(datos_arr))

        with st.expander("Ajustes avanzados"):
            sensibilidad = st.slider("Sensibilidad de estabilidad", 0.80, 1.20, float(st.session_state.ultimo["sensibilidad"]), 0.01)
            umbral_margen = st.slider("Umbral de inestabilidad (margen)", 0.10, 0.30, float(st.session_state.ultimo["umbral_margen"]), 0.01)

        submit_simple = st.button("Calcular (simple)", use_container_width=True)

    else:
        # Controles de administración de conjuntos
        cols_btn = st.columns(3)
        if cols_btn[0].button("Añadir conjunto", use_container_width=True):
            st.session_state.num_conjuntos += 1
        if cols_btn[1].button("Quitar último", use_container_width=True) and st.session_state.num_conjuntos > 1:
            st.session_state.num_conjuntos -= 1
        if cols_btn[2].button("Limpiar todos", use_container_width=True):
            st.session_state.num_conjuntos = 1

        combinar = st.radio(
            "Cómo combinar",
            ["Promedio de valores desdifuzificados", "Unión lógica (máx) y desdifuzificar la unión"],
            index=0
        )

        forzar_dom = st.checkbox("Forzar dominio", value=bool(st.session_state.get("forzar_dom", False)))
        if forzar_dom:
            dom_a = st.number_input("x mínimo (dominio)", value=float(st.session_state.get("dom_a", 0.0)),
                                    step=0.01, format="%.3f", key="dom_a")
            dom_b = st.number_input("x máximo (dominio)", value=float(st.session_state.get("dom_b", 1.1)),
                                    step=0.01, format="%.3f", key="dom_b")
        else:
            dom_a, dom_b = None, None

        ver_individuales = st.checkbox("Graficar funciones individuales", value=True)
        ver_union = st.checkbox("Graficar curva unida", value=True)

        submit_multi = st.button("Calcular (multi-atributo)", use_container_width=True)

# Modo simple
if modo.startswith("Un solo"):
    try:
        modo_triangular = "Triangular" in forma_num_dif
        forma_key = "tri" if modo_triangular else "trap"

        if submit_simple:
            if modo_triangular:
                vals = [vmin, vavg, vmax]
                if any([np.isnan(x) or np.isinf(x) for x in vals]):
                    st.error("Entradas no válidas (NaN/∞)."); st.stop()
                vals = sorted(vals)
                if np.isclose(vals[0], vals[2]):
                    vmin, vavg, vmax = vals[0] - 0.5, vals[1], vals[2] + 0.5
                else:
                    vmin, vavg, vmax = vals
            else:
                if np.any(~np.isfinite(datos_arr)):
                    st.error("Hay datos no válidos (NaN/∞)."); st.stop()
                datos_arr = np.sort(datos_arr)
                vmin, vmax, vavg = float(datos_arr[0]), float(datos_arr[-1]), float(np.mean(datos_arr))
                if np.isclose(vmin, vmax):
                    vmin -= 0.5; vmax += 0.5

            fig, x, L, M, H, comb = fuzificar_resumen(vmin, vavg, vmax, forma=forma_key)
            Q1, Q2, Q3 = q_from_min_prom_max(vmin, vavg, vmax)
            valor = float(desdifuzificar(x, comb, metodo=metodo_key))

            mu_L = float(fuzz.interp_membership(x, L, valor))
            mu_M = float(fuzz.interp_membership(x, M, valor))
            mu_H = float(fuzz.interp_membership(x, H, valor))
            _marcar_valor_y_cuartiles(fig, x, L, M, H, valor, (Q1, Q2, Q3))

            t_plot, t_res, t_det = st.tabs(["Gráfica", "Resultados", "Detalles"])
            with t_plot:
                st.plotly_chart(fig, use_container_width=True)
            with t_res:
                st.markdown(f"""
                <div class="kpi">
                  <div class="kpi-label">Valor desdifusificado</div>
                  <div class="kpi-value">{valor:,.2f}</div>
                  <div class="pill">Método: {metodo_key}</div>
                </div>
                """, unsafe_allow_html=True)
            with t_det:
                c1, c2, c3 = st.columns(3)
                c1.metric("μ(Baja)", f"{mu_L:.2f}")
                c2.metric("μ(Media)", f"{mu_M:.2f}")
                c3.metric("μ(Alta)",  f"{mu_H:.2f}")
                d1, d2, d3 = st.columns(3)
                d1.metric("Q1", f"{Q1:,.2f}")
                d2.metric("Q2 (≈ mediana)", f"{Q2:,.2f}")
                d3.metric("Q3", f"{Q3:,.2f}")
                if not modo_triangular:
                    st.write(pd.DataFrame({"dato": datos_arr}))

            st.session_state.ultimo = {
                "vmin": float(vmin), "vavg": float(vavg), "vmax": float(vmax),
                "metodo": metodo_key, "forma": forma_key,
                "sensibilidad": float(sensibilidad), "umbral_margen": float(umbral_margen)
            }
    except Exception as e:
        st.error(f"Ocurrió un error en modo simple: {e}")

# Modo multi-atributo
else:
    try:
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

        conjuntos = []
        tabs = st.tabs([f"Conjunto {i+1}" for i in range(st.session_state.num_conjuntos)])

        for i in range(st.session_state.num_conjuntos):
            with tabs[i]:
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
                invertir = st.checkbox("Invertir atributo (usar 1-μ)", value=bool(st.session_state.get(f"inv_{i}", False)), key=f"inv_{i}")

                editar = st.checkbox("Editar parámetros manualmente", value=False, key=f"edit_{i}")
                parametros = None
                if editar:
                    try:
                        sugeridos = estimar_parametros(datos_i, tipo)
                        sug = next(iter(sugeridos.values()))
                    except Exception:
                        sug = (np.min(datos_i), np.median(datos_i), np.max(datos_i)) if tipo == "tri" else (0.0, 0.25, 0.75, 1.0)

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
                    invertir=invertir
                )
                conjuntos.append(c)

        if submit_multi:
            if combinar.startswith("Promedio"):
                crisp_por_conjunto, total = evaluar_multiconjuntos(
                    conjuntos=conjuntos,
                    metodo=metodo_key,
                    pesos=None,
                    agregacion="media"
                )

                t_res, t_tab = st.tabs(["Resultados", "Tabla"])
                with t_res:
                    st.markdown(f"""
                    <div class="kpi">
                      <div class="kpi-label">Valor desdifusificado global (promedio de crisp)</div>
                      <div class="kpi-value">{total:,.4f}</div>
                      <div class="pill">Método: {metodo_key}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t_tab:
                    df = pd.DataFrame({
                        "conjunto": list(crisp_por_conjunto.keys()),
                        "valor_desdifuzificado": [crisp_por_conjunto[k] for k in crisp_por_conjunto.keys()]
                    })
                    st.dataframe(df, use_container_width=True)

            else:
                # Unión lógica sobre una malla común
                if forzar_dom and dom_a is not None and dom_b is not None:
                    x = np.linspace(float(dom_a), float(dom_b), 10001)
                else:
                    # dominio automático a partir de los datos de todos los conjuntos
                    all_vals = np.concatenate([np.asarray(c.datos, dtype=float) for c in conjuntos])
                    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                    span = max(hi - lo, 1e-6)
                    x = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 10001)

                mus = []
                for c in conjuntos:
                    if c.parametros is None and getattr(c, "parametros_usados", None) is None:
                        c.construir()
                    params = c.parametros if c.parametros is not None else c.parametros_usados
                    mu = construir_membresia(x, c.tipo_membresia, params, invertir=c.invertir)
                    mus.append(mu)

                mu_union = np.maximum.reduce(mus)
                total = float(desdifuzificar(x, mu_union, metodo=metodo_key))

                # Gráficas como en el libro
                figs = []
                if ver_individuales:
                    fig1 = go.Figure()
                    for c, mu in zip(conjuntos, mus):
                        fig1.add_trace(go.Scatter(x=x, y=mu, mode="lines", name=c.nombre))
                    fig1.update_layout(template="plotly_dark", title="Funciones de membresía individuales",
                                       xaxis_title="x", yaxis_title="μ(x)")
                    figs.append(fig1)

                if ver_union:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=x, y=mu_union, mode="lines", name="Unión (máx)"))
                    fig2.add_trace(go.Scatter(x=[total,total], y=[0,1], mode="lines",
                                              name=f"z* = {total:.4f}", line=dict(dash="dot")))
                    fig2.update_layout(template="plotly_dark", title="Unión de funciones (máx) y desdifuzificación",
                                       xaxis_title="x", yaxis_title="μ(x)")
                    figs.append(fig2)

                t_res, t_plot, t_tab = st.tabs(["Resultado", "Gráficas", "Tabla unión"])
                with t_res:
                    st.markdown(f"""
                    <div class="kpi">
                      <div class="kpi-label">Valor desdifusificado de la UNIÓN (máx)</div>
                      <div class="kpi-value">{total:,.4f}</div>
                      <div class="pill">Método: {metodo_key}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t_plot:
                    for f in figs:
                        st.plotly_chart(f, use_container_width=True)
                with t_tab:
                    dfu = pd.DataFrame({"x": x, "mu_union": mu_union})
                    st.dataframe(dfu.iloc[::200, :], use_container_width=True)
                    st.download_button("Descargar unión (CSV)", data=dfu.to_csv(index=False).encode("utf-8"),
                                       file_name="union_max.csv", mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error en modo multi-atributo: {e}")
