import numpy as np
import pandas as pd
import streamlit as st
import skfuzzy as fuzz
import plotly.graph_objs as go

from base_difusa import fuzificar_resumen, desdifuzificar

# ──────────────────────────────
# Configuración de la página
# ──────────────────────────────
st.set_page_config(
    page_title="Estimador de datos con Lógica difusa",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      .hero {
        margin: 24px 0 8px 0;
        padding: 28px 36px;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(212,175,55,0.35);
        box-shadow: 0 2px 14px rgba(0,0,0,0.25) inset, 0 4px 20px rgba(212,175,55,0.10);
      }
      .hero h1 {
        margin: 0 0 6px 0;
        font-size: clamp(28px, 4vw, 48px);
        line-height: 1.1;
        letter-spacing: .3px;
        background: linear-gradient(90deg, #fff 0%, #e9d699 40%, #d4af37 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .hero p {
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        color: #e6e6e6;
        opacity: .85;
      }
      /* KPI grande del resultado */
      .kpi {
        margin-top: 8px;
        padding: 18px 22px;
        border-radius: 16px;
        border: 1px solid rgba(212,175,55,0.40);
        background: rgba(212,175,55,0.06);
      }
      .kpi-label {
        font-size: 0.95rem;
        opacity: .85;
        letter-spacing: .2px;
      }
      .kpi-value {
        font-size: clamp(36px, 6vw, 68px);
        font-weight: 700;
        line-height: 1.05;
        margin-top: 4px;
      }
      .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.25);
        margin-top: 8px;
      }
      .pill-baja  { background: rgba(0,128,255,0.12); }
      .pill-media { background: rgba(255,165,0,0.14); }
      .pill-alta  { background: rgba(0,200,83,0.14); }
      .pill-inst  { background: rgba(255,0,0,0.14); }
    </style>
    <div class="hero">
      <h1>Estimador de datos con Lógica difusa</h1>
      <p>
        El sistema estima un valor representativo y
        los grados de pertenencia (Baja/Media/Alta).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Estado
if "escenarios" not in st.session_state:
    st.session_state.escenarios = []

DEFAULTS = {
    "vmin": 100.0, "vavg": 150.0, "vmax": 220.0,
    "metodo": "centroid", "forma": "trap",
    "sensibilidad": 1.00, "umbral_margen": 0.20
}
if "ultimo" not in st.session_state:
    st.session_state.ultimo = DEFAULTS.copy()

def q_from_min_prom_max(vmin: float, vavg: float, vmax: float):
    q1 = (2 * vmin + vavg) / 3.0
    q2 = vavg
    q3 = (2 * vmax + vavg) / 3.0
    return float(q1), float(q2), float(q3)

def _marcar_valor_y_cuartiles(fig, x, L, M, H, valor: float, Qs):
    vline_y = np.array([0.0, 1.0])
    for r in (1, 2, 3):
        fig.add_trace(
            go.Scatter(
                x=[valor, valor], y=vline_y, mode="lines",
                name="Valor desdifusificado" if r == 1 else None,
                showlegend=(r == 1), line=dict(width=2, dash="dot")
            ),
            row=r, col=1
        )
    for q in Qs:
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, L, q)],
                                 mode="markers", name="Q", marker=dict(size=8)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, M, q)],
                                 mode="markers", showlegend=False, marker=dict(size=8)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=[q], y=[fuzz.interp_membership(x, H, q)],
                                 mode="markers", showlegend=False, marker=dict(size=8)),
                      row=3, col=1)

# ──────────────────────────────
# Sidebar — controles reactivos (SIN form)
# ──────────────────────────────
with st.sidebar:
    st.header("Parámetros de entrada")

    forma_num_dif = st.selectbox(
    "Función de membresía (forma del número difuso)",
    options=["Triangular (3 datos)", "Trapezoidal (N datos)"],
    index=0,
    help="Define la forma de las funciones de membresía L–M–H: Triangular (pico nítido en el centro) o Trapezoidal (meseta que aporta tolerancia)."
)

    modo_triangular = forma_num_dif.startswith("Triangular")

    if modo_triangular:
        vmin = st.number_input(
            "Dato mínimo",
            value=float(st.session_state.ultimo["vmin"]),
            step=0.01, format="%.2f", key="vmin_tri"
        )
        vavg = st.number_input(
            "Dato promedio",
            value=float(st.session_state.ultimo["vavg"]),
            step=0.01, format="%.2f", key="vavg_tri"
        )
        vmax = st.number_input(
            "Dato máximo",
            value=float(st.session_state.ultimo["vmax"]),
            step=0.01, format="%.2f", key="vmax_tri"
        )
        datos_arr = None
    else:
        N = int(st.number_input(
            "Cantidad de datos (N)", min_value=3, max_value=30, value=6, step=1, key="N_trap"
        ))
        base_min = float(st.session_state.ultimo["vmin"])
        base_max = float(st.session_state.ultimo["vmax"])
        base_vals = np.linspace(base_min, base_max, N)

        cols = st.columns(3)
        datos = []
        for i in range(N):
            col = cols[i % 3]
            default_val = float(base_vals[i])
            datos.append(col.number_input(
                f"Dato #{i+1}", value=default_val, step=0.01, format="%.2f", key=f"dato_{i}"
            ))
        datos_arr = np.array(datos, dtype=float)
        vmin, vmax, vavg = float(np.min(datos_arr)), float(np.max(datos_arr)), float(np.mean(datos_arr))

    # Controles de configuración + botón dentro de form
    with st.form("form_config", clear_on_submit=False):
        metodo = st.selectbox(
            "Método de desdifusificación",
            options=[
                "Centroide (promedio ponderado)",
                "Bisector (divide el área en dos partes iguales)",
                "Media de los máximos",
                "Mínimo de los máximos",
                "Máximo de los máximos"
            ],
            index=0
        )
        with st.expander("Ajustes avanzados"):
            sensibilidad = st.slider(
                "Sensibilidad de estabilidad",
                min_value=0.80, max_value=1.20,
                value=float(st.session_state.ultimo["sensibilidad"]),
                step=0.01
            )
            umbral_margen = st.slider(
                "Umbral de inestabilidad (margen)",
                min_value=0.10, max_value=0.30,
                value=float(st.session_state.ultimo["umbral_margen"]),
                step=0.01
            )
        submit = st.form_submit_button("Calcular diagnóstico", use_container_width=True)

# ──────────────────────────────
# Lógica de mapeo y cálculo
# ──────────────────────────────
mapa_metodos = {
    "Centroide (promedio ponderado)": "centroid",
    "Bisector (divide el área en dos partes iguales)": "bisector",
    "Media de los máximos": "mom",
    "Mínimo de los máximos": "som",
    "Máximo de los máximos": "lom",
}
metodo_key = mapa_metodos[metodo]
forma_key = "tri" if modo_triangular else "trap"

if submit:
    try:
        # Validaciones y normalización
        if modo_triangular:
            if (np.isnan(vmin) or np.isnan(vavg) or np.isnan(vmax) or
                np.isinf(vmin) or np.isinf(vavg) or np.isinf(vmax)):
                st.error("Entradas no válidas (NaN/∞).")
                st.stop()
            vals = sorted([vmin, vavg, vmax])
            if tuple(vals) != (vmin, vavg, vmax):
                st.warning("Se reordenaron valores para cumplir mín ≤ prom ≤ máx.")
            vmin, vavg, vmax = vals
            if np.isclose(vmin, vavg) and np.isclose(vavg, vmax):
                st.warning("Los tres valores son iguales; se amplía el rango para graficar.")
                vmin -= 0.5; vmax += 0.5
        else:
            if np.any(~np.isfinite(datos_arr)):
                st.error("Hay datos no válidos (NaN/∞).")
                st.stop()
            datos_arr = np.sort(datos_arr)
            vmin, vmax, vavg = float(datos_arr[0]), float(datos_arr[-1]), float(np.mean(datos_arr))
            if np.isclose(vmin, vmax):
                st.warning("Todos los datos son iguales; se amplía el rango para graficar.")
                vmin -= 0.5; vmax += 0.5

        # Fuzzificación y combinación
        fig, x, L, M, H, comb = fuzificar_resumen(vmin, vavg, vmax, forma=forma_key)

        # Desdifusificación
        Q1, Q2, Q3 = q_from_min_prom_max(vmin, vavg, vmax)
        valor = float(desdifuzificar(x, comb, metodo=metodo_key))

        # Grados de membresía en el valor (para Detalles)
        mu_L = float(fuzz.interp_membership(x, L, valor))
        mu_M = float(fuzz.interp_membership(x, M, valor))
        mu_H = float(fuzz.interp_membership(x, H, valor))

        _marcar_valor_y_cuartiles(fig, x, L, M, H, valor, (Q1, Q2, Q3))

        # Salidas
        t_plot, t_res, t_det = st.tabs(["Gráfica", "Resultados", "Detalles"])
        with t_plot:
            st.plotly_chart(fig, use_container_width=True)

        # =================== RESULTADOS (limpio, grande) ===================
        with t_res:
            grados = np.array([mu_L, mu_M, mu_H])
            mayor = float(np.max(grados))
            segundo = float(np.partition(grados, -2)[-2])
            margen = mayor - segundo

            dictamen = "Inestable" if margen < float(umbral_margen) else ["Baja","Media","Alta"][int(np.argmax(grados))]
            # Píldora de color
            pill_class = {
                "Baja": "pill-baja",
                "Media": "pill-media",
                "Alta": "pill-alta",
                "Inestable": "pill-inst"
            }[dictamen]

            # KPI grande
            st.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Valor desdifusificado</div>
              <div class="kpi-value">{valor:,.2f}</div>
              <div class="pill {pill_class}">{dictamen}</div>
            </div>
            """, unsafe_allow_html=True)

            # Interpretación en palabras (objetivo lingüístico)
            if dictamen == "Baja":
                interpretacion = "Los valores se comportan predominantemente como BAJOS en la escala Baja–Media–Alta."
            elif dictamen == "Media":
                interpretacion = "Los valores se concentran alrededor de un NIVEL MEDIO en la escala Baja–Media–Alta."
            elif dictamen == "Alta":
                interpretacion = "Los valores se comportan predominantemente como ALTOS en la escala Baja–Media–Alta."
            else:
                interpretacion = "No hay una categoría dominante; la pertenencia es ambigua y sujeta a mayor revisión."

            st.markdown(f"*Interpretación difusa:* {interpretacion}")

        # =================== DETALLES (μ y cuartiles) ===================
        with t_det:
            st.subheader("Pertenencias en el valor desdifusificado")
            c1, c2, c3 = st.columns(3)
            c1.metric("μ(Baja)", f"{mu_L:.2f}")
            c2.metric("μ(Media)", f"{mu_M:.2f}")
            c3.metric("μ(Alta)",  f"{mu_H:.2f}")

            st.subheader("Cuartiles (aprox. desde mín–prom–máx)")
            d1, d2, d3 = st.columns(3)
            d1.metric("Q1", f"{Q1:,.2f}")
            d2.metric("Q2 (≈ mediana)", f"{Q2:,.2f}")
            d3.metric("Q3", f"{Q3:,.2f}")

            if not modo_triangular:
                st.markdown("**Resumen de los datos (modo trapezoidal):**")
                st.write(pd.DataFrame({"dato": datos_arr}))

        # Persistencia
        st.session_state.ultimo = {
            "vmin": float(vmin), "vavg": float(vavg), "vmax": float(vmax),
            "metodo": metodo_key, "forma": forma_key,
            "sensibilidad": float(sensibilidad), "umbral_margen": float(umbral_margen)
        }

        # Descarga
        data = {
            "modo": ["triangular" if modo_triangular else "trapezoidal"],
            "dato_min": [vmin], "dato_prom": [vavg], "dato_max": [vmax],
            "Q1": [Q1], "Q2": [Q2], "Q3": [Q3],
            "metodo": [metodo],
            "forma_LMH": ["Triangular" if forma_key == "tri" else "Trapezoidal"],
            "valor_representativo": [valor],
            "mu_baja_en_valor": [round(mu_L, 4)],
            "mu_media_en_valor": [round(mu_M, 4)],
            "mu_alta_en_valor": [round(mu_H, 4)],
            "margen_dominancia": [round(margen, 4)],
            "sensibilidad": [float(sensibilidad)],
            "umbral_margen": [float(umbral_margen)],
            "dictamen": [dictamen]
        }
        df = pd.DataFrame(data)
        st.download_button(
            "Descargar reporte (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="diagnostico_difuso.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
