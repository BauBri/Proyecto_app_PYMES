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
    page_title="PyME · Analítica Difusa de Ventas",
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
        border: 1px solid rgba(212,175,55,0.35); /* dorado suave */
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
    </style>
    <div class="hero">
      <h1>Proyecto Servicio Social para PyMEs</h1>
      <p>Ingrese ventas mínima, promedio y máxima. El sistema aplica lógica difusa para estimar un valor representativo y pertenencias (Baja/Media/Alta).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if "escenarios" not in st.session_state:
    st.session_state.escenarios = []

DEFAULTS = {
    "vmin": 100.0, "vavg": 150.0, "vmax": 220.0,
    "metodo": "centroid", "forma": "trap",
    "sensibilidad": 1.00,
    "umbral_margen": 0.20
}
if "ultimo" not in st.session_state:
    st.session_state.ultimo = DEFAULTS.copy()


def q_from_min_prom_max(vmin: float, vavg: float, vmax: float):
    """Aproxima Q1, Q2, Q3 a partir de (mín, prom, máx)."""
    q1 = (2 * vmin + vavg) / 3.0
    q2 = vavg
    q3 = (2 * vmax + vavg) / 3.0
    return float(q1), float(q2), float(q3)

def _marcar_valor_y_cuartiles(fig, x, L, M, H, valor: float, Qs):
    """Añade la línea vertical del valor y los puntos Q1–Q3 en L, M y H."""
    vline_y = np.array([0.0, 1.0])
    for r in (1, 2, 3):
        fig.add_trace(
            go.Scatter(
                x=[valor, valor], y=vline_y, mode="lines",
                name="Valor desdifuzificado" if r == 1 else None,
                showlegend=(r == 1),
                line=dict(width=2, dash="dot")
            ),
            row=r, col=1
        )
    for q, lab in zip(Qs, ("Q1", "Q2", "Q3")):
        fig.add_trace(
            go.Scatter(
                x=[q], y=[fuzz.interp_membership(x, L, q)],
                mode="markers", name=lab, marker=dict(size=8)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[q], y=[fuzz.interp_membership(x, M, q)],
                mode="markers", showlegend=False, marker=dict(size=8)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[q], y=[fuzz.interp_membership(x, H, q)],
                mode="markers", showlegend=False, marker=dict(size=8)
            ),
            row=3, col=1
        )


with st.sidebar:
    st.image("unam_logo.png", use_container_width=True)
    st.header("Parámetros de entrada")

    with st.form("form_inputs", clear_on_submit=False):
        vmin = st.number_input(
            "Venta mínima",
            value=float(st.session_state.ultimo["vmin"]),
            step=10.0, format="%.2f",
            help="Valor más bajo observado en el periodo."
        )
        vavg = st.number_input(
            "Venta promedio",
            value=float(st.session_state.ultimo["vavg"]),
            step=10.0, format="%.2f",
            help="Promedio del periodo. Centro del número difuso."
        )
        vmax = st.number_input(
            "Venta máxima",
            value=float(st.session_state.ultimo["vmax"]),
            step=10.0, format="%.2f",
            help="Valor más alto observado en el periodo."
        )

        st.divider()
        st.subheader("Configuración difusa")

        forma_lbl = st.selectbox(
            "Función de membresía (L–M–H)",
            options=[
                "Trapezoidal (L/H) + Triangular (M)",
                "Triangular (L–M–H)",
            ],
            index=0,
            help="Triangular = más nítida. Trapezoidal = tolerante (meseta)."
        )

        metodo = st.selectbox(
            "Desdifusificación",
            options=[
                "Centroide (promedio ponderado)",
                "Bisector (divide el área en dos partes iguales)",
                "Media de los máximos",
                "Mínimo de los máximos",
                "Máximo de los máximos"
            ],
            index=0,
            help="Cómo obtener el valor representativo."
        )

        with st.expander("Ajustes avanzados"):
            sensibilidad = st.slider(
                "Sensibilidad de estabilidad",
                min_value=0.80, max_value=1.20,
                value=float(st.session_state.ultimo["sensibilidad"]),
                step=0.01,
                help="0.8 = más estricta, 1.2 = más laxa."
            )
            umbral_margen = st.slider(
                "Umbral de inestabilidad (margen)",
                min_value=0.10, max_value=0.30,
                value=float(st.session_state.ultimo["umbral_margen"]),
                step=0.01,
                help="Si la diferencia entre el 1º y 2º mayor grado < umbral ⇒ 'Inestable'."
            )

        submit = st.form_submit_button("Calcular diagnóstico", use_container_width=True)


mapa_metodos = {
    "Centroide (promedio ponderado)": "centroid",
    "Bisector (divide el área en dos partes iguales)": "bisector",
    "Media de los máximos": "mom",
    "Mínimo de los máximos": "som",
    "Máximo de los máximos": "lom",
}
metodo_key = mapa_metodos[metodo]

mapa_forma = {
    "Trapezoidal (L/H) + Triangular (M)": "trap",
    "Triangular (L–M–H)": "tri",
}
forma_key = mapa_forma[forma_lbl]


if submit:
    warns = []
    if (np.isnan(vmin) or np.isnan(vavg) or np.isnan(vmax) or
        np.isinf(vmin) or np.isinf(vavg) or np.isinf(vmax)):
        st.error("Entradas no válidas (NaN/∞). Corrija los valores.")
        st.stop()

    orig = (vmin, vavg, vmax)
    vals = sorted([vmin, vavg, vmax])
    if tuple(vals) != orig:
        warns.append("Se reordenaron valores para cumplir mín ≤ prom ≤ máx.")
    vmin, vavg, vmax = vals

    if np.isclose(vmin, vavg) and np.isclose(vavg, vmax):
        warns.append("Los tres valores son iguales; se amplía el rango mínimamente para graficar.")
        vmin -= 0.5
        vmax += 0.5

    if warns:
        for w in warns:
            st.warning(w)

    try:
        # Fuzzificación y combinación (L, M, H) sobre el universo x
        fig, x, L, M, H, comb = fuzificar_resumen(vmin, vavg, vmax, forma=forma_key)

        # Submuestreo ligero para rangos pequeños
        span = max(vmax - vmin, 1e-9)
        step = 2 if span < 20 else 1
        xs = x[::step]; Ls = L[::step]; Ms = M[::step]; Hs = H[::step]; combs = comb[::step]

        # Desdifusificación (valor crisp) y cuartiles aproximados
        Q1, Q2, Q3 = q_from_min_prom_max(vmin, vavg, vmax)
        valor = float(desdifuzificar(x, comb, metodo=metodo_key))

        # Grados de membresía en el valor representativo
        mu_L = float(fuzz.interp_membership(x, L, valor))
        mu_M = float(fuzz.interp_membership(x, M, valor))
        mu_H = float(fuzz.interp_membership(x, H, valor))

        _marcar_valor_y_cuartiles(fig, x, L, M, H, valor, (Q1, Q2, Q3))

        # Pestañas de salida
        t_plot, t_res, t_det = st.tabs(["Gráfica", "Resultados", "Detalles"])

        with t_plot:
            st.plotly_chart(fig, use_container_width=True)

        with t_res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valor representativo", f"{valor:,.2f}")
            c2.metric("μ(Baja) en valor", f"{mu_L:.2f}")
            c3.metric("μ(Media) en valor", f"{mu_M:.2f}")
            c4.metric("μ(Alta) en valor",  f"{mu_H:.2f}")

            grados = np.array([mu_L, mu_M, mu_H])
            mayor = float(np.max(grados))
            segundo = float(np.partition(grados, -2)[-2])
            margen = mayor - segundo

            if margen < float(umbral_margen):
                dictamen = "Inestable"
            else:
                etiquetas = ["Baja", "Media", "Alta"]
                dictamen = etiquetas[int(np.argmax(grados))]

            # Estabilidad basada en la amplitud relativa del rango
            c1_thr = 0.35 * float(sensibilidad)
            c2_thr = 0.75 * float(sensibilidad)
            spread = (vmax - vmin) / max(vavg, 1e-9)
            if spread < c1_thr:
                variacion_txt = "variación baja → alta estabilidad."
            elif spread < c2_thr:
                variacion_txt = "variación moderada → estabilidad consistente."
            else:
                variacion_txt = "variación alta → posible inestabilidad."

            if dictamen == "Baja":
                recomendacion = f"Nivel de ventas bajo; {variacion_txt} Revise inventario/precios y considere promoción puntual."
            elif dictamen == "Media":
                recomendacion = f"Promedio centrado; {variacion_txt} Mantenga estrategia y monitoree."
            elif dictamen == "Alta":
                recomendacion = f"Nivel de ventas alto; {variacion_txt} Refuerce campañas y prepare inventario."
            else:
                recomendacion = "Resultados inestables: ventas sin concentración clara. Revise estacionalidad y factores externos."

            st.markdown(f"**Dictamen:** {dictamen}")
            st.caption(recomendacion)

        with t_det:
            st.subheader("Cuartiles (aprox. desde mín–prom–máx)")
            cqa, cqb, cqc = st.columns(3)
            cqa.metric("Q1", f"{Q1:,.2f}")
            cqb.metric("Q2 (≈ mediana)", f"{Q2:,.2f}")
            cqc.metric("Q3", f"{Q3:,.2f}")

        # Persistencia y descarga
        st.session_state.ultimo = {
            "vmin": float(vmin), "vavg": float(vavg), "vmax": float(vmax),
            "metodo": metodo_key, "forma": forma_key,
            "sensibilidad": float(sensibilidad),
            "umbral_margen": float(umbral_margen)
        }

        data = {
            "venta_min": [vmin],
            "venta_prom": [vavg],
            "venta_max": [vmax],
            "Q1": [Q1], "Q2": [Q2], "Q3": [Q3],
            "forma": [forma_lbl],
            "metodo": [metodo],
            "valor_representativo": [valor],
            "mu_baja_en_valor": [round(mu_L, 4)],
            "mu_media_en_valor": [round(mu_M, 4)],
            "mu_alta_en_valor": [round(mu_H, 4)],
            "margen_dominancia": [round(margen, 4)],
            "spread": [round(spread, 4)],
            "sensibilidad": [float(sensibilidad)],
            "umbral_margen": [float(umbral_margen)],
            "dictamen": [dictamen],
            "recomendacion": [recomendacion],
        }
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar reporte (CSV)",
            data=csv,
            file_name="diagnostico_difuso.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
