import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Constantes e Física
# =========================
K = 9.0e9  # 1/(4*pi*epsilon0) em N·m²/C²

def circumference(a: float) -> float:
    return 2.0 * np.pi * a

def total_charge(lmbda: float, a: float) -> float:
    return lmbda * circumference(a)

def field_on_axis(x: float, a: float, Q: float) -> float:
    denom = (a*a + x*x) ** 1.5
    return 0.0 if denom == 0 else K * x * Q / denom


# =========================
# Formatação: 10^n (sem e±)
# =========================
def sci_parts(val: float, sig: int = 3):
    if val == 0:
        return 0.0, 0
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10 ** exp)
    mant = float(f"{mant:.{sig-1}f}")
    if abs(mant) >= 10:
        mant /= 10
        exp += 1
    return mant, exp

def fmt_latex_10(val: float, unit: str = "", sig: int = 3):
    if val == 0:
        s = "0"
        return f"{s}\\,\\text{{{unit}}}" if unit else s
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    s = f"{mant_str}\\times 10^{{{exp}}}"
    return f"{s}\\,\\text{{{unit}}}" if unit else s

def fmt_html_10(val: float, unit: str = "", sig: int = 3):
    if val == 0:
        return f"0 {unit}".strip()
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    return f"{mant_str}×10<sup>{exp}</sup> {unit}".strip()

def fmt_dec_pt(val: float, nd: int = 3):
    return f"{val:.{nd}f}".replace(".", ",")


# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Simulador Campo Elétrico do Aro – Física II",
    layout="wide"
)

# CSS responsivo para celular (gráficos menores no mobile)
st.markdown(
    """
    <style>
    /* Reduz altura dos gráficos no celular */
    @media (max-width: 768px){
      div[data-testid="stPlotlyChart"] iframe,
      div[data-testid="stPlotlyChart"] > div {
        height: 300px !important;
        min-height: 300px !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Cabeçalho (duas colunas)
# =========================
c1, c2 = st.columns([1, 4], vertical_alignment="center")
with c1:
    try:
        st.image("logo_maua.png", use_container_width=True)
    except Exception:
        st.warning("Adicione **logo_maua.png** na pasta do app para exibir o logo.")
with c2:
    st.markdown(
        """
        # Simulador Campo Elétrico do Aro Física II
        **Estude o campo elétrico gerado por um aro carregado em um ponto P sobre o eixo do aro.**
        """
    )

st.divider()

# =========================
# Parâmetros
# =========================
st.subheader("Parâmetros")

# x agora permite 0 (pedido)
X_MIN, X_MAX = 0.00, 2.00     # m (apenas positivo)
A_MIN, A_MAX = 0.05, 1.00     # m

# λ: para evitar “travamento”/quantização, usamos µC/m com passo fino (muitos valores)
# Não exibimos legenda extra abaixo do slider (pedido 1).
L_U_MIN, L_U_MAX = -20.0, 20.0  # µC/m => [-20e-6, 20e-6] C/m
L_U_STEP = 0.1                  # 0,1 µC/m => 1e-7 C/m

colp1, colp2, colp3 = st.columns(3)

with colp1:
    x = st.slider("Distância x (m)", min_value=float(X_MIN), max_value=float(X_MAX),
                  value=0.40, step=0.01)

with colp2:
    lmbda_u = st.slider("Densidade linear λ (µC/m)", min_value=float(L_U_MIN), max_value=float(L_U_MAX),
                        value=2.0, step=float(L_U_STEP))
    lmbda = lmbda_u * 1e-6  # converte para C/m

with colp3:
    a = st.slider("Raio a (m)", min_value=float(A_MIN), max_value=float(A_MAX),
                  value=0.25, step=0.01)

# Cálculos principais
L = circumference(a)
Q = total_charge(lmbda, a)
Ex = field_on_axis(x, a, Q)

# Sentido do campo (no eixo, x>=0)
if Ex > 0:
    sentido_seta = "→"
    sentido_texto = "para a direita"
elif Ex < 0:
    sentido_seta = "←"
    sentido_texto = "para a esquerda"
else:
    sentido_seta = "•"
    sentido_texto = "nulo"

st.divider()

# =========================
# Emax global (para escala de seta e eixos)
# =========================
@st.cache_data(show_spinner=False)
def compute_global_emax_for_scene():
    xs = np.linspace(X_MIN, X_MAX, 260)
    aas = np.linspace(A_MIN, A_MAX, 220)
    lam_abs = max(abs(L_U_MIN), abs(L_U_MAX)) * 1e-6

    X, A = np.meshgrid(xs, aas)
    Qg = lam_abs * 2*np.pi*A
    E = K * X * Qg / (A*A + X*X)**1.5
    return float(1.15 * np.nanmax(np.abs(E)))

E_MAX_SCENE = compute_global_emax_for_scene()


# =========================
# Imagem (escala fixa + cotas) + orientação acima
# =========================
st.subheader("Imagem")
st.caption("📱 No celular: arraste a figura para os lados (pan) para ver tudo sem perder detalhes.")

# janela fixa (não muda com sliders)
BASE = max(A_MAX, X_MAX)
X_LEFT, X_RIGHT = -0.85 * BASE, 2.10 * BASE
Y_LIM = 1.25 * A_MAX

def make_scene_figure(x, a, lmbda, Q, Ex):
    # cor do aro
    if Q > 0:
        ring_color = "red"
    elif Q < 0:
        ring_color = "blue"
    else:
        ring_color = "black"

    fig = go.Figure()

    # eixo tracejado
    fig.add_trace(go.Scatter(
        x=[X_LEFT, X_RIGHT], y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # aro em perspectiva
    t = np.linspace(0, 2*np.pi, 600)
    persp = 0.35
    ring_x = 0.0 + persp * a * np.sin(t)
    ring_y = a * np.cos(t)

    fig.add_trace(go.Scatter(
        x=ring_x, y=ring_y,
        mode="lines",
        line=dict(color=ring_color, width=5),
        hoverinfo="skip",
        showlegend=False
    ))

    # ponto P
    fig.add_trace(go.Scatter(
        x=[x], y=[0],
        mode="markers+text",
        marker=dict(size=12, color="black"),
        text=["P"],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    ))

    # =========================
    # Vetor E em P: seta verde + rótulo E com seta desenhada acima (robusto)
    # =========================
    max_arrow_len = 0.35 * (X_RIGHT - X_LEFT)
    min_arrow_len = 0.08 * (X_RIGHT - X_LEFT)
    frac = 0.0 if E_MAX_SCENE == 0 else min(1.0, abs(Ex) / E_MAX_SCENE)
    arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * np.sqrt(frac)
    dx = arrow_len if Ex >= 0 else -arrow_len

    x_end = x + dx
    x_end = max(X_LEFT + 0.03*(X_RIGHT-X_LEFT), min(X_RIGHT - 0.03*(X_RIGHT-X_LEFT), x_end))

    # seta principal do campo
    fig.add_annotation(
        x=x_end, y=0,
        ax=x, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=4,
        arrowcolor="green"
    )

    # rótulo E e valor
    midx = (x + x_end) / 2
    label_y = 0.18 * Y_LIM

    # "E" (letra)
    fig.add_annotation(
        x=midx, y=label_y,
        text="E",
        showarrow=False,
        font=dict(color="green", size=16),
        align="center"
    )

    # seta pequena ACIMA do "E" (para indicar vetor)
    small_arrow_half = 0.035 * BASE
    fig.add_annotation(
        x=midx + small_arrow_half, y=label_y + 0.06*Y_LIM,
        ax=midx - small_arrow_half, ay=label_y + 0.06*Y_LIM,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="green",
        text=""
    )

    # "= valor"
    fig.add_annotation(
        x=midx + 0.17*BASE, y=label_y,
        text=f"= {fmt_html_10(Ex, 'N/C', sig=3)}",
        showarrow=False,
        font=dict(color="green", size=14),
        align="left"
    )

    # =========================
    # COTAS
    # =========================
    # Cota x
    y_dimx = -0.35 * Y_LIM

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, y_dimx],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[x, x], y=[0, y_dimx],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))

    fig.add_annotation(
        x=x, y=y_dimx,
        ax=0, ay=y_dimx,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black",
        text=""
    )

    fig.add_annotation(
        x=(x/2 if x > 0 else 0.08*BASE), y=y_dimx - 0.06*Y_LIM,
        text=f"x = {fmt_dec_pt(x, 3)} m",
        showarrow=False,
        font=dict(color="black", size=13)
    )

    # Cota a
    x_dima = -0.55 * BASE

    fig.add_trace(go.Scatter(
        x=[0, x_dima], y=[0, 0],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, x_dima], y=[a, a],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))

    fig.add_annotation(
        x=x_dima, y=a,
        ax=x_dima, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black",
        text=""
    )

    # texto do raio deslocado para não sobrepor a cota
    fig.add_annotation(
        x=x_dima - 0.10*BASE, y=a/2,
        text=f"a = {fmt_dec_pt(a, 3)} m",
        showarrow=False,
        font=dict(color="black", size=13),
        textangle=-90
    )

    # Caixa de informação (SEM Q, pedido 4)
    info = (
        f"λ = {fmt_html_10(lmbda, 'C/m', sig=3)}<br>"
        f"a = {fmt_dec_pt(a, 3)} m<br>"
        f"x = {fmt_dec_pt(x, 3)} m"
    )
    fig.add_annotation(
        x=X_LEFT + 0.02*(X_RIGHT-X_LEFT),
        y=Y_LIM - 0.05*(2*Y_LIM),
        text=info,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=13, color="black")
    )

    # Layout
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="pan"
    )

    fig.update_xaxes(range=[X_LEFT, X_RIGHT], visible=False, fixedrange=False)
    fig.update_yaxes(range=[-Y_LIM, Y_LIM], visible=False, scaleanchor="x", scaleratio=1, fixedrange=False)

    return fig

scene = make_scene_figure(x, a, lmbda, Q, Ex)
st.plotly_chart(scene, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})

st.divider()

# =========================
# Equações
# =========================
st.subheader("Equações")

st.markdown("**Circunferência L**")
st.latex(r"L = 2\pi a")

st.markdown("**Carga Q**")
st.latex(r"Q = \lambda\,L = \lambda(2\pi a)")

st.markdown("**Campo elétrico E_x**")
st.latex(r"E_x = \frac{1}{4\pi\varepsilon_0}\,\frac{x\,Q}{(a^2+x^2)^{3/2}}")
st.latex(r"\frac{1}{4\pi\varepsilon_0} = 9,0\times10^9\ \text{N·m}^2/\text{C}^2")

st.divider()

# =========================
# Cálculos (seta final)
# =========================
st.subheader("Cálculos")

st.latex(rf"L = 2\pi a = 2\pi({fmt_dec_pt(a,3)}) = {fmt_latex_10(L,'m',sig=4)}")

st.latex(
    rf"Q = \lambda L = \left({fmt_latex_10(lmbda,'C/m',sig=3)}\right)\left({fmt_latex_10(L,'m',sig=4)}\right)"
    rf" = {fmt_latex_10(Q,'C',sig=4)}"
)

st.latex(rf"E_x = (9,0\times 10^9)\,\frac{{xQ}}{{(a^2+x^2)^{{3/2}}}}")

st.latex(
    rf"E_x = (9,0\times 10^9)\,\frac{{({fmt_dec_pt(x,3)})\left({fmt_latex_10(Q,'C',sig=4)}\right)}}"
    rf"{{\left(({fmt_dec_pt(a,3)})^2+({fmt_dec_pt(x,3)})^2\right)^{{3/2}}}}"
)

st.latex(rf"E_x = {fmt_latex_10(Ex,'N/C',sig=4)}\quad {sentido_seta}")
st.markdown(f"**Sentido do campo em P:** {sentido_texto} **{sentido_seta}**")

st.divider()

# =========================
# Gráficos
# - Eixo Y auto-ajustado (preenche) e igual nos 3 gráficos (pedido 3)
# =========================
st.subheader("Gráficos")

def curve_E_vs_x(a, Q):
    xs = np.linspace(X_MIN, X_MAX, 450)
    E = K * xs * Q / (a*a + xs*xs)**1.5
    return xs, E

def curve_E_vs_a(x, lmbda):
    aas = np.linspace(A_MIN, A_MAX, 450)
    Qs = lmbda * 2*np.pi*aas
    E = K * x * Qs / (aas*aas + x*x)**1.5
    return aas, E

def curve_E_vs_Q(x, a):
    Qmin = (L_U_MIN*1e-6) * 2*np.pi*A_MAX
    Qmax = (L_U_MAX*1e-6) * 2*np.pi*A_MAX
    Qs = np.linspace(Qmin, Qmax, 450)
    E = K * x * Qs / (a*a + x*x)**1.5
    return Qs, E

def style_axes_black(fig):
    fig.update_xaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power"
    )
    fig.update_yaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power"
    )
    return fig

# Curvas atuais
xs, Es = curve_E_vs_x(a, Q)
aas, Ea = curve_E_vs_a(x, lmbda)
Qs, EQ = curve_E_vs_Q(x, a)

# Escala Y: calcula o maior |E| considerando os 3 gráficos e usa a mesma nos três
max_abs = float(np.max(np.abs(np.concatenate([Es, Ea, EQ]))))
if max_abs == 0:
    max_abs = 1.0  # evita range nulo quando tudo é zero
YMAX = 1.08 * max_abs

# Limites de Q fixos (para o eixo X do gráfico 3)
Q_MIN_AXIS = float((L_U_MIN*1e-6) * 2*np.pi*A_MAX)
Q_MAX_AXIS = float((L_U_MAX*1e-6) * 2*np.pi*A_MAX)

# Layout: no desktop 3 colunas; no celular empilha automaticamente
gx1, gx2, gx3 = st.columns(3)

with gx1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xs, y=Es, mode="lines", line=dict(color="#1f77b4", width=3)))
    fig1.add_trace(go.Scatter(x=[x], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig1.update_layout(
        title="Campo elétrico em função da distância x",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig1.update_xaxes(title="x (m)", range=[X_MIN, X_MAX], zeroline=True)
    fig1.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with gx2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=aas, y=Ea, mode="lines", line=dict(color="#2ca02c", width=3)))
    fig2.add_trace(go.Scatter(x=[a], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig2.update_layout(
        title="Campo elétrico em função do raio a",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig2.update_xaxes(title="a (m)", range=[A_MIN, A_MAX], zeroline=True)
    fig2.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with gx3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Qs, y=EQ, mode="lines", line=dict(color="#9467bd", width=3)))
    fig3.add_trace(go.Scatter(x=[Q], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig3.update_layout(
        title="Campo elétrico em função da carga total Q",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig3.update_xaxes(title="Q (C)", range=[Q_MIN_AXIS, Q_MAX_AXIS], zeroline=True)
    fig3.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig3)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.caption("🔴 O ponto vermelho indica a situação atual. A escala vertical é a mesma nos três gráficos e se ajusta automaticamente.")
