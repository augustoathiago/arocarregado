import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Constantes e Física
# =========================
K = 9.0e9  # 1/(4*pi*epsilon0) em N·m²/C² (conforme solicitado)

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
    """Retorna (mantissa, expoente) em base 10 para val != 0."""
    if val == 0:
        return 0.0, 0
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10 ** exp)
    # arredonda mantissa para sig algarismos significativos
    mant = float(f"{mant:.{sig-1}f}")
    # reajusta caso vire 10.0 por arredondamento
    if abs(mant) >= 10:
        mant /= 10
        exp += 1
    return mant, exp

def fmt_latex_10(val: float, unit: str = "", sig: int = 3):
    """Formata em LaTeX com ×10^{n} e vírgula decimal pt-BR."""
    if val == 0:
        s = "0"
        return f"{s}\\,\\text{{{unit}}}" if unit else s
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    s = f"{mant_str}\\times 10^{{{exp}}}"
    return f"{s}\\,\\text{{{unit}}}" if unit else s

def fmt_html_10(val: float, unit: str = "", sig: int = 3):
    """Formata em HTML com ×10<sup>n</sup>."""
    if val == 0:
        return f"0 {unit}".strip()
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    return f"{mant_str}×10<sup>{exp}</sup> {unit}".strip()

def fmt_dec_pt(val: float, nd: int = 3):
    """Decimal com vírgula (quando quiser números não científicos)."""
    return f"{val:.{nd}f}".replace(".", ",")


# =========================
# Página
# =========================
st.set_page_config(
    page_title="Simulador Campo Elétrico do Aro – Física II",
    layout="wide"
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

# Limites (x apenas positivo, conforme pedido)
X_MIN, X_MAX = 0.01, 2.00     # m
A_MIN, A_MAX = 0.05, 1.00     # m

# Lambda: faixa maior e passo menor (muitos valores)
L_MIN, L_MAX = -2.0e-5, 2.0e-5  # C/m
L_STEP = 1.0e-7                # 401 passos -> responsivo, e muito mais que "3 valores"

colp1, colp2, colp3 = st.columns(3)

with colp1:
    x = st.slider("Distância x (m)", min_value=float(X_MIN), max_value=float(X_MAX),
                  value=0.40, step=0.01)

with colp2:
    lmbda = st.slider("Densidade linear λ (C/m)", min_value=float(L_MIN), max_value=float(L_MAX),
                      value=2.0e-6, step=float(L_STEP), format="%.7f")

with colp3:
    a = st.slider("Raio a (m)", min_value=float(A_MIN), max_value=float(A_MAX),
                  value=0.25, step=0.01)

# Cálculos principais
L = circumference(a)
Q = total_charge(lmbda, a)
Ex = field_on_axis(x, a, Q)

# Sentido do campo (no eixo)
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
# EMAX GLOBAL (eixos fixos p/ gráficos e seta)
# =========================
@st.cache_data(show_spinner=False)
def compute_global_emax():
    xs = np.linspace(X_MIN, X_MAX, 260)
    aas = np.linspace(A_MIN, A_MAX, 220)
    lam_abs = max(abs(L_MIN), abs(L_MAX))

    X, A = np.meshgrid(xs, aas)
    Qg = lam_abs * 2*np.pi*A
    E = K * X * Qg / (A*A + X*X)**1.5
    return float(1.15 * np.nanmax(np.abs(E)))

E_MAX_GLOBAL = compute_global_emax()


# =========================
# Imagem (escala fixa + cotas)
# =========================
st.subheader("Imagem")

# Janela fixa baseada nos limites máximos (não varia com sliders)
# Aro fixo em x=0
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

    # eixo do aro (tracejado)
    fig.add_trace(go.Scatter(
        x=[X_LEFT, X_RIGHT], y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # Aro em perspectiva (projeção)
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

    # Ponto P (apenas "P")
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
    # Vetor E em P (tamanho varia, mas sem vazar a janela fixa)
    # =========================
    # comprimento controlado por sqrt(|E|/Emax)
    max_arrow_len = 0.35 * (X_RIGHT - X_LEFT)
    min_arrow_len = 0.08 * (X_RIGHT - X_LEFT)
    frac = 0.0 if E_MAX_GLOBAL == 0 else min(1.0, abs(Ex) / E_MAX_GLOBAL)
    arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * np.sqrt(frac)
    dx = arrow_len if Ex >= 0 else -arrow_len

    x_end = x + dx
    # evita ultrapassar borda
    x_end = max(X_LEFT + 0.03*(X_RIGHT-X_LEFT), min(X_RIGHT - 0.03*(X_RIGHT-X_LEFT), x_end))

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

    # Texto do vetor E (valor) próximo ao vetor
    midx = (x + x_end) / 2
    fig.add_annotation(
        x=midx, y=0.15*Y_LIM,
        text=f"E = {fmt_html_10(Ex, 'N/C', sig=3)}",
        showarrow=False,
        font=dict(color="green", size=14),
        align="center"
    )

    # =========================
    # COTAS (dimension lines)
    # - Cota de x: do centro do aro (0) até P (x), abaixo do eixo
    # - Cota de a: do centro até topo do aro, à esquerda
    # =========================

    # Cota x (dupla seta)
    y_dimx = -0.35 * Y_LIM
    fig.add_annotation(
        x=x, y=y_dimx,
        ax=0, ay=y_dimx,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black"
    )
    fig.add_annotation(
        x=(x/2), y=y_dimx - 0.06*Y_LIM,
        text=f"x = {fmt_dec_pt(x, 3)} m",
        showarrow=False,
        font=dict(color="black", size=13)
    )

    # Linhas auxiliares (cota x)
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

    # Cota a (vertical à esquerda do aro)
    x_dima = -0.55 * BASE
    fig.add_annotation(
        x=x_dima, y=a,
        ax=x_dima, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black"
    )
    fig.add_annotation(
        x=x_dima, y=a/2,
        text=f"a = {fmt_dec_pt(a, 3)} m",
        showarrow=False,
        font=dict(color="black", size=13),
        textangle=-90
    )

    # Linhas auxiliares (cota a)
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

    # Caixa de informação (λ, a, x, Q)
    info = (
        f"λ = {fmt_html_10(lmbda, 'C/m', sig=3)}<br>"
        f"Q = {fmt_html_10(Q, 'C', sig=3)}<br>"
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

    # Layout fixo + pan (celular)
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="pan"  # permite "deslizar"
    )

    # Escalas fixas (não mudam com sliders)
    fig.update_xaxes(range=[X_LEFT, X_RIGHT], visible=False, fixedrange=False)
    fig.update_yaxes(range=[-Y_LIM, Y_LIM], visible=False, scaleanchor="x", scaleratio=1, fixedrange=False)

    return fig

scene = make_scene_figure(x, a, lmbda, Q, Ex)
st.plotly_chart(
    scene,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": False}
)
st.caption("📱 No celular: arraste a figura para os lados (pan) para ver tudo sem perder detalhes.")

st.divider()

# =========================
# Equações (com títulos pedidos)
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
# Cálculos (com 10^n e seta final)
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

# Resultado + seta final (pedido 8)
st.latex(rf"E_x = {fmt_latex_10(Ex,'N/C',sig=4)}\quad {sentido_seta}")

st.markdown(f"**Sentido do campo em P:** {sentido_texto} **{sentido_seta}**")

st.divider()

# =========================
# Gráficos (títulos + eixos pretos + 10^n)
# =========================
st.subheader("Gráficos")

def curve_E_vs_x(a, Q):
    xs = np.linspace(X_MIN, X_MAX, 400)
    E = K * xs * Q / (a*a + xs*xs)**1.5
    return xs, E

def curve_E_vs_a(x, lmbda):
    aas = np.linspace(A_MIN, A_MAX, 400)
    Qs = lmbda * 2*np.pi*aas
    E = K * x * Qs / (aas*aas + x*x)**1.5
    return aas, E

def curve_E_vs_Q(x, a):
    # varia Q diretamente (mantendo x e a fixos)
    # limites coerentes com lambda max e a max (para faixa ampla)
    Qmin = L_MIN * 2*np.pi*A_MAX
    Qmax = L_MAX * 2*np.pi*A_MAX
    Qs = np.linspace(Qmin, Qmax, 400)
    E = K * x * Qs / (a*a + x*x)**1.5
    return Qs, E

def style_axes_black(fig):
    fig.update_xaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power"  # 10^n
    )
    fig.update_yaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power"  # 10^n
    )
    return fig

gx1, gx2, gx3 = st.columns(3)

with gx1:
    xs, Es = curve_E_vs_x(a, Q)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xs, y=Es, mode="lines",
                              line=dict(color="#1f77b4", width=3)))
    fig1.add_trace(go.Scatter(x=[x], y=[Ex], mode="markers",
                              marker=dict(color="red", size=10)))
    fig1.update_layout(
        title="Campo elétrico em função da distância x",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig1.update_xaxes(title="x (m)", range=[X_MIN, X_MAX], zeroline=True)
    fig1.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    style_axes_black(fig1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with gx2:
    aas, Ea = curve_E_vs_a(x, lmbda)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=aas, y=Ea, mode="lines",
                              line=dict(color="#2ca02c", width=3)))
    fig2.add_trace(go.Scatter(x=[a], y=[Ex], mode="markers",
                              marker=dict(color="red", size=10)))
    fig2.update_layout(
        title="Campo elétrico em função do raio a",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig2.update_xaxes(title="a (m)", range=[A_MIN, A_MAX], zeroline=True)
    fig2.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    style_axes_black(fig2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with gx3:
    Qs, EQ = curve_E_vs_Q(x, a)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Qs, y=EQ, mode="lines",
                              line=dict(color="#9467bd", width=3)))
    fig3.add_trace(go.Scatter(x=[Q], y=[Ex], mode="markers",
                              marker=dict(color="red", size=10)))
    fig3.update_layout(
        title="Campo elétrico em função da carga total Q",
        title_font=dict(color="black"),
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig3.update_xaxes(title="Q (C)", range=[float(Qs.min()), float(Qs.max())], zeroline=True)
    fig3.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    style_axes_black(fig3)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.caption("🔴 O ponto vermelho indica a situação atual escolhida nos sliders. Eixos fixos para facilitar a comparação.")
