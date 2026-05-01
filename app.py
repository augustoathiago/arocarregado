import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Constantes e funções físicas
# -----------------------------
K = 9.0e9  # 1/(4*pi*epsilon0) em N·m²/C² (conforme solicitado)

def circumference(a: float) -> float:
    return 2.0 * np.pi * a

def total_charge(lmbda: float, a: float) -> float:
    return lmbda * circumference(a)

def field_on_axis(x: float, a: float, Q: float) -> float:
    # Ex = k * x Q / (a^2 + x^2)^(3/2)
    denom = (a*a + x*x) ** 1.5
    if denom == 0:
        return 0.0
    return K * x * Q / denom

def fmt_si(x, unit="", digits=3):
    # formatação simples com notação científica quando necessário
    if x == 0:
        return f"0 {unit}".strip()
    ax = abs(x)
    if ax < 1e-2 or ax >= 1e3:
        return f"{x:.{digits}e} {unit}".strip()
    return f"{x:.{digits}f} {unit}".strip()

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Simulador Campo Elétrico do Aro – Física II",
    layout="wide"
)

# -----------------------------
# Cabeçalho (duas colunas)
# -----------------------------
c1, c2 = st.columns([1, 4], vertical_alignment="center")
with c1:
    try:
        st.image("logo_maua.png", use_container_width=True)
    except Exception:
        st.warning("Coloque o arquivo **logo_maua.png** na pasta do app para exibir o logo.")

with c2:
    st.markdown(
        """
        # Simulador Campo Elétrico do Aro Física II
        **Estude o campo elétrico gerado por um aro carregado em um ponto P sobre o eixo do aro.**
        """
    )

st.divider()

# -----------------------------
# Sliders (Parâmetros)
# -----------------------------
st.subheader("Parâmetros")

# Limites escolhidos para cobrir casos típicos e manter gráficos bem comportados
X_MIN, X_MAX = 0.01, 2.00      # m
A_MIN, A_MAX = 0.05, 1.00      # m
L_MIN, L_MAX = -5e-6, 5e-6     # C/m (permite negativo)

colp1, colp2, colp3 = st.columns(3)
with colp1:
    x = st.slider("Distância x (m)", min_value=float(X_MIN), max_value=float(X_MAX),
                  value=0.40, step=0.01)
with colp2:
    lmbda = st.slider("Densidade linear λ (C/m)", min_value=float(L_MIN), max_value=float(L_MAX),
                      value=2.0e-6, step=1.0e-7, format="%.2e")
with colp3:
    a = st.slider("Raio a (m)", min_value=float(A_MIN), max_value=float(A_MAX),
                  value=0.25, step=0.01)

L = circumference(a)
Q = total_charge(lmbda, a)
Ex = field_on_axis(x, a, Q)

# Sentido do campo
if Ex > 0:
    sentido = "→ (para a direita)"
    seta = "→"
elif Ex < 0:
    sentido = "← (para a esquerda)"
    seta = "←"
else:
    sentido = "0 (nulo)"
    seta = "•"

st.divider()

# ------------------------------------------------
# Função para estimar Emax global (para eixos fixos)
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_global_emax():
    # Grid pequeno para obter um Emax robusto (mantém eixos fixos)
    xs = np.linspace(X_MIN, X_MAX, 160)
    aas = np.linspace(A_MIN, A_MAX, 140)
    # usamos |lambda| max e variamos sinais separadamente
    lam_abs = max(abs(L_MIN), abs(L_MAX))

    # Q = lambda * 2pi a
    # Ex(x,a) = k * x * (lambda*2pi a) / (a^2+x^2)^(3/2)
    X, A = np.meshgrid(xs, aas)
    Qgrid = lam_abs * 2*np.pi*A
    E = K * X * Qgrid / (A*A + X*X)**1.5
    Emax = float(np.nanmax(np.abs(E)))
    # margem extra para não encostar
    return 1.15 * Emax

E_MAX_GLOBAL = compute_global_emax()

# -----------------------------
# Seção Imagem
# -----------------------------
st.subheader("Imagem")

def make_scene_figure(x, a, lmbda, Q, Ex):
    # Cores do aro conforme sinal da carga total
    if Q > 0:
        ring_color = "red"
    elif Q < 0:
        ring_color = "blue"
    else:
        ring_color = "black"

    # Definimos uma "janela" que garanta que nada ultrapasse os limites
    # Mantemos o aro (centro em 0) à esquerda e o ponto P à direita.
    # Usamos margem baseada no maior entre a e x.
    base = max(a, x)
    x_left = -1.25 * base
    x_right = 1.65 * base
    y_lim = 1.35 * a

    # Aro em perspectiva: círculo no plano yz projetado em (x,y)
    t = np.linspace(0, 2*np.pi, 500)
    # projeção: x recebe um "deslocamento" proporcional a sin(t) para dar perspectiva
    persp = 0.35
    ring_x = 0.0 + persp * a * np.sin(t)
    ring_y = a * np.cos(t)

    fig = go.Figure()

    # eixo do aro (linha tracejada)
    fig.add_trace(go.Scatter(
        x=[x_left, x_right],
        y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Eixo"
    ))

    # aro
    fig.add_trace(go.Scatter(
        x=ring_x, y=ring_y,
        mode="lines",
        line=dict(color=ring_color, width=5),
        name="Aro"
    ))

    # ponto P
    fig.add_trace(go.Scatter(
        x=[x], y=[0],
        mode="markers+text",
        marker=dict(size=12, color="black"),
        text=["P"],
        textposition="top center",
        name="Ponto P"
    ))

    # vetor E em P (seta)
    # Escala não linear para caber sempre (sqrt) e usar E_MAX_GLOBAL
    # Comprimento máximo da seta em unidades do gráfico:
    max_arrow_len = 0.55 * (x_right - x_left)
    # comprimento mínimo perceptível
    min_arrow_len = 0.08 * (x_right - x_left)

    frac = 0.0 if E_MAX_GLOBAL == 0 else min(1.0, (abs(Ex) / E_MAX_GLOBAL))
    arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * np.sqrt(frac)

    # Direção da seta depende do sinal de Ex
    dx = arrow_len if Ex >= 0 else -arrow_len
    x_end = x + dx

    # Garante que a seta não "vaze" da janela
    x_end = max(x_left + 0.05*(x_right-x_left), min(x_right - 0.05*(x_right-x_left), x_end))

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

    # Label do vetor E (com seta acima via LaTeX não renderiza dentro do plotly nativamente,
    # então usamos texto "E" próximo da seta e no texto da página usamos \vec{E})
    midx = (x + x_end) / 2
    fig.add_annotation(
        x=midx, y=0.14*y_lim,
        text=f"E = {fmt_si(Ex, 'N/C', digits=3)}",
        showarrow=False,
        font=dict(color="green", size=14)
    )

    # Anotações dos valores
    text_info = (
        f"λ = {fmt_si(lmbda, 'C/m', digits=3)}<br>"
        f"a = {fmt_si(a, 'm', digits=3)}<br>"
        f"x = {fmt_si(x, 'm', digits=3)}<br>"
        f"Q = {fmt_si(Q, 'C', digits=3)}"
    )
    fig.add_annotation(
        x=x_left + 0.02*(x_right-x_left),
        y=y_lim - 0.02*(2*y_lim),
        text=text_info,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=13, color="black")
    )

    # Layout: fundo branco, pan habilitado (bom no celular para "deslizar" a imagem)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="pan"
    )
    fig.update_xaxes(range=[x_left, x_right], visible=False)
    fig.update_yaxes(range=[-y_lim, y_lim], visible=False, scaleanchor="x", scaleratio=1)

    return fig

scene = make_scene_figure(x, a, lmbda, Q, Ex)
st.plotly_chart(
    scene,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": False}
)

st.caption("💡 No celular, você pode **arrastar** (pan) para os lados com o dedo para visualizar toda a imagem.")

st.divider()

# -----------------------------
# Seção Equações
# -----------------------------
st.subheader("Equações")

st.latex(r"L = 2\pi a")
st.latex(r"Q = \lambda\,L = \lambda(2\pi a)")
st.latex(r"E_x = \frac{1}{4\pi\varepsilon_0}\,\frac{x\,Q}{(a^2+x^2)^{3/2}}")
st.latex(r"\frac{1}{4\pi\varepsilon_0} = 9,0\times10^9\ \text{N·m}^2/\text{C}^2")

st.divider()

# -----------------------------
# Seção Cálculos (substituição numérica)
# -----------------------------
st.subheader("Cálculos")

st.latex(rf"L = 2\pi a = 2\pi({a:.3f}) = {L:.6f}\ \text{{m}}")
st.latex(rf"Q = \lambda L = ({lmbda:.3e})({L:.6f}) = {Q:.6e}\ \text{{C}}")
st.latex(rf"E_x = (9,0\times 10^9)\,\frac{{xQ}}{{(a^2+x^2)^{{3/2}}}}")
st.latex(rf"E_x = (9,0\times 10^9)\,\frac{{({x:.3f})({Q:.6e})}}{{(({a:.3f})^2+({x:.3f})^2)^{{3/2}}}}")
st.latex(rf"E_x = {Ex:.6e}\ \text{{N/C}}")

st.markdown(f"**Sentido do campo em P:** {sentido}  **{seta}**")

st.divider()

# -----------------------------
# Seção Gráficos
# -----------------------------
st.subheader("Gráficos")

# Funções para curvas
def curve_E_vs_x(a, Q):
    xs = np.linspace(X_MIN, X_MAX, 300)
    E = K * xs * Q / (a*a + xs*xs)**1.5
    return xs, E

def curve_E_vs_a(x, lmbda):
    aas = np.linspace(A_MIN, A_MAX, 300)
    Qs = lmbda * 2*np.pi*aas
    E = K * x * Qs / (aas*aas + x*x)**1.5
    return aas, E

def curve_E_vs_Q(x, a):
    # Q independente (varia linearmente), mantendo x e a fixos
    Qmin = L_MIN * 2*np.pi*A_MAX
    Qmax = L_MAX * 2*np.pi*A_MAX
    Qs = np.linspace(Qmin, Qmax, 300)
    E = K * x * Qs / (a*a + x*x)**1.5
    return Qs, E

# Gráfico 1: E(x)
gx1, gx2, gx3 = st.columns(3)

with gx1:
    xs, Es = curve_E_vs_x(a, Q)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xs, y=Es, mode="lines", line=dict(color="#1f77b4", width=3), name="E(x)"))
    fig1.add_trace(go.Scatter(x=[x], y=[Ex], mode="markers", marker=dict(color="red", size=10), name="Atual"))
    fig1.update_layout(
        title="Campo elétrico em função da distância x",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig1.update_xaxes(title="x (m)", range=[X_MIN, X_MAX], zeroline=True)
    fig1.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with gx2:
    aas, Ea = curve_E_vs_a(x, lmbda)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=aas, y=Ea, mode="lines", line=dict(color="#2ca02c", width=3), name="E(a)"))
    fig2.add_trace(go.Scatter(x=[a], y=[Ex], mode="markers", marker=dict(color="red", size=10), name="Atual"))
    fig2.update_layout(
        title="Campo elétrico em função do raio a",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig2.update_xaxes(title="a (m)", range=[A_MIN, A_MAX], zeroline=True)
    fig2.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with gx3:
    Qs, EQ = curve_E_vs_Q(x, a)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Qs, y=EQ, mode="lines", line=dict(color="#9467bd", width=3), name="E(Q)"))
    fig3.add_trace(go.Scatter(x=[Q], y=[Ex], mode="markers", marker=dict(color="red", size=10), name="Atual"))
    fig3.update_layout(
        title="Campo elétrico em função da carga total Q",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig3.update_xaxes(title="Q (C)", range=[float(Qs.min()), float(Qs.max())], zeroline=True)
    fig3.update_yaxes(title="Eₓ (N/C)", range=[-E_MAX_GLOBAL, E_MAX_GLOBAL], zeroline=True)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.caption("🔴 O ponto vermelho indica a situação atual escolhida nos sliders. Eixos fixos para comparação visual consistente.")
