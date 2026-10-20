import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
import io
import re
import math

st.set_page_config(page_title="Métodos Numéricos - Raízes", layout="wide")

# ============================================================
# FUNÇÃO
# ============================================================
def criar_funcao(expressao):
    expr = expressao.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("X", "x")
    expr = expr.replace(" ", "")
    expr = re.sub(r'(?<=\d),(?=\d)', '.', expr)

    substituicoes = {
        "sen": "sin",
        "tg": "tan",
        "raiz": "sqrt",
        "ln": "log",
    }
    for a, b in substituicoes.items():
        expr = re.sub(rf"\b{a}\b", b, expr, flags=re.IGNORECASE)

    # Multiplicação implícita: 2x, 3(x+1), x(x+1), )(, 2pi, xpi
    expr = re.sub(r'(?<=\d)(?=x)', '*', expr)
    expr = re.sub(r'(?<=\d)(?=\()', '*', expr)
    expr = re.sub(r'(?<=x)(?=\()', '*', expr)
    expr = re.sub(r'(?<=\))(?=\()', '*', expr)
    expr = re.sub(r'(?<=\))(?=x)', '*', expr)
    expr = re.sub(r'(?<=\d)(?=pi\b)', '*', expr)
    expr = re.sub(r'(?<=x)(?=pi\b)', '*', expr)

    def f(x):
        ambiente = {
            "x": x, "e": math.e, "pi": math.pi,
            "exp": math.exp, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "sqrt": math.sqrt, "log": math.log,
            "abs": abs, "pow": pow
        }
        return float(eval(expr, {"__builtins__": {}}, ambiente))

    # Testa a expressão
    f(0.0)
    return f, expr


# ============================================================
# MÉTODOS
# ============================================================
def metodo_bissecao(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Na Bisseção, f(a) e f(b) precisam ter sinais opostos.")

    linhas = []
    geometria = []

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        erro = abs(b - a) / 2

        linhas.append([i, a, b, c, fa, fb, fc, erro])
        geometria.append((a, b, c))

        if abs(fc) < tol or erro < tol:
            return c, pd.DataFrame(
                linhas,
                columns=["Iteração","a","b","c","f(a)","f(b)","f(c)","Erro"]
            ), geometria

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return c, pd.DataFrame(
        linhas,
        columns=["Iteração","a","b","c","f(a)","f(b)","f(c)","Erro"]
    ), geometria


def metodo_falsa_posicao(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Na Falsa Posição, f(a) e f(b) precisam ter sinais opostos.")

    linhas = []
    geometria = []

    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        erro = abs(fc)

        linhas.append([i, a, b, c, fa, fb, fc, erro])
        geometria.append((a, fa, b, fb, c, fc))

        if abs(fc) < tol:
            return c, pd.DataFrame(
                linhas,
                columns=["Iteração","a","b","c","f(a)","f(b)","f(c)","Erro"]
            ), geometria

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return c, pd.DataFrame(
        linhas,
        columns=["Iteração","a","b","c","f(a)","f(b)","f(c)","Erro"]
    ), geometria


def metodo_ponto_fixo(f, g, x0, tol, max_iter):
    linhas = []
    geometria = []
    x = x0

    for i in range(1, max_iter + 1):
        x1 = g(x)
        erro = abs(x1 - x)

        linhas.append([i, x, x1, f(x1), erro])
        geometria.append((x, x1))

        if erro < tol:
            return x1, pd.DataFrame(
                linhas,
                columns=["Iteração","xᵢ","xᵢ₊₁","f(xᵢ₊₁)","Erro"]
            ), geometria
        x = x1

    return x, pd.DataFrame(
        linhas,
        columns=["Iteração","xᵢ","xᵢ₊₁","f(xᵢ₊₁)","Erro"]
    ), geometria


def metodo_newton(f, df, x0, tol, max_iter):
    linhas = []
    geometria = []
    x = x0

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-14:
            raise ValueError("Derivada muito próxima de zero durante Newton-Raphson.")

        x1 = x - fx / dfx
        erro = abs(x1 - x)

        linhas.append([i, x, fx, dfx, x1, erro])
        geometria.append((x, fx, dfx, x1))

        if erro < tol:
            return x1, pd.DataFrame(
                linhas,
                columns=["Iteração","xᵢ","f(xᵢ)","f'(xᵢ)","xᵢ₊₁","Erro"]
            ), geometria
        x = x1

    return x, pd.DataFrame(
        linhas,
        columns=["Iteração","xᵢ","f(xᵢ)","f'(xᵢ)","xᵢ₊₁","Erro"]
    ), geometria


def metodo_secantes(f, x0, x1, tol, max_iter):
    linhas = []
    geometria = []

    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < 1e-14:
            raise ValueError("Denominador muito próximo de zero no método das Secantes.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        erro = abs(x2 - x1)

        linhas.append([i, x0, x1, x2, f0, f1, f2, erro])
        geometria.append((x0, f0, x1, f1, x2, f2))

        if abs(f2) < tol or erro < tol:
            return x2, pd.DataFrame(
                linhas,
                columns=["Iteração","x₀","x₁","x₂","f(x₀)","f(x₁)","f(x₂)","Erro"]
            ), geometria

        x0, x1 = x1, x2

    return x2, pd.DataFrame(
        linhas,
        columns=["Iteração","x₀","x₁","x₂","f(x₀)","f(x₁)","f(x₂)","Erro"]
    ), geometria


# ============================================================
# GRÁFICO POR ITERAÇÃO
# ============================================================
def criar_grafico(f, metodo, geometria, iteracao_exibir=None, g=None):
    if not geometria:
        return None

    n = len(geometria) if iteracao_exibir is None else min(iteracao_exibir, len(geometria))
    dados = geometria[:n]

    xs = []
    for item in dados:
        xs.extend([v for v in item if isinstance(v, (int, float)) and math.isfinite(v)])

    if xs:
        xmin, xmax = min(xs), max(xs)
        margem = max((xmax - xmin) * 0.30, 1.0)
    else:
        xmin, xmax, margem = -5, 5, 1

    xmin -= margem
    xmax += margem

    xgrid = np.linspace(xmin, xmax, 700)
    ygrid = []
    for x in xgrid:
        try:
            y = f(float(x))
            ygrid.append(y if math.isfinite(y) and abs(y) < 1e5 else np.nan)
        except Exception:
            ygrid.append(np.nan)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xgrid, ygrid, label="f(x)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # Mostra a geometria de TODAS as iterações até a escolhida.
    for k, item in enumerate(dados, start=1):
        alpha = 0.25 + 0.65 * (k / max(n, 1))
        if metodo == "Bisseção":
            a, b, c = item
            try:
                ax.plot([a, b], [f(a), f(b)], linestyle="--", alpha=alpha)
                ax.plot(c, f(c), "o", alpha=alpha)
                ax.vlines(c, 0, f(c), linestyle=":", alpha=alpha)
                ax.annotate(f"I{k}: c={c:.6g}", (c, f(c)),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
            except Exception:
                pass

        elif metodo == "Falsa Posição":
            a, fa, b, fb, c, fc = item
            ax.plot([a, b], [fa, fb], linestyle="--", alpha=alpha)
            ax.plot(c, fc, "o", alpha=alpha)
            ax.vlines(c, 0, fc, linestyle=":", alpha=alpha)
            ax.annotate(f"I{k}", (c, fc), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)

        elif metodo == "Ponto Fixo":
            x0, x1 = item
            try:
                # Cobweb: (x0,x0) -> (x0,g(x0)) -> (x1,x1)
                gx0 = g(x0)
                ax.plot([x0, x0], [x0, gx0], linestyle="--", alpha=alpha)
                ax.plot([x0, x1], [gx0, gx0], linestyle="--", alpha=alpha)
                ax.plot(x1, x1, "o", alpha=alpha)
                ax.annotate(f"I{k}", (x1, x1), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
            except Exception:
                pass

        elif metodo == "Newton-Raphson":
            x0, fx0, dfx0, x1 = item
            tx = np.array([xmin, xmax])
            ty = fx0 + dfx0 * (tx - x0)
            ax.plot(tx, ty, linestyle="--", alpha=alpha)
            ax.plot(x0, fx0, "o", alpha=alpha)
            ax.plot(x1, 0, "o", alpha=alpha)
            ax.vlines(x1, 0, f(x1), linestyle=":", alpha=alpha)
            ax.annotate(f"I{k}", (x1, 0), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)

        elif metodo == "Secantes":
            x0, f0, x1, f1, x2, f2 = item
            if abs(x1 - x0) > 1e-14:
                m = (f1 - f0) / (x1 - x0)
                tx = np.array([xmin, xmax])
                ty = f0 + m * (tx - x0)
                ax.plot(tx, ty, linestyle="--", alpha=alpha)
            ax.plot([x0, x1], [f0, f1], "o", alpha=alpha)
            ax.plot(x2, 0, "o", alpha=alpha)
            ax.vlines(x2, 0, f2, linestyle=":", alpha=alpha)
            ax.annotate(f"I{k}", (x2, 0), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)

    if metodo == "Ponto Fixo" and g is not None:
        try:
            gy = np.array([g(float(x)) for x in xgrid], dtype=float)
            gy[~np.isfinite(gy)] = np.nan
            ax.plot(xgrid, gy, label="g(x)")
            ax.plot(xgrid, xgrid, linestyle="-.", label="y=x")
        except Exception:
            pass

    ax.set_title(f"{metodo} — evolução até a iteração {n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig



# ============================================================
# GRÁFICO INTERATIVO - PLOTLY
# ============================================================
def criar_grafico_interativo(f, metodo, geometria, iteracao_exibir=None, g=None):
    if not geometria:
        return None

    n = len(geometria) if iteracao_exibir is None else min(iteracao_exibir, len(geometria))
    dados = geometria[:n]

    xs = []
    for item in dados:
        for v in item:
            if isinstance(v, (int, float)) and math.isfinite(v):
                xs.append(float(v))

    if xs:
        xmin, xmax = min(xs), max(xs)
    else:
        xmin, xmax = -2, 2

    faixa = xmax - xmin
    if faixa < 0.5:
        centro = (xmin + xmax) / 2
        xmin, xmax = centro - 1, centro + 1
    else:
        margem = faixa * 0.25
        xmin -= margem
        xmax += margem

    xgrid = np.linspace(xmin, xmax, 350)
    ygrid = []
    for x in xgrid:
        try:
            y = float(f(float(x)))
            ygrid.append(y if math.isfinite(y) and abs(y) < 1e6 else np.nan)
        except Exception:
            ygrid.append(np.nan)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xgrid, y=ygrid, mode="lines", name="f(x)",
        line=dict(width=3),
        hovertemplate="x = %{x:.6f}<br>f(x) = %{y:.6f}<extra>f(x)</extra>"
    ))
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)

    for k, item in enumerate(dados, start=1):
        if metodo == "Bisseção":
            a, b, c = item
            fa, fb, fc = f(a), f(b), f(c)

            fig.add_trace(go.Scatter(
                x=[a, b], y=[fa, fb], mode="lines",
                name=f"I{k} - intervalo",
                line=dict(dash="dash", width=1.5),
                hovertemplate=f"I{k}<br>x = %{{x:.6f}}<br>f(x) = %{{y:.6f}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[c, c], y=[0, fc], mode="lines",
                line=dict(dash="dot", width=1),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[c], y=[fc], mode="markers+text",
                text=[f"I{k}"], textposition="top center",
                marker=dict(size=9), showlegend=False,
                hovertemplate=f"I{k}<br>c = %{{x:.10f}}<br>f(c) = %{{y:.10f}}<extra></extra>"
            ))

        elif metodo == "Falsa Posição":
            a, fa, b, fb, c, fc = item
            fig.add_trace(go.Scatter(
                x=[a, b], y=[fa, fb], mode="lines",
                name=f"I{k} - reta secante",
                line=dict(dash="dash", width=1.8),
                hovertemplate=f"I{k} - secante<br>x = %{{x:.6f}}<br>y = %{{y:.6f}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[c, c], y=[0, fc], mode="lines",
                line=dict(dash="dot", width=1),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[c], y=[fc], mode="markers+text",
                text=[f"I{k}"], textposition="top center",
                marker=dict(size=9), showlegend=False,
                hovertemplate=f"I{k}<br>c = %{{x:.10f}}<br>f(c) = %{{y:.10f}}<extra></extra>"
            ))

        elif metodo == "Ponto Fixo":
            x0, x1 = item
            try:
                gx0 = g(x0)
                fig.add_trace(go.Scatter(
                    x=[x0, x0], y=[x0, gx0], mode="lines",
                    line=dict(dash="dash", width=1.5),
                    hoverinfo="skip", showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[gx0, gx0], mode="lines",
                    line=dict(dash="dash", width=1.5),
                    hovertemplate=f"I{k}<br>x = %{{x:.6f}}<br>g(x) = %{{y:.6f}}<extra></extra>",
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[x1], y=[x1], mode="markers+text",
                    text=[f"I{k}"], textposition="top center",
                    marker=dict(size=9), name=f"I{k}",
                    hovertemplate=f"I{k}<br>x = %{{x:.10f}}<extra></extra>"
                ))
            except Exception:
                pass

        elif metodo == "Newton-Raphson":
            x0, fx0, dfx0, x1 = item
            tx = np.array([xmin, xmax])
            ty = fx0 + dfx0 * (tx - x0)

            fig.add_trace(go.Scatter(
                x=tx, y=ty, mode="lines",
                name=f"I{k} - tangente",
                line=dict(dash="dash", width=1.8),
                hovertemplate=f"I{k} - tangente<br>x = %{{x:.6f}}<br>y = %{{y:.6f}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[x1, x1], y=[0, f(x1)], mode="lines",
                line=dict(dash="dot", width=1),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x1], y=[0], mode="markers+text",
                text=[f"I{k}"], textposition="top center",
                marker=dict(size=9), showlegend=False,
                hovertemplate=f"I{k}<br>x = %{{x:.10f}}<extra></extra>"
            ))

        elif metodo == "Secantes":
            x0, f0, x1, f1, x2, f2 = item
            if abs(x1 - x0) > 1e-14:
                m = (f1 - f0) / (x1 - x0)
                tx = np.array([xmin, xmax])
                ty = f0 + m * (tx - x0)
                fig.add_trace(go.Scatter(
                    x=tx, y=ty, mode="lines",
                    name=f"I{k} - reta secante",
                    line=dict(dash="dash", width=1.8),
                    hovertemplate=f"I{k} - secante<br>x = %{{x:.6f}}<br>y = %{{y:.6f}}<extra></extra>"
                ))
            fig.add_trace(go.Scatter(
                x=[x2, x2], y=[0, f2], mode="lines",
                line=dict(dash="dot", width=1),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x2], y=[0], mode="markers+text",
                text=[f"I{k}"], textposition="top center",
                marker=dict(size=9), showlegend=False,
                hovertemplate=f"I{k}<br>x = %{{x:.10f}}<extra></extra>"
            ))

    if metodo == "Ponto Fixo" and g is not None:
        gy = []
        for x in xgrid:
            try:
                val = float(g(float(x)))
                gy.append(val if math.isfinite(val) and abs(val) < 1e6 else np.nan)
            except Exception:
                gy.append(np.nan)

        fig.add_trace(go.Scatter(
            x=xgrid, y=gy, mode="lines", name="g(x)",
            line=dict(width=3),
            hovertemplate="x = %{x:.6f}<br>g(x) = %{y:.6f}<extra>g(x)</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=xgrid, y=xgrid, mode="lines", name="y = x",
            line=dict(dash="dash", width=2),
            hovertemplate="x = %{x:.6f}<br>y = %{y:.6f}<extra>y=x</extra>"
        ))

    # Escala automática: o Plotly escolhe as divisões do eixo conforme o zoom.
    # Isso evita milhares de marcas decimais e deixa o gráfico mais leve.
    fig.update_xaxes(showgrid=True, zeroline=True, nticks=12)
    fig.update_yaxes(showgrid=True, zeroline=True, nticks=12)

    fig.update_layout(
        title=f"{metodo} — evolução até a iteração {n}",
        xaxis_title="x",
        yaxis_title="y",
        height=650,
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="zoom"
    )
    return fig

# ============================================================
# PDF
# ============================================================
def gerar_pdf(expressao, metodo, raiz, tabela, fig):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig_pdf = fig
        pdf.savefig(fig_pdf, bbox_inches="tight")

        fig2, ax2 = plt.subplots(figsize=(11, 8.5))
        ax2.axis("off")
        texto = (
            f"RELATÓRIO - MÉTODOS NUMÉRICOS\n\n"
            f"Método: {metodo}\n"
            f"Função: {expressao}\n"
            f"Raiz aproximada: {raiz:.12g}\n"
            f"Iterações: {len(tabela)}\n"
        )
        ax2.text(0.03, 0.95, texto, va="top", fontsize=13)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    buffer.seek(0)
    return buffer


# ============================================================
# INTERFACE
# ============================================================
st.title("🔢 Métodos Numéricos para Determinação de Raízes - Amanda, David, Samuel e Marcos - Prof.: José Carlos")
st.write("Escolha um método. Os campos serão solicitados de acordo com o método escolhido.")

metodo = st.sidebar.selectbox(
    "Escolha o método:",
    ["Bisseção", "Falsa Posição", "Ponto Fixo", "Newton-Raphson", "Secantes"]
)

expressao = st.sidebar.text_input("f(x)", value="x^3-x-2")

a = b = x0 = x1 = None
expressao_g = None

if metodo in ["Bisseção", "Falsa Posição"]:
    a = st.sidebar.number_input("a", value=1.0)
    b = st.sidebar.number_input("b", value=2.0)
elif metodo == "Ponto Fixo":
    expressao_g = st.sidebar.text_input("g(x)", value="(x+2)^(1/3)")
    x0 = st.sidebar.number_input("x₀", value=1.5)
elif metodo == "Newton-Raphson":
    expressao_g = st.sidebar.text_input("f'(x)", value="3*x^2-1")
    x0 = st.sidebar.number_input("x₀", value=1.5)
else:
    x0 = st.sidebar.number_input("x₀", value=1.0)
    x1 = st.sidebar.number_input("x₁", value=2.0)

tol = st.sidebar.number_input("Tolerância", value=0.000001, format="%.10f")
max_iter = st.sidebar.number_input("Máximo de iterações", min_value=1, value=100, step=1)

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if st.sidebar.button("🚀 CALCULAR"):
    try:
        f, expr_ok = criar_funcao(expressao)

        if metodo == "Bisseção":
            raiz, tabela, geometria = metodo_bissecao(f, a, b, tol, max_iter)
            gfun = None

        elif metodo == "Falsa Posição":
            raiz, tabela, geometria = metodo_falsa_posicao(f, a, b, tol, max_iter)
            gfun = None

        elif metodo == "Ponto Fixo":
            gfun, _ = criar_funcao(expressao_g)
            raiz, tabela, geometria = metodo_ponto_fixo(f, gfun, x0, tol, max_iter)

        elif metodo == "Newton-Raphson":
            dfun, _ = criar_funcao(expressao_g)
            raiz, tabela, geometria = metodo_newton(f, dfun, x0, tol, max_iter)
            gfun = None

        else:
            raiz, tabela, geometria = metodo_secantes(f, x0, x1, tol, max_iter)
            gfun = None

        st.session_state.resultado = {
            "raiz": raiz,
            "tabela": tabela,
            "geometria": geometria,
            "f": f,
            "metodo": metodo,
            "g": gfun,
            "expressao": expr_ok
        }
        st.success("Cálculo realizado com sucesso!")

    except Exception as e:
        st.session_state.resultado = None
        st.error(f"Não foi possível realizar o cálculo.\n\nErro: {e}")

# ============================================================
# RESULTADOS
# ============================================================
res = st.session_state.resultado

if res:
    st.subheader("Resultado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Método", res["metodo"])
    col2.metric("Raiz aproximada", f'{res["raiz"]:.12g}')
    col3.metric("Iterações", len(res["tabela"]))

    st.subheader("📈 Comportamento das retas em cada iteração")

    total = len(res["geometria"])
    opcoes = ["Todas as iterações"] + list(range(1, total + 1))
    escolha = st.selectbox(
        "Escolha quantas iterações deseja visualizar:",
        opcoes
    )

    if escolha == "Todas as iterações":
        n_exibir = total
    else:
        n_exibir = int(escolha)

    fig_interativo = criar_grafico_interativo(
        res["f"],
        res["metodo"],
        res["geometria"],
        iteracao_exibir=n_exibir,
        g=res["g"]
    )

    if fig_interativo:
        st.plotly_chart(
            fig_interativo,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "responsive": True
            }
        )

    st.info(
        "💡 Use a roda do mouse para dar zoom, arraste para navegar e use "
        "a barra de ferramentas para restaurar o enquadramento. A escala "
        "dos eixos agora é automática para deixar o gráfico mais leve."
    )

    st.subheader("📋 Tabela de iterações")
    st.dataframe(res["tabela"], use_container_width=True)

    fig_pdf = criar_grafico(
        res["f"], res["metodo"], res["geometria"],
        iteracao_exibir=total, g=res["g"]
    )
    if fig_pdf:
        pdf = gerar_pdf(
            res["expressao"], res["metodo"], res["raiz"],
            res["tabela"], fig_pdf
        )
        plt.close(fig_pdf)

        nome = re.sub(r"[^a-zA-Z0-9_.-]+", "_", res["expressao"])[:80]
        st.download_button(
            "📄 Baixar relatório PDF",
            data=pdf,
            file_name=f"{nome}_{res['metodo'].replace(' ','_')}.pdf",
            mime="application/pdf"
        )