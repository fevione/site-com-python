import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
# -----------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoBrasil Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    /* Deixar o layout mais "clean" */
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .css-18e3th9 {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------------------------------------------------------
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados(caminho: str = "base_ficticia_vendas_carros.csv") -> pd.DataFrame:
    df_ = pd.read_csv(caminho)
    # Garantir tipos adequados
    df_["Data_Venda"] = pd.to_datetime(df_["Data_Venda"], errors="coerce")
    df_["Ano_Venda"] = df_["Data_Venda"].dt.year
    df_["Mes_Venda"] = df_["Data_Venda"].dt.month
    df_["Mes_Ano"] = df_["Data_Venda"].dt.to_period("M").astype(str)
    # Conversões numéricas
    df_["Valor_Venda"] = pd.to_numeric(df_["Valor_Venda"], errors="coerce")
    df_["Ano_Modelo"] = pd.to_numeric(df_["Ano_Modelo"], errors="coerce")
    df_["Idade"] = pd.to_numeric(df_["Idade"], errors="coerce")
    # Criação de faixas de idade
    bins = [0, 29, 39, 49, 59, 120]
    labels = ["<=29", "30-39", "40-49", "50-59", "60+"]
    df_["Faixa_Idade"] = pd.cut(df_["Idade"], bins=bins, labels=labels, right=True)
    # Situação simulada de devolução (alvo 1)
    # Regra: carros mais caros, clientes jovens e consórcio tendem a devolver mais (apenas simulação)
    rng = np.random.default_rng(seed=42)
    prob_dev = (
        0.10
        + 0.0000003 * df_["Valor_Venda"].fillna(df_["Valor_Venda"].median())
        + 0.05 * (df_["Faixa_Idade"].isin(["<=29", "30-39"]).astype(int))
        + 0.07 * (df_["Forma_Pagamento"].eq("Consórcio").astype(int))
        + rng.normal(0, 0.03, len(df_))
    )
    prob_dev = np.clip(prob_dev, 0, 0.9)
    df_["Devolucao"] = (rng.random(len(df_)) < prob_dev).astype(int)
    # Situação simulada de necessidade de manutenção em 12 meses (alvo 2)
    # Regra: carros mais antigos, maior quilate de valor e uso severo simulado por idade + estado quente (N/NE)
    estados_quentes = ["AM", "PA", "BA", "PE", "CE"]
    prob_maint = (
        0.08
        + 0.03 * (2026 - df_["Ano_Modelo"].clip(1995, 2026))  # ano mais antigo, prob maior
        / 10
        + 0.06 * (df_["Estado"].isin(estados_quentes)).astype(int)
        + 0.05 * (df_["Faixa_Idade"].eq("60+")).astype(int)
        + rng.normal(0, 0.04, len(df_))
    )
    prob_maint = np.clip(prob_maint, 0, 0.95)
    df_["Manutencao_12m"] = (rng.random(len(df_)) < prob_maint).astype(int)
    return df_
df_raw = carregar_dados()
df = df_raw.copy()
# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
def aplicar_filtros(df_: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros Globais")
    estados = st.sidebar.multiselect(
        "Estados",
        options=sorted(df_["Estado"].dropna().unique()),
        default=sorted(df_["Estado"].dropna().unique())
    )
    anos = st.sidebar.multiselect(
        "Ano da Venda",
        options=sorted(df_["Ano_Venda"].dropna().unique()),
        default=sorted(df_["Ano_Venda"].dropna().unique())
    )
    marcas = st.sidebar.multiselect(
        "Marca",
        options=sorted(df_["Marca"].dropna().unique()),
        default=sorted(df_["Marca"].dropna().unique())
    )
    df_f = df_.copy()
    if estados:
        df_f = df_f[df_f["Estado"].isin(estados)]
    if anos:
        df_f = df_f[df_f["Ano_Venda"].isin(anos)]
    if marcas:
        df_f = df_f[df_f["Marca"].isin(marcas)]
    return df_f
def kpi_card(label: str, valor: str | float, delta: str | float | None = None, col=None):
    if col is None:
        col = st
    if isinstance(valor, (int, float)):
        valor_str = f"{valor:,.0f}".replace(",", ".")
    else:
        valor_str = valor
    col.metric(label, valor_str, delta)
def treinar_modelos_ml(df_base: pd.DataFrame):
    """
    Treina dois modelos (Random Forest) para:
      - Devolução
      - Manutenção em 12 meses
    Retorna dicionário com modelos, encoders e scalers.
    """
    df_ml = df_base.copy()
    # Seleção de features (mix numéricas e categóricas)
    features_categoricas = ["Estado", "Marca", "Modelo", "Forma_Pagamento", "Sexo", "Cor"]
    features_numericas = ["Idade", "Valor_Venda", "Ano_Modelo"]
    # LabelEncoding de categóricas
    encoders = {}
    for col in features_categoricas:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
    X = df_ml[features_numericas + features_categoricas].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Devolução
    y_dev = df_ml["Devolucao"].values
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_scaled, y_dev, test_size=0.2, random_state=42, stratify=y_dev
    )
    modelo_dev = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    modelo_dev.fit(X_train_d, y_train_d)
    y_pred_d = modelo_dev.predict(X_test_d)
    y_proba_d = modelo_dev.predict_proba(X_test_d)[:, 1]
    acc_d = accuracy_score(y_test_d, y_pred_d)
    auc_d = roc_auc_score(y_test_d, y_proba_d)
    # Manutenção
    y_maint = df_ml["Manutencao_12m"].values
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_scaled, y_maint, test_size=0.2, random_state=42, stratify=y_maint
    )
    modelo_m = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    modelo_m.fit(X_train_m, y_train_m)
    y_pred_m = modelo_m.predict(X_test_m)
    y_proba_m = modelo_m.predict_proba(X_test_m)[:, 1]
    acc_m = accuracy_score(y_test_m, y_pred_m)
    auc_m = roc_auc_score(y_test_m, y_proba_m)
    resultados = {
        "model_devolucao": modelo_dev,
        "model_manutencao": modelo_m,
        "encoders": encoders,
        "scaler": scaler,
        "features": features_numericas + features_categoricas,
        "metrica_devolucao": {"acc": acc_d, "auc": auc_d},
        "metrica_manutencao": {"acc": acc_m, "auc": auc_m},
    }
    return resultados
@st.cache_resource
def get_modelos_treinados():
    return treinar_modelos_ml(df)
# -----------------------------------------------------------------------------
# PÁGINAS
# -----------------------------------------------------------------------------
def pagina_dashboard(df_: pd.DataFrame):
    st.title("📊 Painel Executivo de Vendas - AutoBrasil")
    df_f = aplicar_filtros(df_)
    if df_f.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_vendas = len(df_f)
    faturamento = df_f["Valor_Venda"].sum()
    ticket_medio = faturamento / total_vendas if total_vendas > 0 else 0
    idade_media = df_f["Idade"].mean()
    kpi_card("Total de Vendas", total_vendas, col=col1)
    kpi_card("Faturamento Total (R$)", f"R$ {faturamento:,.2f}".replace(",", "."), col=col2)
    kpi_card("Ticket Médio (R$)", f"R$ {ticket_medio:,.2f}".replace(",", "."), col=col3)
    kpi_card("Idade Média do Cliente", f"{idade_media:.1f} anos", col=col4)
    st.markdown("---")
    # Linha 1 de gráficos
    c1, c2 = st.columns(2)
    with c1:
        faturamento_estado = (
            df_f.groupby("Estado")["Valor_Venda"].sum().reset_index().sort_values("Valor_Venda", ascending=False)
        )
        fig1 = px.bar(
            faturamento_estado,
            x="Estado",
            y="Valor_Venda",
            title="Faturamento por Estado",
            text_auto=".0f",
            color="Estado",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(showlegend=False, yaxis_title="Valor (R$)")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        faturamento_mes = (
            df_f.groupby("Mes_Ano")["Valor_Venda"].sum().reset_index().sort_values("Mes_Ano")
        )
        fig2 = px.line(
            faturamento_mes,
            x="Mes_Ano",
            y="Valor_Venda",
            markers=True,
            title="Faturamento ao Longo do Tempo (Mês/Ano)"
        )
        fig2.update_layout(xaxis_tickangle=-45, yaxis_title="Valor (R$)")
        st.plotly_chart(fig2, use_container_width=True)
    # Linha 2 de gráficos
    c3, c4 = st.columns(2)
    with c3:
        vendas_marca = (
            df_f.groupby("Marca")["ID_Venda"].count().reset_index().rename(columns={"ID_Venda": "Qtde"})
        )
        vendas_marca = vendas_marca.sort_values("Qtde", ascending=False)
        fig3 = px.bar(
            vendas_marca,
            x="Marca",
            y="Qtde",
            title="Volume de Vendas por Marca",
            text_auto=True,
            color="Marca",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        forma = (
            df_f.groupby("Forma_Pagamento")["Valor_Venda"].sum().reset_index()
        )
        fig4 = px.pie(
            forma,
            values="Valor_Venda",
            names="Forma_Pagamento",
            title="Participação por Forma de Pagamento (Faturamento)",
            hole=0.4
        )
        st.plotly_chart(fig4, use_container_width=True)
    # Tabela de detalhes
    st.markdown("### 🔎 Detalhamento das Vendas Filtradas")
    st.dataframe(
        df_f[
            [
                "ID_Venda",
                "Data_Venda",
                "Estado",
                "Cidade",
                "Marca",
                "Modelo",
                "Ano_Modelo",
                "Cor",
                "Valor_Venda",
                "Forma_Pagamento",
                "Sexo",
                "Idade",
                "Vendedor",
            ]
        ].sort_values("Data_Venda", ascending=False),
        use_container_width=True,
        height=400
    )
def pagina_analise_estatistica(df_: pd.DataFrame):
    st.title("📈 Análises Estatísticas e Exploratórias")
    df_f = aplicar_filtros(df_)
    if df_f.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return
    st.subheader("Distribuições de Variáveis-Chave")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Distribuição de Idades**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df_f["Idade"], bins=15, kde=True, ax=ax, color="#4e79a7")
        ax.set_xlabel("Idade")
        ax.set_ylabel("Frequência")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.markdown("**Distribuição de Valor de Venda**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df_f["Valor_Venda"], bins=20, kde=True, ax=ax, color="#f28e2b")
        ax.set_xlabel("Valor da Venda (R$)")
        ax.set_ylabel("Frequência")
        st.pyplot(fig, clear_figure=True)
    with c3:
        st.markdown("**Ano do Modelo**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Ano_Modelo", data=df_f, ax=ax, color="#59a14f")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig, clear_figure=True)
    st.markdown("---")
    st.subheader("Correlação Entre Variáveis Numéricas")
    cols_corr = ["Idade", "Valor_Venda", "Ano_Modelo"]
    corr = df_f[cols_corr].corr()
    fig_corr, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5, ax=ax)
    st.pyplot(fig_corr, clear_figure=True)
    st.markdown("---")
    st.subheader("Perfil de Clientes por Marca / Faixa de Idade")
    c4, c5 = st.columns(2)
    with c4:
        top_marcas = (
            df_f.groupby("Marca")["ID_Venda"].count().reset_index().rename(columns={"ID_Venda": "Qtde"})
        )
        top_marcas = top_marcas.sort_values("Qtde", ascending=False).head(10)
        fig5 = px.bar(
            top_marcas,
            x="Qtde",
            y="Marca",
            orientation="h",
            title="Top 10 Marcas por Volume de Vendas",
            text_auto=True,
            color="Marca",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
    with c5:
        df_idade_marca = (
            df_f.groupby(["Marca", "Faixa_Idade"])["ID_Venda"].count()
            .reset_index()
            .rename(columns={"ID_Venda": "Qtde"})
        )
        fig6 = px.bar(
            df_idade_marca,
            x="Marca",
            y="Qtde",
            color="Faixa_Idade",
            barmode="group",
            title="Distribuição de Vendas por Marca e Faixa de Idade"
        )
        fig6.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig6, use_container_width=True)
    st.markdown("---")
    st.subheader("Impacto Simulado de Devoluções e Manutenções")
    c6, c7 = st.columns(2)
    with c6:
        taxa_dev = df_f["Devolucao"].mean()
        taxa_maint = df_f["Manutencao_12m"].mean()
        st.metric("Taxa Simulada de Devolução", f"{taxa_dev*100:.1f} %")
        st.metric("Taxa Simulada de Manutenção (12m)", f"{taxa_maint*100:.1f} %")
        impacto_dev = df_f.loc[df_f["Devolucao"] == 1, "Valor_Venda"].sum()
        impacto_maint = df_f.loc[df_f["Manutencao_12m"] == 1, "Valor_Venda"].sum()
        st.write(f"• **Faturamento associado a vendas que podem ser devolvidas**: R$ {impacto_dev:,.2f}".replace(",", "."))
        st.write(f"• **Faturamento associado a veículos com alta chance de manutenção (12m)**: R$ {impacto_maint:,.2f}".replace(",", "."))
    with c7:
        df_risco = df_f.groupby("Marca")[["Devolucao", "Manutencao_12m"]].mean().reset_index()
        df_risco["Devolucao"] *= 100
        df_risco["Manutencao_12m"] *= 100
        fig7 = go.Figure()
        fig7.add_trace(
            go.Bar(
                x=df_risco["Marca"],
                y=df_risco["Devolucao"],
                name="Devolução (%)",
                marker_color="#e15759"
            )
        )
        fig7.add_trace(
            go.Bar(
                x=df_risco["Marca"],
                y=df_risco["Manutencao_12m"],
                name="Manutenção 12m (%)",
                marker_color="#76b7b2"
            )
        )
        fig7.update_layout(
            title="Taxas Simuladas de Risco por Marca",
            barmode="group",
            xaxis_tickangle=-45,
            yaxis_title="Taxa (%)"
        )
        st.plotly_chart(fig7, use_container_width=True)
def pagina_predicoes(df_: pd.DataFrame):
    st.title("🤖 Predição de Devoluções e Manutenções")
    st.write(
        """
        Esta página utiliza **modelos de Machine Learning (Random Forest)** para 
        **estimar a probabilidade de devolução** da venda e a **probabilidade de 
        necessidade de manutenção em 12 meses**, com base em características do cliente
        e do veículo.
        
        **Importante:** Como a base não possui essas informações reais, os alvos de
        devolução e manutenção foram **simulados** a partir de regras de negócio 
        aproximadas, apenas para demonstrar a arquitetura analítica.
        """
    )
    modelos = get_modelos_treinados()
    # Mostrar métricas dos modelos
    st.subheader("Desempenho dos Modelos (Treinados na Base Completa)")
    col1, col2 = st.columns(2)
    met_dev = modelos["metrica_devolucao"]
    met_maint = modelos["metrica_manutencao"]
    with col1:
        st.markdown("**Modelo de Devolução**")
        st.write(f"Acurácia: **{met_dev['acc']:.2%}**")
        st.write(f"AUC ROC: **{met_dev['auc']:.2f}**")
    with col2:
        st.markdown("**Modelo de Manutenção em 12 meses**")
        st.write(f"Acurácia: **{met_maint['acc']:.2%}**")
        st.write(f"AUC ROC: **{met_maint['auc']:.2f}**")
    st.markdown("---")
    # Formulário de input do usuário
    st.subheader("Simulador de Risco para uma Nova Venda")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        idade = st.number_input("Idade do Cliente", min_value=18, max_value=100, value=40)
        valor_venda = st.number_input("Valor da Venda (R$)", min_value=10000.0, max_value=600000.0, value=150000.0, step=1000.0)
        ano_modelo = st.number_input("Ano do Modelo", min_value=2010, max_value=2026, value=2022)
    with col_i2:
        estado = st.selectbox("Estado", sorted(df_["Estado"].dropna().unique()))
        marca = st.selectbox("Marca", sorted(df_["Marca"].dropna().unique()))
        modelo = st.selectbox("Modelo", sorted(df_["Modelo"].dropna().unique()))
    with col_i3:
        forma_pg = st.selectbox("Forma de Pagamento", sorted(df_["Forma_Pagamento"].dropna().unique()))
        sexo = st.selectbox("Sexo", sorted(df_["Sexo"].dropna().unique()))
        cor = st.selectbox("Cor do Veículo", sorted(df_["Cor"].dropna().unique()))
    if st.button("Calcular Probabilidades"):
        # Montar o mesmo vetor de features usado no treinamento
        entrada = pd.DataFrame(
            {
                "Idade": [idade],
                "Valor_Venda": [valor_venda],
                "Ano_Modelo": [ano_modelo],
                "Estado": [estado],
                "Marca": [marca],
                "Modelo": [modelo],
                "Forma_Pagamento": [forma_pg],
                "Sexo": [sexo],
                "Cor": [cor],
            }
        )
        # Aplicar os encoders
        for col in modelos["encoders"].keys():
            le = modelos["encoders"][col]
            # Se o valor não existir no encoder (categoria nova), usar a 1ª classe como fallback
            entrada[col] = entrada[col].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            entrada[col] = le.transform(entrada[col])
        # Escalonar
        X_input = entrada[modelos["features"]]
        scaler = modelos["scaler"]
        X_scaled = scaler.transform(X_input)
        # Predições
        m_dev = modelos["model_devolucao"]
        m_maint = modelos["model_manutencao"]
        prob_dev = m_dev.predict_proba(X_scaled)[0, 1]
        prob_maint = m_maint.predict_proba(X_scaled)[0, 1]
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.markdown("### Probabilidade de Devolução")
            st.metric(
                label="Risco de Devolução",
                value=f"{prob_dev*100:.1f} %",
            )
            risco_label_dev = (
                "ALTO" if prob_dev >= 0.6 else "MÉDIO" if prob_dev >= 0.3 else "BAIXO"
            )
            st.write(f"Classificação: **Risco {risco_label_dev}**")
        with c_r2:
            st.markdown("### Probabilidade de Manutenção em 12 Meses")
            st.metric(
                label="Risco de Manutenção",
                value=f"{prob_maint*100:.1f} %",
            )
            risco_label_maint = (
                "ALTO" if prob_maint >= 0.6 else "MÉDIO" if prob_maint >= 0.3 else "BAIXO"
            )
            st.write(f"Classificação: **Risco {risco_label_maint}**")
        st.markdown("#### Recomendações Operacionais (Simuladas)")
        bullets = []
        if prob_dev >= 0.6:
            bullets.append("- Priorizar **contato proativo pós-venda** e reforçar garantias/benefícios.")
            bullets.append("- Avaliar condições comerciais (prazo, entrada, seguro) para evitar arrependimento.")
        elif prob_dev >= 0.3:
            bullets.append("- Sugerir **pacotes de serviços** (revisão, seguro, acessórios) para aumentar satisfação.")
        else:
            bullets.append("- Perfil de devolução baixo. Manter rotina padrão de acompanhamento.")
        if prob_maint >= 0.6:
            bullets.append("- Incluir o cliente em **campanhas de revisão antecipada** (manutenção preventiva).")
            bullets.append("- Oferecer plano de manutenção estendida / garantia adicional.")
        elif prob_maint >= 0.3:
            bullets.append("- Reforçar importância de revisões programadas e uso de peças originais.")
        else:
            bullets.append("- Manutenção estimada como de baixo risco no curto prazo.")
        for b in bullets:
            st.write(b)
# -----------------------------------------------------------------------------
# LAYOUT PRINCIPAL / NAVEGAÇÃO
# -----------------------------------------------------------------------------
def main():
    menu = st.sidebar.radio(
        "Navegação",
        [
            "Dashboard Executivo",
            "Análises Estatísticas",
            "Predição de Devolução & Manutenção",
        ]
    )
    if menu == "Dashboard Executivo":
        pagina_dashboard(df)
    elif menu == "Análises Estatísticas":
        pagina_analise_estatistica(df)
    else:
        pagina_predicoes(df)
if __name__ == "__main__":
    main()