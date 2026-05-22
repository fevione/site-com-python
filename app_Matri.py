import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from streamlit_option_menu import option_menu

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduTech Analytics - SENAI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# CSS Personalizado
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.9;
    }
    
    .stAlert > div {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
    }
    
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
    
    .css-1d391kg {
        padding: 1rem;
    }
    
    .plotly-graph-div {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .highlight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GERAÇÃO DE DADOS FICTÍCIOS COMPLETOS (2500 REGISTROS)
# -----------------------------------------------------------------------------
@st.cache_data
def gerar_base_matriculas_completa(n_registros: int = 2500) -> pd.DataFrame:
    """
    Gera uma base completa de matrículas educacionais com dados relevantes
    """
    np.random.seed(42)
    
    # Arrays de dados realistas
    nomes = [
        "Ana Silva", "Bruno Costa", "Carla Oliveira", "Daniel Ferreira", "Eduarda Santos",
        "Felipe Rocha", "Gabriela Mendes", "Henrique Alves", "Isabel Lima", "João Pedro",
        "Larissa Freitas", "Marcos Vinícius", "Natália Costa", "Otávio Barbosa", "Paula Rodrigues",
        "Rafael Gomes", "Sofia Martins", "Thiago Pereira", "Vitória Dias", "Wagner Silva",
        "Amanda Ribeiro", "Carlos Eduardo", "Diana Santos", "Eduardo Lima", "Fernanda Costa",
        "Gustavo Santos", "Helena Rocha", "Igor Almeida", "Juliana Barbosa", "Kevin Nascimento",
        "Luana Cardoso", "Miguel Torres", "Nicole Silva", "Pedro Henrique", "Roberta Lima",
        "Samuel Ferreira", "Tatiana Costa", "Vinícius Souza", "Yasmin Oliveira", "Zeca Moreira"
    ]
    
    estados = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "DF", "ES", "CE", "PE", "AM", "PA", "MT", "MS"]
    cidades = {
        "SP": ["São Paulo", "Campinas", "Santos", "Ribeirão Preto", "São José dos Campos", "Sorocaba"],
        "RJ": ["Rio de Janeiro", "Niterói", "Nova Iguaçu", "Duque de Caxias", "Campos", "Petrópolis"],
        "MG": ["Belo Horizonte", "Uberlândia", "Contagem", "Juiz de Fora", "Montes Claros", "Betim"],
        "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas", "Canoas", "Santa Maria", "Gravataí"],
        "PR": 
