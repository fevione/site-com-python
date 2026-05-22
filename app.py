import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='AutoBrasil Analytics', layout='wide')

@st.cache_data
def carregar_dados():
    return pd.read_csv('base_ficticia_vendas_carros.csv')

df = carregar_dados()
np.random.seed(42)
df['Prob_Devolucao'] = np.random.randint(0,2,size=len(df))

st.title('Dashboard Inteligente de Vendas Automotivas')

estado = st.sidebar.multiselect('Estado', options=df['Estado'].unique(), default=df['Estado'].unique())
base = df[df['Estado'].isin(estado)]

col1, col2 = st.columns(2)
with col1:
    st.metric('Total de Vendas', len(base))
with col2:
    st.metric('Faturamento', f"R$ {base['Valor_Venda'].sum():,.2f}")

fig = px.bar(base.groupby('Estado')['Valor_Venda'].sum().reset_index(), x='Estado', y='Valor_Venda', title='Faturamento por Estado')
st.plotly_chart(fig, use_container_width=True)

base_ml = base.copy()
encoder = LabelEncoder()
for col in ['Estado','Marca','Modelo']:
    base_ml[col] = encoder.fit_transform(base_ml[col])

features = ['Idade','Valor_Venda','Ano_Modelo','Estado','Marca','Modelo']
X = base_ml[features]
y = base_ml['Prob_Devolucao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)

st.success(f'Acurácia do modelo: {acc:.2%}')
