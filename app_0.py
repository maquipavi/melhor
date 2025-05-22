import streamlit as st
import os
from google.colab import userdata  # This won't work outside of Colab; adapt for local deployment
import google.generativeai as genai  # Correct import
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.generativeai import types  # Correct import, including types
from datetime import date, datetime
import textwrap
import pandas as pd
import requests
import warnings
import re

warnings.filterwarnings("ignore")

# --- Setup/Config ---
st.title("Personalized Life Purpose and Personality Analysis System")

# Sidebar for API Key (adapt for local deployment if not using Colab)
api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")
if not api_key:
    st.sidebar.warning("Please enter your Google Gemini API key.  Get one from makersuite.google.com.")
    st.stop()


os.environ["GOOGLE_API_KEY"] = AIzaSyBarB5CfRsl_M0nkQjgg-ystWV-CyzN0jU

# Configure Gemini client
genai.configure(api_key=api_key)
# client = genai.Client() # Old way, not needed with configure()


# Define models
MODELO_RAPIDO = "gemini-pro"  # Standard Gemini Pro
MODELO_ROBUSTO = "gemini-pro" # Same as fast,  adapt to Gemini Ultra (if you have access and want to pay)


# Session State (for Streamlit)
if 'session_service' not in st.session_state:
    st.session_state['session_service'] = InMemorySessionService()  # Initialize only once


# --- Helper Functions ---
async def call_agent(agent: Agent, message_text: str) -> str:
    session_service = st.session_state['session_service']
    session = await session_service.create_session(app_name=agent.name, user_id="user1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)]) # Correct way to create content

    final_response = ""
    async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text
                    final_response += "\n"
    return final_response


def to_markdown(text):
    text = text.replace('•', '  *')
    return text


# --- Agent Definitions ---
async def agente_analisador(data_nascimento):
    analisador = Agent(
        name="agente_analisador",
        model=MODELO_RAPIDO,
        instruction="""
        Você é um analista de personalidade e propósito de vida com base na data de nascimento.
        Sua tarefa é fornecer análises profundas e precisas sobre a personalidade, padrões emocionais,
        caminhos de carreira e desafios pessoais com base na data de nascimento fornecida.
        Use a ferramenta de busca do Google (google_search) para obter informações relevantes e
        garantir que as análises sejam fundamentadas e úteis.
        """,
        description="Agente que analisa a personalidade e o propósito de vida com base na data de nascimento",
        tools=[google_search]
    )

    entrada_do_agente_analisador = f"""
    Data de Nascimento: {data_nascimento}

    Realize as seguintes análises:

    1. Decodificador de Personalidade pela Data de Nascimento
    “Com base na data de nascimento {data_nascimento}, descreva meus pontos fortes naturais, padrões emocionais e como me comporto em relacionamentos — que seja profundo, específico e psicologicamente preciso.”

    2. Roteiro da Infância
    “Usando a data de nascimento {data_nascimento}, escreva um perfil psicológico de como minha infância moldou minha personalidade, hábitos e tomada de decisões hoje — seja gentil, mas revelador.”

    3. Analisador de Propósito Profissional
    “Dada a data de nascimento {data_nascimento}, quais caminhos de carreira combinam com meus traços de personalidade, valores e talentos naturais? Sugira áreas, funções e ambientes de trabalho.”

    4. Detector de Auto-Sabotagem
    “Com base na data {data_nascimento}, quais são meus hábitos de auto-sabotagem mais prováveis e como eles aparecem no dia a dia? Dê soluções práticas com base na psicologia.”

    5. Mapa de Gatilhos Emocionais
    “Usando a data de nascimento {data_nascimento}, explique o que geralmente me desencadeia emocionalmente, como eu costumo reagir e como posso desenvolver resiliência emocional em torno desses padrões.”

    6. Escaneamento de Energia nos Relacionamentos
    “Com base na data de nascimento {data_nascimento}, descreva como eu dou e recebo amor, o que preciso de um parceiro e que tipo de pessoa eu naturalmente atraio.”
    """

    analises = await call_agent(analisador, entrada_do_agente_analisador)
    return analises


async def agente_melhorias(data_nascimento, analises_agente1):
    melhorias = Agent(
        name="agente_melhorias",
        model=MODELO_RAPIDO,
        instruction="""
        Você é um consultor de desenvolvimento pessoal. Sua tarefa é analisar as análises fornecidas
        pelo Agente 1 (analisador de nascimento) e identificar áreas de melhoria em cada uma das seis
        categorias. Seja específico e forneça sugestões práticas para o desenvolvimento pessoal.
        """,
        description="Agente que identifica pontos de melhoria nas análises do Agente 1",
        tools=[google_search]
    )

    entrada_do_agente_melhorias = f"""
    Data de Nascimento: {data_nascimento}
    Análises do Agente 1: {analises_agente1}

    Para cada uma das seis análises fornecidas pelo Agente 1, identifique áreas de melhoria e
    forneça sugestões práticas para o desenvolvimento pessoal.
    """

    pontos_de_melhoria = await call_agent(melhorias, entrada_do_agente_melhorias)
    return pontos_de_melhoria


async def agente_buscador_sucesso(data_nascimento):
    buscador_sucesso = Agent(
        name="agente_buscador_sucesso",
        model=MODELO_ROBUSTO,
        instruction="""
            Você é um pesquisador de pessoas de sucesso brasileiras. Sua tarefa é buscar na internet 5 homens e 5 mulheres
            que nasceram na mesma data fornecida e que alcançaram sucesso em suas áreas de atuação, e que sejam brasileiros.
            Monte uma tabela com as seguintes colunas: nome, profissão, no que a pessoa tem sucesso e site da informação.
            Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" para garantir que os resultados sejam apenas de pessoas do Brasil.
            Use a ferramenta de busca do Google (google_search) para encontrar as informações e o site de onde tirou a informação.
            """,
        description="Agente que busca pessoas de sucesso nascidas na mesma data",
        tools=[google_search]
    )

    entrada_do_agente_buscador_sucesso = f"""
    Data de Nascimento: {data_nascimento}

    Busque na internet 5 homens e 5 mulheres que nasceram na mesma data e que alcançaram sucesso
    em suas áreas de atuação e que sejam brasileiros. Monte uma tabela com as seguintes colunas:
    nome, profissão, no que a pessoa tem sucesso e site da informação. Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" para garantir que os resultados sejam apenas de pessoas do Brasil. Seja claro e objetivo
    """

    tabela_sucesso = await call_agent(buscador_sucesso, entrada_do_agente_buscador_sucesso)

    # Extract data for DataFrame
    data_lines = tabela_sucesso.strip().split('\n')

    header_index = -1
    for i, line in enumerate(data_lines):
        if "Nome" in line and "Profissão" in line and "Sucesso" in line and "Site da Informação" in line:
            header_index = i
            break

    data = []
    if header_index != -1:
        for line in data_lines[header_index+2:]: # Skip header and separator line
            if line.strip() and '|' in line:
                values = re.split(r"\|", line.strip())  # Corrected splitting
                cleaned_values = [v.strip() for v in values if v.strip()]

                if len(cleaned_values) >= 4:
                    nome, profissao, sucesso, site = cleaned_values[:4]
                    data.append([nome, profissao, sucesso, site])
                elif len(cleaned_values) > 0:
                    data.append(cleaned_values + [''] * (4 - len(cleaned_values)))

    df = pd.DataFrame(data, columns=["Nome", "Profissão", "Sucesso", "Site da Informação"])
    return df


async def agente_relatorio_final(data_nascimento, analises, melhorias, tabela_sucesso):
    relatorio = Agent(
        name="agente_relatorio",
        model=MODELO_RAPIDO,
        instruction="""
        Você é um gerador de relatórios finais. Sua tarefa é combinar as análises do Agente 1,
        os pontos de melhoria do Agente 2 e a tabela de pessoas de sucesso do Agente 3 em um
        relatório final otimista e motivador. Conclua o relatório com uma mensagem de incentivo.
        """,
        description="Agente que gera o relatório final",
    )

    entrada_do_agente_relatorio = f"""
    Data de Nascimento: {data_nascimento}
    Análises do Agente 1: {analises}
    Pontos de Melhoria do Agente 2: {melhorias}
    Tabela de Pessoas de Sucesso do Agente 3: {tabela_sucesso.to_string()} # Pass DataFrame as string

    Combine as informações acima em um relatório final otimista e motivador.
    Conclua o relatório com uma mensagem de incentivo.
    """

    relatorio_final = await call_agent(relatorio, entrada_do_agente_relatorio)
    return relatorio_final


# --- Streamlit UI ---
data_nascimento = st.text_input("Enter your date of birth (DD/MM/YYYY):")

if st.button("Analyze"):
    if not data_nascimento:
        st.warning("Please enter your date of birth.")
    else:
        try:
            datetime.strptime(data_nascimento, '%d/%m/%Y')

            with st.spinner("Running Analysis..."):
                # --- Execute Agents ---
                st.subheader("Agent 1: Personality Analyzer")
                analises_agente1 = await agente_analisador(data_nascimento)
                st.markdown(to_markdown(analises_agente1))

                st.subheader("Agent 2: Improvement Identifier")
                pontos_de_melhoria = await agente_melhorias(data_nascimento, analises_agente1)
                st.markdown(to_markdown(pontos_de_melhoria))

                st.subheader("Agent 3: Successful People Finder")
                tabela_sucesso = await agente_buscador_sucesso(data_nascimento)
                st.dataframe(tabela_sucesso)

                st.subheader("Agent 4: Final Report Generator")
                relatorio_final = await agente_relatorio_final(data_nascimento, analises_agente1, pontos_de_melhoria, tabela_sucesso)
                st.markdown(to_markdown(relatorio_final))


        except ValueError:
            st.error("Invalid date format. Please use DD/MM/YYYY.")