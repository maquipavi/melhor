import streamlit as st
import os
import asyncio
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from datetime import datetime
import re
import pandas as pd

# --- Configuração da API Key (NÃO RECOMENDADO para produção) ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBarB5CfRsl_M0nkQjgg-ystWV-CyzN0jU"

# Inicializa cliente GenAI
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Erro ao inicializar o cliente da API Google GenAI: {e}")
    st.stop()

MODELO_RAPIDO = "gemini-2.0-flash"
MODELO_ROBUSTO = "gemini-2.0-flash"

session_service = InMemorySessionService()

async def call_agent(agent: Agent, message_text: str) -> str:
    session_id = f"{agent.name}_session"
    user_id = "streamlit_user"
    try:
        session = await session_service.create_session(app_name=agent.name, user_id=user_id)
    except:
        session = await session_service.create_session(app_name=agent.name, user_id=user_id)

    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])
    final_response = ""
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text:
                    final_response += part.text + ("\n" if not part.text.endswith("\n") else "")
    return final_response

def to_markdown_string(text):
    return text.replace('•', '*')

async def agente_analisador(data_nascimento):
    with st.spinner("Executando Agente 1..."):
        analisador = Agent(
            name="agente_analisador",
            model=MODELO_RAPIDO,
            instruction="""
            Você é um analista de personalidade e propósito de vida com base na data de nascimento.
            Formate a saída usando Markdown, com títulos para cada seção (1 a 6).
            """,
            description="Análise baseada na data de nascimento",
            tools=[google_search]
        )
        entrada = f"""
        Data de Nascimento: {data_nascimento}

        Realize as seguintes análises, formatando com Markdown:

        1. Decodificador de Personalidade
        2. Roteiro da Infância
        3. Propósito Profissional
        4. Auto-Sabotagem
        5. Gatilhos Emocionais
        6. Energia nos Relacionamentos
        """
        return await call_agent(analisador, entrada)

async def agente_melhorias(data_nascimento, analises):
    with st.spinner("Executando Agente 2..."):
        melhorias = Agent(
            name="agente_melhorias",
            model=MODELO_RAPIDO,
            instruction="""
            Você é um consultor de desenvolvimento pessoal. Analise as seis áreas descritas e sugira melhorias práticas para cada uma.
            Use títulos Markdown.
            """,
            description="Identifica pontos de melhoria"
        )
        entrada = f"""
        Data: {data_nascimento}
        Análises:
        {analises}
        """
        return await call_agent(melhorias, entrada)

async def agente_buscador_sucesso(data_nascimento):
    with st.spinner("Executando Agente 3..."):
        buscador = Agent(
            name="agente_sucesso",
            model=MODELO_ROBUSTO,
            instruction="""
            Busque 5 homens e 5 mulheres brasileiros nascidos em {data_nascimento} com sucesso profissional.
            Use o formato: "* Nome: [Nome] | Profissão: [Profissão] | Sucesso: [Descrição] | Site: [URL]"
            """,
            description="Busca pessoas brasileiras de sucesso",
            tools=[google_search]
        )
        entrada = f"Data de nascimento: {data_nascimento}"
        resposta = await call_agent(buscador, entrada)

        pattern = re.compile(r"\*\s*Nome:\s*(.*?)\s*\|\s*Profissão:\s*(.*?)\s*\|\s*Sucesso:\s*(.*?)\s*\|\s*Site:\s*(.*?)\s*$", re.MULTILINE)
        data = [list(map(str.strip, match.groups())) for match in pattern.finditer(resposta)]
        return pd.DataFrame(data, columns=["Nome", "Profissão", "Sucesso", "Site da Informação"])

async def agente_relatorio_final(data_nascimento, analises, melhorias, tabela_sucesso_df):
    with st.spinner("Executando Agente 4..."):
        relatorio = Agent(
            name="agente_relatorio",
            model=MODELO_RAPIDO,
            instruction="""
            Combine todas as análises anteriores em um relatório final motivador.
            Use Markdown com títulos claros. Inclua a tabela de pessoas de sucesso.
            Finalize com uma mensagem encorajadora.
            """,
            description="Gera o relatório final"
        )
        tabela_md = tabela_sucesso_df.to_markdown(index=False)
        entrada = f"""
        Data: {data_nascimento}

        Análises:
        {analises}

        Melhorias:
        {melhorias}

        Pessoas de Sucesso:
        {tabela_md}
        """
        return await call_agent(relatorio, entrada)

# --- INTERFACE STREAMLIT ---
st.title("🌟 Analisador de Personalidade e Propósito de Vida")
st.markdown("Com base na sua data de nascimento, descubra traços da sua personalidade, propósitos e muito mais.")

data_nascimento_str = st.text_input("Digite sua data de nascimento (DD/MM/AAAA):")
run_button = st.button("✨ Gerar Relatório ✨")
report_container = st.empty()

if run_button:
    if not data_nascimento_str:
        st.warning("Digite sua data de nascimento.")
    else:
        try:
            datetime.strptime(data_nascimento_str, '%d/%m/%Y')
            st.info(f"Analisando: {data_nascimento_str}")
            async def executar_tudo(dob):
                analises = await agente_analisador(dob)
                melhorias = await agente_melhorias(dob, analises)
                df_sucesso = await agente_buscador_sucesso(dob)
                st.session_state['sucesso_df'] = df_sucesso
                return await agente_relatorio_final(dob, analises, melhorias, df_sucesso)

            final_md = asyncio.run(executar_tudo(data_nascimento_str))
            final_md_formatado = to_markdown_string(final_md)
            st.session_state['final_report_md'] = final_md_formatado

            report_container.markdown("## 📝 Relatório Final")
            report_container.markdown(final_md_formatado)

            if 'sucesso_df' in st.session_state:
                report_container.markdown("### Pessoas de Sucesso")
                report_container.dataframe(st.session_state['sucesso_df'])

        except ValueError:
            st.error("Formato de data inválido. Use DD/MM/AAAA.")
        except Exception as e:
            st.error(f"Erro: {e}")

if 'final_report_md' in st.session_state:
    report_container.download_button(
        label="💾 Baixar Relatório",
        data=st.session_state['final_report_md'],
        file_name=f"relatorio_{data_nascimento_str.replace('/', '-')}.md",
        mime="text/markdown"
    )

st.markdown("---")
st.caption("Desenvolvido com Google AI.")
