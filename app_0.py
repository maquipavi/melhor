import streamlit as st
import asyncio
import os
from datetime import datetime, date
import pandas as pd
import re
import warnings

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise de Personalidade",
    page_icon="🚀",
    layout="wide"
)

# Suprimir warnings
warnings.filterwarnings("ignore")

# Título principal
st.title("Sistema de Análise de Personalidade e Propósito de Vida!")

# Configurar a API Key (fixo)
api_key = "AIzaSyBarB5CfRsl_M0nkQjgg-ystWV-CyzN0jU"
os.environ["GOOGLE_API_KEY"] = api_key

# Importações necessárias (só depois de configurar a API)
try:
    from google import genai
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import google_search
    from google.genai import types
except ImportError as e:
    st.error(f"Erro ao importar bibliotecas necessárias: {e}")
    st.info("Execute: pip install google-genai google-adk")
    st.stop()

# Configurar cliente
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Erro ao configurar cliente: {e}")
    st.stop()

# Definir modelos
MODELO_RAPIDO = "gemini-2.0-flash"
MODELO_ROBUSTO = "gemini-2.0-flash"

# Inicializar serviço de sessão (cache para evitar reinicialização)
@st.cache_resource
def get_session_service():
    return InMemorySessionService()

session_service = get_session_service()

# Função auxiliar para chamar agentes
async def call_agent(agent: Agent, message_text: str) -> str:
    try:
        session = await session_service.create_session(app_name=agent.name, user_id="user1")
        runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
        content = types.Content(role="user", parts=[types.Part(text=message_text)])

        final_response = ""
        async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=content):
            if event.is_final_response():
                for part in event.content.parts:
                    if part.text is not None:
                        final_response += part.text
                        final_response += "\n"
        return final_response
    except Exception as e:
        return f"Erro ao executar agente: {str(e)}"

# Agente 1: Analisador de Nascimento
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
    "Com base na data de nascimento {data_nascimento}, descreva meus pontos fortes naturais, padrões emocionais e como me comporto em relacionamentos — que seja profundo, específico e psicologicamente preciso."

    2. Roteiro da Infância
    "Usando a data de nascimento {data_nascimento}, escreva um perfil psicológico de como minha infância moldou minha personalidade, hábitos e tomada de decisões hoje — seja gentil, mas revelador."

    3. Analisador de Propósito Profissional
    "Dada a data de nascimento {data_nascimento}, quais caminhos de carreira combinam com meus traços de personalidade, valores e talentos naturais? Sugira áreas, funções e ambientes de trabalho."

    4. Detector de Auto-Sabotagem
    "Com base na data {data_nascimento}, quais são meus hábitos de auto-sabotagem mais prováveis e como eles aparecem no dia a dia? Dê soluções práticas com base na psicologia."

    5. Mapa de Gatilhos Emocionais
    "Usando a data de nascimento {data_nascimento}, explique o que geralmente me desencadeia emocionalmente, como eu costumo reagir e como posso desenvolver resiliência emocional em torno desses padrões."

    6. Escaneamento de Energia nos Relacionamentos
    "Com base na data de nascimento {data_nascimento}, descreva como eu dou e recebo amor, o que preciso de um parceiro e que tipo de pessoa eu naturalmente atraio."
    """

    return await call_agent(analisador, entrada_do_agente_analisador)

# Agente 2: Identificador de Melhorias
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

    return await call_agent(melhorias, entrada_do_agente_melhorias)

# Agente 3: Buscador de Pessoas de Sucesso
async def agente_buscador_sucesso(data_nascimento):
    buscador_sucesso = Agent(
        name="agente_buscador_sucesso",
        model=MODELO_ROBUSTO,
        instruction="""
            Você é um pesquisador de pessoas de sucesso brasileiras. Sua tarefa é buscar na internet homens e mulheres
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

    Busque na internet homens e mulheres que nasceram na mesma data e que alcançaram sucesso
    em suas áreas de atuação e que sejam brasileiros. Monte uma tabela com as seguintes colunas:
    nome, profissão, no que a pessoa tem sucesso e site da informação. Ao realizar a busca no Google, certifique-se de incluir o termo "brasileiro" ou "brasileira" para garantir que os resultados sejam apenas de pessoas do Brasil. Seja claro e objetivo
    """

    tabela_sucesso = await call_agent(buscador_sucesso, entrada_do_agente_buscador_sucesso)
    
    # Processar dados para DataFrame
    data_lines = tabela_sucesso.strip().split('\n')
    header_index = -1
    
    for i, line in enumerate(data_lines):
        if "Nome" in line and "Profissão" in line and "Sucesso" in line and "Site da Informação" in line:
            header_index = i
            break

    data = []
    if header_index != -1:
        for line in data_lines[header_index+2:]:
            if line.strip() and '|' in line:
                values = re.split(r"\|", line.strip())
                cleaned_values = [v.strip() for v in values if v.strip()]

                if len(cleaned_values) >= 4:
                    nome, profissao, sucesso, site = cleaned_values[:4]
                    data.append([nome, profissao, sucesso, site])
                elif len(cleaned_values) > 0:
                    data.append(cleaned_values + [''] * (4 - len(cleaned_values)))

    df = pd.DataFrame(data, columns=["Nome", "Profissão", "Sucesso", "Site da Informação"])
    return df, tabela_sucesso

# Agente 4: Gerador de Relatório Final
async def agente_relatorio_final(data_nascimento, analises, melhorias, tabela_sucesso):
    relatorio = Agent(
        name="agente_relatorio",
        model=MODELO_RAPIDO,
        instruction="""
        Você é um gerador de relatórios finais. Sua tarefa é combinar as análises do Agente 1,
        os pontos de melhoria do Agente 2 e a tabela de pessoas de sucesso do Agente 3 em um
        relatório final otimista e motivador. Conclua o relatório com uma mensagem de incentivo.
        """,
        description="Agente que gera o relatório final"
    )

    entrada_do_agente_relatorio = f"""
    Data de Nascimento: {data_nascimento}
    Análises do Agente 1: {analises}
    Pontos de Melhoria do Agente 2: {melhorias}
    Tabela de Pessoas de Sucesso do Agente 3: {tabela_sucesso}

    Combine as informações acima em um relatório final otimista e motivador.
    Conclua o relatório com uma mensagem de incentivo.
    """

    return await call_agent(relatorio, entrada_do_agente_relatorio)

# Interface principal
def main():
    st.markdown("---")
    
    # Input da data de nascimento
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input de data personalizado com formato brasileiro
        st.write("📅 **Selecione sua data de nascimento:**")
        
        # Criar três colunas para dia, mês e ano
        col_dia, col_mes, col_ano = st.columns(3)
        
        with col_dia:
            dia = st.selectbox("Dia", range(1, 32), index=0)
        
        with col_mes:
            meses = [
                "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
            ]
            mes_nome = st.selectbox("Mês", meses, index=0)
            mes = meses.index(mes_nome) + 1
        
        with col_ano:
            ano = st.selectbox("Ano", range(1900, datetime.now().year + 1), 
                             index=len(range(1900, datetime.now().year + 1)) - 25)  # Padrão: 25 anos atrás
        
        # Validar data
        try:
            data_nascimento = date(ano, mes, dia)
            st.success(f"Data selecionada: {data_nascimento.strftime('%d/%m/%Y')}")
            data_valida = True
        except ValueError:
            st.error("❌ Data inválida! Por favor, selecione uma data válida.")
            data_valida = False
    
    with col2:
        st.write("")
        st.write("")
        iniciar_analise = st.button("🚀 Iniciar Análise", type="primary")

    if iniciar_analise and data_valida:
        data_str = data_nascimento.strftime("%d/%m/%Y")
        
        st.success(f"Maravilha! Vamos analisar sua personalidade e propósito de vida com base em {data_str}")
        
        # Criar tabs para organizar melhor a interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Análise de Personalidade", 
            "🔧 Pontos de Melhoria", 
            "⭐ Pessoas de Sucesso", 
            "📋 Relatório Final",
            "💾 Download"
        ])
        
        # Executar análises
        with st.spinner("Executando análises... Isso pode levar alguns minutos."):
            
            # Agente 1
            with tab1:
                st.subheader("📝 Análise de Personalidade")
                with st.spinner("Analisando sua personalidade..."):
                    analises_agente1 = asyncio.run(agente_analisador(data_str))
                st.markdown(analises_agente1)
            
            # Agente 2
            with tab2:
                st.subheader("🔧 Identificação de Melhorias")
                with st.spinner("Identificando pontos de melhoria..."):
                    pontos_de_melhoria = asyncio.run(agente_melhorias(data_str, analises_agente1))
                st.markdown(pontos_de_melhoria)
            
            # Agente 3
            with tab3:
                st.subheader("⭐ Pessoas de Sucesso")
                with st.spinner("Buscando pessoas de sucesso..."):
                    df_sucesso, tabela_sucesso_raw = asyncio.run(agente_buscador_sucesso(data_str))
                
                if not df_sucesso.empty:
                    st.dataframe(df_sucesso, use_container_width=True)
                else:
                    st.warning("Não foi possível processar a tabela de pessoas de sucesso.")
                    st.text(tabela_sucesso_raw)
            
            # Agente 4
            with tab4:
                st.subheader("📋 Relatório Final")
                with st.spinner("Gerando relatório final..."):
                    relatorio_final = asyncio.run(agente_relatorio_final(
                        data_str, analises_agente1, pontos_de_melhoria, tabela_sucesso_raw
                    ))
                st.markdown(relatorio_final)
            
            # Download
            with tab5:
                st.subheader("💾 Download do Relatório")
                
                # Compilar relatório completo
                relatorio_completo = f"""
# Relatório de Análise de Personalidade
**Data de Nascimento:** {data_str}
**Data do Relatório:** {datetime.now().strftime("%d/%m/%Y %H:%M")}

---

## 📝 Análise de Personalidade
{analises_agente1}

---

## 🔧 Pontos de Melhoria
{pontos_de_melhoria}

---

## ⭐ Pessoas de Sucesso
{tabela_sucesso_raw}

---

## 📋 Relatório Final
{relatorio_final}
                """
                
                st.download_button(
                    label="📄 Baixar Relatório Completo (.txt)",
                    data=relatorio_completo,
                    file_name=f"relatorio_personalidade_{data_str.replace('/', '_')}.txt",
                    mime="text/plain"
                )
                
                # Também oferecer download do CSV das pessoas de sucesso
                if not df_sucesso.empty:
                    csv = df_sucesso.to_csv(index=False)
                    st.download_button(
                        label="📊 Baixar Tabela de Pessoas de Sucesso (.csv)",
                        data=csv,
                        file_name=f"pessoas_sucesso_{data_str.replace('/', '_')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>💡 Sistema de Análise de Personalidade e Propósito de Vida</p>
        <p>Desenvolvido com ❤️ por Engº Paulo Rogério Veiga Silva!</p>
    </div>
    """, 
    unsafe_allow_html=True
)
