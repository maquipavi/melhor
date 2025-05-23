import streamlit as st
import asyncio
import os
from datetime import datetime
import pandas as pd
import re
import warnings
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# Configurações da página
st.set_page_config(
    page_title="Análise de Personalidade e Propósito de Vida",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configurações e inicialização
warnings.filterwarnings("ignore")

@st.cache_resource
def initialize_services():
    """Inicializa os serviços necessários"""
    # Configurar API Key (você deve definir isso como uma variável de ambiente ou secret)
    api_key = st.secrets.get("GOOGLE_API_KEY", "AIzaSyBarB5CfRsl_M0nkQjgg-ystWV-CyzN0jU")
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Configurar cliente
    client = genai.Client()
    
    # Modelos
    MODELO_RAPIDO = "gemini-2.0-flash"
    MODELO_ROBUSTO = "gemini-2.0-flash"
    
    # Serviço de sessão
    session_service = InMemorySessionService()
    
    return client, MODELO_RAPIDO, MODELO_ROBUSTO, session_service

# Função auxiliar para chamar agentes
async def call_agent(agent: Agent, message_text: str, session_service) -> str:
    """Chama um agente e retorna a resposta"""
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

# Agente 1: Analisador de Nascimento
async def agente_analisador(data_nascimento, modelo_rapido, session_service):
    """Agente que analisa personalidade baseada na data de nascimento"""
    analisador = Agent(
        name="agente_analisador",
        model=modelo_rapido,
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

    return await call_agent(analisador, entrada_do_agente_analisador, session_service)

# Agente 2: Identificador de Melhorias
async def agente_melhorias(data_nascimento, analises_agente1, modelo_rapido, session_service):
    """Agente que identifica pontos de melhoria"""
    melhorias = Agent(
        name="agente_melhorias",
        model=modelo_rapido,
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

    return await call_agent(melhorias, entrada_do_agente_melhorias, session_service)

# Agente 3: Buscador de Pessoas de Sucesso
async def agente_buscador_sucesso(data_nascimento, modelo_robusto, session_service):
    """Agente que busca pessoas de sucesso nascidas na mesma data"""
    buscador_sucesso = Agent(
        name="agente_buscador_sucesso",
        model=modelo_robusto,
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

    tabela_sucesso = await call_agent(buscador_sucesso, entrada_do_agente_buscador_sucesso, session_service)
    
    # Processar a tabela para DataFrame
    try:
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
    except Exception as e:
        st.error(f"Erro ao processar tabela: {e}")
        return pd.DataFrame(), tabela_sucesso

# Agente 4: Gerador de Relatório Final
async def agente_relatorio_final(data_nascimento, analises, melhorias, tabela_sucesso, modelo_rapido, session_service):
    """Agente que gera o relatório final"""
    relatorio = Agent(
        name="agente_relatorio",
        model=modelo_rapido,
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

    return await call_agent(relatorio, entrada_do_agente_relatorio, session_service)

# Função principal para executar a análise
async def executar_analise_completa(data_nascimento):
    """Executa todo o sistema de análise"""
    try:
        client, modelo_rapido, modelo_robusto, session_service = initialize_services()
        
        # Execução sequencial dos agentes
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Agente 1
        status_text.text("🔍 Executando análise de personalidade...")
        progress_bar.progress(0.25)
        analises_agente1 = await agente_analisador(data_nascimento, modelo_rapido, session_service)
        
        # Agente 2
        status_text.text("📈 Identificando pontos de melhoria...")
        progress_bar.progress(0.50)
        pontos_de_melhoria = await agente_melhorias(data_nascimento, analises_agente1, modelo_rapido, session_service)
        
        # Agente 3
        status_text.text("🌟 Buscando pessoas de sucesso...")
        progress_bar.progress(0.75)
        df_sucesso, tabela_sucesso_raw = await agente_buscador_sucesso(data_nascimento, modelo_robusto, session_service)
        
        # Agente 4
        status_text.text("📋 Gerando relatório final...")
        progress_bar.progress(0.90)
        relatorio_final = await agente_relatorio_final(data_nascimento, analises_agente1, pontos_de_melhoria, tabela_sucesso_raw, modelo_rapido, session_service)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Análise concluída!")
        
        return analises_agente1, pontos_de_melhoria, df_sucesso, relatorio_final
        
    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")
        return None, None, None, None

# Interface principal
def main():
    """Função principal da aplicação Streamlit"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Sistema de Análise de Personalidade e Propósito de Vida</h1>
        <p>Descubra insights profundos sobre sua personalidade baseados na sua data de nascimento</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Configurações")
    st.sidebar.markdown("### Como funciona:")
    st.sidebar.markdown("""
    1. **Digite sua data de nascimento**
    2. **Clique em 'Iniciar Análise'**
    3. **Aguarde os 4 agentes trabalharem:**
       - 🔍 Analisador de Personalidade
       - 📈 Identificador de Melhorias
       - 🌟 Buscador de Pessoas de Sucesso
       - 📋 Gerador de Relatório Final
    """)
    
    # Input da data de nascimento
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📅 Sua Data de Nascimento")
        data_nascimento = st.date_input(
            "Selecione sua data de nascimento:",
            value=datetime(1990, 1, 1),
            min_value=datetime(1900, 1, 1),
            max_value=datetime.now()
        )
        
        data_formatada = data_nascimento.strftime('%d/%m/%Y')
        
        if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
            st.session_state.executar_analise = True
            st.session_state.data_nascimento = data_formatada
    
    # Executar análise se solicitado
    if st.session_state.get('executar_analise', False):
        data_para_analise = st.session_state.get('data_nascimento', data_formatada)
        
        st.markdown(f"""
        <div class="success-message">
            <strong>✅ Análise iniciada para: {data_para_analise}</strong><br>
            Aguarde enquanto nossos agentes especializados trabalham para você...
        </div>
        """, unsafe_allow_html=True)
        
        # Executar análise
        analises, melhorias, df_sucesso, relatorio = asyncio.run(
            executar_analise_completa(data_para_analise)
        )
        
        if analises:
            # Exibir resultados
            st.markdown("## 📊 Resultados da Análise")
            
            # Criar tabs para organizar os resultados
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Análise de Personalidade", 
                "📈 Pontos de Melhoria", 
                "🌟 Pessoas de Sucesso", 
                "📋 Relatório Final"
            ])
            
            with tab1:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🔍 Análise Detalhada de Personalidade")
                st.markdown(analises)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 📈 Áreas de Melhoria e Desenvolvimento")
                st.markdown(melhorias)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🌟 Pessoas de Sucesso - Mesma Data de Nascimento")
                if not df_sucesso.empty:
                    st.dataframe(df_sucesso, use_container_width=True)
                else:
                    st.info("Não foram encontradas pessoas de sucesso para esta data específica.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 📋 Relatório Final Motivacional")
                st.markdown(relatorio)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Botão para nova análise
            if st.button("🔄 Fazer Nova Análise", type="secondary"):
                st.session_state.executar_analise = False
                st.rerun()
        
        # Limpar flag de execução
        st.session_state.executar_analise = False
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 Desenvolvido com Google Gemini AI Agents | 
        💫 Sua jornada de autoconhecimento começa aqui</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()