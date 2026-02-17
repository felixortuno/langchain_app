import streamlit as st
import os
from typing import Annotated, Literal, TypedDict
import functools

# LangChain & LangGraph
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Conchita AI News Writer", page_icon="📝", layout="wide")

# --- CSS PERSONALIZADO (Opcional) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: GESTIÓN DE CREDENCIALES ---
with st.sidebar:
    st.header("🔑 Configuración")
    
    # Inputs para las claves
    google_key_input = st.text_input("Google API Key", type="password", key="google_key_input")
    tavily_key_input = st.text_input("Tavily API Key", type="password", key="tavily_key_input")
    
    # Asignación a variables de entorno si existen
    if google_key_input:
        os.environ["GOOGLE_API_KEY"] = google_key_input
    if tavily_key_input:
        os.environ["TAVILY_API_KEY"] = tavily_key_input

    st.divider()
    st.info("Modelo seleccionado: **gemini-2.5-flash**")
    st.caption("Asegúrate de que tus librerías estén actualizadas para usar este modelo.")

# --- LÓGICA DE LANGGRAPH ---

# 1. Definición del Estado
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Función para crear agentes
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

# 3. Nodo genérico de agente
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Convertimos el resultado a un formato compatible si es necesario
    return {"messages": [result]}

# 4. Lógica condicional (Router)
def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    # Si el LLM pide usar una herramienta, vamos al nodo "tools"
    if last_message.tool_calls:
        return "tools"
    #