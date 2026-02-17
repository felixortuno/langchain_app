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
st.set_page_config(page_title="Conchita AI News Writer", page_icon="📝")

# --- SIDEBAR: GESTIÓN DE CREDENCIALES ---
with st.sidebar:
    st.title("Configuración de API")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.info("Estas llaves son necesarias para que los agentes puedan investigar y escribir.")

# --- LÓGICA DE LANGGRAPH ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "outliner"

def init_graph():
    if not google_key or not tavily_key:
        return None

    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    tools = [TavilySearchResults(max_results=5)]
    
    # Templates
    search_template = "Job: Search web for relevant news for the user's article. DO NOT write the article. Forward news to outliner."
    outliner_template = "Job: Take web articles and user intent to generate a structured outline for the article."
    writer_template = "Job: Write the article. Format: TITLE: <title> \n BODY: <body>. Do not copy outline, use it to write original content."

    # Agents
    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    # Nodes
    workflow = StateGraph(AgentState)
    workflow.add_node("search", functools.partial(agent_node, agent=search_agent, name="Search Agent"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent"))
    workflow.add_node("writer", functools.partial(agent_node, agent=writer_agent, name="Writer Agent"))

    # Edges
    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# --- INTERFAZ DE USUARIO ---
st.title("📝 Conchita: News Writer Agent")
st.markdown("Genera artículos profesionales basados en tendencias de la web en tiempo real.")

topic = st.text_input("¿Sobre qué quieres que escriba hoy?", placeholder="Ej: Avances en computación cuántica 2024")

if st.button("Generar Artículo"):
    if not google_key or not tavily_key:
        st.error("Por favor, introduce ambas API Keys en la barra lateral.")
    elif not topic:
        st.warning("Introduce un tema para comenzar.")
    else:
        graph = init_graph()
        
        try:
            with st.status("🚀 Orquestando agentes...", expanded=True) as status:
                input_data = {"messages": [HumanMessage(content=topic)]}
                
                # Stream de eventos para mostrar progreso
                for event in graph.stream(input_data, stream_mode="values"):
                    last_msg = event["messages"][-1]
                    
                    if hasattr(last_msg, "name") and last_msg.name:
                        st.write(f"✅ **{last_msg.name}** ha terminado su tarea.")
                    elif last_msg.type == "ai" and not last_msg.tool_calls:
                        # Identificar qué nodo generó el mensaje basándonos en el flujo
                        # (Simplificado para la UI)
                        pass
                
                status.update(label="✅ ¡Artículo completado!", state="complete", expanded=False)

            # --- RESULTADO FINAL ---
            # El último mensaje del grafo (del Writer Agent) contiene el artículo
            final_state = graph.get_state({"messages": [HumanMessage(content=topic)]}) # Nota: en un flujo real usamos el historial
            # Obtenemos la última respuesta del Writer
            all_events = list(graph.stream(input_data, stream_mode="values"))
            final_text = all_events[-1]["messages"][-1].content

            st.divider()
            st.subheader("Resultado Final")
            st.markdown(final_text)
            
        except Exception as e:
            st.error(f"Ocurrió un error durante la ejecución: {str(e)}")