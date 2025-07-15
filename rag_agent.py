import os
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph, END
from db_utils import list_tables, describe_table, execute_query

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Define Tools ===
@tool
def list_tables_tool() -> list[str]:
    """Lists all tables in the pharmacy database."""
    return list_tables()

@tool
def describe_table_tool(table_name: str) -> list[tuple[str, str]]:
    """Describes the schema of a given table."""
    return describe_table(table_name)

@tool
def execute_query_tool(sql: str) -> list[list[str]]:
    """Executes a SQL query and returns the result."""
    return execute_query(sql)

tools = [list_tables_tool, describe_table_tool, execute_query_tool]

# === Define LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
)

# === Define Prompt ===
system_msg = SystemMessage(content="""
You are a pharmacy assistant. You will receive a JSON prescription including medicine names, dosage, frequency, and duration.

Your job is to:
1. If all required information is available, calculate quantity = frequency × duration.
2. Then check availability of the drug from the 'drugs' table.
3. If the drug is available in sufficient quantity:
    - Subtract the quantity from the inventory.
    - Insert a sale record into the 'sales' table.
    - Respond with: "Inventory update successful."
4. If **any** information is missing or the drug is unavailable:
    - Respond with: "Information insufficient, inventory update not successful."
Never ask the user for additional information.
Never guess. Never output JSON. Just return one of the two exact messages.
""")


prompt = ChatPromptTemplate.from_messages([
    system_msg,
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# === Create Agent ===
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

# === Agent State ===
class AgentState:
    def __init__(self, parsed=None, input=None, db_result=None):
        self.parsed = parsed
        self.input = input
        self.db_result = db_result

    def to_dict(self):
        return {
            "parsed": self.parsed,
            "input": self.input,
            "db_result": self.db_result
        }

    def __getitem__(self, key):
        return self.to_dict().get(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(self.to_dict())

# === LangGraph Nodes ===
def parse_node(state: dict) -> dict:
    print("[DEBUG] Full state in parse_node:", state)
    parsed_json = state.get("parsed")

    if not parsed_json:
        raise ValueError("[ERROR] Missing 'parsed' key in state")

    state["input"] = json.dumps(parsed_json)
    return state

def db_check_node(state: dict) -> dict:
    result = agent_executor.invoke({"input": state["input"]})
    state["db_result"] = result
    return state

def confirm_node(state: dict) -> dict:
    return state

# === LangGraph ===
graph = StateGraph(dict)
graph.add_node("Parse", parse_node)
graph.add_node("CheckDB", db_check_node)
graph.set_entry_point("Parse")
graph.add_edge("Parse", "CheckDB")
graph.add_edge("CheckDB", END)

rag_graph = graph.compile()

def run_agent(parsed_json: dict):
    try:
        initial_state = AgentState(parsed=parsed_json)
        final_state = rag_graph.invoke(initial_state.to_dict())
        result = final_state.get("db_result")

        # Normalize the result to our expected messages
        if isinstance(result, dict):
            output_text = result.get("output", "")
        elif isinstance(result, str):
            output_text = result
        else:
            output_text = str(result)

        if "Inventory update successful" in output_text:
            return "✅ Inventory update successful."
        else:
            return "❌ Information insufficient, inventory update not successful."
    except Exception as e:
        return f"[ERROR] {str(e)}"

