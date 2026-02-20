import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import calculator, wikipedia_search, get_current_time


@pytest.fixture(scope="module")
def llm():
    return ChatOllama(model="llama3.2")


@pytest.fixture(scope="module")
def agent(llm):
    tools = [calculator, wikipedia_search, get_current_time]
    checkpointer = MemorySaver()
    return create_agent(llm, tools, checkpointer=checkpointer)


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test"}}
