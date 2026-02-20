import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import calculator, wikipedia_search, get_current_time

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY and LANGSMITH_API_KEY != "your_key_here":
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "langgraph-hello-world"

llm = ChatOllama(model="llama3.2")

tools = [calculator, wikipedia_search, get_current_time]

checkpointer = MemorySaver()
agent = create_agent(llm, tools, checkpointer=checkpointer)


def main():
    config = {"configurable": {"thread_id": "main"}}

    print("LangGraph Agent with Ollama (llama3.2)")
    print("Tools: calculator, wikipedia_search, get_current_time")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        if not user_input.strip():
            continue

        result = agent.invoke({"messages": [("user", user_input)]}, config)
        response = result["messages"][-1].content
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
