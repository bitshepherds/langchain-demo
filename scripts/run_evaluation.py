import os
import sys
from dotenv import load_dotenv
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

load_dotenv()

from langsmith import Client
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import calculator, wikipedia_search, get_current_time

DATASET_NAME = "langgraph-hello-world-golden"
JUDGE_MODEL = "llama3.2"

llm = ChatOllama(model="llama3.2")
judge_llm = ChatOllama(model=JUDGE_MODEL)

tools = [calculator, wikipedia_search, get_current_time]
checkpointer = MemorySaver()
agent = create_agent(llm, tools, checkpointer=checkpointer)

client = Client()


def run_agent(inputs: dict) -> dict:
    question = inputs.get("question", "")
    config = {"configurable": {"thread_id": "eval"}}
    result = agent.invoke({"messages": [("user", question)]}, config)
    return {
        "question": question,
        "response": result["messages"][-1].content,
    }


def correctness_evaluator(run, example) -> dict:
    reference = example.outputs.get("answer", "")
    response = run.outputs.get("response", "")

    prompt = f"""You are an expert evaluator. Compare the reference answer to the model's response.

Reference Answer: {reference}
Model Response: {response}

Evaluate if the model response contains the correct answer. Consider:
1. Does the response contain the key information from the reference?
2. Is the answer factually correct?
3. Partial matches count as correct if the essential info is present.

Respond with ONLY a JSON object:
{{"score": 1.0 if correct, 0.0 if incorrect, "reason": "brief explanation"}}"""

    result = judge_llm.invoke(prompt)
    content = result.content

    try:
        import json
        for line in content.split('\n'):
            if '{' in line and '}' in line:
                start = line.find('{')
                end = line.rfind('}') + 1
                eval_result = json.loads(line[start:end])
                return {
                    "key": "correctness",
                    "score": eval_result.get("score", 0.0),
                    "comment": eval_result.get("reason", ""),
                }
    except:
        pass

    return {
        "key": "correctness",
        "score": 0.5,
        "comment": "Could not parse evaluation",
    }


def tool_calling_evaluator(run, example) -> dict:
    question = example.inputs.get("question", "")
    reference_tool = get_expected_tool(question)

    config = {"configurable": {"thread_id": "eval"}}
    result = agent.invoke({"messages": [("user", question)]}, config)

    tool_names = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names.append(tc.get("name", ""))

    expected_tool = reference_tool
    is_correct = expected_tool in tool_names

    return {
        "key": "correct_tool",
        "score": 1.0 if is_correct else 0.0,
        "comment": f"Expected {expected_tool}, found {tool_names}",
    }


def get_expected_tool(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in ["calculate", "what is", "sum", "multiply", "divide", "+", "-", "*", "/", "100 *"]):
        return "calculator"
    if any(kw in q for kw in ["who", "what year", "capital", "president"]):
        return "wikipedia_search"
    if "time" in q:
        return "get_current_time"
    return "unknown"


def main():
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    if not dataset:
        print(f"Dataset '{DATASET_NAME}' not found. Run scripts/upload_dataset.py first.")
        sys.exit(1)

    print(f"Running evaluation on dataset: {DATASET_NAME}")
    print(f"Using judge model: {JUDGE_MODEL}\n")

    experiment_results = client.evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, tool_calling_evaluator],
        experiment_prefix="llama3.2-baseline",
    )

    print("\nEvaluation complete!")
    print(f"View results at: https://smith.langchain.com/evaluation/experiments")


if __name__ == "__main__":
    main()
