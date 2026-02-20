import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "langgraph-hello-world-golden"

examples = [
    # Original simple questions
    ("What is 5 + 7?", "12"),
    ("What is 15 * 4?", "60"),
    ("What is 100 * 100?", "10000"),
    ("What is 50 / 2?", "25"),
    ("What is 200 - 75?", "125"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("Who was Albert Einstein?", "Einstein"),
    ("Who was the first US president?", "Washington"),
    ("What year did WWII end?", "1945"),
    ("What is the capital of France?", "Paris"),
    # Harder questions where baseline may struggle
    # These require specific formatting, nuance, or multi-step reasoning
    ("What is 2 + 2?", "4"),  # Very simple - should work
    ("Who is the author of Harry Potter?", "JK Rowling"),
    ("What is 17 + 83?", "100"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What year did the Titanic sink?", "1912"),
    ("What is 999 + 1?", "1000"),
]

dataset = client.create_dataset(DATASET_NAME)
print(f"Created dataset: {DATASET_NAME}")

for question, answer in examples:
    client.create_example(
        inputs={"question": question}, outputs={"answer": answer}, dataset_id=dataset.id
    )
    print(f"  Added: {question[:40]}...")

print(f"\nTotal examples: {len(examples)}")
print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")
