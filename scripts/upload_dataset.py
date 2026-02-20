import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "langgraph-hello-world-golden"

examples = [
    # Simple questions (easy for baseline)
    ("What is 5 + 7?", "12"),
    ("What is 15 * 4?", "60"),
    ("What is 100 * 100?", "10000"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("Who was the first US president?", "Washington"),
    ("What is the capital of France?", "Paris"),
    # Harder questions where DSPy can help optimization
    # These require nuance, multi-step reasoning, or specific knowledge
    ("If I have 3 apples and eat 1, then buy 5 more, how many do I have?", "7"),
    ("What is the square root of 144?", "12"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What year did the US declare independence?", "1776"),
    ("What is 25% of 80?", "20"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is 7 to the power of 2?", "49"),
    ("In what year did man first land on the moon?", "1969"),
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
