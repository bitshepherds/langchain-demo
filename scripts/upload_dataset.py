import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "langgraph-hello-world-golden"

# Check if dataset already exists
existing = client.read_dataset(dataset_name=DATASET_NAME)
if existing:
    print(f"Dataset '{DATASET_NAME}' already exists. Deleting and recreating...")
    client.delete_dataset(dataset_id=existing.id)

examples = [
    # Harder questions where DSPy optimization can help
    # These have specific edge cases, nuances, or tricky phrasing
    ("If I have 3 apples and eat 1, then buy 5 more, how many do I have?", "7"),
    ("What is the square root of 144?", "12"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What year did the US declare independence?", "1776"),
    ("What is 25% of 80?", "20"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is 7 to the power of 2?", "49"),
    ("In what year did man first land on the moon?", "1969"),
    # Edge cases - tricky phrasing or specific answers
    ("Who is the current President of the United States?", "Trump"),  # Time-sensitive
    ("What is 123 + 456?", "579"),
    ("What is 999 * 999?", "998001"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"),
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
