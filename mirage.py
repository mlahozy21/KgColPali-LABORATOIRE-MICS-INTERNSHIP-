import json
import os
from tqdm import tqdm
from funciones import medrag_answer
from utilsmirage import QADataset
import random

# Load benchmark
#benchmark = json.load(open("benchmark.json"))
kg = 1  # Modes: 1. LLM only, 2. RAG only, 3. RAG and KG context, 4. RAG and KG retrieve
# Initialize predictions dictionary
predictions = {}
for dataset_name in ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]:
    # Fix a seed so the selection is always the same
    random.seed(42)
    dataset = QADataset(dataset_name) 
    my_list = list(range(len(QADataset(dataset_name))))  
    selection = random.sample(my_list, 200)
    dataset = [dataset[i] for i in selection]
    predictions[dataset_name] = []

    for idx, item in tqdm(enumerate(dataset), desc=f"Processing {dataset_name}"):
        question = item["question"]
        options = item["options"]
        # Use the dataset index as the ID if "id" is not in the item
        answer = medrag_answer(question=question, options=options, kg=kg, k=3, thresholdrag=0.25, thresholdkg=0.2)
        predictions[dataset_name].append({
            "question": question,
            "prediction": answer,
            "ground_truth": item["answer"]
        })

    # Save predictions
    os.makedirs("prediction", exist_ok=True)
    with open(f"prediction/{dataset_name}_predictions.json", "w") as f:
        json.dump(predictions[dataset_name], f, indent=2)