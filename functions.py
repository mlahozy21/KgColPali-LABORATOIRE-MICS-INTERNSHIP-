import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
import google.generativeai.types as genai_types
from io import BytesIO
import sys
import numpy as np
from PIL import Image
from api import gemini, apitemplates
from colpali import colpalimodel, processor
from embeddings import dataset, image_embeddings
from kg import get_entity_context
device = "cuda:0"

            
def select(threshold, query_embeddings, image_embeddings, k):
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    if threshold > 0:  # Assuming 'seuil' is a boolean condition
        # Create a boolean mask for elements greater than 0.15
        mask = scores > threshold
        # Get the indices of the elements that meet the condition
        indices = torch.nonzero(mask, as_tuple=False)  # Shape: [n, 2]
        # Take only the column indices (dimension 1)
        if indices.numel() > 0:
            indices = indices[:, 1]  # Select the column (indices of dimension 1)
        else:
            indices = torch.tensor([], dtype=torch.long)
        # Filter the values that meet the condition
        filtered_scores = scores[mask]
        if filtered_scores.numel() == 0:
            print(f"No values greater than {threshold}")
            top_k_scores = torch.tensor([])
            top_k_indices = torch.tensor([])
        else:
            # Apply topk to the filtered values
            top_k_scores, top_k_filtered_indices = torch.topk(
                filtered_scores, k=min(k, filtered_scores.numel()), largest=True
            )
            # Map the filtered indices to the original indices
            top_k_indices = indices[top_k_filtered_indices]
            print(f"Filtered {len(filtered_scores)} scores greater than {threshold}")
            print(f"Selected these scores after applying topk: {top_k_scores}")
    else:
        # If no filtering is applied, use the original tensor
        top_k_scores, top_k_indices = torch.topk(scores[0], k=k, largest=True)
    return top_k_scores, top_k_indices

def get_relevant_documents(question,kg=2,kg_context='', k=32,thresholdrag=0,thresholdkg=0, **kwarg):
    assert type(question) == str
    scores = []
    retrieved_snippets = [] 
    current_embeddings = image_embeddings
    question = [question]
    batch_queries = processor.process_queries(question).to(colpalimodel.device)
    query_embeddings= colpalimodel(**batch_queries)
    query_embeddings=query_embeddings.unsqueeze(1)
    if k>0: 
        top_k_scores, top_k_indices = select(thresholdrag, query_embeddings, current_embeddings,k)
        for index_in_dataset in top_k_indices:
            # Convertir el índice a un entero de Python
            original_idx = index_in_dataset.item()  # Esto funciona porque top_k_indices es 1D
            # Recuperar el número de página del dataset usando el índice
            snippet_image = dataset[original_idx]
            retrieved_snippets.append(snippet_image)             
    if kg == 4 or kg == 3:
        batch_kg= processor.process_queries(kg_context).to(colpalimodel.device)
        kg_embeddings = colpalimodel(**batch_kg)
        kg_embeddings = kg_embeddings.unsqueeze(1)
        if kg == 3:
            kg_scores = processor.score_multi_vector(query_embeddings,kg_embeddings)
            kg_scores = kg_scores.squeeze(0)
            print(f"KG scores: {kg_scores.mean()}")  # Debugging line
            if kg_scores.mean()<thresholdkg:
                kg_context = ''
        elif kg == 4:       
            top_k_scores, top_k_indices = select(threshold=thresholdkg,query_embeddings=kg_embeddings,image_embeddings=current_embeddings,k=k)
            for index_in_dataset in top_k_indices:
                # Convertir el índice a un entero de Python
                original_idx = index_in_dataset.item()  # Esto funciona porque top_k_indices es 1D
                # Recuperar el número de página del dataset usando el índice
                snippet_image = dataset[original_idx]
                retrieved_snippets.append(snippet_image)
    return retrieved_snippets, kg_context

def prepare_snippets_for_gemini(retrieved_snippets,content):
    for snippet in retrieved_snippets:
        try:
            # Convertir snippet a formato compatible con Gemini API
            if isinstance(snippet, str):
                img = Image.open(snippet)
                # Convertir imagen a bytes
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_data = buffer.getvalue()
                content.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                })
            elif isinstance(snippet, Image.Image):
                # Convertir objeto PIL.Image a bytes
                buffer = BytesIO()
                snippet.save(buffer, format="PNG")
                img_data = buffer.getvalue()
                content.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                })
            else:
                print(f"Snippet no soportado: {snippet}")
        except Exception as e:
            print(f"Error al procesar {snippet}: {str(e)}")
            continue
    
    return content

def medrag_answer(question, options=None,k=32, kg=1,thresholdrag=0,thresholdkg=0,**kwargs):
    # Formatear opciones si existen
    if options is not None:
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
    else:
        options_text = ''
    retrieved_snippets = []
    scores = [] #Modos: 1. nada, 2. solo RAG, 3. RAG y KG context, 4. RAG y KG retrieve
    if kg == 1:
        #Meter aquí template sin KG
        system_prompt = apitemplates["cot_system"]
        prompt_template = apitemplates["cot_prompt"]
        prompt = prompt_template.render(
            question=question,
            options=options_text
        ) 
    elif kg == 2:
        #Meter aquí template con RAG
        retrieved_snippets,_= get_relevant_documents(question,thresholdrag=thresholdrag,thresholdkg=thresholdkg, k=k)
        if len(retrieved_snippets) == 0:
            system_prompt = apitemplates["cot_system"]
            prompt_template = apitemplates["cot_prompt"]
            prompt = prompt_template.render(
                question=question,
                options=options_text
            ) 
        else:
            system_prompt = apitemplates["medrag_system"]
            prompt_template = apitemplates["medrag_prompt"]
            prompt = prompt_template.render(
                question=question,
                options=options_text
            ) 
    elif kg == 3:
        kg_context = get_entity_context(question)
        retrieved_snippets, kg_context= get_relevant_documents(question,thresholdrag=thresholdrag,thresholdkg=thresholdkg, k=k,kg=3,kg_context=kg_context)
        if kg_context == 'No medical nouns identified in the query.':
            kg_context = ''
        if len(retrieved_snippets) == 0 and kg_context == '':
            system_prompt = apitemplates["cot_system"]
            prompt_template = apitemplates["cot_prompt"]
            prompt = prompt_template.render(
                kg_context=kg_context,
                question=question,
                options=options_text
            )
        else:
            system_prompt = apitemplates["kgcontext_system"]
            prompt_template = apitemplates["kgcontext_prompt"]
            prompt = prompt_template.render(
                kg_context=kg_context,
                question=question,
                options=options_text
            ) 
    elif kg == 4:
        kg_context = get_entity_context(question)
        retrieved_snippets,kg_context = get_relevant_documents(question=question,thresholdrag=thresholdrag,thresholdkg=thresholdkg,k=k, kg=4, kg_context=kg_context)
        system_prompt = apitemplates["medrag_system"]
        prompt_template = apitemplates["medrag_prompt"]
        prompt = prompt_template.render(
            question=question,
            options=options_text
        ) 
    # Preparar entrada (texto y imágenes recuperadas)
    content = [
        {"text": system_prompt},
        {"text": prompt}
    ]   
    # Añadir imágenes recuperadas
    content= prepare_snippets_for_gemini(retrieved_snippets,content)
    
    response = gemini.generate_content(content)
    answer = response.text.strip()
    
    return answer
