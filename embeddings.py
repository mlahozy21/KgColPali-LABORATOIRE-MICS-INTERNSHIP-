import os
import torch
from datasets import load_dataset
from paginasunicas import split_and_save_pdf, load_pdf_pages
device = "cuda:0"

colpali_embeddings_dir ="directory where you have the embeddings"
colpali_image_embeddings_path = os.path.join(colpali_embeddings_dir, "image_embeddings.pt")
dataset = load_dataset("./syntheticDocQA_artificial_intelligence_test", split="test")['image']
if os.path.exists(colpali_image_embeddings_path):
    image_embeddings= torch.load(colpali_image_embeddings_path).to(device)
    image_embeddings=image_embeddings.unsqueeze(1)


