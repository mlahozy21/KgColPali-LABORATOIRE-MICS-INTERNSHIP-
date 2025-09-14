import os
import torch

from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
device = "cuda:0"
model_name_colpali = "nomic-ai/nomic-embed-multimodal-3b"
os.environ['HF_TOKEN']= "YOUR_HF_TOKEN" 
# Initialize the BiQwen model (a version of colpali)
colpalimodel = BiQwen2_5.from_pretrained(
    model_name_colpali,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = BiQwen2_5_Processor.from_pretrained(model_name_colpali, use_fast=True)


