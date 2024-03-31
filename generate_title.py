import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
  
if __name__ == "__main__":
    print("Model loading...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    PEFT_MODEL = "checkpoint-1000"

    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, PEFT_MODEL)

    text = input("You can insert an abstract of the article here and I will generate for you a possible title for it:\n")
    number = int(input("How many example to generate?\n"))
    while text != 'exit':
        # generation params
        generation_params = {
            "max_new_tokens": 100,
            "num_beams": max(4, number),
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "num_return_sequences": number,
        }
        prompt = f"### Imagine yourself as a scientist who needs to come up with a title for an article. \
        Your task is with known abstract of paper create possible title of it. Abstract: {text}\n\n### Title: "

        device = "cuda:0"
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            beam_outputs = model.generate(
                input_ids = encoding.input_ids,
                attention_mask = encoding.attention_mask,
                **generation_params,
        )
        for i, beam_output in enumerate(beam_outputs):
            title = tokenizer.decode(beam_output, skip_special_tokens=True)
            sequence_start = title.find("### Title: ") + 11
            title = title[sequence_start:]
            print(f"{i}: {title}")
        text = input("You can insert an abstract of the article here and I will generate for you a possible abstract for it:\n")
        if text == "exit" or text == "":
            break
        number = int(input("How many example to generate?\n"))
    print("Generation ended")