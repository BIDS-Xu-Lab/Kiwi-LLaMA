from glob import glob
import random,torch,os
from transformers import AutoTokenizer, GenerationConfig
from accelerate import Accelerator
import os
import pandas as pd
import spacy
from glob import glob
import re
import argparse
from peft import AutoPeftModelForCausalLM
from bs4 import BeautifulSoup
import csv

device='cuda:0'
token = '' ### your huggingface token


finetuned_model_name = './models/Llama-2-7b-2epoch/'
if 'Llama-2-7b' in finetuned_model_name:
    model_id = "meta-llama/Llama-2-7b-chat-hf"
if 'Llama-2-13b' in finetuned_model_name:
    model_id = "meta-llama/Llama-2-13b-chat-hf"
if 'Llama-2-70b' in finetuned_model_name:
    model_id = "meta-llama/Llama-2-70b-chat-hf"
# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_id,token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = '|'
tokenizer.padding_side = "left"
model = AutoPeftModelForCausalLM.from_pretrained(f"{finetuned_model_name}",token=token, device_map='auto', torch_dtype=torch.bfloat16,cache_dir="/data/yhu5/huggingface_models/")

from peft import AutoPeftModelForCausalLM

def batch_list(input_list, batch_size):
    batched_list = []
    for i in range(0, len(input_list), batch_size):
        batched_list.append(input_list[i:i + batch_size])
    return batched_list

def run_generation(prompts,dataset,separator,batch_size,finetuned_model_name):
    prompts_list = batch_list(prompts, batch_size)

    outputs = []
    for i,prompt_list in enumerate(prompts_list):
        print (f'batch:{i + 1} of total:{len(prompts_list)}', flush=True)
        inputs = tokenizer(prompt_list, return_tensors="pt",padding=True).to(device)
        output = model.generate(input_ids=inputs["input_ids"].to(device), 
                                attention_mask=inputs["attention_mask"], 
                                max_new_tokens=512, 
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False,
                                num_beams=1,
                                eos_token_id=tokenizer.eos_token_id)
        outputs += tokenizer.batch_decode(output,skip_special_tokens=True)

    model_name = finetuned_model_name.split('/')[-2]
    dir_path = f'./finetuned_chat_output/{model_name}/{dataset}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    outputs = [seq[len(prompts[i])+1:].split('<EOS>')[0] for i,seq in enumerate(outputs)]
    return outputs

def get_RE_instance(NER_output):
    soup = BeautifulSoup(NER_output, 'html.parser')
    span_tags = soup.find_all('span')
    html_snippets = []
    for i, span in enumerate(span_tags):
        type = span.get('class')[0]
        new_soup = BeautifulSoup('', 'html.parser')
        
        new_soup.append(span)
        before_text = NER_output[:NER_output.find(str(span))]
        before_text = BeautifulSoup(before_text, 'html.parser')
        for span_tmp in before_text.find_all('span'):
            span_tmp.unwrap()
    
        after_text = NER_output[NER_output.find(str(span)) + len(str(span)):]
        after_text = BeautifulSoup(after_text, 'html.parser')
        for span_tmp in after_text.find_all('span'):
            span_tmp.unwrap()
        
        new_html = str(before_text) + str(new_soup) + str(after_text)
        html_snippets.append((type,new_html))
    
    return html_snippets

NER_prompt = '''### Task:
Your task is to generate an HTML version of an input text, using HTML <span> tags to mark up specific entities.

### Entity Markup Guides:
Use <span class="problem"> to denote a medical problem.
Use <span class="treatment"> to denote a treatment.
Use <span class="test"> to denote a test.
Use <span class="drug"> to denote a drug.

### Entity Definitions:
Medical Problem: The abnormal condition that happens physically or mentally to a patient.
Treatment: The procedures, interventions, and substances given to a patient for treating a problem.
Drug: Generic or brand name of a single medication or a collective name of a group of medication.
Test: A medical procedure performed (i) to detect or diagnose a problem, (ii) to monitor diseases, disease processes, and susceptibility, or (iii) to determine a course of treatment.

### Input Text: {} |<EOS>|
### Output Text:'''

test_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="labvalue"> to denote a numeric value or a normal description of the result of a lab test.
Use <span class="reference_range"> to denote the range or interval of values that are deemed as normal for a test in a healthy person.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a test.

### Input Text: {} |<EOS>|
### Output Text:'''

drug_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="form"> to denote the form of drug.
Use <span class="frequency"> to denote the frequency of taking a drug.
Use <span class="dosage"> to denote the amount of active ingredient from the number of drugs prescribed.
Use <span class="duration"> to denote the time period a patient should take a drug.
Use <span class="strength"> to denote the amount of active ingredient in a given dosage form.
Use <span class="route"> to denote the way by which a drug, fluid, poison, or other substance is taken into the body.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a drug.

### Input Text: {} |<EOS>|
### Output Text:'''

problem_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="uncertain"> to denote a measure of doubt.
Use <span class="condition"> to denote a phrase that indicates the problems existing in a certain situation.
Use <span class="subject"> to denote the person entity who is experiencing the disorder.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="bodyloc"> to denote the location on the body where the observation is present.
Use <span class="severity"> to denote the degree of intensity of a clinical condition.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a problem.
Use <span class="course"> to denote the development or alteration of a problem.

### Input Text: {} |<EOS>|
### Output Text:'''

treatment_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="temporal"> to denote a calendar date, time, or duration related to a treatment.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.

### Input Text: {} |<EOS>|
### Output Text:'''


def main():
    dataset = 'MTSample_test'
    file = f'./data/e2e/{dataset}.tsv'

    
    unprocessed = []
    with open(file) as f:
        lines=f.readlines()
        for line in lines:
            unprocessed.append(NER_prompt.format(line.split('\t')[0]))
    
    batch_size = 5
    separator = '\t'
    NER_outputs = run_generation(unprocessed,dataset,separator,batch_size,finetuned_model_name)

    RE_unprocessed = []
    types = []
    data_idx = []
    for i, NER_output in enumerate(NER_outputs):
        Re_instances = get_RE_instance(NER_output)
    
        for Re_instance in Re_instances:
            type = Re_instance[0]
            instance = Re_instance[1]

            types.append(type)
            data_idx.append(i)
            if type == 'problem': RE_unprocessed.append(problem_prompt.format(instance))
            if type == 'treatment': RE_unprocessed.append(treatment_prompt.format(instance))
            if type == 'test': RE_unprocessed.append(test_prompt.format(instance))
            if type == 'drug': RE_unprocessed.append(drug_prompt.format(instance))

    RE_outputs = run_generation(RE_unprocessed,dataset,separator,batch_size,finetuned_model_name)

    with open(f'./output/7b_e2e_{dataset}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['data_idx','Type', 'RE_input', 'RE_output'])
        for idx, type, RE_input, RE_output in zip(data_idx, types, RE_unprocessed, RE_outputs):
            writer.writerow([idx, type, RE_input, RE_output])
if __name__ == "__main__":
    main()