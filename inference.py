import os
import spacy
import pandas as pd
import re
from glob import glob
from bs4 import BeautifulSoup
import csv

gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

model_name = '../../models/llama2/mix/Llama-3-8b-mix-2epoch/merged/'

device_map = [f"cuda:{i}" for i in gpus.split(",")]

from vllm import LLM,SamplingParams
sampling_params = SamplingParams(max_tokens=512,stop='<EOS>',temperature=0)
llm = LLM(model=f"{model_name}", tensor_parallel_size=len(device_map))  # Create an LLM.


def batch_list(input_list, batch_size):
    batched_list = []
    for i in range(0, len(input_list), batch_size):
        batched_list.append(input_list[i:i + batch_size])
    return batched_list


def replace_entities_with_types(sent, entities):
    sent_text = str(sent)
    if isinstance(entities, list):
        for e in entities:
            ent_type, start, end =e
            sent_text = sent_text[:start - sent.start_char]+f'<span class="{ent_type}">{sent_text[start - sent.start_char:end - sent.start_char]}</span>'+sent_text[end - sent.start_char:] 
    else:
        ent_type, start, end =entities
        sent_text = sent_text[:start - sent.start_char]+f'<span class="{ent_type}">{sent_text[start - sent.start_char:end - sent.start_char]}</span>'+sent_text[end - sent.start_char:] 
    return sent_text

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

### Input Text: {} <EOS>
### Output Text:'''

test_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="labvalue"> to denote a numeric value or a normal description of the result of a lab test.
Use <span class="reference_range"> to denote the range or interval of values that are deemed as normal for a test in a healthy person.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a test.

### Input Text: {} <EOS>
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

### Input Text: {} <EOS>
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

### Input Text: {} <EOS>
### Output Text:'''

treatment_prompt = '''### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="temporal"> to denote a calendar date, time, or duration related to a treatment.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.

### Input Text: {} <EOS>
### Output Text:'''

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


separator = '\t'
batch_size = 100
for dataset in ['MTSample']:
    print(dataset)
    files = glob(f'./data/test/{dataset}/sentence_level_bio/*.bio')

    prompts = []
    for i, file in enumerate(files):
        with open(file, 'r', encoding='utf-8') as f_read:
            text = ' '.join([line.split(separator)[0] for line in f_read.read().splitlines()])
        file_name = file.split('/')[-1].split('.')[0]
        prompts.append(NER_prompt.format(text))

    prompts_list = batch_list(prompts, batch_size)

    NER_outputs = []
    
    print ("Running NER inference")
    for i, prompt_list in enumerate(prompts_list):
        # Generate the output
        output = llm.generate(prompt_list, sampling_params, use_tqdm=False)
        
        NER_outputs += output
        
    for i,seq in enumerate(NER_outputs):
        file_name = files[i].split('/')[-1].split('.')[0]
        with open(f'./output/NER/{file_name}.html','w',encoding='utf-8') as f_write:
            f_write.write(seq.outputs[0].text)
            
    print ("NER inference done")
    print ("Running RE inference")
            
    RE_unprocessed = []
    types = []
    data_idx = []
    for i, seq in enumerate(NER_outputs):
        NER_output = seq.outputs[0].text
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
                
    prompts_list = batch_list(RE_unprocessed, batch_size)
    RE_outputs = []
    
    for i, prompt_list in enumerate(prompts_list):
        # Generate the output
        output = llm.generate(prompt_list, sampling_params, use_tqdm=False)
        RE_outputs += output
        
    with open('./output/RE/MTSample.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['data_idx','Type', 'RE_input', 'RE_output'])
        for idx, type, RE_input, RE_output in zip(data_idx, types, RE_unprocessed, RE_outputs):
            writer.writerow([idx, type, RE_input, RE_output.outputs[0].text])
    print ("RE inference done")