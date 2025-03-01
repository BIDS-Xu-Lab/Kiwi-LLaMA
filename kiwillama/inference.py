import os
from glob import glob
from bs4 import BeautifulSoup
import csv


from vllm import LLM, SamplingParams

from kiwillama.constants import (
    NER_prompt,
    test_prompt,
    drug_prompt,
    problem_prompt,
    treatment_prompt,
)
from kiwillama import settings

os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES

device_map = [f"cuda:{i}" for i in settings.CUDA_VISIBLE_DEVICES.split(",")]


sampling_params = SamplingParams(max_tokens=512, stop="<EOS>", temperature=0)
llm = LLM(
    model=f"{settings.KIWI_LLAMA_MODEL_DIR}", tensor_parallel_size=len(device_map)
)  # Create an LLM.


def batch_list(input_list, batch_size):
    return [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]


def replace_entities_with_types(sent, entities):
    sent_text = str(sent)
    if isinstance(entities, list):
        for e in entities:
            ent_type, start, end = e
            sent_text = (
                sent_text[: start - sent.start_char]
                + f'<span class="{ent_type}">{sent_text[start - sent.start_char:end - sent.start_char]}</span>'
                + sent_text[end - sent.start_char :]
            )
    else:
        ent_type, start, end = entities
        sent_text = (
            sent_text[: start - sent.start_char]
            + f'<span class="{ent_type}">{sent_text[start - sent.start_char:end - sent.start_char]}</span>'
            + sent_text[end - sent.start_char :]
        )
    return sent_text


def get_RE_instance(NER_output):
    soup = BeautifulSoup(NER_output, "html.parser")
    span_tags = soup.find_all("span")
    html_snippets = []
    for i, span in enumerate(span_tags):
        tag_type = span.get("class")[0]
        new_soup = BeautifulSoup("", "html.parser")

        new_soup.append(span)
        before_text = NER_output[: NER_output.find(str(span))]
        before_text = BeautifulSoup(before_text, "html.parser")
        for span_tmp in before_text.find_all("span"):
            span_tmp.unwrap()

        after_text = NER_output[NER_output.find(str(span)) + len(str(span)) :]
        after_text = BeautifulSoup(after_text, "html.parser")
        for span_tmp in after_text.find_all("span"):
            span_tmp.unwrap()

        new_html = str(before_text) + str(new_soup) + str(after_text)
        html_snippets.append((tag_type, new_html))

    return html_snippets


separator = "\t"
for dataset in ["MTSample"]:
    print(dataset)
    files = glob(f"./data/test/{dataset}/sentence_level_bio/*.bio")

    prompts = []
    for i, file in enumerate(files):
        with open(file, "r", encoding="utf-8") as f_read:
            text = " ".join(
                [line.split(separator)[0] for line in f_read.read().splitlines()]
            )
        file_name = file.split("/")[-1].split(".")[0]
        prompts.append(NER_prompt().format(text))

    prompts_list = batch_list(prompts, settings.PROMPT_BATCH_SIZE)

    NER_outputs = []

    print("Running NER inference")

    # NOTE: This is the bottleneck.
    for i, prompt_list in enumerate(prompts_list):
        # Generate the output
        output = llm.generate(prompt_list, sampling_params, use_tqdm=False)

        NER_outputs += output

    for i, seq in enumerate(NER_outputs):
        file_name = files[i].split("/")[-1].split(".")[0]
        with open(f"./output/NER/{file_name}.html", "w", encoding="utf-8") as f_write:
            f_write.write(seq.outputs[0].text)

    print("NER inference done")
    print("Running RE inference")

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
            if type == "problem":
                RE_unprocessed.append(problem_prompt().format(instance))
            if type == "treatment":
                RE_unprocessed.append(treatment_prompt().format(instance))
            if type == "test":
                RE_unprocessed.append(test_prompt().format(instance))
            if type == "drug":
                RE_unprocessed.append(drug_prompt().format(instance))

    prompts_list = batch_list(RE_unprocessed, settings.PROMPT_BATCH_SIZE)
    RE_outputs = []

    for i, prompt_list in enumerate(prompts_list):
        # Generate the output
        output = llm.generate(prompt_list, sampling_params, use_tqdm=False)
        RE_outputs += output

    with open("./output/RE/MTSample.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["data_idx", "Type", "RE_input", "RE_output"])
        for idx, type, RE_input, RE_output in zip(
            data_idx, types, RE_unprocessed, RE_outputs
        ):
            writer.writerow([idx, type, RE_input, RE_output.outputs[0].text])
    print("RE inference done")
