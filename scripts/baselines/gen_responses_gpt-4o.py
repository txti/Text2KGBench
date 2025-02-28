import argparse
import json
import os
import sys
import time
from typing import Dict, List

from openai import OpenAI

from kgbench.utils.io import read_json


def parse_triples(response_text: str) -> List[List[str]]:
    """Parse the response text to extract triples"""
    triples = []
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '(' in line and ')' in line:
            try:
                relation_part, args_part = line.split('(', 1)
                args_part = args_part.rstrip(')')
                args = args_part.split(',', 1)
                if len(args) == 2:
                    sub = args[0].strip()
                    obj = args[1].strip()
                    rel = relation_part.strip()
                    triples.append([sub, rel, obj])
            except Exception as e:
                print(f"Error parsing line: {line}, Error: {str(e)}")
    return triples
def get_file_paths(config: dict) -> Dict[str, dict]:
    """Generate file paths from config"""
    try:
        # Extract dataset and type  from path patterns
        prompt_pattern = config['path_patterns']['prompt']

        # Extract base dataset path (e.g., "./data/dbpedia_webnlg" or "./data/wikidata_tekgen")
        base_path = prompt_pattern.split('/baselines')[0]

        # Construct response directory path
        response_base = f"{base_path}/baselines/OpenAI-GPT-4o"
        response_dir = f"{response_base}/llm_responses"

        file_paths = {}
        for onto in config['onto_list']:
            file_paths[onto] = {
                'prompt_file': prompt_pattern.replace('$$onto$$', onto),
                'response_dir': response_dir
            }
        return file_paths
    except Exception as e:
        print(f"Error generating file paths: {str(e)}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='Path to prompt generation config file')
    parser.add_argument('--api_key', required=False, help='OpenAI API key')
    args = parser.parse_args()

    config = read_json(args.config_path)
    if not config:
        sys.exit(1)

    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OpenAI API key not provided. Please set it as an argument or environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    file_paths = get_file_paths(config)
    if not file_paths:
        sys.exit(1)

    # Get response directory from the first onto (all have same response_dir)
    output_dir = next(iter(file_paths.values()))['response_dir']
    os.makedirs(output_dir, exist_ok=True)

    for onto in config['onto_list']:
        print(f"\nProcessing ontology: {onto}")
        paths = file_paths[onto]
        prompt_file = paths['prompt_file']

        if not os.path.exists(prompt_file):
            print(f"Prompt file {prompt_file} not found. Skipping ontology {onto}.")
            continue

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading prompt file {prompt_file}: {str(e)}")
            continue

        responses = []
        for prompt_data in prompts:
            prompt_id = prompt_data.get('id')
            prompt_text = prompt_data.get('prompt')
            if not prompt_id or not prompt_text:
                continue

            print(f"Processing prompt {prompt_id}")

            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=250,
                    temperature=0
                )
                end_time = time.time()
                print(f"Query OpenAI loading time: {end_time - start_time:.2f} seconds")

                response_text = response.choices[0].message.content
                triples = parse_triples(response_text)

                response_data = {
                    'id': prompt_id,
                    'response': response_text,
                    'triples': triples
                }

                responses.append(response_data)

            except Exception as e:
                print(f"Error processing prompt {prompt_id}: {str(e)}")
                continue

        output_file = os.path.join(output_dir, f'ont_{onto}_responses.jsonl')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for response_data in responses:
                    json_line = json.dumps(response_data, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"Successfully wrote responses to {output_file}")
        except Exception as e:
            print(f"Error writing responses: {str(e)}")
            continue
