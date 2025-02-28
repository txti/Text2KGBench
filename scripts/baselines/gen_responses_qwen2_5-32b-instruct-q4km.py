import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from kgbench.utils.io import read_json


def download_model() -> Optional[str]:
    """
    Downloads the model from Hugging Face Hub if not already present.

    Returns:
        The path to the downloaded model or None if download fails.
    """
    models_dir = "/data/johnsonv/models"
    model_name = "qwen2.5-32b-instruct-q4_k_m.gguf"
    model_path = os.path.join(models_dir, model_name)

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path

    print("Downloading model... This may take a while.")
    try:
        model_path = hf_hub_download(
            repo_id="TheRains/Qwen2.5-32B-Instruct-Q4_K_M-GGUF",
            filename=model_name,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None


def initialize_model() -> Optional[Llama]:
    """
    Initializes the model using llama_cpp.

    Returns:
        An instance of the Llama model or None if initialization fails.
    """
    model_path = download_model()
    if not model_path:
        return None

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,        # Maximum layers for RTX 3090
            n_ctx=2048,             # Keep default context size
            n_threads=24,           # Match your CPU core count
            offload_kqv=True,       # Beneficial for large models
            use_mlock=True,
			verbose=False
        )
        print("Model initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None

def parse_triples(response_text: str) -> List[List[str]]:
    """Parse the response text to extract triples."""
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
    """Generate file paths from config."""
    try:
        # Extract dataset and type from path patterns
        prompt_pattern = config['path_patterns']['prompt']


        # Extract base dataset path (e.g., "./data/dbpedia_webnlg" or "./data/wikidata_tekgen")
        base_path = prompt_pattern.split('/baselines')[0]

        # Construct response directory path
        response_base = f"{base_path}/baselines/Qwen2_5-32B-Instruct-Q4KM"
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

def generate_response(llm: Llama, prompt: str) -> Optional[str]:
    """
    Generates a response from the model given a prompt using chat completion.

    Args:
        llm: The initialized model.
        prompt: The input prompt string.

    Returns:
        The generated response string or None if generation fails.
    """
    try:
        start_time = time.time()
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        end_time = time.time()

        # Extract content from chat completion response
        response_text = response['choices'][0]['message']['content']

        print(f"Response generated in {end_time - start_time:.2f} seconds.")
        return response_text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process prompts using model and store responses.")
    parser.add_argument('--config_path', required=True, help='Path to prompt generation config file')
    args = parser.parse_args()

    config = read_json(args.config_path)
    if not config:
        sys.exit(1)

    llm = initialize_model()
    if not llm:
        print("Failed to initialize model.")
        sys.exit(1)

    file_paths = get_file_paths(config)
    if not file_paths:
        sys.exit(1)

    # Get response directory from the first ontology (all have same response_dir)
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

            response_text = generate_response(llm, prompt_text)
            if response_text:
                triples = parse_triples(response_text)

                response_data = {
                    'id': prompt_id,
                    'response': response_text,
                    'triples': triples
                }

                responses.append(response_data)
                print(f"Prompt {prompt_id} processed successfully.")
            else:
                print(f"Failed to generate response for prompt {prompt_id}.")

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

if __name__ == "__main__":
    main()
