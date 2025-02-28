import argparse
import json
from pathlib import Path

from kgbench.prompts import parse_triples
from kgbench.utils.io import read_json
from kgbench.utils.llm import download_ollama_model, get_llm_response


def main():
    parser = argparse.ArgumentParser(
        description="Process prompts using model and store responses."
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to prompt generation config file",
    )
    args = parser.parse_args()
    config = read_json(args.config_path)

    # Model configuration
    model_tag = config["model_config"]["tag"]
    temperature = config["model_config"]["temperature"]
    prefix = config["model_config"]["provider"]
    base_url = config["model_config"]["base_url"]
    api_key = config["model_config"]["api_key"]

    # Download model
    download_ollama_model(model_tag)

    # Process prompts for each ontology
    for onto in config["onto_list"]:

        prompt_file = Path(config["path_patterns"]["prompt"].replace("$$onto$$", onto))

        if not prompt_file.exists():
            print(f"Prompt file {prompt_file} not found. Skipping ontology {onto}.")
            continue
        else:
            print(f"\nProcessing ontology: {onto}")

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading prompt file {prompt_file}: {str(e)}")
            continue

        responses = []
        for prompt_data in prompts:
            prompt_id = prompt_data.get("id")
            prompt_text = prompt_data.get("prompt")
            if not prompt_id or not prompt_text:
                continue

            print(f"Processing prompt {prompt_id}")

            response = get_llm_response(
                prompt_text, model_tag, temperature, prefix, base_url, api_key)
            if response:
                triples = parse_triples(response["response"])
                responses.append({
                    "id": prompt_id,
                    "response": response,
                    "triples": triples,
                })
                print(f"Prompt {prompt_id} processed successfully.")
            else:
                print(f"Failed to generate response for prompt {prompt_id}.")

        # Write responses to file
        output_file = Path(
            config["path_patterns"]["sys"].replace("$$onto$$", onto))

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write responses to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for response_data in responses:
                    json_line = json.dumps(response_data, ensure_ascii=False)
                    f.write(json_line + "\n")
            print(f"Successfully wrote responses to {output_file}")
        except Exception as e:
            print(f"Error writing responses: {str(e)}")
            continue


if __name__ == "__main__":
    main()
