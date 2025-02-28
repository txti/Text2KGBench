import json
import os
from typing import Dict, List, Optional


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


def get_concept_label(ontology: dict, concept_qid: str) -> str:
    """
    Retrieve the label for a given concept QID from the ontology.
    :param ontology: The ontology dictionary containing concepts.
    :param concept_qid: The QID of the concept.
    :return: The label of the concept or an empty string if not found.
    """
    for concept in ontology.get("concepts", []):
        if concept.get("qid") == concept_qid:
            return concept.get("label", "")
    return ""


def get_ontology_concepts(ontology: dict) -> str:
    """Extract concepts from ontology"""
    try:
        if isinstance(ontology, dict) and "concepts" in ontology:
            concepts = []
            for concept in ontology["concepts"]:
                if isinstance(concept, dict) and "label" in concept:
                    concepts.append(concept["label"])
            return ", ".join(concepts)
        return ""
    except Exception as e:
        print(f"Error getting ontology concepts: {str(e)}")
        return ""


def get_ontology_relations(ontology: dict) -> str:
    """
    Extract and format ontology relations.
    :param ontology: The ontology dictionary containing relations and concepts.
    :return: A comma-separated string of formatted relations.
    """
    try:
        if not isinstance(ontology, dict) or "relations" not in ontology:
            return ""

        relations = []
        for relation in ontology["relations"]:
            label = relation.get("label", "").replace(
                " ", "_"
            )  # Replace spaces with underscores

            domain_qid = relation.get("domain", "")
            range_qid = relation.get("range", "")

            # Retrieve labels using QIDs
            domain = get_concept_label(ontology, domain_qid)
            range_ = get_concept_label(ontology, range_qid)

            # Skip if any part is missing
            if not label or not domain or not range_:
                continue

            relations.append(f"{label}({domain},{range_})")

        return ", ".join(relations)
    except Exception as e:
        print(f"Error getting ontology relations: {str(e)}")
        return ""


def get_example_prompt(train_sent: dict) -> str:
    """Generate example prompt with proper triple formatting"""
    try:
        example_prompt = "\n\nExample Sentence: "

        if isinstance(train_sent, dict):
            # Add the sentence
            sentence = train_sent.get("sent", "")
            example_prompt += sentence
            example_prompt += "\nExample Output:"

            # Format triples
            triples = train_sent.get("triples", [])
            if triples:
                for triple in triples:
                    rel = triple.get("rel", "").replace("_", " ")
                    sub = triple.get("sub", "").replace("_", " ")
                    obj = triple.get("obj", "").replace("_", " ")
                    example_prompt += f"\n{rel}({sub}, {obj})"

        return example_prompt
    except Exception as e:
        print(f"Error processing train sentence: {str(e)}")
        return "\n\nExample Sentence: [Error processing example]"


def get_test_prompt(test_sentence: str) -> str:
    """Generate test prompt"""
    return f"\n\nTest Sentence: {test_sentence}\nOutput:"


def get_similar_sentences(test_id: str, similarity_dict: dict) -> List[str]:
    """Get similar sentences with improved error handling"""
    try:
        if isinstance(similarity_dict, dict):
            return similarity_dict.get(test_id, [])
        elif isinstance(similarity_dict, list):
            for item in similarity_dict:
                if item.get("test_id") == test_id:
                    return item.get("similar_sentences", [])
        return []
    except Exception:
        return []


def get_train_sentence(
    simil_sent_id: str, train_sentences: List[dict]
) -> Optional[dict]:
    """Get training sentence with proper triple extraction"""
    try:
        if isinstance(train_sentences, list):
            for sent in train_sentences:
                if sent.get("id") == simil_sent_id:
                    # Get the sentence text
                    sentence = sent.get("text", sent.get("sent", ""))

                    # Get triples from the training data
                    triples = []
                    if "triples" in sent:
                        triples = sent["triples"]
                    elif all(k in sent for k in ["relation", "subject", "object"]):
                        triples = [
                            {
                                "rel": sent["relation"],
                                "sub": sent["subject"],
                                "obj": sent["object"],
                            }
                        ]

                    return {"sent": sentence, "triples": triples}
        return None
    except Exception as e:
        print(f"Error getting train sentence: {str(e)}")
        return None


def prepare_prompt(
    ontology: dict, test_sentence: str, train_sent: dict
) -> Optional[str]:
    """Prepare prompt with proper formatting"""
    try:
        if not all([ontology, test_sentence, train_sent]):
            return None

        prompt = (
            "Given the following ontology and sentences, please extract the triples from the sentence according "
            "to the relations in the ontology. In the output, only include the triples in the given output format."
            "\n\nCONTEXT:\n\n"
        )

        # Add concepts and relations
        concepts = get_ontology_concepts(ontology)
        relations = get_ontology_relations(ontology)

        prompt += f"Ontology Concepts: {concepts}\n"
        prompt += f"Ontology Relations: {relations}"

        # Add example with triples
        prompt += get_example_prompt(train_sent)

        # Add test sentence
        prompt += f"\n\nTest Sentence: {test_sentence}\nOutput:"

        return prompt
    except Exception:
        return None


def write_prompts(prompts_json: List[dict], prompt_file: str) -> None:
    """Write prompts to JSONL file with proper formatting"""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(prompt_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(prompt_file, "w", encoding="utf-8") as f:
            for prompt_data in prompts_json:
                # Ensure consistent formatting
                formatted_prompt = {
                    "id": prompt_data["id"],
                    "prompt": prompt_data["prompt"].replace("\n        ", "\n").strip(),
                }
                json_line = json.dumps(formatted_prompt, ensure_ascii=False)
                f.write(json_line + "\n")
        print(f"Successfully wrote prompts to {prompt_file}")
    except Exception as e:
        print(f"Error writing prompts: {str(e)}")


def get_file_paths(config: dict) -> Dict[str, dict]:
    """Generate file paths from config based on path_patterns"""
    try:
        onto_list = config["onto_list"]
        path_patterns = config["path_patterns"]

        file_paths = {}
        for onto in onto_list:
            file_paths[onto] = {
                "test_train_similarity_file": path_patterns["sent_sim"].replace(
                    "$$onto$$", onto
                ),
                "train_file": path_patterns["train"].replace("$$onto$$", onto),
                "test_file": path_patterns["test"].replace("$$onto$$", onto),
                "ontology_file": path_patterns["onto"].replace("$$onto$$", onto),
                "prompt_file": path_patterns["prompt"].replace("$$onto$$", onto),
            }
        return file_paths
    except Exception as e:
        print(f"Error generating file paths: {str(e)}")
        return {}
