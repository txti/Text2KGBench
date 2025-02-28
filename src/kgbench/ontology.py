import re
from pathlib import Path
from typing import Dict


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


def update_onto_list(json_data):
    try:
        # Extract path pattern for ontologies
        onto_pattern = json_data["path_patterns"]["onto"]

        # Extract base directory from the pattern (up to the last '/')
        base_dir = Path(onto_pattern.split("$$onto$$")[0]).resolve()

        # Extract filenames from the directory
        filenames = []
        pattern = r'^(l?\d+_\w+)_ontology\.json$'
        for f in base_dir.glob('*.json'):
            if f.is_file() and re.match(pattern, f.name):
                # Extract number (with optional 'l' prefix) and category name
                match = re.search(r'^(l?\d+_\w+)_ontology', f.stem)
                if match:
                    filenames.append(match.group(1))

        filenames.sort()

        if filenames:
            json_data['onto_list'] = filenames
            return json_data
        else:
            print("No matching files found in directory")
            return None

    except FileNotFoundError:
        print(f"Directory not found: {base_dir}")
        return None
    except KeyError as e:
        print(f"Missing key in configuration: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None
