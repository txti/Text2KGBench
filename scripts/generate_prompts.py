import argparse

from kgbench.ontology import get_file_paths
from kgbench.prompts import (
    get_similar_sentences,
    get_train_sentence,
    prepare_prompt,
    write_prompts,
)
from kgbench.utils.io import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to prompt generation config file",
    )
    args = parser.parse_args()

    # load the prompt generation configuration with details of files needed for prompt generation
    # The config has a list of ontologies and patterns for generating for paths for the ontology file, test sentences
    # file, training sentences file, test-train sentence similarity file, and the output file.
    gen_config = read_json(args.config_path)
    file_paths = get_file_paths(gen_config)

    # for each of the ontology, we load the corresponding files and generate the prompts for each test sentence
    for onto in gen_config["onto_list"]:
        print(f"\nProcessing ontology: {onto}")
        paths = file_paths[onto]

        # load the ontology which has the concepts and relations (with domain / range constraints)
        ontology = read_json(paths["ontology_file"])

        # In the prompt, for each test sentence, we are using the most similar train sentence as the example for
        # in-context learning. This files contains pre-calculated similarities for each test sentence using a T5XXL
        # SBERT model.
        test_train_similarity = read_json(paths["test_train_similarity_file"])
        # load the list of train sentences. We use the train sentences with aligned triples to find the examples to
        # include in the prompt.
        train_sentences = read_json(paths["train_file"])
        # load the list of test sentences for which we need to generate the prompts.
        test_sentences = read_json(paths["test_file"])

        if not all([test_train_similarity, train_sentences, test_sentences, ontology]):
            print(f"Skipping {onto} due to missing files")
            continue

        try:
            prompts_json = []

            # iterate through all test sentences while generating prompts
            for test_sentence in test_sentences:
                test_sentence_id = test_sentence['id']
                # test sentence for which the prompt to be generated
                test_text = test_sentence['sent']

                # get the similar train sentences for the test sentence
                similar_sents = get_similar_sentences(test_sentence_id, test_train_similarity)
                if not similar_sents:
                    continue
                # we retrieve by default the first similar sentence from the list of similar sentences
                # we get the train sentence from the train sentences list and from there we process each field sub_label, obj_label, rel_label
                train_sent = get_train_sentence(similar_sents[0], train_sentences)

                # prompt generation logic
                prompt = prepare_prompt(ontology, test_text, train_sent)
                prompt_data = {"id": test_sentence_id, "prompt": prompt}
                prompts_json.append(prompt_data)

            write_prompts(prompts_json, paths["prompt_file"])
            print(f"Generated {len(prompts_json)} prompts for {onto}")

        except Exception as e:
            print(f"Error processing ontology {onto}: {str(e)}")
            continue
