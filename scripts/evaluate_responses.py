import argparse
import os
import sys
from typing import Dict

from kgbench.utils.eval import (
    calculate_precision_recall_f1,
    convert_to_dict,
    get_ontology_conformance,
    get_subject_object_hallucinations,
    normalize_triple,
)
from kgbench.utils.io import append_jsonl, read_json, save_jsonl


def load_config(config_path: str) -> Dict:
    """
    Load the evaluation configuration file
    :param config_path: path to the evaluation configuration file
    :return: a new config object where paths to the files are resolved based on the path patterns
    """
    raw_config = read_json(config_path)
    onto_list = raw_config["onto_list"]
    path_patterns = raw_config["path_patterns"]
    new_config = dict()
    expanded_onto_list = list()
    for onto in onto_list:
        onto_data = dict()
        onto_data["id"] = onto
        for key in path_patterns:
            onto_data[key] = path_patterns[key].replace("$$onto$$", onto)
        expanded_onto_list.append(onto_data)
    new_config["onto_list"] = expanded_onto_list
    new_config["avg_out_file"] = raw_config["avg_out_file"]
    return new_config


def main():

    parser = argparse.ArgumentParser()
    # please have a look at src/evaluation/config for examples of evaluation configs.
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    # load the files needed for evaluation from a user provided config file, it contains the system generated
    # output, the ground truth files, path to ontology file, and the path to store the evaluation output.
    config_path = args.config_path

    if not os.path.exists(config_path):
        print(f"Evaluation config file is not found in path: {config_path}")
    eval_inputs = load_config(config_path)

    # initialize the global variables for the total evaluation metrics
    (
        global_p,
        global_r,
        global_f1,
        global_onto_conf,
        global_rel_halluc,
        global_sub_halluc,
        global_obj_halluc,
    ) = 0, 0, 0, 0, 0, 0, 0

    # evaluate the output of each of the ontologies
    for onto in eval_inputs["onto_list"]:
        # initialize the local variables for the evaluation metrics for each ontology
        (
            t_p,
            t_r,
            t_f1,
            t_onto_conf,
            t_rel_halluc,
            t_sub_halluc,
            t_obj_halluc,
        ) = 0, 0, 0, 0, 0, 0, 0

        # initialize the local variables for the evaluation metrics for each ontology for the selected triples
        (
            sel_t_p,
            sel_t_r,
            sel_t_f1,
            sel_t_onto_conf,
            sel_t_rel_halluc,
            sel_t_sub_halluc,
            sel_t_obj_halluc,
        ) = 0, 0, 0, 0, 0, 0, 0
        eval_metrics_list = list()
        onto_id = onto["id"]
        system_output = convert_to_dict(read_json(onto["sys"]))
        ground_truth = convert_to_dict(read_json(onto["gt"]))
        ontology = read_json(onto["onto"])
        if "selected_ids" in onto:
            selected_ids = read_json(onto["selected_ids"], is_json=False)
        else:
            selected_ids = []

        # iterate through each element in the ground truth and evaluate the system output
        for sent_id in list(ground_truth.keys()):
            # collect the ground truth triples
            gt_triples = [
                [tr["sub"], tr["rel"], tr["obj"]]
                for tr in ground_truth[sent_id]["triples"]
            ]
            sentence = ground_truth[sent_id]["sent"]

            # check if system output as an entry for this sentence
            if sent_id in system_output:
                system_triples = system_output[sent_id]["triples"]

                # collect the set of relations in ground truth triples, spaces are converted to "_" to make them
                # comparable with system triples
                gt_relations = {tr[1].replace(" ", "_") for tr in gt_triples}

                # filter out any triples in system output that does not match with ground truth relations
                filtered_system_triples = [
                    tr for tr in system_triples if tr[1] in gt_relations
                ]

                # create a normalized string from subject, relation, object of each triple for comparison
                normalized_system_triples = {
                    normalize_triple(tr[0], tr[1], tr[2])
                    for tr in filtered_system_triples
                }
                normalized_gt_triples = {
                    normalize_triple(tr[0], tr[1], tr[2])
                    for tr in gt_triples
                }

                # compare the system output triples with ground truth triples and calculate precision, recall, f1
                precision, recall, f1 = calculate_precision_recall_f1(
                    normalized_gt_triples, normalized_system_triples
                )

                # calculate ontology conformance and relation hallucination
                ont_conformance, rel_hallucination = get_ontology_conformance(
                    ontology, system_triples
                )

                # calculate subject and object hallucination
                subj_hallucination, obj_hallucination = (
                    get_subject_object_hallucinations(
                        ontology, sentence, system_triples
                    )
                )
                if (
                    f1 < 1
                    and len(filtered_system_triples) > 0
                    and subj_hallucination == 0
                    and obj_hallucination == 0
                ):
                    print(
                        f"sent: {sentence}\nf1: {f1}\nsys:{filtered_system_triples}\nground:{gt_triples}\n\n"
                    )

                eval_metrics = {
                    "id": sent_id,
                    "precision": f"{precision:.2f}",
                    "recall": f"{recall:.2f}",
                    "f1": f"{f1:.2f}",
                    "onto_conf": f"{ont_conformance:.2f}",
                    "rel_halluc": f"{rel_hallucination:.2f}",
                    "sub_halluc": f"{subj_hallucination:.2f}",
                    "obj_halluc": f"{obj_hallucination:.2f}",
                    "llm_triples": system_triples,
                    "filtered_llm_triples": filtered_system_triples,
                    "gt_triples": gt_triples,
                    "sent": sentence,
                }
                eval_metrics_list.append(eval_metrics)

                # aggregate precision, recall, f1 for later averaging
                t_p += precision
                t_r += recall
                t_f1 += f1
                t_onto_conf += ont_conformance
                t_rel_halluc += rel_hallucination
                t_sub_halluc += subj_hallucination
                t_obj_halluc += obj_hallucination

                # aggregate precision, recall, f1 for later averaging for selected ids
                if sent_id in selected_ids:
                    sel_t_p += precision
                    sel_t_r += recall
                    sel_t_f1 += f1
                    sel_t_onto_conf += ont_conformance
                    sel_t_rel_halluc += rel_hallucination
                    sel_t_sub_halluc += subj_hallucination
                    sel_t_obj_halluc += obj_hallucination

        save_jsonl(eval_metrics_list, onto["output"])
        total_test_cases = len(ground_truth)
        total_selected_test_cases = len(selected_ids)
        # average metrics calculate the average of evaluate metrics for all test cases in a given ontology
        average_metrics = {
            "onto": onto_id,
            "type": "all_test_cases",
            "avg_precision": f"{t_p / total_test_cases:.2f}",
            "avg_recall": f"{t_r / total_test_cases:.2f}",
            "avg_f1": f"{t_f1 / total_test_cases:.2f}",
            "avg_onto_conf": f"{t_onto_conf / total_test_cases:.2f}",
            "avg_sub_halluc": f"{t_sub_halluc / total_test_cases:.2f}",
            "avg_rel_halluc": f"{t_rel_halluc / total_test_cases:.2f}",
            "avg_obj_halluc": f"{t_obj_halluc / total_test_cases:.2f}",
        }
        append_jsonl(average_metrics, eval_inputs["avg_out_file"])
        global_p += t_p / total_test_cases
        global_r += t_r / total_test_cases
        global_f1 += t_f1 / total_test_cases
        global_onto_conf += t_onto_conf / total_test_cases
        global_sub_halluc += t_sub_halluc / total_test_cases
        global_rel_halluc += t_rel_halluc / total_test_cases
        global_obj_halluc += t_obj_halluc / total_test_cases
        # in some cases, we have a subset of selected test cases for which we report the average numbers separately
        if total_selected_test_cases > 0:
            selected_average_metrics = {
                "onto": onto_id,
                "type": "selected_test_cases",
                "avg_precision": f"{sel_t_p / total_selected_test_cases:.2f}",
                "avg_recall": f"{sel_t_r / total_selected_test_cases:.2f}",
                "avg_f1": f"{sel_t_f1 / total_selected_test_cases:.2f}",
                "avg_onto_conf": f"{sel_t_onto_conf / total_selected_test_cases:.2f}",
                "avg_sub_halluc": f"{sel_t_sub_halluc / total_selected_test_cases:.2f}",
                "avg_rel_halluc": f"{sel_t_rel_halluc / total_selected_test_cases:.2f}",
                "avg_obj_halluc": f"{sel_t_obj_halluc / total_selected_test_cases:.2f}",
            }
            append_jsonl(selected_average_metrics, eval_inputs["avg_out_file"])

    # global metrics calculate the average total metrics for all ontologies that are part of the evaluation
    num_ontologies = len(eval_inputs["onto_list"])
    global_metrics = {
        "id": "global",
        "type": "global",
        "avg_precision": f"{global_p / num_ontologies:.2f}",
        "avg_recall": f"{global_r / num_ontologies:.2f}",
        "avg_f1": f"{global_f1 / num_ontologies:.2f}",
        "avg_onto_conf": f"{global_onto_conf / num_ontologies:.2f}",
        "avg_sub_halluc": f"{global_sub_halluc / num_ontologies:.2f}",
        "avg_rel_halluc": f"{global_rel_halluc / num_ontologies:.2f}",
        "avg_obj_halluc": f"{global_obj_halluc / num_ontologies:.2f}",
        "onto_list": eval_inputs["onto_list"],
    }
    append_jsonl(global_metrics, eval_inputs["avg_out_file"])


if __name__ == "__main__":
    sys.exit(main())
