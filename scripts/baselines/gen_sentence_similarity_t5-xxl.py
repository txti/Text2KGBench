import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from kgbench.utils.io import read_json


def load_sentences(file_path: str) -> Tuple[List[str], List[str], str]:
    """Load sentences, IDs, and compute file content hash from JSONL file"""
    sentences, ids = [], []
    content = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content.append(line)
                data = json.loads(line.strip())
                sentences.append(data['sent'])
                ids.append(data['id'])
        file_content = ''.join(content)
        file_hash = hashlib.sha1(file_content.encode('utf-8')).hexdigest()
        return sentences, ids, file_hash
    except Exception as e:
        print(f"Error loading sentences from {file_path}: {str(e)}")
        return [], [], ''

def compute_similarities(test_embeddings: torch.Tensor,
                       train_embeddings: torch.Tensor,
                       test_ids: List[str],
                       train_ids: List[str],
                       top_k: int) -> dict:
    """Compute similarities between test and train embeddings"""
    try:
        # Compute similarities and find top-k similar sentences
        similarity_results = {}
        print('Computing similarities and finding top similar sentences...')
        for idx, test_embedding in enumerate(tqdm(test_embeddings)):
            cosine_scores = util.cos_sim(test_embedding, train_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=top_k)
            similar_train_ids = [train_ids[i] for i in top_results[1]]
            similarity_results[test_ids[idx]] = similar_train_ids
        return similarity_results
    except Exception as e:
        print(f"Error computing similarities: {str(e)}")
        return {}

def process_ontology(onto: str,
                    config: dict,
                    model: SentenceTransformer,
                    model_name: str) -> None:
    """Process single ontology with cross-config cache support"""
    try:
        # Get file paths using patterns
        test_file = config['path_patterns']['test'].replace('$$onto$$', onto)
        train_file = config['path_patterns']['train'].replace('$$onto$$', onto)
        output_file = config['path_patterns']['sent_sim'].replace('$$onto$$', onto)

        print(f'\n{"-"*40}\nProcessing ontology: {onto}\n{"-"*40}')

        # Load test and train data
        test_sentences, test_ids, _ = load_sentences(test_file)
        train_sentences, train_ids, train_hash = load_sentences(train_file)

        if not test_sentences or not train_sentences:
            print(f"Skipping {onto} due to missing data")
            return

        # Set up cache paths
        output_dir = os.path.dirname(output_file)
        current_cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(current_cache_dir, exist_ok=True)

        # Check regular config cache directory if available
        potential_sources = [current_cache_dir]

        # Generate cache filename components
        safe_model_name = model_name.replace('/', '_')
        cache_filename = f"{onto}__{safe_model_name}__{train_hash}.pt"
        current_cache_path = os.path.join(current_cache_dir, cache_filename)

        # Look for existing caches in potential locations
        existing_caches = []
        for source_dir in potential_sources:
            candidate = os.path.join(source_dir, cache_filename)
            if os.path.exists(candidate):
                existing_caches.append(candidate)
                print(f"Found valid cache in: {candidate}")

        # Copy cache from regular config if available
        if existing_caches:
            best_source = existing_caches[0]
            if best_source != current_cache_path:
                print(f"\nCopying shared cache from: {best_source}")
                try:
                    shutil.copyfile(best_source, current_cache_path)
                    print(f"Successfully copied cache to: {current_cache_path}")
                except Exception as e:
                    print(f"Cache copy failed: {str(e)}")
                    existing_caches = []  # Force recompute if copy failed

        # Cleanup old caches in current directory
        cache_pattern = os.path.join(current_cache_dir, f"{onto}__{safe_model_name}__*.pt")
        existing_cache_files = glob.glob(cache_pattern)
        for file_path in existing_cache_files:
            if file_path != current_cache_path:
                try:
                    os.remove(file_path)
                    print(f"Removed outdated cache: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Failed to remove old cache: {str(e)}")

        # Load or compute embeddings
        train_embeddings = None
        if os.path.exists(current_cache_path):
            print(f"\nLoading cached embeddings from: {current_cache_path}")
            try:
                cache_data = torch.load(current_cache_path)
                train_embeddings = cache_data['train_embeddings']
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                train_embeddings = train_embeddings.to(device)
                print(f"Successfully loaded cached embeddings (shape: {train_embeddings.shape})")
            except Exception as e:
                print(f"Error loading cache file: {str(e)}")
                train_embeddings = None

        if train_embeddings is None:
            print('\nComputing fresh embeddings for train sentences...')
            train_embeddings = model.encode(train_sentences, convert_to_tensor=True, show_progress_bar=True)
            try:
                torch.save({'train_embeddings': train_embeddings}, current_cache_path)
                print(f"\nSaved new cache to: {current_cache_path}")
            except Exception as e:
                print(f"Error saving cache file: {str(e)}")

        # Compute test embeddings and similarities
        print('\nComputing embeddings for test sentences...')
        test_embeddings = model.encode(test_sentences, convert_to_tensor=True, show_progress_bar=True)

        # Compute similarities
        similarity_results = compute_similarities(
            test_embeddings=test_embeddings,
            train_embeddings=train_embeddings,
            test_ids=test_ids,
            train_ids=train_ids,
            top_k=config.get('top_k', 5)
        )

        if similarity_results:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(similarity_results, f, indent=4)
            print(f'\n{"-"*40}\nResults saved to {output_file}\n{"-"*40}')

    except Exception as e:
        print(f"Error processing ontology {onto}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = read_json(args.config_path)
    if not config:
        sys.exit(1)

    try:
        # Initialize model
        model_name = config.get('model_name', 'sentence-t5-xxl')
        model = SentenceTransformer(model_name)

        # Process each ontology
        for onto in config['onto_list']:
            process_ontology(onto, config, model, model_name)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
