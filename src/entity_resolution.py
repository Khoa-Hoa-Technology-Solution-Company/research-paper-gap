import json
import os
from src import config

def resolve_entities(triples: list[dict]) -> tuple[list[dict], dict]:
    """
    Perform Entity Resolution using a two-pass approach:
    1. Lexical Matching: Fuzzy string matching using token sort ratio.
    2. Semantic Matching: Cosine similarity using Sentence-BERT embeddings.
    """
    # 1. Collect all unique entities and their types
    entity_types = {} # entity_name -> type
    for t in triples:
        entity_types[t["subject"]] = t["subject_type"]
        entity_types[t["object"]] = t["object_type"]
        
    unique_names = list(entity_types.keys())
    print(f"[*] Total unique entities found: {len(unique_names)}")
    
    # Mapping to store target canonical names: entity -> canonical_name
    mapping = {name: name for name in unique_names}
    
    # --- Pass 1: Fuzzy Lexical Matching ---
    try:
        from rapidfuzz import fuzz
        print("[*] Pass 1: Running Fuzzy Lexical Matching...")
        for i in range(len(unique_names)):
            name_a = unique_names[i]
            type_a = entity_types[name_a]
            
            for j in range(i + 1, len(unique_names)):
                name_b = unique_names[j]
                type_b = entity_types[name_b]
                
                # Only merge entities of the same type
                if type_a != type_b:
                    continue
                    
                # Calculate fuzzy similarity
                score = fuzz.token_sort_ratio(name_a.lower(), name_b.lower())
                if score >= config.FUZZY_MATCH_THRESHOLD:
                    # Merge to the shorter name (often the cleaner canonical form) 
                    # or keep consistency by merging b to the root of a
                    root_a = find_root(mapping, name_a)
                    root_b = find_root(mapping, name_b)
                    if root_a != root_b:
                        if len(root_a) <= len(root_b):
                            mapping[root_b] = root_a
                        else:
                            mapping[root_a] = root_b
    except ImportError:
        print("[!] rapidfuzz not installed. Skipping fuzzy matching pass.")
        
    # --- Pass 2: Semantic Matching using Sentence-BERT ---
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        print("[*] Pass 2: Running Semantic Embedding Matching using Sentence-BERT...")
        # Use a lightweight, high-performance model (~80MB)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # We only match entities that are not already merged in Pass 1
        current_canonical_set = set(find_root(mapping, name) for name in unique_names)
        canonical_list = list(current_canonical_set)
        
        if len(canonical_list) > 1:
            embeddings = model.encode(canonical_list, show_progress_bar=False)
            
            # Compute Cosine Similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            for i in range(len(canonical_list)):
                name_a = canonical_list[i]
                type_a = entity_types[name_a]
                
                for j in range(i + 1, len(canonical_list)):
                    name_b = canonical_list[j]
                    type_b = entity_types[name_b]
                    
                    if type_a != type_b:
                        continue
                        
                    sim = similarity_matrix[i, j]
                    if sim >= config.COSINE_SIMILARITY_THRESHOLD:
                        root_a = find_root(mapping, name_a)
                        root_b = find_root(mapping, name_b)
                        if root_a != root_b:
                            if len(root_a) <= len(root_b):
                                mapping[root_b] = root_a
                            else:
                                mapping[root_a] = root_b
    except ImportError:
        print("[!] sentence-transformers not installed. Skipping semantic matching pass.")
    except Exception as e:
        print(f"[!] Error during semantic matching: {e}. Skipping pass.")

    # Flatten mapping to direct canonical values
    final_mapping = {}
    for name in unique_names:
        final_mapping[name] = find_root(mapping, name)
        
    # Apply mapping to triples
    resolved_triples = []
    for t in triples:
        sub = t["subject"]
        obj = t["object"]
        canonical_sub = final_mapping[sub]
        canonical_obj = final_mapping[obj]
        
        # Avoid self-loops after merge
        if canonical_sub == canonical_obj:
            continue
            
        resolved_triples.append({
            "subject": canonical_sub,
            "subject_type": t["subject_type"],
            "relation": t["relation"],
            "object": canonical_obj,
            "object_type": t["object_type"],
            "confidence": t["confidence"],
            "evidence_quote": t["evidence_quote"],
            "year": t["year"]
        })
        
    # Group mapping entries for output clarity (canonical -> list of aliases)
    reversed_mapping = {}
    for alias, canonical in final_mapping.items():
        if alias != canonical:
            reversed_mapping.setdefault(canonical, []).append(alias)
            
    print(f"[+] Entity resolution complete. Merged {len(unique_names) - len(set(final_mapping.values()))} entities.")
    return resolved_triples, reversed_mapping

def find_root(mapping: dict, name: str) -> str:
    """Helper to traverse mapping path to canonical root."""
    path = []
    curr = name
    while mapping[curr] != curr:
        path.append(curr)
        curr = mapping[curr]
    # Path compression
    for p in path:
        mapping[p] = curr
    return curr

def run_resolution():
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    if not os.path.exists(raw_path):
        print(f"[!] File {raw_path} not found. Please run extract_triples first.")
        return
        
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_triples = json.load(f)
        
    resolved_triples, entity_mapping = resolve_entities(raw_triples)
    
    # Save results
    mapping_path = os.path.join(config.TRIPLES_DIR, "entity_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(entity_mapping, f, ensure_ascii=False, indent=2)
        
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(resolved_triples, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved resolved triples to {resolved_path}")
    print(f"[+] Saved mapping rules to {mapping_path}")

if __name__ == "__main__":
    run_resolution()
