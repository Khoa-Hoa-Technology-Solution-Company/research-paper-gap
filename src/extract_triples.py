import json
import os
import argparse
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

def extract_triples_from_chunk(chunk_text: str, year: int) -> list[dict]:
    """
    Call LLM to extract triples from a single chunk.
    """
    client = get_llm_client()
    model = get_llm_model()
    
    prompt = f"""
    You are an expert in Software Engineering research. Read the following text chunk and extract typed triples of the form <Subject, Relation, Object>.
    
    CONSTRAINTS:
    - Subject and Object entity types MUST belong to: {config.ENTITY_TYPES}
    - Relation types MUST belong to: {config.RELATION_TYPES}
    - Assign a Confidence Score (0.0 to 1.0) for each triple.
    - Provide a short "evidence_quote" from the text showing where the relation is stated.
    
    Text Chunk:
    "{chunk_text}"
    
    Output strictly as a JSON array of objects, with no markdown formatting (like ```json), backticks, or explanation.
    JSON Schema:
    [
      {{
        "subject": "Name of subject entity (canonical, noun phrase)",
        "subject_type": "One of: {config.ENTITY_TYPES}",
        "relation": "One of: {config.RELATION_TYPES}",
        "object": "Name of object entity (canonical, noun phrase)",
        "object_type": "One of: {config.ENTITY_TYPES}",
        "confidence": 0.95,
        "evidence_quote": "Exact quote from text"
      }}
    ]
    """
    
    try:
        content = call_llm(prompt, temperature=0.1)
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
                
        triples = json.loads(content)
        valid_triples = []
        
        for t in triples:
            sub = t.get("subject", "").strip()
            sub_type = t.get("subject_type", "").strip().upper()
            rel = t.get("relation", "").strip().upper()
            obj = t.get("object", "").strip()
            obj_type = t.get("object_type", "").strip().upper()
            conf = float(t.get("confidence", 0.0))
            quote = t.get("evidence_quote", "").strip()
            
            # Validation
            if not sub or not obj or not rel:
                continue
            if sub_type not in config.ENTITY_TYPES:
                continue
            if obj_type not in config.ENTITY_TYPES:
                continue
            if rel not in config.RELATION_TYPES:
                continue
            if conf < config.TRIPLE_CONFIDENCE_THRESHOLD:
                continue
                
            valid_triples.append({
                "subject": sub,
                "subject_type": sub_type,
                "relation": rel,
                "object": obj,
                "object_type": obj_type,
                "confidence": conf,
                "evidence_quote": quote,
                "year": year
            })
            
        return valid_triples
    except Exception as e:
        print(f"[!] Error extracting triples: {e}")
        return []

def run_extraction():
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print(f"[!] File {chunks_path} not found. Please run fetch_papers first.")
        return
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"[*] Processing {len(chunks)} text chunks for triple extraction...")
    all_triples = []
    
    for i, c in enumerate(chunks):
        print(f"[*] Extracting triples from chunk {i+1}/{len(chunks)} of '{c['title'][:40]}...'")
        triples = extract_triples_from_chunk(c["text"], c["year"])
        print(f"[+] Extracted {len(triples)} valid triples from chunk.")
        all_triples.extend(triples)
        
    output_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {len(all_triples)} raw triples to {output_path}")

if __name__ == "__main__":
    run_extraction()
