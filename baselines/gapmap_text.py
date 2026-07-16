import json
import os
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

def run_gapmap_baseline(chunks: list[dict]) -> list[dict]:
    """
    Evaluate text-only TABI inference directly on paragraphs (B3: GAPMAP baseline).
    """
    all_gaps = []
    
    print(f"[*] Running GAPMAP Text-Only Baseline (B3) on {len(chunks)} chunks...")
    
    # Run on a subset or all chunks depending on limit
    for idx, c in enumerate(chunks):
        title = c["title"]
        text = c["text"]
        print(f"[*] Processing chunk {idx+1}/{len(chunks)} of '{title[:40]}...'")
        
        prompt = f"""
        Act as a Senior Software Engineering Researcher. Read the following scientific paper paragraph and identify an implicit research gap based on its premises using the Toulmin-Abductive Bucketed Inference (TABI) framework.
        
        Text Chunk:
        "{text}"
        
        Output strictly as a JSON object with no markdown formatting (like ```json), backticks, or explanation.
        JSON Schema:
        {{
          "Grounds": "The explicit text sentences/premises from the chunk that imply a missing piece of evidence.",
          "Claim": "The implied research gap statement (the inferred conclusion).",
          "Warrant": "The technical explanation reasoning why this gap is important for the Software Engineering field.",
          "Bucket": "Classify as 'more_probable' or 'least_probable' feasibility."
        }}
        """
        
        try:
            content = call_llm(prompt, temperature=0.1)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
                    
            gap_data = json.loads(content)
            gap_data["source_paper"] = title
            gap_data["chunk_index"] = c.get("chunk_index", 0)
            gap_data["type"] = "gapmap_text"
            all_gaps.append(gap_data)
        except Exception as e:
            print(f"[!] Error running GAPMAP baseline for chunk {idx}: {e}")
            
    return all_gaps

def main():
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print(f"[!] File {chunks_path} not found. Please run fetch_papers first.")
        return
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    gaps = run_gapmap_baseline(chunks)
    
    output_path = os.path.join(config.GAPS_DIR, "baseline_gapmap.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B3 baseline gaps to {output_path}")

if __name__ == "__main__":
    main()
