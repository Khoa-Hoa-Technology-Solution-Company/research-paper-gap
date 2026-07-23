import json
import os
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

def get_top_k_similar_abstracts(target_idx: int, papers: list[dict], k: int = 3) -> list[str]:
    """
    Find top-k similar abstracts for the paper at target_idx.
    Uses Sentence-BERT if available, falls back to a simple keyword overlap method.
    """
    target_paper = papers[target_idx]
    target_text = target_paper["abstract"]
    
    other_papers = [(idx, p) for idx, p in enumerate(papers) if idx != target_idx]
    if not other_papers:
        return []
        
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Load lightweight embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        abstracts = [p["abstract"] for idx, p in other_papers]
        embeddings = model.encode([target_text] + abstracts, show_progress_bar=False)
        
        target_emb = embeddings[0]
        other_embs = embeddings[1:]
        
        # Compute Cosine Similarity
        target_norm = np.linalg.norm(target_emb)
        other_norms = np.linalg.norm(other_embs, axis=1)
        
        # Avoid division by zero
        if target_norm == 0:
            similarities = np.zeros(len(abstracts))
        else:
            similarities = np.dot(other_embs, target_emb) / (target_norm * other_norms + 1e-8)
            
        top_indices = np.argsort(similarities)[::-1][:k]
        return [abstracts[idx] for idx in top_indices]
        
    except Exception as e:
        print(f"[*] SentenceTransformer fallback for RAG: {e}. Running simple word overlap matching.")
        # Fallback: Count overlapping unique words (simple TF-IDF-like similarity)
        target_words = set(target_text.lower().split())
        similarities = []
        
        for idx, p in other_papers:
            other_words = set(p["abstract"].lower().split())
            intersection = target_words.intersection(other_words)
            sim = len(intersection) / (len(target_words) + len(other_words) - len(intersection) + 1e-8)
            similarities.append((sim, p["abstract"]))
            
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [abstract for sim, abstract in similarities[:k]]

def run_mulla_rag_baseline(papers: list[dict]) -> list[dict]:
    """
    Generate research gaps using Mulla et al. RAG baseline (B1).
    Retrieves 3 most similar abstracts and prompts LLM to output 4 structured fields.
    """
    all_gaps = []
    
    print(f"[*] Running Mulla RAG Baseline (B1) on {len(papers)} papers...")
    
    for idx, p in enumerate(papers):
        title = p["title"]
        print(f"[*] Processing paper {idx+1}/{len(papers)}: '{title[:40]}...'")
        
        # Retrieve context
        similar_abstracts = get_top_k_similar_abstracts(idx, papers, k=3)
        context = "\n\n".join([f"Reference Abstract {i+1}:\n{abs_text}" for i, abs_text in enumerate(similar_abstracts)])
        
        prompt = f"""
        Act as a Software Engineering researcher. Using the retrieved reference papers as context, 
        identify the research gaps left open by the target paper.
        
        Context (Similar Papers):
        {context}
        
        Target Paper:
        Title: {title}
        Abstract: {p['abstract']}
        
        Output strictly as a JSON object with the following four fields (no markdown formatting, backticks, or other text):
        {{
          "RESEARCH_GAPS": "Main research gaps identified in the target paper.",
          "RESEARCH_DIRECTION": "Suggested future research directions to solve these gaps.",
          "SOLUTION_APPROACH": "Possible technical solution approaches.",
          "REMAINING_GAPS": "Other secondary or remaining open problems."
        }}
        """
        
        try:
            content = call_llm(prompt, temperature=0.3)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
                    
            gap_data = json.loads(content)
            
            # Map B1 fields to standard evaluation columns for comparison
            # We map the main RESEARCH_GAPS to "Claim", and RESEARCH_DIRECTION/SOLUTION_APPROACH to "Warrant".
            all_gaps.append({
                "source_paper": title,
                "type": "mulla_rag",
                "Grounds": f"RAG Context (3 similar papers)",
                "Claim": gap_data.get("RESEARCH_GAPS", ""),
                "Warrant": f"Direction: {gap_data.get('RESEARCH_DIRECTION', '')}. Approach: {gap_data.get('SOLUTION_APPROACH', '')}",
                "Bucket": "near_term_feasible", # schema placeholder; not evidence of feasibility
                "raw_b1_data": gap_data
            })
        except Exception as e:
            print(f"[!] Error running Mulla RAG baseline for '{title[:30]}': {e}")
            
    return all_gaps

def main():
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    if not os.path.exists(screened_path):
        print(f"[!] File {screened_path} not found. Please run fetch_papers first.")
        return
        
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    gaps = run_mulla_rag_baseline(papers)
    
    output_path = os.path.join(config.GAPS_DIR, "baseline_mulla_rag.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B1 baseline gaps to {output_path}")

if __name__ == "__main__":
    main()
