import json
import os
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

def run_simple_llm_baseline(papers: list[dict]) -> list[dict]:
    """
    Generate research gaps directly from abstracts without any graph structure (B2: Simple LLM).
    """
    all_gaps = []
    
    print(f"[*] Running Simple LLM Baseline (B2) on {len(papers)} papers...")
    
    for idx, p in enumerate(papers):
        title = p["title"]
        abstract = p["abstract"]
        print(f"[*] Processing paper {idx+1}/{len(papers)}: '{title[:40]}...'")
        
        prompt = f"""
        Read the following scientific paper title and abstract and identify 3 research gaps it leaves open.
        
        Title: {title}
        Abstract: {abstract}
        
        Output strictly as a JSON array of objects representing gaps in the TABI framework.
        Do not add any markdown formatting, backticks, or explanation.
        JSON Schema:
        [
          {{
            "Grounds": "The explicit text evidence or limitations mentioned in the abstract.",
            "Claim": "The research gap statement representing what needs to be studied.",
            "Warrant": "The technical explanation/reasoning of why this gap is important to resolve.",
            "Bucket": "Classify feasibility only as 'near_term_feasible' or 'long_term_or_speculative', never as probability that the claim is true."
          }}
        ]
        """
        
        try:
            content = call_llm(prompt, temperature=0.3)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
                    
            gaps = json.loads(content)
            for g in gaps:
                g["source_paper"] = title
                g["type"] = "simple_llm"
                all_gaps.append(g)
        except Exception as e:
            print(f"[!] Error running Simple LLM baseline for '{title[:30]}': {e}")
            
    return all_gaps

def main():
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    if not os.path.exists(screened_path):
        print(f"[!] File {screened_path} not found. Please run fetch_papers first.")
        return
        
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    gaps = run_simple_llm_baseline(papers)
    
    output_path = os.path.join(config.GAPS_DIR, "baseline_simple_llm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved B2 baseline gaps to {output_path}")

if __name__ == "__main__":
    main()
