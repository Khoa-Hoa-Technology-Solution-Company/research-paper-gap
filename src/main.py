import json
import os
import argparse
import time
from src import config

# Import pipeline steps
from src.fetch_papers import expand_query, search_papers, screen_relevance, chunk_text
from src.extract_triples import extract_triples_from_chunk
from src.entity_resolution import resolve_entities
from src.graph_analysis import build_graph, detect_orphan_clusters, compute_temporal_decay
from src.tabi_inference import infer_gaps_from_clusters, infer_gaps_from_decay
from src.evaluate import run_evaluation

# Import baselines
from baselines.simple_llm import run_simple_llm_baseline
from baselines.gapmap_text import run_gapmap_baseline
from baselines.mulla_rag import run_mulla_rag_baseline

def run_stage_1(topic: str, limit: int, use_sample: bool = False):
    print("\n" + "="*50 + "\n[Stage 1] Literature Search & Screening\n" + "="*50)
    
    if use_sample:
        sample_path = os.path.join(config.RAW_PAPERS_DIR, "sample_papers.json")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample data file not found at {sample_path}. Please run Task 15 or specify `--topic` without `--sample`.")
            
        print(f"[*] Loading sample papers from {sample_path}...")
        with open(sample_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
            
        # Mock relevance screen
        for p in papers:
            p["relevance_score"] = 0.95
            p["relevance_reason"] = "Loaded from sample data (pre-screened)"
        screened = papers
    else:
        # Expand query using LLM
        expanded = expand_query(topic)
        
        # Search Semantic Scholar
        fetched = search_papers(expanded, limit_per_query=limit)
        
        # Save raw fetched papers
        raw_path = os.path.join(config.RAW_PAPERS_DIR, "papers_metadata.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(fetched, f, ensure_ascii=False, indent=2)
            
        # Screen papers for relevance
        screened = screen_relevance(fetched, topic)
        
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    with open(screened_path, "w", encoding="utf-8") as f:
        json.dump(screened, f, ensure_ascii=False, indent=2)
    print(f"[+] Screened papers saved to {screened_path}")
    
    # Chunk text
    all_chunks = []
    for p in screened:
        paper_chunks = chunk_text(p["abstract"], max_words=1000)
        for i, chunk in enumerate(paper_chunks):
            all_chunks.append({
                "paperId": p["paperId"],
                "title": p["title"],
                "year": p["year"],
                "chunk_index": i,
                "text": chunk
            })
            
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[+] Chunking completed: {len(all_chunks)} chunks written to {chunks_path}")

def run_stage_2():
    print("\n" + "="*50 + "\n[Stage 2] Knowledge Graph Construction\n" + "="*50)
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("Chunks file chunks.json not found! Please run Stage 1 first.")
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # 2a. Triple Extraction
    print("[*] Extracting triples from chunks...")
    all_triples = []
    for i, c in enumerate(chunks):
        print(f"[*] Chunk {i+1}/{len(chunks)} of '{c['title'][:40]}...'")
        triples = extract_triples_from_chunk(c["text"], c["year"])
        print(f"[+] Found {len(triples)} triples.")
        all_triples.extend(triples)
        
    raw_triples_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    with open(raw_triples_path, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
        
    # 2b. Entity Resolution (Deduplication)
    print("[*] Resolving entities...")
    resolved_triples, entity_mapping = resolve_entities(all_triples)
    
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(resolved_triples, f, ensure_ascii=False, indent=2)
        
    mapping_path = os.path.join(config.TRIPLES_DIR, "entity_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(entity_mapping, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Triple extraction and resolution completed. Saved resolved triples to {resolved_path}")

def run_stage_3():
    print("\n" + "="*50 + "\n[Stage 3] Topological Gap Detection\n" + "="*50)
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError("Resolved triples file not found! Please run Stage 2 first.")
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_triples = json.load(f)
        
    # Build Graph
    G = build_graph(resolved_triples)
    
    # Save Graph as GML
    import networkx as nx
    gml_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.gml")
    G_export = G.copy()
    for u, v, data in G_export.edges(data=True):
        if "years" in data:
            data["years"] = ",".join(map(str, data["years"]))
    nx.write_gml(G_export, gml_path)
    print(f"[+] Saved knowledge graph to {gml_path}")
    
    # Run Louvain Orphan Cluster Detection
    orphans = detect_orphan_clusters(G)
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    with open(orphans_path, "w", encoding="utf-8") as f:
        json.dump(orphans, f, ensure_ascii=False, indent=2)
        
    # Run Temporal Decay Analysis
    stagnant = compute_temporal_decay(G)
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")
    with open(stagnant_path, "w", encoding="utf-8") as f:
        json.dump(stagnant, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Topological gap detection completed. Found {len(orphans)} orphan clusters and {len(stagnant)} stagnant concepts.")

def run_stage_4():
    print("\n" + "="*50 + "\n[Stage 4] Information Synthesis via TABI\n" + "="*50)
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    stagnant_path = os.path.join(config.GRAPH_DIR, "temporal_decay.json")
    
    if not os.path.exists(orphans_path) or not os.path.exists(stagnant_path):
        raise FileNotFoundError("Topology reports not found! Please run Stage 3 first.")
        
    with open(orphans_path, "r", encoding="utf-8") as f:
        orphans = json.load(f)
        
    with open(stagnant_path, "r", encoding="utf-8") as f:
        stagnant = json.load(f)
        
    # Run TABI Inference
    cluster_gaps = infer_gaps_from_clusters(orphans)
    decay_gaps = infer_gaps_from_decay(stagnant)
    all_gaps = cluster_gaps + decay_gaps
    
    output_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, ensure_ascii=False, indent=2)
    print(f"[+] TABI Inference completed. Surfaced {len(all_gaps)} research gaps saved to {output_path}")

def run_baselines_and_evaluation():
    print("\n" + "="*50 + "\n[Phase 4] Baselines Execution & Evaluation\n" + "="*50)
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    
    if not os.path.exists(screened_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Literature files not found! Please run Stage 1 first.")
        
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # B2: Simple LLM
    gaps_b2 = run_simple_llm_baseline(papers)
    b2_path = os.path.join(config.GAPS_DIR, "baseline_simple_llm.json")
    with open(b2_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b2, f, ensure_ascii=False, indent=2)
        
    # B3: GAPMAP text-only
    gaps_b3 = run_gapmap_baseline(chunks)
    b3_path = os.path.join(config.GAPS_DIR, "baseline_gapmap.json")
    with open(b3_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b3, f, ensure_ascii=False, indent=2)
        
    # B1: Mulla RAG
    gaps_b1 = run_mulla_rag_baseline(papers)
    b1_path = os.path.join(config.GAPS_DIR, "baseline_mulla_rag.json")
    with open(b1_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b1, f, ensure_ascii=False, indent=2)
        
    # Run evaluation metrics
    print("[*] Launching automatic evaluation...")
    run_evaluation()

def main():
    parser = argparse.ArgumentParser(description="KG-TABI Pipeline Runner")
    parser.add_argument("--topic", type=str, default="Microservices security", help="The topic to search for.")
    parser.add_argument("--limit", type=int, default=5, help="Number of papers to pull per keyword query.")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], 
                        help="Run a specific stage (1-4, 5 for baselines/evaluation, 0 for end-to-end pipeline).")
    parser.add_argument("--sample", action="store_true", help="Use local pre-screened synthetic samples.")
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.stage == 0:
            print("[*] Running KG-TABI end-to-end pipeline...")
            run_stage_1(args.topic, args.limit, args.sample)
            run_stage_2()
            run_stage_3()
            run_stage_4()
            run_baselines_and_evaluation()
        elif args.stage == 1:
            run_stage_1(args.topic, args.limit, args.sample)
        elif args.stage == 2:
            run_stage_2()
        elif args.stage == 3:
            run_stage_3()
        elif args.stage == 4:
            run_stage_4()
        elif args.stage == 5:
            run_baselines_and_evaluation()
            
        print(f"\n[+] Pipeline run completed in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"\n[!] Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
