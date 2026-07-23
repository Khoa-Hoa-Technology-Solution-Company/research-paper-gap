import json
import os
import argparse
import time
from src import config
from src.provenance import file_sha256

# Import pipeline steps
from src.fetch_papers import (
    build_chunk_records, expand_query_with_provenance, search_papers,
    screen_relevance, screen_relevance_lexical, write_stage1_provenance_manifest,
)
from src.extract_triples import run_extraction
from src.entity_resolution import resolve_entities
from src.graph_analysis import (
    build_graph,
    compute_temporal_decay,
    detect_orphan_clusters,
    gml_export_copy,
    save_topology_run_config,
)
from src.tabi_inference import infer_gaps_from_clusters, infer_gaps_from_decay
from src.evaluate import run_evaluation

# Import baselines
from baselines.simple_llm import run_simple_llm_baseline
from baselines.gapmap_text import run_gapmap_baseline
from baselines.mulla_rag import run_mulla_rag_baseline
from baselines.graphrag import run_graphrag_baseline
from baselines.lightrag import run_lightrag_baseline
from baselines.hipporag import run_hipporag_baseline

def run_stage_1(topic: str, limit: int, max_papers: int, use_sample: bool = False,
                reuse_raw: bool = False, screening_mode: str = "llm"):
    print("\n" + "="*50 + "\n[Stage 1] Literature Search & Screening\n" + "="*50)
    
    raw_path = os.path.join(config.RAW_PAPERS_DIR, "papers_metadata.json")
    if use_sample:
        sample_path = os.path.join(config.RAW_PAPERS_DIR, "sample_papers.json")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample data file not found at {sample_path}. Please run Task 15 or specify `--topic` without `--sample`.")
            
        print(f"[*] Loading sample papers from {sample_path}...")
        with open(sample_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        raw_path = sample_path
            
        # Mock relevance screen
        for p in papers:
            p["relevance_score"] = 0.95
            p["relevance_reason"] = "Loaded from sample data (pre-screened)"
            p["screening_method"] = "local-pre-screened-sample"
        screened = papers
        expansion_manifest = {
            "operation": "local_sample",
            "status": "not_applicable",
            "reason": "--sample bypasses retrieval and LLM query expansion",
        }
        retrieval_manifest = dict(expansion_manifest)
        screening_manifest = {
            "operation": "local_sample",
            "status": "not_applicable",
            "reason": "sample records are supplied pre-screened",
        }
    else:
        if reuse_raw:
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"No raw corpus found to reuse at {raw_path}")
            with open(raw_path, "r", encoding="utf-8") as f:
                fetched = json.load(f)
            print(f"[*] Reusing {len(fetched)} previously retrieved raw papers from {raw_path}")
            expansion_manifest = {
                "operation": "query_expansion",
                "status": "not_replayed",
                "reason": "--reuse-raw was selected; inspect the existing Stage 1 manifest if present.",
            }
            retrieval_manifest = {
                "operation": "reused_raw_metadata",
                "status": "reused",
                "source_artifact": os.path.basename(raw_path),
                "source_artifact_sha256": file_sha256(raw_path),
            }
        else:
            # Expand query using LLM
            expanded, expansion_manifest = expand_query_with_provenance(topic)

            # Search Semantic Scholar
            fetched, retrieval_manifest = search_papers(
                expanded, limit_per_query=limit, return_manifest=True
            )

            # Save raw fetched papers
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(fetched, f, ensure_ascii=False, indent=2)
            
        # Screen papers for relevance. The lexical mode is a transparent,
        # auditable retrieval filter for large initial corpus construction.
        screened, screening_manifest = (
            screen_relevance_lexical(fetched, max_papers=max_papers, return_manifest=True)
            if screening_mode == "lexical"
            else screen_relevance(fetched, topic, max_papers=max_papers, return_manifest=True)
        )
        
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    with open(screened_path, "w", encoding="utf-8") as f:
        json.dump(screened, f, ensure_ascii=False, indent=2)
    print(f"[+] Screened papers saved to {screened_path}")
    
    # Chunk text
    all_chunks = []
    for p in screened:
        all_chunks.extend(build_chunk_records(p, max_words=1000, section_label="abstract"))
            
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[+] Chunking completed: {len(all_chunks)} chunks written to {chunks_path}")
    manifest_path = write_stage1_provenance_manifest(
        topic=topic,
        query_expansion=expansion_manifest,
        retrieval=retrieval_manifest,
        screening=screening_manifest,
        artifact_paths={
            "papers_metadata": raw_path,
            "screened_papers": screened_path,
            "chunks": chunks_path,
        },
    )
    print(f"[+] Saved Stage 1 provenance manifest to {manifest_path}")

def run_stage_2():
    print("\n" + "="*50 + "\n[Stage 2] Knowledge Graph Construction\n" + "="*50)
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("Chunks file chunks.json not found! Please run Stage 1 first.")
        
    # 2a. Triple Extraction
    print("[*] Extracting triples from chunks with resumable provenance checkpoints...")
    all_triples = run_extraction()
        
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
    gml_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.gml")
    import networkx as nx
    nx.write_gml(gml_export_copy(G), gml_path)
    print(f"[+] Saved knowledge graph to {gml_path}")
    
    # Run Louvain Orphan Cluster Detection
    orphans = detect_orphan_clusters(G)
    orphans_path = os.path.join(config.GRAPH_DIR, "orphan_clusters.json")
    with open(orphans_path, "w", encoding="utf-8") as f:
        json.dump(orphans, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved topology configuration to {save_topology_run_config()}")
        
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
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    
    if not os.path.exists(screened_path) or not os.path.exists(chunks_path) or not os.path.exists(resolved_path):
        raise FileNotFoundError("Literature or triples files not found! Please run Stage 1 and 2 first.")
        
    with open(screened_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(resolved_path, "r", encoding="utf-8") as f:
        resolved_triples = json.load(f)
        
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

    # B4: GraphRAG
    G = build_graph(resolved_triples)
    gaps_b4 = run_graphrag_baseline(G, papers)
    b4_path = os.path.join(config.GAPS_DIR, "baseline_graphrag.json")
    with open(b4_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b4, f, ensure_ascii=False, indent=2)

    # B5: LightRAG
    gaps_b5 = run_lightrag_baseline(papers, resolved_triples)
    b5_path = os.path.join(config.GAPS_DIR, "baseline_lightrag.json")
    with open(b5_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b5, f, ensure_ascii=False, indent=2)

    # B6: HippoRAG
    gaps_b6 = run_hipporag_baseline(papers, chunks, resolved_triples)
    b6_path = os.path.join(config.GAPS_DIR, "baseline_hipporag.json")
    with open(b6_path, "w", encoding="utf-8") as f:
        json.dump(gaps_b6, f, ensure_ascii=False, indent=2)
        
    # Run evaluation metrics
    print("[*] Launching automatic evaluation...")
    run_evaluation()

def main():
    parser = argparse.ArgumentParser(description="KG-TABI Pipeline Runner")
    parser.add_argument("--topic", type=str, default="Microservices security", help="The topic to search for.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum records to pull per expanded query.")
    parser.add_argument("--max-papers", type=int, default=300, help="Maximum screened papers to retain.")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], 
                        help="Run a specific stage (1-4, 5 for baselines/evaluation, 0 for end-to-end pipeline).")
    parser.add_argument("--sample", action="store_true", help="Use local pre-screened synthetic samples.")
    parser.add_argument("--reuse-raw", action="store_true",
                        help="Reuse raw_papers/papers_metadata.json for this run instead of querying again.")
    parser.add_argument("--screening-mode", choices=["llm", "lexical"], default="llm",
                        help="Use batched LLM screening or the auditable lexical first-pass screen.")
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.stage == 0:
            print("[*] Running KG-TABI end-to-end pipeline...")
            run_stage_1(args.topic, args.limit, args.max_papers, args.sample,
                        args.reuse_raw, args.screening_mode)
            run_stage_2()
            run_stage_3()
            run_stage_4()
            run_baselines_and_evaluation()
        elif args.stage == 1:
            run_stage_1(args.topic, args.limit, args.max_papers, args.sample,
                        args.reuse_raw, args.screening_mode)
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
