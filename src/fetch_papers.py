import json
import os
import time
import requests
import argparse
from src import config
from src.llm_client import get_llm_client, get_llm_model, call_llm

def expand_query(topic: str) -> list[str]:
    """
    Use LLM to generate diverse search terms for a research subdomain.
    """
    print(f"[*] Expanding query for topic: '{topic}' using LLM...")
    client = get_llm_client()
    model = get_llm_model()
    
    prompt = f"""
    You are an expert academic researcher. Generate 5 diverse, highly specific search terms or phrases 
    to retrieve scientific articles about the following topic from Semantic Scholar:
    Topic: "{topic}"
    
    Output strictly as a JSON array of strings, with no additional text or markdown formatting (like ```json).
    Example: ["term 1", "term 2", "term 3", "term 4", "term 5"]
    """
    
    try:
        content = call_llm(prompt, temperature=0.2)
        # Clean up any potential markdown backticks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        
        queries = json.loads(content)
        if isinstance(queries, list):
            # Include original topic
            if topic not in queries:
                queries.insert(0, topic)
            print(f"[+] Expanded queries: {queries}")
            return queries
    except Exception as e:
        print(f"[!] Error expanding query: {e}. Falling back to original topic.")
    
    return [topic]

def search_papers(queries: list[str], limit_per_query: int = 20) -> list[dict]:
    """
    Retrieve paper metadata from Semantic Scholar API.
    """
    print(f"[*] Querying Semantic Scholar API for {len(queries)} search terms...")
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    seen_paper_ids = set()
    papers = []

    for q in queries:
        params = {
            "query": q,
            "limit": limit_per_query,
            "fields": "title,abstract,year,citationCount"
        }
        
        print(f"[*] Querying: '{q}'...")
        try:
            # Semantic Scholar API rate limits: sleep to avoid 429
            time.sleep(3.0)
            
            response = requests.get(endpoint, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", [])
                print(f"[+] Found {len(results)} papers for query '{q}'")
                
                for r in results:
                    paper_id = r.get("paperId")
                    if paper_id and paper_id not in seen_paper_ids:
                        seen_paper_ids.add(paper_id)
                        # We only want papers that have abstracts
                        if r.get("abstract"):
                            papers.append({
                                "paperId": paper_id,
                                "title": r.get("title"),
                                "abstract": r.get("abstract"),
                                "year": r.get("year"),
                                "citationCount": r.get("citationCount", 0)
                            })
            else:
                print(f"[!] API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[!] Error querying Semantic Scholar for query '{q}': {e}")
            
    print(f"[+] Total unique papers with abstracts retrieved: {len(papers)}")
    return papers

def sent_tokenize_fallback(text: str) -> list[str]:
    """
    Split text into sentences. Uses NLTK if available, falls back to regex.
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
        return nltk.sent_tokenize(text)
    except Exception as e:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def screen_relevance(papers: list[dict], topic: str) -> list[dict]:
    """
    Use LLM to grade paper relevance to the topic, filtering out those below threshold.
    """
    print(f"[*] Screening {len(papers)} papers for relevance to topic '{topic}'...")
    client = get_llm_client()
    model = get_llm_model()
    screened = []
    
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        
        prompt = f"""
        Evaluate the relevance of the following scientific paper to the research topic: "{topic}".
        
        Title: {title}
        Abstract: {abstract}
        
        Assign a score between 0.0 (completely irrelevant) and 1.0 (highly relevant).
        Output strictly as a JSON object with two fields:
        - "score": a float between 0.0 and 1.0.
        - "reason": a short explanation.
        
        Output format example:
        {{"score": 0.85, "reason": "Mentions Envoy proxy authorization which is core to the topic."}}
        Do not add any markdown formatting, backticks, or explanation surrounding the JSON.
        """
        
        try:
            content = call_llm(prompt, temperature=0.0)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])
            
            result = json.loads(content)
            score = float(result.get("score", 0.0))
            print(f"[*] Paper: '{title[:50]}...' -> Score: {score}")
            
            if score >= config.RELEVANCE_THRESHOLD:
                p["relevance_score"] = score
                p["relevance_reason"] = result.get("reason", "")
                screened.append(p)
        except Exception as e:
            print(f"[!] Error screening paper '{title[:30]}': {e}. Keeping by default.")
            p["relevance_score"] = 0.8
            p["relevance_reason"] = "Error screening, kept by default"
            screened.append(p)
            
    print(f"[+] {len(screened)} papers survived relevance screening (threshold >= {config.RELEVANCE_THRESHOLD})")
    
    # Sort by relevance_score descending, then citationCount descending
    screened.sort(key=lambda x: (x.get("relevance_score", 0.0), x.get("citationCount", 0)), reverse=True)
    
    if len(screened) > 100:
        print(f"[*] Truncating screened papers from {len(screened)} to 100 based on relevance and citations.")
        screened = screened[:100]
        
    return screened

def chunk_text(text: str, max_words: int = 1000) -> list[str]:
    """
    Split text into chunks of at most max_words, keeping sentences intact.
    """
    sentences = sent_tokenize_fallback(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        if current_word_count + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and screen papers from Semantic Scholar.")
    parser.add_argument("--topic", type=str, default="Microservices security", help="The topic to search for.")
    parser.add_argument("--limit", type=int, default=5, help="Limit per query.")
    args = parser.parse_args()
    
    # 1. Expand Query
    expanded = expand_query(args.topic)
    
    # 2. Search Papers
    fetched = search_papers(expanded, limit_per_query=args.limit)
    
    # Save raw papers metadata
    raw_path = os.path.join(config.RAW_PAPERS_DIR, "papers_metadata.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(fetched, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved raw papers metadata to {raw_path}")
    
    # 3. Screen Relevance
    screened = screen_relevance(fetched, args.topic)
    screened_path = os.path.join(config.RAW_PAPERS_DIR, "screened_papers.json")
    with open(screened_path, "w", encoding="utf-8") as f:
        json.dump(screened, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved screened papers to {screened_path}")
    
    # 4. Chunk Screened Papers
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
    print(f"[+] Created {len(all_chunks)} chunks from screened papers. Saved to {chunks_path}")

