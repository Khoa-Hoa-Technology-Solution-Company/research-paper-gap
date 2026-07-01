import json
import os
import datetime
import streamlit as st
import networkx as nx
import streamlit.components.v1 as components

# Import config
from src import config

# Set page config for beautiful HCAI Dashboard look
st.set_page_config(
    page_title="KG-TABI: Expert Research Gap Review Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 KG-TABI Expert Review Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Human-Centered AI (HCAI) audit trail loop for Software Engineering Research Gap validation</div>', unsafe_allow_html=True)

# Helper to load GML graph
@st.cache_data
def load_graph():
    gml_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.gml")
    if os.path.exists(gml_path):
        try:
            return nx.read_gml(gml_path)
        except Exception as e:
            st.warning(f"Failed to load GML graph: {e}")
    return None

# Load generated gaps
gaps_path = os.path.join(config.GAPS_DIR, "kgtabi_gaps.json")
if not os.path.exists(gaps_path):
    st.error("No gaps found! Please run the KG-TABI pipeline first to generate `kgtabi_gaps.json`.")
    st.info("Run command: `python -m src.main --sample` or configure your API key to run a live pipeline.")
else:
    with open(gaps_path, "r", encoding="utf-8") as f:
        gaps = json.load(f)
        
    G = load_graph()
    
    # Sidebar stats
    st.sidebar.header("📊 Pipeline Statistics")
    st.sidebar.markdown(f"**Total Gaps Surfaced:** `{len(gaps)}`")
    if G:
        st.sidebar.markdown(f"**KG Nodes:** `{G.number_of_nodes()}`")
        st.sidebar.markdown(f"**KG Edges:** `{G.number_of_edges()}`")
        
    # Expert profile
    st.sidebar.subheader("👤 Reviewer Identity")
    reviewer_name = st.sidebar.text_input("Reviewer Name/ID", value="Expert_Reviewer_1")
    
    # Selected Gap Selector
    st.subheader("📋 Select Research Gap to Audit")
    if not gaps:
        st.warning("Gaps list is empty.")
    else:
        gap_options = [f"[{g.get('type', 'Unknown').upper()}] {g.get('Claim', '')[:80]}..." for g in gaps]
        selected_idx = st.selectbox("Choose a gap entry", range(len(gap_options)), format_func=lambda x: gap_options[x])
        
        selected_gap = gaps[selected_idx]
        
        # Display TABI fields
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💬 Toulmin-Abductive Reasoning (TABI)")
            
            st.markdown("### 🧬 Grounds (Evidence)")
            st.info(selected_gap.get("Grounds", "No grounds provided."))
            
            st.markdown("### 🎯 Claim (Research Gap)")
            st.success(selected_gap.get("Claim", "No claim provided."))
            
            st.markdown("### 🔗 Warrant (Technical Rationale)")
            st.warning(selected_gap.get("Warrant", "No warrant provided."))
            
            bucket = selected_gap.get("Bucket", "more_probable")
            st.markdown(f"**Feasibility Bucket:** <span class='metric-badge'>{bucket}</span>", unsafe_allow_html=True)
            st.markdown(f"**Source:** `{selected_gap.get('source', 'Unknown')}`")
            
        with col2:
            st.subheader("🕸️ Subgraph Evidence Trail")
            if G:
                # Visualize community nodes referenced
                # Extract words/entities from the Grounds text or source string to find related nodes
                source_str = selected_gap.get("source", "")
                # Simple community matching: e.g. "Community 3 vs Community 5"
                communities_matched = re.findall(r'Community (\d+)', source_str)
                
                # Render interactive pyvis network of related nodes
                try:
                    from pyvis.network import Network
                    
                    net = Network(height="400px", width="100%", notebook=False, heading="")
                    net.repulsion(node_distance=150, spring_length=100)
                    
                    # If communities are matched, extract nodes in them (we can save community ID as node attribute during analysis, or find adjacent nodes)
                    # Let's find nodes mentioned in target claim/grounds
                    highlight_nodes = []
                    for node in G.nodes():
                        if node.lower() in selected_gap.get("Grounds", "").lower() or node.lower() in selected_gap.get("Claim", "").lower():
                            highlight_nodes.append(node)
                            
                    # Get the ego network around highlighted nodes
                    subgraph_nodes = set(highlight_nodes)
                    for n in highlight_nodes:
                        subgraph_nodes.update(G.neighbors(n))
                        # For DiGraph, also add predecessors
                        subgraph_nodes.update(G.predecessors(n))
                        
                    # Build pyvis graph
                    H = G.subgraph(subgraph_nodes)
                    
                    for node in H.nodes():
                        is_highlight = node in highlight_nodes
                        color = "#EF4444" if is_highlight else "#3B82F6"
                        size = 25 if is_highlight else 15
                        net.add_node(node, label=node, color=color, size=size, title=f"Type: {G.nodes[node].get('type', 'CONCEPT')}")
                        
                    for u, v in H.edges():
                        net.add_edge(u, v, label=G[u][v].get("relation", "CONNECTS"))
                        
                    # Save HTML
                    html_path = os.path.join(config.GRAPH_DIR, "subgraph.html")
                    net.save_graph(html_path)
                    
                    # Read and render HTML
                    with open(html_path, "r", encoding="utf-8") as html_file:
                        components.html(html_file.read(), height=420, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not render PyVis subgraph: {e}. Showing text list of nodes instead.")
                    # Fallback text representation
                    highlight_nodes = [node for node in G.nodes() if node.lower() in selected_gap.get("Grounds", "").lower() or node.lower() in selected_gap.get("Claim", "").lower()]
                    st.write("**Related entities in KG:**")
                    st.write(highlight_nodes)
            else:
                st.info("Knowledge graph structure file (.gml) not found. Visualization unavailable.")
                
        # Audit Decisions (HCAI Loop)
        st.markdown("---")
        st.subheader("🛠️ Auditor Verdict & Custom Modifications")
        
        # Inputs for custom modification
        custom_claim = st.text_input("Modify Claim (optional)", value=selected_gap.get("Claim", ""))
        custom_warrant = st.text_area("Modify Warrant (optional)", value=selected_gap.get("Warrant", ""))
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        # Load existing reviews to append to log
        reviews_log_path = os.path.join(config.DATA_DIR, "expert_reviews.json")
        reviews_log = []
        if os.path.exists(reviews_log_path):
            try:
                with open(reviews_log_path, "r", encoding="utf-8") as log_f:
                    reviews_log = json.load(log_f)
            except Exception:
                reviews_log = []
                
        def save_review(verdict, claim_text, warrant_text):
            log_entry = {
                "timestamp": str(datetime.datetime.now()),
                "reviewer": reviewer_name,
                "original_claim": selected_gap.get("Claim"),
                "original_warrant": selected_gap.get("Warrant"),
                "verdict": verdict,
                "modified_claim": claim_text,
                "modified_warrant": warrant_text,
                "source": selected_gap.get("source"),
                "type": selected_gap.get("type")
            }
            reviews_log.append(log_entry)
            with open(reviews_log_path, "w", encoding="utf-8") as log_w:
                json.dump(reviews_log, log_w, ensure_ascii=False, indent=2)
            st.success(f"Successfully recorded verdict: '{verdict}' to {reviews_log_path}!")
            
        with col_btn1:
            if st.button("🟢 Accept Gap", use_container_width=True):
                save_review("Accept", custom_claim, custom_warrant)
                
        with col_btn2:
            if st.button("🟡 Modify & Save Gap", use_container_width=True):
                save_review("Modify", custom_claim, custom_warrant)
                
        with col_btn3:
            if st.button("🔴 Reject Gap", use_container_width=True):
                save_review("Reject", custom_claim, custom_warrant)
                
        # Show past reviews table
        st.markdown("### 📜 Recent Audits")
        if reviews_log:
            st.write(reviews_log[-5:])
        else:
            st.write("No verification reviews recorded yet.")
            
        import re # For community regex
