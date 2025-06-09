import streamlit as st
import json
import pandas as pd
from src.components.ContextRetriever import ContextRetriever
from src.components.GroverTopK import GroverTopK
from src.components.AgentHandler import AgentHandler
import os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Build absolute paths
EMBEDDINGS_FILE = os.path.join(script_dir, "saved_embeddings", "squad_embeddings_mixedbread_ai_mxbai_embed_large_v1.npy")
DOCS_LIST_FILE = os.path.join(script_dir, "saved_embeddings", "squad_docs_mixedbread_ai_mxbai_embed_large_v1.json")

# Debug: Check if files exist
print(f"Embeddings file exists: {os.path.exists(EMBEDDINGS_FILE)}")
print(f"Docs file exists: {os.path.exists(DOCS_LIST_FILE)}")
print(f"Current working directory: {os.getcwd()}")

#from gpt_final_answer import generate_answer_from_contexts
grover_top_k = GroverTopK()
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K_FIRST = 10
TOP_K_FINAL = 3

# Context retriever
retriever = ContextRetriever(
    model_name=MODEL_NAME,
    embeddings_file=EMBEDDINGS_FILE,
    docs_list_file=DOCS_LIST_FILE
)

with open(DOCS_LIST_FILE, 'r', encoding='utf-8') as f:
    docs = json.load(f)
retriever.build_index(docs)

# Streamlit UI
st.title("QA with Grover and LLMs")
st.markdown("Ask a question and get an answer using Grover and LLM.")

user_question = st.text_input("Your question:", placeholder="Type your question here...")

if user_question:
    query = QUERY_PREFIX + user_question
    with st.spinner("Searching for relevant contexts..."):
        # Use the retriever to get the top 10 results
        top_10_results = retriever.search(query, top_k=TOP_K_FIRST)

    if not top_10_results:
        st.error("No relevant contexts found.")
    else:
        with st.spinner("Searching for relevant contexts with Grover..."):
            # Use Grover to get the top k contexts
            # grover_output = grover_top_k(top_10_results, k=TOP_K_FINAL)
            grover_output=grover_top_k.select(top_10_results, k=TOP_K_FINAL)
            top_k_contexts = grover_output["contexts"]


        with st.spinner("Generating answer..."):
            # Generate the final answer from the top k contexts
            agent_handler = AgentHandler()
            agents_output = agent_handler.compare_all(user_question, top_k_contexts)
            

        st.subheader("Answer:")
        if agents_output is None:
            st.error("No answer found.")
        else:
            st.success("Answer found!")
        # st.write(final_answer)
        # Prepare data for DataFrame
            for model_key, outputs in agents_output.items():
                with st.expander(f"Model: {model_key}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ With Context:**")
                        st.info(outputs["with_context"])
                    
                    with col2:
                        st.markdown("**❓ Without Context:**")
                        st.warning(outputs["without_context"])
                    
                    # Add a small comparison summary
                    match_percentage = len(set(outputs["with_context"].split()) & set(outputs["without_context"].split())) / max(len(set(outputs["with_context"].split())), len(set(outputs["without_context"].split()))) * 100
                    st.caption(f"Word overlap: ~{match_percentage:.1f}% similarity between responses")
        
        st.markdown(f"**Treshold:** {grover_output['threshold']:.4f}")



        with st.expander("Show top 3 contexts"):
            for i, ctx in enumerate(top_k_contexts):
                st.markdown(f"**Context {i+1} (score: {ctx['similarity_score']:.4f})**")
                st.write(ctx["document"])

        with st.expander("Show all top 10 contexts"):
            for res in top_10_results:
                st.markdown(f"**Rank {res['rank']} (score: {res['similarity_score']:.4f})**")
                st.write(res["document"][:500] + "...")
                
        st.markdown("### Grover info:")
        st.markdown(f"**Qiskit version:** {grover_output['qiskit_version']}")
        st.markdown(f"**Runtime version:** {grover_output['runtime_version']}")
