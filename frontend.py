import streamlit as st
import requests
import json

# Custom theme for aesthetics
st.set_page_config(page_title="Research Brief Generator", page_icon="🔍", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f4f8;}
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px;}
    .stTextInput>div>div>input {border-radius: 5px;}
    .stNumberInput>div>div>input {border-radius: 5px;}
    .stCheckbox>div {align-items: center;}
    .block-container {padding: 2rem;}
    h1 {color: #333;}
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Research Brief Generator")
st.markdown("Generate structured research briefs with context-aware follow-ups.")

# Sidebar for inputs
with st.sidebar:
    st.header("Query Parameters")
    topic = st.text_input("Topic", placeholder="e.g., AI Ethics")
    depth = st.number_input("Depth (1-5)", min_value=1, max_value=5, value=3)
    follow_up = st.checkbox("Follow-up Query")
    user_id = st.text_input("User ID", value="user1")
    st.markdown("---")
    view_history = st.checkbox("View History")

# Main content
if st.button("Generate Brief"):
    if not topic:
        st.error("Please enter a topic.")
    else:
        with st.spinner("Generating brief..."):
            try:
                response = requests.post(
                    "http://localhost:8000/brief",  # Or deployed URL
                    json={"topic": topic, "depth": depth, "follow_up": follow_up, "user_id": user_id}
                )
                response.raise_for_status()
                brief = response.json()
                
                st.success("Brief Generated!")
                st.markdown(f"### {brief['topic']}")
                st.markdown(f"**Summary:** {brief['summary']}")
                
                for section in brief['sections']:
                    with st.expander(section['title']):
                        st.markdown(section['content'])
                
                st.markdown("### References")
                for ref in brief['references']:
                    st.markdown(f"- [{ref['source_url']}]({ref['source_url']}) (Relevance: {ref['relevance']})")
                    st.write("Key Points: " + "; ".join(ref['key_points']))
            except Exception as e:
                st.error(f"Error: {e}")

if view_history:
    with st.spinner("Loading history..."):
        try:
            response = requests.get(f"http://localhost:8000/history/{user_id}")
            response.raise_for_status()
            history = response.json()["history"]
            st.markdown("### User History")
            st.text_area("Past Briefs", history, height=300)
        except Exception as e:
            st.error(f"Error loading history: {e}")