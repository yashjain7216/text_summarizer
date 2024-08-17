import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import nltk
nltk.download('punkt')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("GROQ_API_KEY", value="", type="password")

generic_url = st.text_input("Enter the URL to summarize", label_visibility="collapsed")

# Check if the API key is provided
if not groq_api_key.strip():
    st.error("Please provide your GROQ API key in the sidebar.")
else:
    # Gemma Model Using Groq API
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    # Define the prompt template
    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Summarize the content when the button is clicked
    if st.button("Summarize the Content from YT or Website"):
        # Validate all inputs
        if not generic_url.strip():
            st.error("Please enter a URL to get started.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YT video URL or website URL.")
        else:
            try:
                with st.spinner("Loading and summarizing content..."):
                    # Load the website or YT video data
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                    # Chain for summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")