######################################################################
#
# Spreetail Product Description Optimization Workflow
#
######################################################################

import re
import time
import json
import boto3
import operator
import streamlit as st
from curl_cffi import requests
from textwrap import dedent
from omegaconf import OmegaConf
from bs4 import BeautifulSoup
from random import randint
from pydantic import HttpUrl
from typing import Annotated, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, HttpUrl, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

client = boto3.client("bedrock-runtime")

# Load the configuration from the YAML file
conf = OmegaConf.load('config.yaml')

#-----------------------------------------------------------------
# Product Pydantic Model

class Product(BaseModel):
    asin: str = Field(..., description="ASIN of the product")
    title: str = Field(..., description="Title of the product")
    description: str = Field(..., description="Description of the product")
    bullets: List[str] = Field(..., description="Bullets of the product")
    brand: str = Field(..., description="Brand of the product")
    color: str = Field(..., description="Color of the product")
    asin_link: HttpUrl = Field(..., description="Link to the product on Amazon")

#-----------------------------------------------------------------
# LangGraph State

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    # messages: Annotated[list, operator.add]
    asin: str
    title: str
    description: str
    bullets: List[str]
    brand: str
    color: str
    keywords: str
    optimized_title: str
    optimized_description: str
    optimized_bullets: List[str]

restrictive_terms=conf.restrictive_terms

#-----------------------------------------------------------------
# Scrape Amazon Product using ASIN

def scrape_amazon_product(asin):
    url = f"https://www.amazon.com/gp/product/{asin}"

    def new_session():
        session = requests.Session(impersonate="chrome120")
        return session
        
    session = new_session()
    response = session.get(url)
    
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract title
    title = soup.find("span", {"id": "productTitle"}).text.strip() if soup.find("span", {"id": "productTitle"}) else "N/A"

    # Extract description
    description = "N/A"
    description_elem = soup.find("div", {"id": "productDescription"})
    if description_elem:
        description = description_elem.text.strip()
    else:
        # Try to find description in product features
        feature_div = soup.find("div", {"id": "feature-bullets"})
        if feature_div:
            description = " ".join([li.text.strip() for li in feature_div.find_all("li")])

    # Extract bullet points
    feature_div = soup.find("div", {"id": "feature-bullets"})
    bullets = [li.text.strip() for li in feature_div.find_all("li")] if feature_div else []

    # Extract brand
    brand = "N/A"
    brand_elem = soup.find("a", {"id": "bylineInfo"})
    if brand_elem:
        brand = brand_elem.text.strip().replace("Visit the ", "").replace(" Store", "")
    
    # Extract color
    color = "N/A"
    if color == "N/A":
        detail_table = soup.find("table", {"id": "productDetails_detailBullets_sections1"})
        if detail_table:
            rows = detail_table.find_all("tr")
            for row in rows:
                header = row.find("th")
                if header and "color" in header.text.lower():
                    color = row.find("td").text.strip()
                    break

    return {
        "asin": asin,
        "title": title,
        "description": description,
        "bullets": bullets,
        "brand": brand,
        "color": color,
        "asin_link": HttpUrl(url),
    }

#-----------------------------------------------------------------
# LangGraph - Nodes

def product_retrieval_node(state: State):
    asin = state['asin']

    amazon_product = scrape_amazon_product(asin)
    st.write(amazon_product)
    product = Product(**amazon_product)
    
    return {
        "asin": product.asin,
        "title": product.title,
        "description": product.description,
        "bullets": product.bullets,
        "brand": product.brand,
        "color": product.color,
    }

def search_keywords_node(state: State):
    asin = state['asin']
    title = state['title']
    description = state['description']
    bullets = state['bullets']
    brand = state['brand']
    color = state['color']

    search_keywords_prompt = conf.prompts.search_keywords_prompt.format(
        title=title,
        description=description,
        bullets=bullets,
        brand=brand,
        color=color
    )
    
    messages = [{"role": "user","content": [{"text": search_keywords_prompt}]}]
    response = client.converse(
        modelId=conf.aws.model_id,
        messages=messages,
        system=[{"text": conf.prompts.system_prompt}],
        inferenceConfig={
            'maxTokens': conf.aws.maxTokens,
            'temperature': conf.aws.temperature,
            'topP': conf.aws.topP,
        },
    )
    product_keywords = response['output']['message']['content'][0]['text']

    return {"keywords":product_keywords}
    
def title_node(state: State):
    asin = state['asin']
    title = state['title']
    description = state['description']
    bullets = state['bullets']
    brand = state['brand']
    color = state['color']
    keywords = state['keywords']

    product_title_prompt = conf.prompts.product_title_prompt.format(
        asin=asin,
        title=title,
        description=description,
        bullets=bullets,
        brand=brand,
        color=color,
        keywords=keywords,
        restrictive_terms=restrictive_terms,
        title_char_range=title_char_range,
    )
    
    messages = [{"role": "user","content": [{"text": product_title_prompt}]}]
    response = client.converse(
        modelId=conf.aws.model_id,
        messages=messages,
        system=[{"text": conf.prompts.system_prompt}],
        inferenceConfig={
            'maxTokens': conf.aws.maxTokens,
            'temperature': conf.aws.temperature,
            'topP': conf.aws.topP,
        },
    )

    product_title = response['output']['message']['content'][0]['text']

    return {"optimized_title":product_title}


def description_node(state: State):
    asin = state['asin']
    title = state['title']
    description = state['description']
    bullets = state['bullets']
    brand = state['brand']
    color = state['color']
    keywords = state['keywords']

    product_description_prompt = conf.prompts.product_description_prompt.format(
        asin=asin,
        title=title,
        description=description,
        bullets=bullets,
        brand=brand,
        color=color,
        keywords=keywords,
        restrictive_terms=restrictive_terms,
        description_char_range=description_char_range
    )

    messages = [{"role": "user","content": [{"text": product_description_prompt}]}]
    response = client.converse(
        modelId=conf.aws.model_id,
        messages=messages,
        system=[{"text": conf.prompts.system_prompt}],
        inferenceConfig={
            'maxTokens': conf.aws.maxTokens,
            'temperature': conf.aws.temperature,
            'topP': conf.aws.topP,
        },
    )
    product_description = response['output']['message']['content'][0]['text']
    
    return {"optimized_description":product_description}

def bullets_node(state: State):
    asin = state['asin']
    title = state['title']
    description = state['description']
    bullets = state['bullets']
    brand = state['brand']
    color = state['color']
    keywords = state['keywords']

    product_bullets_prompt = conf.prompts.product_bullets_prompt.format(
        title=title,
        description=description,
        bullets=bullets,
        brand=brand,
        color=color,
        keywords=keywords,
        restrictive_terms=restrictive_terms,
        bullets_char_range=bullets_char_range
    )

    converse_tools = [{'toolSpec': {'name': 'Optimized_Bullets',
      'description': 'Optimized Bullets',
      'inputSchema': {'json': {'properties': {'optimized_bullets': {'description': 'Optimized Bullets',
          'items': {'type': 'string'},
          'title': 'Optimized Bullets',
          'type': 'array'}},
        'title': 'Optimized_Bullets',
        'type': 'object'}}}}]
    
    messages = [{"role": "user","content": [{"text": product_bullets_prompt}]}]
    response = client.converse(
        modelId=conf.aws.model_id,
        messages=messages,
        system=[{"text": conf.prompts.system_prompt}],
        inferenceConfig={
            'maxTokens': conf.aws.maxTokens,
            'temperature': conf.aws.temperature,
            'topP': conf.aws.topP,
        },
        toolConfig={"tools": converse_tools},
    )
    tool_message = response['output']['message']
    tool_results_content_blocks = []
    for content_block in tool_message['content']:
        if 'toolUse' in content_block:
            tool_use = content_block['toolUse']
            optimized_bullets = tool_use['input']

    return optimized_bullets

#-----------------------------------------------------------------
# Build Graph
def create_graph():
    workflow = StateGraph(State)
    workflow.add_node("product_retrieval", product_retrieval_node)
    workflow.add_node("search_keywords", search_keywords_node)
    workflow.add_node("product_title", title_node)
    workflow.add_node("product_description", description_node)
    workflow.add_node("product_bullets", bullets_node)
    
    workflow.add_edge(START, "product_retrieval")
    workflow.add_edge("product_retrieval", "search_keywords")
    workflow.add_edge("search_keywords", "product_title")
    workflow.add_edge("search_keywords", "product_description")
    workflow.add_edge("search_keywords", "product_bullets")
    workflow.add_edge("product_bullets", END)
    workflow.add_edge("product_description", END)
    workflow.add_edge("product_title", END)

    graph = workflow.compile()

    return graph

def create_graph_image():
    return create_graph().get_graph().draw_mermaid_png()
# ------------------------------------------------------------------------
# Streamlit App

greeting_message = "Welcome to the Amazon Product Optimization Workflow. Please enter an ASIN for a product you want to optimize."

# Clear Chat History fuction
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": greeting_message}]

def validate_asin(asin):
    if not isinstance(asin, str):
        return False
    if len(asin) != 10:
        return False
    if not asin.isalnum():
        return False
    asin_pattern = r'^B[\dA-Z]{9}$'
    if not re.match(asin_pattern, asin):
        return False
    return True

with st.sidebar:
    st.image("amazon.png")
    st.title('Product Optimization Workflow')
    title_char_range = st.slider("Title: character range", 120, 160, (140, 150))
    description_char_range = st.slider("Description: character range", 1000, 1500, (1300, 1400))
    bullets_char_range = st.slider("Bullets: character range", 100, 250, (150, 200))
    st.button('Clear Screen', on_click=clear_screen)
    st.markdown("""
    ## How to use:

    * Enter an ASIN(Amazon Standard Identification Number).
    """)

# Create Graph Image
st.image(create_graph_image())

# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": greeting_message}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input - User Prompt
if asin_input := st.chat_input("Enter an ASIN:"):
    st.session_state.messages.append({"role": "user", "content": asin_input})
    with st.chat_message("user"):
        st.write(asin_input)

    if validate_asin(asin_input):
        response = f"The ASIN '{asin_input}' is valid!"
        with st.spinner(f"Optimizing product description..."):
            # Create Graph
            app = create_graph()
            
            output = {}
            start_time = time.time()
            st.success(f"Amazon.com product page - https://www.amazon.com/gp/product/{asin_input}")
            
            for chunk in app.stream({'asin': asin_input}):
                for node, value in chunk.items():
                    st.info(f"**{node}**", icon="🤖")
                    with st.expander("message"):
                        st.write(value)
                    for k,v in value.items():
                        output[k] = v
            end_time = time.time()

            def create_bullet_list(items):
                return "\n".join(f"* {item}" for item in items)
            # Output Format Template
            output_format = dedent(f"""
**Run Time**: {end_time - start_time:.2f}s\n

### Title
**Current Title**\n
{output['title']}

**Optimized Title**\n
{output['optimized_title']}

### Description
**Current Description**\n
{output['description']}

**Optimized Description**\n
{output['optimized_description']}

### Bullets
**Current Bullets**\n
{create_bullet_list(output['bullets'])}

**Optimized Bullets**\n
{create_bullet_list(output['optimized_bullets'])}

### Keywords
**Optimized Keywords**\n
{output['keywords']}
""")
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(output_format)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": output_format})

    else:
        response = f"The ASIN '{asin_input}' is not valid. Please enter a valid ASIN."
        with st.chat_message("assistant"):
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
