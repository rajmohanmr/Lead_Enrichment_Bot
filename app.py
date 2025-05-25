import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
# import google.generativeai as genai # REMOVE OR COMMENT OUT THIS LINE
import os
from dotenv import load_dotenv
import re
import time
import io # For file handling in Streamlit

# --- NEW: Import Cohere library ---
import cohere

# --- Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="AI Lead Enrichment Bot", layout="wide")

# --- Configuration and Setup ---
# Load environment variables from .env file
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY") # NEW: Get Cohere API Key

if COHERE_API_KEY: # NEW: Check for Cohere Key
    try:
        # NEW: Initialize Cohere client
        co = cohere.Client(COHERE_API_KEY)
        # st.success("Cohere API configured successfully!") # Optional: Removed for cleaner output
    except Exception as e:
        st.error(f"Error configuring Cohere API: {e}. Please check your COHERE_API_KEY.")
        st.stop()
else:
    st.error("COHERE_API_KEY not found in .env file. Please create a .env file with your key.")
    st.markdown("Get your key from [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).")
    st.stop()

# --- Helper Functions for Data Enrichment ---
# (No changes needed for get_company_website and scrape_website_text)

def get_company_website(company_name):
    """
    Attempts to find a company's website by trying common URL patterns.
    Returns the confirmed website URL or None if not found.
    """
    # Clean the company name for URL generation
    cleaned_name = re.sub(r'(\s*(inc|ltd|llc|corp|co|group|holdings|solutions|technologies|systems)\.?\s*)$', '', company_name, flags=re.IGNORECASE).strip()
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s-]', '', cleaned_name)
    cleaned_name = cleaned_name.replace(" ", "").lower()

    if not cleaned_name:
        return None

    common_tlds = ['com', 'io', 'net', 'org', 'co', 'ai', 'tech', 'app', 'cloud', 'dev', 'xyz']
    potential_urls = []

    for tld in common_tlds:
        potential_urls.append(f"https://www.{cleaned_name}.{tld}")
        potential_urls.append(f"https://{cleaned_name}.{tld}")
        potential_urls.append(f"http://www.{cleaned_name}.{tld}")
        potential_urls.append(f"http://{cleaned_name}.{tld}")

    for url in potential_urls:
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200 and 'google.com' not in response.url and 'bing.com' not in response.url:
                st.session_state.log_messages.append(f"Found website for {company_name}: {url}")
                return url
        except requests.exceptions.RequestException:
            pass

    st.session_state.log_messages.append(f"Could not find a valid website for {company_name}")
    return None

def scrape_website_text(url):
    """
    Fetches the main text content from a given URL using requests and BeautifulSoup.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        main_content_tags = ['main', 'article', 'div.main-content', 'div#content', 'div.container', 'body']
        main_content = None
        for tag_selector in main_content_tags:
            main_content = soup.select_one(tag_selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
            if not main_content:
                return ""

        texts = []
        for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            text = tag.get_text(separator=' ', strip=True)
            if text:
                texts.append(text)

        full_text = ' '.join(texts)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        max_text_length = 2000
        if len(full_text) > max_text_length:
            st.session_state.log_messages.append(f"Warning: Text for {url} is very long ({len(full_text)} chars). Truncating.")
            full_text = full_text[:max_text_length]

        return full_text

    except requests.exceptions.RequestException as e:
        st.session_state.log_messages.append(f"Error scraping {url}: {e}")
        return ""
    except Exception as e:
        st.session_state.log_messages.append(f"An unexpected error occurred while scraping {url}: {e}")
        return ""

# --- MODIFIED: get_llm_insights for Cohere with enhanced extraction ---
def get_llm_insights(text, company_name):
    """
    Uses the Cohere LLM (Command model) to summarize, identify target customers,
    suggest automation ideas, and extract additional company details.
    """
    if not text or len(text.strip()) < 50:
        return {
            "summary": "N/A (Insufficient text)",
            "industry": "N/A (Insufficient text)",
            "target_customer": "N/A (Insufficient text)",
            "employee_count": "N/A (Insufficient text)",
            "founding_year": "N/A (Insufficient text)",
            "automation_pitch": "N/A (Insufficient text)"
        }

    # Prompt engineering for Cohere
    prompt_content = f"""
    Analyze the following text about '{company_name}' and provide the following information concisely.
    Extract directly from the provided text if available, or infer based on context.

    1.  **Company Summary:** A concise summary (2-3 sentences) of what the company does, their main products/services, and core mission based on the provided text.
    2.  **Industry:** The primary industry or sector the company operates in (e.g., "Artificial Intelligence", "Cloud Software", "Financial Technology", "E-commerce"). Be specific.
    3.  **Target Customer:** Who is the company's primary target customer or audience? (e.g., "Small businesses", "Enterprise IT departments", "Individual consumers", "Developers", "Marketing agencies").
    4.  **Employee Count:** The estimated number of employees if mentioned or strongly inferable from the text (e.g., "100-200", "500+", "N/A").
    5.  **Founding Year:** The year the company was founded if mentioned or strongly inferable from the text (e.g., "2005", "1998", "N/A").
    6.  **AI Automation Idea for {company_name} (QF Innovate Pitch):** Suggest a specific, innovative AI automation idea that 'QF Innovate' could pitch to {company_name} to help them improve their operations, enhance their products, or expand their market. Focus on a clear problem {company_name} might face and an AI-driven solution.

    Company Text for Analysis:
    ---
    {text}
    ---

    Please format your response clearly with the exact labels as requested, and ensure each section is on its own line or clearly separated.
    Example Format:
    Company Summary: ...
    Industry: ...
    Target Customer: ...
    Employee Count: ...
    Founding Year: ...
    AI Automation Idea for [Company Name]: ...
    """
    
    try:
        # Call Cohere Chat API
        response = co.chat(
            model='command', # Using 'command' model, which is generally available in the free tier.
            message=prompt_content,
            temperature=0.7,
            max_tokens=500
        )
        response_text = response.text.strip() # Cohere returns response.text
        
        # Initialize with N/A
        summary = "N/A"
        industry = "N/A"
        target_customer = "N/A"
        employee_count = "N/A"
        founding_year = "N/A"
        automation_pitch = "N/A"

        # Using regex to extract information based on the labels in the prompt
        summary_match = re.search(r'Company Summary:\s*(.*?)(?=\n(?:Industry:|Target Customer:|Employee Count:|Founding Year:|AI Automation Idea for|$))', response_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
        
        industry_match = re.search(r'Industry:\s*(.*?)(?=\n(?:Target Customer:|Employee Count:|Founding Year:|AI Automation Idea for|$))', response_text, re.DOTALL | re.IGNORECASE)
        if industry_match:
            industry = industry_match.group(1).strip()

        target_customer_match = re.search(r'Target Customer:\s*(.*?)(?=\n(?:Employee Count:|Founding Year:|AI Automation Idea for|$))', response_text, re.DOTALL | re.IGNORECASE)
        if target_customer_match:
            target_customer = target_customer_match.group(1).strip()

        employee_count_match = re.search(r'Employee Count:\s*(.*?)(?=\n(?:Founding Year:|AI Automation Idea for|$))', response_text, re.DOTALL | re.IGNORECASE)
        if employee_count_match:
            employee_count = employee_count_match.group(1).strip()

        founding_year_match = re.search(r'Founding Year:\s*(.*?)(?=\n(?:AI Automation Idea for|$))', response_text, re.DOTALL | re.IGNORECASE)
        if founding_year_match:
            founding_year = founding_year_match.group(1).strip()
            
        automation_pitch_match = re.search(r'AI Automation Idea for .*?:\s*(.*)', response_text, re.DOTALL | re.IGNORECASE)
        if automation_pitch_match:
            automation_pitch = automation_pitch_match.group(1).strip()

        return {
            "summary": summary,
            "industry": industry,
            "target_customer": target_customer,
            "employee_count": employee_count,
            "founding_year": founding_year,
            "automation_pitch": automation_pitch
        }

    except Exception as e:
        st.session_state.log_messages.append(f"LLM Error for {company_name}: {e}")
        return {
            "summary": "LLM Error",
            "industry": "LLM Error",
            "target_customer": "LLM Error",
            "employee_count": "LLM Error",
            "founding_year": "LLM Error",
            "automation_pitch": "LLM Error"
        }

# --- Main Enrichment Process Function ---
def run_enrichment(input_df):
    """
    Main function to orchestrate the lead enrichment process.
    """
    enriched_data = []
    
    st.session_state.log_messages = [] # Initialize log messages for this run

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty() # For displaying log messages

    # Add new columns if they don't exist
    for col in ['website', 'industry', 'summary_from_llm', 'target_customer', 'employee_count', 'founding_year', 'automation_pitch_from_llm']:
        if col not in input_df.columns:
            input_df[col] = None

    for i, row in input_df.iterrows():
        company_name = row['company_name']
        status_text.text(f"Processing company {i+1}/{len(input_df)}: {company_name}")
        log_area.text("\n".join(st.session_state.log_messages[-5:])) # Show last 5 log messages

        # 1. Find Website
        website = get_company_website(company_name)
        input_df.at[i, 'website'] = website
        
        if website:
            # 2. Scrape Website Text
            website_text = scrape_website_text(website)
            if website_text:
                # 3. Get LLM Insights
                llm_data = get_llm_insights(website_text, company_name)
                input_df.at[i, 'industry'] = llm_data['industry']
                input_df.at[i, 'summary_from_llm'] = llm_data['summary']
                input_df.at[i, 'target_customer'] = llm_data['target_customer']
                input_df.at[i, 'employee_count'] = llm_data['employee_count']
                input_df.at[i, 'founding_year'] = llm_data['founding_year']
                input_df.at[i, 'automation_pitch_from_llm'] = llm_data['automation_pitch']
            else:
                input_df.at[i, ['industry', 'summary_from_llm', 'target_customer', 'employee_count', 'founding_year', 'automation_pitch_from_llm']] = "N/A (Scrape Failed)"
        else:
            input_df.at[i, ['industry', 'summary_from_llm', 'target_customer', 'employee_count', 'founding_year', 'automation_pitch_from_llm']] = "N/A (Website Not Found)"

        progress_bar.progress((i + 1) / len(input_df))
        time.sleep(1.5) # Increased delay to be polite to websites and LLM API (adjust as needed)

    status_text.text("Enrichment complete!")
    log_area.empty() # Clear log messages after completion
    return input_df # Return the updated DataFrame

# --- Streamlit UI Layout ---

st.title("🤖 AI-Powered Lead Enrichment Bot")
st.markdown("""
This tool automates the process of enriching company lead data.
Upload a CSV with company names, and the bot will:
* Discover company websites.
* Scrape relevant information.
* Use Cohere AI to generate comprehensive summaries, identify target customers, suggest AI automation pitches, and extract additional company details like employee count and founding year.
""")

# Initialize session state for logging
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

st.subheader("1. Upload your Company CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with a 'company_name' column", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if 'company_name' not in input_df.columns:
            st.error("Error: The uploaded CSV must contain a column named 'company_name'.")
            st.stop()
        
        st.success("CSV uploaded successfully!")
        st.write("Preview of your input data:")
        st.dataframe(input_df[['company_name']].head())
        st.info(f"Detected {len(input_df)} companies for enrichment.")

        if st.button("🚀 Start Lead Enrichment"):
            with st.spinner("Processing companies and enriching data... This may take some time."):
                final_enriched_df = run_enrichment(input_df.copy())
            
            st.subheader("✅ Enrichment Complete!")
            st.write("Here is your enriched lead data:")
            st.dataframe(final_enriched_df)

            csv_buffer = io.StringIO()
            final_enriched_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Enriched Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="enriched_leads.csv",
                mime="text/csv"
            )
            st.success("Your enriched CSV file is ready for download!")

    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded CSV file is empty.")
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the file: {e}")

st.markdown("---")
st.markdown("""
**How the Bot Works:**
1.  **Company Name to Website:** Attempts to intelligently guess and validate the official website for each company based on common URL patterns.
2.  **Website Scraping:** Fetches the main textual content from the confirmed company homepage using `requests` and `BeautifulSoup`.
3.  **AI Analysis with Cohere:** The scraped text is sent to the Cohere LLM with carefully crafted prompts to:
    * Generate a concise **Company Summary**.
    * Identify the company's primary **Industry**.
    * Determine their **Target Customer**.
    * Extract **Employee Count** and **Founding Year** (if available on the website).
    * Propose a **Custom AI Automation Idea** that 'QF Innovate' could pitch, highlighting a problem and an AI-driven solution.

**Important Notes:**
* **API Key:** Ensure your `COHERE_API_KEY` is correctly set in a `.env` file in the same directory as this script.
* **Rate Limits:** The bot includes a small delay between requests to avoid hitting API rate limits and to be polite to websites and the LLM API.
* **Scraping Robustness:** Website structures vary, so the scraping logic is a heuristic. Some websites might not yield optimal results.
* **LLM Variability:** While prompt engineering is used, LLM output can sometimes be inconsistent, especially for data points like employee count or founding year which might not be clearly present on all website homepages.
""")

st.markdown("---")
st.caption("Developed by Your Coding Partner for Lead Enrichment Automation")