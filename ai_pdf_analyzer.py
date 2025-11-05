# 🚀 ULTIMATE AI PDF ANALYZER WITH GEMINI-2.5-PRO
# Complete Application: Extract everything from PDF, Ask Questions, Get AI Analysis
# Single File: ai_pdf_analyzer.py

import streamlit as st
import pymupdf
import pandas as pd
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json
from pathlib import Path
import uuid
import google.generativeai as genai
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🤖 AI PDF Analyzer with Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-container { max-width: 1200px; margin: 0 auto; }
    .header-title { color: #1f77b4; font-size: 3rem; font-weight: bold; text-align: center; }
    .feature-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1.5rem; border-radius: 0.75rem; color: white; margin: 0.5rem 0; }
    .ai-response { background: #e8f4f8; padding: 1.5rem; border-radius: 0.75rem; 
                   border-left: 5px solid #667eea; margin: 1rem 0; }
    .data-table { background: #f9f9f9; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .success-box { background: #d4edda; padding: 1rem; border-radius: 0.5rem; color: #155724; }
    .error-box { background: #f8d7da; padding: 1rem; border-radius: 0.5rem; color: #721c24; }
    .image-gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                     gap: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'current_doc_id' not in st.session_state:
    st.session_state.current_doc_id = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ============================================================================
# GEMINI API INITIALIZATION
# ============================================================================
def init_gemini(api_key):
    """Initialize Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        return model
    except Exception as e:
        st.error(f"❌ Gemini API Error: {str(e)}")
        return None

# ============================================================================
# PDF EXTRACTION SERVICES
# ============================================================================

class AdvancedPDFExtractor:
    """Extract text, tables, images, and metadata from PDF"""
    
    @staticmethod
    def extract_all(pdf_path):
        """Complete extraction: text, images, tables, metadata"""
        try:
            doc = pymupdf.open(pdf_path)
            
            data = {
                'metadata': {
                    'title': doc.metadata.get('title', 'Unknown') if doc.metadata else 'Unknown',
                    'author': doc.metadata.get('author', 'Unknown') if doc.metadata else 'Unknown',
                    'pages': len(doc),
                    'size_kb': os.path.getsize(pdf_path) / 1024,
                },
                'pages': {},
                'images': [],
                'tables': [],
                'full_text': "",
            }
            
            # Extract page-by-page
            for page_num, page in enumerate(doc):
                page_data = {
                    'number': page_num + 1,
                    'text': '',
                    'images': [],
                    'tables': [],
                    'width': page.rect.width,
                    'height': page.rect.height,
                }
                
                # Extract text
                text = page.get_text()
                page_data['text'] = text
                data['full_text'] += f"\n--- PAGE {page_num + 1} ---\n{text}"
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = pymupdf.Pixmap(doc, xref)
                        
                        if pix.n < 5:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                        else:  # CMYK
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                            img_data = pix.tobytes("ppm")
                        
                        img_info = {
                            'page': page_num + 1,
                            'index': img_index,
                            'data': base64.b64encode(img_data).decode(),
                            'format': 'ppm',
                        }
                        page_data['images'].append(img_info)
                        data['images'].append(img_info)
                    except:
                        pass
                
                # Extract tables (basic detection via structured text)
                if '\t' in text or '|' in text or '\n' in text.split('\n')[0] * 3:
                    table_info = {
                        'page': page_num + 1,
                        'content': text[:500],
                    }
                    page_data['tables'].append(table_info)
                    data['tables'].append(table_info)
                
                data['pages'][page_num + 1] = page_data
            
            doc.close()
            return data
        
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    @staticmethod
    def extract_structured_text(full_text):
        """Extract structured data: numbers, emails, dates, etc."""
        import re
        
        structured = {
            'numbers': re.findall(r'\b\d+(?:\.\d+)?\b', full_text),
            'percentages': re.findall(r'\b\d+(?:\.\d+)?%\b', full_text),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', full_text),
            'phone_numbers': re.findall(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', full_text),
            'currency': re.findall(r'(?:$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d{2})?', full_text),
        }
        
        return structured


class GeminiAnalyzer:
    """Analyze PDF content using Gemini-2.5-Pro"""
    
    def __init__(self, model):
        self.model = model
    
    def ask_question(self, question, pdf_text, conversation_history=None):
        """Ask question about PDF content"""
        try:
            # Build context
            context = f"""You are an expert PDF analyzer. Analyze the following PDF content and answer the user's question accurately.

PDF CONTENT:
{pdf_text[:10000]}  # Use first 10k chars to stay within limits

CONVERSATION HISTORY:
{json.dumps(conversation_history[-5:]) if conversation_history else "No previous questions"}

USER QUESTION: {question}

Please provide a detailed, accurate answer based on the PDF content."""
            
            response = self.model.generate_content(context)
            return response.text
        
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def analyze_numerical_data(self, pdf_text):
        """Extract and analyze numerical data"""
        try:
            prompt = f"""Analyze this PDF text and extract all numerical data. Provide:
1. List of all numbers found
2. Key statistics (totals, averages, ranges)
3. Any trends or patterns
4. Currency amounts
5. Percentages

PDF TEXT:
{pdf_text[:8000]}

Format response as structured JSON."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing: {str(e)}"
    
    def summarize_pdf(self, pdf_text):
        """Generate comprehensive summary"""
        try:
            prompt = f"""Summarize this PDF in the following structure:
1. Executive Summary (2-3 sentences)
2. Key Points (5-7 bullet points)
3. Main Findings
4. Recommendations
5. Important Data Points

PDF TEXT:
{pdf_text[:8000]}"""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error summarizing: {str(e)}"
    
    def extract_key_entities(self, pdf_text):
        """Extract key entities: names, dates, organizations, etc."""
        try:
            prompt = f"""Extract and categorize all key entities from this PDF:
- People/Names
- Organizations
- Dates/Times
- Locations
- Products/Services
- Technical Terms
- Important Concepts

PDF TEXT:
{pdf_text[:8000]}

Format as structured list."""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error extracting: {str(e)}"
    
    def compare_sections(self, section1_text, section2_text):
        """Compare two sections of text"""
        try:
            prompt = f"""Compare these two sections and highlight:
1. Similarities
2. Differences
3. Contradictions
4. Evolution/Changes

SECTION 1:
{section1_text[:4000]}

SECTION 2:
{section2_text[:4000]}"""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error comparing: {str(e)}"


# ============================================================================
# SIDEBAR - API KEY & NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("# 🤖 AI PDF Analyzer")
    st.markdown("**Powered by Gemini-2.5-Pro**")
    st.markdown("---")
    
    # API Key Setup
    st.markdown("### 🔑 API Configuration")
    api_key_input = st.text_input(
        "Enter Gemini API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Get your key from https://ai.google.dev"
    )
    
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.success("✅ API key configured")
    
    st.markdown("---")
    st.markdown("### 📁 Uploaded Documents")
    
    if st.session_state.documents:
        for doc_id, doc_info in st.session_state.documents.items():
            if st.button(f"📄 {doc_info['filename']}", use_container_width=True):
                st.session_state.current_doc_id = doc_id
                st.rerun()
    else:
        st.info("No documents uploaded yet")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    analysis_mode = st.selectbox(
        "Default Analysis Mode:",
        ["Chat", "Summary", "Numerical Data", "Entities", "All Analyses"]
    )

# ============================================================================
# MAIN PAGE - NAVIGATION
# ============================================================================

main_page = st.radio(
    "Select Feature:",
    ["📤 Upload & Extract", "🔍 View Content", "💬 AI Chat", "📊 Analysis", "📈 Advanced Tools"],
    horizontal=True
)

# ============================================================================
# PAGE 1: UPLOAD & EXTRACT
# ============================================================================

if main_page == "📤 Upload & Extract":
    st.markdown("# 📤 Upload PDF & Extract All Content")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        st.info(f"📄 File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 Extract Everything", use_container_width=True):
            with st.spinner("Extracting content..."):
                # Save to temp
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract
                extractor = AdvancedPDFExtractor()
                data = extractor.extract_all("temp.pdf")
                
                # Store
                doc_id = str(uuid.uuid4())
                st.session_state.documents[doc_id] = {
                    'filename': uploaded_file.name,
                    'data': data,
                    'upload_time': datetime.now().isoformat(),
                }
                st.session_state.current_doc_id = doc_id
                
                # Show results
                st.success("✅ Extraction Complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📖 Pages", data['metadata']['pages'])
                with col2:
                    st.metric("🖼️ Images", len(data['images']))
                with col3:
                    st.metric("📋 Tables", len(data['tables']))
                with col4:
                    st.metric("📝 Characters", len(data['full_text']))
                
                st.markdown("---")
                st.markdown("### 📊 Structured Data Found")
                
                structured = extractor.extract_structured_text(data['full_text'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Numbers:** {len(structured['numbers'])} found")
                    if structured['numbers'][:5]:
                        st.caption(f"Examples: {', '.join(structured['numbers'][:5])}")
                
                with col2:
                    st.write(f"**Percentages:** {len(structured['percentages'])} found")
                    if structured['percentages'][:5]:
                        st.caption(f"Examples: {', '.join(structured['percentages'][:5])}")
                
                with col3:
                    st.write(f"**Currency:** {len(structured['currency'])} found")
                    if structured['currency'][:5]:
                        st.caption(f"Examples: {', '.join(structured['currency'][:5])}")
                
                st.info("✅ Document ready for analysis! Go to 'View Content' or 'AI Chat'")

# ============================================================================
# PAGE 2: VIEW CONTENT
# ============================================================================

elif main_page == "🔍 View Content":
    st.markdown("# 🔍 View Extracted Content")
    
    if not st.session_state.current_doc_id:
        st.warning("⚠️ Please upload and select a document first")
    else:
        doc = st.session_state.documents[st.session_state.current_doc_id]
        data = doc['data']
        
        st.markdown(f"### 📄 {doc['filename']}")
        
        # Tabs for different content types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text", "🖼️ Images", "📋 Tables", "📊 Data", "📋 Metadata"])
        
        # TAB 1: TEXT CONTENT
        with tab1:
            st.markdown("### Full Text Content")
            
            # Page selector
            page_num = st.selectbox("Select Page:", list(data['pages'].keys()))
            page_content = data['pages'][page_num]['text']
            
            st.text_area(
                "Page Content:",
                value=page_content,
                height=300,
                disabled=True
            )
        
        # TAB 2: IMAGES
        with tab2:
            st.markdown("### Extracted Images")
            
            if data['images']:
                col_count = 3
                cols = st.columns(col_count)
                
                for idx, img_info in enumerate(data['images']):
                    col = cols[idx % col_count]
                    with col:
                        try:
                            # Decode image
                            img_bytes = base64.b64decode(img_info['data'])
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption=f"Page {img_info['page']} - Image {img_info['index']}")
                        except:
                            st.write(f"Image from Page {img_info['page']}")
            else:
                st.info("No images found in PDF")
        
        # TAB 3: TABLES
        with tab3:
            st.markdown("### Extracted Tables")
            
            if data['tables']:
                for idx, table_info in enumerate(data['tables']):
                    with st.expander(f"Table {idx + 1} - Page {table_info['page']}"):
                        st.code(table_info['content'])
            else:
                st.info("No tables detected")
        
        # TAB 4: STRUCTURED DATA
        with tab4:
            st.markdown("### Structured Data Analysis")
            
            structured = AdvancedPDFExtractor.extract_structured_text(data['full_text'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Numbers")
                st.write(f"Total: {len(structured['numbers'])}")
                if structured['numbers']:
                    st.dataframe(pd.DataFrame({
                        'Number': structured['numbers'][:20]
                    }))
                
                st.markdown("#### Percentages")
                st.write(f"Total: {len(structured['percentages'])}")
                if structured['percentages']:
                    st.write(structured['percentages'][:10])
            
            with col2:
                st.markdown("#### Currency")
                st.write(f"Total: {len(structured['currency'])}")
                if structured['currency']:
                    st.write(structured['currency'][:10])
                
                st.markdown("#### Emails")
                st.write(f"Total: {len(structured['emails'])}")
                if structured['emails']:
                    st.write(structured['emails'])
        
        # TAB 5: METADATA
        with tab5:
            st.markdown("### Document Metadata")
            
            metadata = data['metadata']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Title", metadata.get('title', 'N/A'))
                st.metric("Pages", metadata['pages'])
            
            with col2:
                st.metric("Author", metadata.get('author', 'N/A'))
                st.metric("Size", f"{metadata['size_kb']:.1f} KB")
            
            st.json(metadata)

# ============================================================================
# PAGE 3: AI CHAT
# ============================================================================

elif main_page == "💬 AI Chat":
    st.markdown("# 💬 Ask Questions to Your PDF")
    
    if not st.session_state.api_key:
        st.error("❌ Please configure Gemini API key in sidebar")
    elif not st.session_state.current_doc_id:
        st.warning("⚠️ Please upload and select a document first")
    else:
        doc = st.session_state.documents[st.session_state.current_doc_id]
        data = doc['data']
        
        # Initialize Gemini
        model = init_gemini(st.session_state.api_key)
        
        if model:
            analyzer = GeminiAnalyzer(model)
            
            st.markdown(f"### Chatting about: {doc['filename']}")
            
            # Chat display
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {message['content']}")
            
            # Input
            st.markdown("---")
            question = st.text_input("Ask a question about the PDF:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📤 Send Question", use_container_width=True) and question:
                    with st.spinner("🤔 Analyzing..."):
                        # Get response
                        response = analyzer.ask_question(
                            question,
                            data['full_text'][:15000],
                            st.session_state.chat_history
                        )
                        
                        # Store in history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question
                        })
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col3:
                if st.button("💾 Save Chat", use_container_width=True):
                    chat_text = "\n".join([f"{m['role']}: {m['content']}\n" for m in st.session_state.chat_history])
                    st.download_button(
                        "Download Chat",
                        chat_text,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )

# ============================================================================
# PAGE 4: ANALYSIS
# ============================================================================

elif main_page == "📊 Analysis":
    st.markdown("# 📊 AI-Powered PDF Analysis")
    
    if not st.session_state.api_key:
        st.error("❌ Please configure Gemini API key in sidebar")
    elif not st.session_state.current_doc_id:
        st.warning("⚠️ Please upload and select a document first")
    else:
        doc = st.session_state.documents[st.session_state.current_doc_id]
        data = doc['data']
        
        model = init_gemini(st.session_state.api_key)
        
        if model:
            analyzer = GeminiAnalyzer(model)
            
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                ["Summary", "Numerical Data", "Key Entities", "Compare Sections", "Custom Analysis"]
            )
            
            if analysis_type == "Summary":
                if st.button("📝 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = analyzer.summarize_pdf(data['full_text'][:10000])
                        st.markdown(summary)
                        
                        if st.button("💾 Download Summary"):
                            st.download_button(
                                "Download",
                                summary,
                                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            )
            
            elif analysis_type == "Numerical Data":
                if st.button("🔢 Analyze Numerical Data"):
                    with st.spinner("Analyzing numerical data..."):
                        analysis = analyzer.analyze_numerical_data(data['full_text'][:10000])
                        st.markdown(analysis)
            
            elif analysis_type == "Key Entities":
                if st.button("🏷️ Extract Key Entities"):
                    with st.spinner("Extracting entities..."):
                        entities = analyzer.extract_key_entities(data['full_text'][:10000])
                        st.markdown(entities)
            
            elif analysis_type == "Compare Sections":
                st.markdown("### Compare Two Sections")
                
                col1, col2 = st.columns(2)
                with col1:
                    page1 = st.selectbox("Select first page:", list(data['pages'].keys()))
                with col2:
                    page2 = st.selectbox("Select second page:", list(data['pages'].keys()), key="page2")
                
                if st.button("🔄 Compare"):
                    with st.spinner("Comparing..."):
                        section1 = data['pages'][page1]['text']
                        section2 = data['pages'][page2]['text']
                        
                        comparison = analyzer.compare_sections(section1, section2)
                        st.markdown(comparison)
            
            elif analysis_type == "Custom Analysis":
                st.markdown("### Custom Analysis")
                
                custom_prompt = st.text_area(
                    "Enter your analysis request:",
                    placeholder="e.g., Find all financial metrics and their values"
                )
                
                if st.button("🔍 Analyze"):
                    with st.spinner("Analyzing..."):
                        custom_prompt_full = f"""Based on this PDF content, {custom_prompt}
                        
PDF CONTENT:
{data['full_text'][:10000]}"""
                        
                        response = model.generate_content(custom_prompt_full)
                        st.markdown(response.text)

# ============================================================================
# PAGE 5: ADVANCED TOOLS
# ============================================================================

elif main_page == "📈 Advanced Tools":
    st.markdown("# 📈 Advanced Tools & Features")
    
    if not st.session_state.current_doc_id:
        st.warning("⚠️ Please upload and select a document first")
    else:
        doc = st.session_state.documents[st.session_state.current_doc_id]
        data = doc['data']
        
        tool = st.selectbox(
            "Select Tool:",
            ["Statistics", "Text Analytics", "Comparison Tool", "Batch Q&A", "Export Data"]
        )
        
        if tool == "Statistics":
            st.markdown("### Document Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pages", data['metadata']['pages'])
            with col2:
                st.metric("Total Characters", len(data['full_text']))
            with col3:
                st.metric("Avg Chars/Page", len(data['full_text']) // data['metadata']['pages'])
            with col4:
                word_count = len(data['full_text'].split())
                st.metric("Word Count", word_count)
            
            st.markdown("---")
            st.markdown("### Data Distribution")
            
            structured = AdvancedPDFExtractor.extract_structured_text(data['full_text'])
            stats_df = pd.DataFrame({
                'Data Type': ['Numbers', 'Percentages', 'Currency', 'Emails', 'URLs'],
                'Count': [
                    len(structured['numbers']),
                    len(structured['percentages']),
                    len(structured['currency']),
                    len(structured['emails']),
                    len(structured['urls']),
                ]
            })
            
            st.bar_chart(stats_df.set_index('Data Type'))
        
        elif tool == "Text Analytics":
            st.markdown("### Text Analytics")
            
            # Readability metrics
            sentences = len(data['full_text'].split('.'))
            words = len(data['full_text'].split())
            paragraphs = len(data['full_text'].split('\n\n'))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sentences", sentences)
            with col2:
                st.metric("Words", words)
            with col3:
                st.metric("Paragraphs", paragraphs)
            with col4:
                avg_word_length = len(data['full_text']) / max(words, 1)
                st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        
        elif tool == "Export Data":
            st.markdown("### Export PDF Data")
            
            export_options = st.multiselect(
                "Select data to export:",
                ["Full Text", "Metadata", "Structured Data", "Chat History"]
            )
            
            if st.button("📥 Prepare Export"):
                export_data = {}
                
                if "Full Text" in export_options:
                    export_data['full_text'] = data['full_text']
                if "Metadata" in export_options:
                    export_data['metadata'] = data['metadata']
                if "Structured Data" in export_options:
                    export_data['structured'] = AdvancedPDFExtractor.extract_structured_text(data['full_text'])
                if "Chat History" in export_options:
                    export_data['chat_history'] = st.session_state.chat_history
                
                export_json = json.dumps(export_data, indent=2)
                
                st.download_button(
                    "📥 Download as JSON",
                    export_json,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 <b>AI PDF Analyzer v1.0</b> | Powered by Gemini-2.5-Pro</p>
    <p>Extract everything • Search semantically • Ask questions • Get intelligent analysis</p>
</div>
""", unsafe_allow_html=True)

# Clean up temp file
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
