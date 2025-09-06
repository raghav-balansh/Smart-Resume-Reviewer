import streamlit as st
import pdfplumber
import openai
import google.generativeai as genai
from io import BytesIO
import json
import re
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import os
from typing import Dict, List, Tuple, Optional
import hashlib
import tempfile

# Configuration
class Config:
    """Application configuration"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    SUPPORTED_FORMATS = ['.pdf', '.txt']
    
    # ATS Keywords Database
    ATS_KEYWORDS = {
        "data_scientist": [
            "python", "machine learning", "sql", "statistics", "tensorflow",
            "pytorch", "pandas", "numpy", "scikit-learn", "deep learning",
            "data visualization", "tableau", "power bi", "aws", "azure",
            "natural language processing", "big data", "hadoop", "spark"
        ],

        "software_engineer": [
            "java", "python", "javascript", "react", "node.js", "sql",
            "git", "docker", "kubernetes", "aws", "agile", "ci/cd",
            "rest api", "microservices", "cloud computing", "c++", "spring boot"
        ],

        "full_stack_developer": [
            "html", "css", "javascript", "react", "angular", "vue.js",
            "node.js", "express.js", "django", "java", "spring boot",
            "graphql", "rest api", "sql", "mongodb", "firebase",
            "git", "docker", "aws", "kubernetes", "devops", "ci/cd"
        ],

        "data_analyst": [
            "sql", "excel", "power bi", "tableau", "python", "r",
            "pandas", "numpy", "data visualization", "statistics",
            "business intelligence", "google analytics", "a/b testing"
        ],

        "devops_engineer": [
            "devops", "ci/cd", "docker", "kubernetes", "ansible", "terraform",
            "jenkins", "git", "bash", "shell scripting", "aws", "azure", "gcp",
            "monitoring", "prometheus", "grafana", "infrastructure as code"
        ],

        "cybersecurity_analyst": [
            "network security", "firewalls", "penetration testing", "siem",
            "incident response", "vulnerability management", "risk assessment",
            "encryption", "ethical hacking", "wireshark", "nmap", "ids/ips",
            "compliance", "iso 27001", "gdpr"
        ],

        "ui_ux_designer": [
            "ui design", "ux design", "wireframing", "prototyping", "figma",
            "adobe xd", "sketch", "user research", "usability testing",
            "interaction design", "information architecture", "design systems",
            "responsive design", "accessibility"
        ],

        "product_manager": [
            "product strategy", "roadmap", "agile", "scrum", "stakeholder management",
            "data analysis", "user research", "kpi", "metrics", "jira",
            "product lifecycle", "market analysis", "competitive analysis",
            "go-to-market", "a/b testing"
        ],

        "marketing_manager": [
            "seo", "sem", "google analytics", "social media", "content marketing",
            "email marketing", "campaign management", "roi", "conversion optimization",
            "marketing automation", "hubspot", "salesforce", "branding", "ppc"
        ],

        "general": [
            "leadership", "communication", "problem-solving", "teamwork", "analytical",
            "project management", "time management", "adaptability",
            "microsoft office", "excel", "powerpoint", "presentation skills"
        ]
    }


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'ats_score' not in st.session_state:
        st.session_state.ats_score = None
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None
    if 'optimized_resume' not in st.session_state:
        st.session_state.optimized_resume = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

# PDF Processing
class PDFProcessor:
    # handle pdf extraction and processing
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        #text extraction by using the pdfplumber
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        #extract common resume sections
        sections = {
            'contact': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'achievements': '',

        }
        
        # Common section headers
        section_patterns = {
            'contact': r'(?i)(contact|email|phone|\+91|tel|address|linkedin|github|portfolio)',
            'summary': r'(?i)(summary|objective|profile|about|overview|highlights)',
            'experience': r'(?i)(experience|employment|work history|professional experience|internship|industry experience|relevant experience)',
            'education': r'(?i)(education|academic|qualification|training & education)',
            'skills': r'(?i)(skills|technical skills|competencies|expertise|tools & technologies|soft skills|other skills)',
            'projects': r'(?i)(projects|portfolio|projects & responsibilities|key projects|projects & accomplishments)',
            'certifications': r'(?i)(certifications?|licenses?|awards?|certification)',
            'achievements': r'(?i)(achievements|accomplishments|awards|recognitions|honors|awards & recognition)'
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            # Check if line matches any section header
            for section, pattern in section_patterns.items():
                if re.search(pattern, line):
                    current_section = section
                    break
            
            # Add line to current section
            if current_section:
                sections[current_section] += line + '\n'
        
        return sections

# ATS Analysis Engine
class ATSAnalyzer:
    #analyze resume for ATS score
    
    def __init__(self, api_key: str, use_google: bool = False):
        self.use_google = use_google
        if use_google:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            openai.api_key = api_key
            self.max_tokens = 1000
            self.temperature = 0.7
            self.model = 'gpt-3.5-turbo'
        
    def calculate_ats_score(self, resume_text: str, job_role: str, job_description: str = "") -> Dict:
        #calculate comprehensive ATS score
        
        # Extract sections
        sections = PDFProcessor.extract_sections(resume_text)
        
        # Initialize scoring components
        scores = {
            'keyword_match': 0,
            'formatting': 0,
            'section_completeness': 0,
            'readability': 0,
            'length_appropriateness': 0
        }
        
        # 1. Keyword Analysis (30% weight)
        keywords = Config.ATS_KEYWORDS.get(job_role.lower().replace(' ', '_'), Config.ATS_KEYWORDS['general'])
        if job_description:

            #extract additional keywords from job description
            job_keywords = self._extract_keywords_from_description(job_description)
            keywords.extend(job_keywords)
        
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in resume_text.lower())
        scores['keyword_match'] = min((keyword_count / len(keywords)) * 100, 100) if keywords else 0
        
        
        # 2. Formatting Analysis (20% weight)
        formatting_checks = {
            'has_bullets': bool(re.search(r'[•·▪▫◦‣⁃●○■□◆◇➔→⇒➤]|^\s*[-*+>]\s+', resume_text, re.MULTILINE)),
            'proper_sections': sum(1 for s in sections.values() if s.strip()) >= 4,
            'no_tables': 'table' not in resume_text.lower(),
            'no_images': True,  # Assuming text-based resume
            'consistent_formatting': self._check_formatting_consistency(resume_text)
        }
        scores['formatting'] = (sum(formatting_checks.values()) / len(formatting_checks)) * 100
        
        # 3. Section Completeness (30% weight)
        essential_sections = ['contact', 'experience', 'education', 'skills', 'projects', 'certifications', 'achievements']
        present_sections = sum(1 for s in essential_sections if sections.get(s, '').strip())
        scores['section_completeness'] = (present_sections / len(essential_sections)) * 100
        
        # 4. Readability (10% weight)
        scores['readability'] = self._calculate_readability(resume_text)
        
        # 5. Length Appropriateness (10% weight)
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:
            scores['length_appropriateness'] = 100
        elif 200 <= word_count < 300 or 800 < word_count <= 1000:
            scores['length_appropriateness'] = 70
        else:
            scores['length_appropriateness'] = 40
        
        # Calculate weighted total
        weights = {
            'keyword_match': 0.3,
            'formatting': 0.2,
            'section_completeness': 0.3,
            'readability': 0.1,
            'length_appropriateness': 0.1
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        # Find missing keywords
        missing_keywords = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
        
        return {
            'total_score': round(total_score, 1),
            'component_scores': scores,
            'missing_keywords': missing_keywords[:10],  # Top 10 missing
            'sections_analysis': sections
        }
    
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        # Simple keyword extraction - can be enhanced with NLP
        tech_pattern = r"""\b(?:
            # Programming Languages
                python|java|javascript|typescript|c\+\+|c\#|go|rust|php|ruby|perl|scala|swift|kotlin|objective-c|
                dart|haskell|ocaml|clojure|lisp|prolog|elixir|erlang|

                # Web & Mobile Frameworks
                react|angular|vue|svelte|next\.js|nuxt\.js|react native|flutter|

                # Cloud Platforms
                aws|azure|gcp|firebase|digitalocean|heroku|

                # DevOps & Infra
                docker|kubernetes|jenkins|terraform|ansible|git|ci/cd|

                # Databases
                sql|mysql|postgresql|oracle|mongodb|cassandra|redis|neo4j|

                # Data & AI
                data engineering|data science|machine learning|deep learning|artificial intelligence|
                natural language processing|nlp|computer vision|robotics|

                # Security & Blockchain
                cybersecurity|ethical hacking|penetration testing|network security|firewalls|
                siem|incident response|vulnerability management|risk assessment|encryption|
                wireshark|nmap|ids/ips|iso 27001|gdpr|compliance|cryptography|blockchain)\b"""
        
        skills_pattern = r"""\b(?:
                # Soft Skills
                leadership|communication|analytical|strategic|innovative|problem-solving|
                teamwork|time management|adaptability|

                # Business & Productivity
                microsoft office|excel|powerpoint|word|outlook|presentation skills|
                data analysis|data visualization|data modeling|data warehousing|data mining)\b"""

        tech_keywords = re.findall(tech_pattern, description.lower())
        skill_keywords = re.findall(skills_pattern, description.lower())
        
        return list(set(tech_keywords + skill_keywords))
    

    def _check_formatting_consistency(self, text: str) -> bool:
        #Check if formatting is consistent
        lines = text.split('\n')
        bullet_styles = set()
        
        for line in lines:
            if re.match(r'^\s*[•·▪▫◦‣⁃●○■□◆◇➔→⇒➤*]\s+', line):
                bullet = re.match(r'^\s*([•·▪▫◦‣⁃●○■□◆◇➔→⇒➤*])\s+', line).group(1)
                bullet_styles.add(bullet)
        
        # Consistent if using 0-1 bullet styles
        return len(bullet_styles) <= 1
    
    def _calculate_readability(self, text: str) -> float:
        #Calculate readability score
        sentences = re.split(r'[.!?;:]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 50
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability: shorter sentences are better for ATS
        if avg_sentence_length < 15:
            return 100
        elif avg_sentence_length < 20:
            return 80
        elif avg_sentence_length < 25:
            return 60
        else:
            return 40

# LLM Feedback Generator
class FeedbackGenerator:
    #Generate detailed feedback using LLMs
    
    def __init__(self, openai_key: str = None, google_key: str = None):
        self.openai_key = openai_key
        self.google_key = google_key
        if google_key:
            genai.configure(api_key=google_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    
    def generate_feedback(self, resume_text: str, ats_analysis: Dict, job_role: str, job_description: str = "") -> str:
        """Generate comprehensive feedback using LLM"""
        
        prompt = f"""
        As an expert resume reviewer and ATS specialist, provide detailed feedback for this resume.
        
        Job Role: {job_role}
        {"Job Description: " + job_description if job_description else ""}
        
        ATS Score: {ats_analysis['total_score']}/100
        Component Scores:
        - Keyword Match: {ats_analysis['component_scores']['keyword_match']:.1f}%
        - Formatting: {ats_analysis['component_scores']['formatting']:.1f}%
        - Section Completeness: {ats_analysis['component_scores']['section_completeness']:.1f}%
        - Readability: {ats_analysis['component_scores']['readability']:.1f}%
        - Length: {ats_analysis['component_scores']['length_appropriateness']:.1f}%
        
        Missing Keywords: {', '.join(ats_analysis['missing_keywords'][:5]) if ats_analysis['missing_keywords'] else 'Your Resume Is AWSOME..🎉'}
        
        Resume Text:
        {resume_text[:2000]}...
        
        Please provide:
        1. **Overall Assessment** (2-3 sentences)
        2. **Strengths** (3-4 bullet points)
        3. **Areas for Improvement** (4-5 specific suggestions)
        4. **ATS Optimization Tips** (3-4 actionable items)
        5. **Content Recommendations** (2-3 suggestions for missing elements)
        
        Be specific, constructive, and actionable in your feedback.
        """
        
        try:
            if self.google_key:
                response = self.gemini_model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    return "Error: Unable to generate feedback from Google AI. Please try again."
                    
            elif self.openai_key:
                # Use legacy OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            else:
                return "⚠️ API keys not configured. Please add your OpenAI or Google API key to generate detailed feedback."
                
        except openai.APIError as e:
            return f"OpenAI API Error: {str(e)}. Please check your API key and try again."
        except Exception as e:
            return f"Error generating feedback: {str(e)}. Please try again or contact support."
    
    def optimize_resume(self, resume_text: str, feedback: str, job_role: str, ats_analysis: Dict) -> str:
        """Generate an optimized version of the resume"""
        
        prompt = f"""
        As an expert resume writer, create an optimized version of this resume for the {job_role} position.
        
        Current ATS Score: {ats_analysis['total_score']}/100
        Missing Keywords to Include: {', '.join(ats_analysis['missing_keywords'][:8])}
        
        Original Resume:
        {resume_text}
        
        Feedback to Address:
        {feedback[:500]}
        
        Please rewrite this resume with:
        1. All missing keywords naturally integrated
        2. Improved formatting for ATS scanning
        3. Stronger action verbs and quantified achievements
        4. Clear section headers
        5. Optimized length (aim for 500-700 words)
        
        Return ONLY the optimized resume text, properly formatted with clear sections.
        """
        
        try:
            if self.google_key:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            elif self.openai_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.6
                )
                return response.choices[0].message.content
                
            else:
                return resume_text  # Return original if no API
        except Exception as e:
            return resume_text

# PDF Generator
class PDFGenerator:
    """Generate enhanced PDF resumes"""
    
    @staticmethod
    def create_optimized_pdf(resume_text: str, filename: str = "optimized_resume.pdf") -> BytesIO:
        """Create a professionally formatted PDF"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle(name='Header', fontSize=16, alignment=TA_CENTER, 
                                 spaceAfter=12, textColor=colors.HexColor("#2E4057")))
        styles.add(ParagraphStyle(name='SectionHeader', fontSize=12, 
                                 textColor=colors.HexColor("#2E4057"),
                                 spaceAfter=6, spaceBefore=12, bold=True))
        
        story = []
        
        # Parse resume text into sections
        sections = resume_text.split('\n\n')
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.split('\n')
            
            # Check if first line is a header
            if lines[0].isupper() or any(keyword in lines[0].lower() 
                                        for keyword in ['experience', 'education', 'skills', 'summary']):
                # Section header
                story.append(Paragraph(lines[0], styles['SectionHeader']))
                story.append(Spacer(1, 6))
                
                # Section content
                for line in lines[1:]:
                    if line.strip():
                        # Handle bullet points
                        if line.strip().startswith(('•', '-', '*')):
                            line = '• ' + line.strip().lstrip('•-*').strip()
                        story.append(Paragraph(line, styles['BodyText']))
                        story.append(Spacer(1, 3))
            else:
                # Regular paragraph
                for line in lines:
                    if line.strip():
                        story.append(Paragraph(line, styles['BodyText']))
                        story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="ATS Resume Optimizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E4057;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.title(" ATS Resume Optimizer & Analyzer")
    st.markdown("### Enhance your resume for Applicant Tracking Systems and land your dream job!")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        api_choice = st.radio("Select LLM Provider:", ["OpenAI (GPT-4)", "Google AI (Gemini)"])
        
        if api_choice == "OpenAI (GPT-4)":
            openai_key = st.text_input("OpenAI API Key:", type="password", value=Config.OPENAI_API_KEY)
            google_key = None
        else:
            google_key = st.text_input("Google AI API Key:", type="password", value=Config.GOOGLE_API_KEY)
            openai_key = None
        
        st.divider()
        
        # Job Configuration
        st.subheader("Job Details")
        job_roles = ["Data Scientist", "Software Engineer", "Product Manager", 
                    "Marketing Manager", "Business Analyst", "Project Manager", "Custom"]
        
        selected_role = st.selectbox("Select Job Role:", job_roles)
        
        if selected_role == "Custom":
            custom_role = st.text_input("Enter Custom Role:")
            job_role = custom_role if custom_role else "General"
        else:
            job_role = selected_role
        
        # Optional Job Description
        with st.expander("Add Job Description (Optional)"):
            job_description = st.text_area("Paste the job description here:", height=200)
        
        st.divider()
        
        # Analysis History
        if st.session_state.analysis_history:
            st.subheader("📊 Analysis History")
            for idx, history in enumerate(st.session_state.analysis_history[-3:], 1):
                st.info(f"{idx}. {history['role']} - Score: {history['score']:.1f}%")
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Resume")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload your resume in PDF or TXT format (max 5MB)"
        )
        
        # Text Input Alternative
        with st.expander("Or paste your resume text"):
            pasted_text = st.text_area("Paste resume text here:", height=300)
        
        # Process Resume
        if st.button("🔍 Analyze Resume", type="primary"):
            # Validate API key
            if not (openai_key or google_key):
                st.error("Please provide an API key in the sidebar!")
                return
            
            # Get resume text
            resume_text = ""
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    resume_text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = str(uploaded_file.read(), "utf-8")
            elif pasted_text:
                resume_text = pasted_text
            else:
                st.warning("Please upload a resume or paste text!")
                return
            
            if not resume_text:
                st.error("Could not extract text from the resume!")
                return
            
            # Store in session state
            st.session_state.resume_text = resume_text
            
            # Perform ATS Analysis
            with st.spinner("Analyzing resume for ATS compatibility..."):
                analyzer = ATSAnalyzer(
                    api_key=google_key if google_key else openai_key,
                    use_google=bool(google_key)
                )
                ats_analysis = analyzer.calculate_ats_score(resume_text, job_role, job_description)
                st.session_state.ats_score = ats_analysis
            
            # Generate Feedback
            with st.spinner("Generating detailed feedback..."):
                feedback_gen = FeedbackGenerator(openai_key, google_key)
                feedback = feedback_gen.generate_feedback(
                    resume_text, ats_analysis, job_role, job_description
                )
                st.session_state.feedback = feedback
            
            # Generate Optimized Version
            with st.spinner("Creating optimized resume..."):
                optimized = feedback_gen.optimize_resume(
                    resume_text, feedback, job_role, ats_analysis
                )
                st.session_state.optimized_resume = optimized
            
            # Add to history
            st.session_state.analysis_history.append({
                'role': job_role,
                'score': ats_analysis['total_score'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            st.success("✅ Analysis Complete!")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.ats_score:
            # ATS Score Display
            score = st.session_state.ats_score['total_score']
            
            # Score color coding
            if score >= 80:
                score_color = "🟢"
                score_message = "Excellent ATS compatibility!"
            elif score >= 60:
                score_color = "🟡"
                score_message = "Good, but room for improvement"
            else:
                score_color = "🔴"
                score_message = "Needs significant optimization"
            
            # Display main score
            st.markdown(f"### {score_color} ATS Score: {score:.1f}/100")
            st.caption(score_message)
            
            # Progress bar
            st.progress(score / 100)
            
            # Component Scores
            st.subheader("📈 Detailed Breakdown")
            
            scores = st.session_state.ats_score['component_scores']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Keyword Match", f"{scores['keyword_match']:.0f}%")
                st.metric("Formatting", f"{scores['formatting']:.0f}%")
                st.metric("Readability", f"{scores['readability']:.0f}%")
            
            with col_b:
                st.metric("Sections", f"{scores['section_completeness']:.0f}%")
                st.metric("Length", f"{scores['length_appropriateness']:.0f}%")
            
            # Missing Keywords
            if st.session_state.ats_score['missing_keywords']:
                st.subheader("🔑 Missing Keywords")
                missing = st.session_state.ats_score['missing_keywords']
                st.warning(f"Add these keywords: {', '.join(missing[:5])}")
    
    # Feedback Section
    if st.session_state.feedback:
        st.divider()
        st.header("💡 Detailed Feedback & Recommendations")
        
        tab1, tab2, tab3 = st.tabs(["📝 Feedback", "✨ Optimized Resume", "⬇️ Download"])
        
        with tab1:
            st.markdown(st.session_state.feedback)
        
        with tab2:
            if st.session_state.optimized_resume:
                st.subheader("Enhanced Resume Content")
                st.text_area("Optimized Resume:", 
                           value=st.session_state.optimized_resume,
                           height=500)
                
                # Show improvement
                if st.session_state.ats_score:
                    original_score = st.session_state.ats_score['total_score']
                    estimated_new_score = min(original_score + 15, 95)
                    st.success(f"Estimated score improvement: {original_score:.0f}% → {estimated_new_score:.0f}%")
        
        with tab3:
            st.subheader("Download Optimized Resume")
            
            if st.session_state.optimized_resume:
                # Generate PDF
                pdf_buffer = PDFGenerator.create_optimized_pdf(st.session_state.optimized_resume)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_buffer,
                        file_name=f"optimized_resume_{job_role.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                
                with col2:
                    st.download_button(
                        label="📝 Download as TXT",
                        data=st.session_state.optimized_resume,
                        file_name=f"optimized_resume_{job_role.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                st.info("💡 Tip: Use the PDF version for professional applications!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔒 Your data is processed securely and not stored</p>
        <p>Built with ❤️ using Streamlit, OpenAI, and Google AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()