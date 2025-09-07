import pdfplumber
import re
import nltk
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import textstat
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class ResumeSection:
    name: str
    content: str
    start_line: int
    end_line: int
    score: float = 0.0

@dataclass
class ATSMetrics:
    keyword_score: float
    formatting_score: float
    readability_score: float
    section_score: float
    overall_score: float
    missing_keywords: List[str]
    suggestions: List[str]

@dataclass
class ATSAnalysis:
    total_score: float
    component_scores: Dict[str, float]
    missing_keywords: List[str]
    sections_analysis: Dict[str, str]
    formatting_checks: Dict[str, bool]

class ResumeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Common resume sections
        self.section_patterns = {
            'contact': r'(contact|personal\s+information|header)',
            'summary': r'(summary|profile|objective|about)',
            'experience': r'(experience|work\s+history|employment|professional\s+experience)',
            'education': r'(education|academic|qualifications)',
            'skills': r'(skills|technical\s+skills|competencies)',
            'projects': r'(projects|portfolio|key\s+projects)',
            'certifications': r'(certifications|certificates|licenses)',
            'achievements': r'(achievements|awards|honors|accomplishments)',
            'languages': r'(languages|language\s+skills)',
            'interests': r'(interests|hobbies|activities)'
        }
        
        # ATS-friendly keywords for different roles (expanded from app.py)
        self.role_keywords = {
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

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def identify_sections(self, text: str) -> List[ResumeSection]:
        """Identify and extract resume sections"""
        lines = text.split('\n')
        sections = []
        
        # Find section headers
        section_starts = {}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 50:
                    section_starts[section_name] = i
                    break
        
        # Extract section content
        section_names = list(section_starts.keys())
        for i, section_name in enumerate(section_names):
            start_line = section_starts[section_name]
            end_line = section_starts[section_names[i + 1]] if i + 1 < len(section_names) else len(lines)
            
            content = '\n'.join(lines[start_line:end_line])
            sections.append(ResumeSection(
                name=section_name,
                content=content,
                start_line=start_line,
                end_line=end_line
            ))
        
        return sections

    def chunk_text(self, text: str) -> List[Document]:
        """Split text into chunks for processing"""
        documents = self.text_splitter.create_documents([text])
        return documents

    def calculate_keyword_score(self, text: str, job_role: str) -> Tuple[float, List[str]]:
        """Calculate keyword relevance score"""
        if job_role not in self.role_keywords:
            return 0.0, []
        
        target_keywords = self.role_keywords[job_role]
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in target_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        score = len(found_keywords) / len(target_keywords) * 100
        missing_keywords = [kw for kw in target_keywords if kw.lower() not in text_lower]
        
        return min(score, 100.0), missing_keywords

    def calculate_formatting_score(self, text: str) -> float:
        """Calculate formatting score based on structure"""
        lines = text.split('\n')
        score = 0.0
        
        # Check for bullet points
        bullet_count = sum(1 for line in lines if re.match(r'^\s*[•\-\*]\s+', line))
        if bullet_count > 0:
            score += 20
        
        # Check for consistent formatting
        if len(lines) > 10:
            score += 20
        
        # Check for section headers (capitalized, short lines)
        header_count = sum(1 for line in lines if len(line.strip()) < 50 and line.strip().isupper())
        if header_count > 3:
            score += 20
        
        # Check for dates (experience/education)
        date_count = len(re.findall(r'\b(19|20)\d{2}\b', text))
        if date_count > 0:
            score += 20
        
        # Check for contact information
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        phone_count = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        if email_count > 0 and phone_count > 0:
            score += 20
        
        return min(score, 100.0)

    def calculate_readability_score(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        try:
            # Remove special characters and numbers for readability calculation
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\d+', '', clean_text)
            
            if len(clean_text.split()) < 10:
                return 50.0  # Default score for very short text
            
            flesch_score = textstat.flesch_reading_ease(clean_text)
            # Convert to 0-100 scale (higher is better for resumes)
            if flesch_score >= 80:
                return 100.0
            elif flesch_score >= 60:
                return 80.0
            elif flesch_score >= 40:
                return 60.0
            else:
                return 40.0
        except:
            return 50.0

    def calculate_section_score(self, sections: List[ResumeSection]) -> float:
        """Calculate score based on resume sections"""
        essential_sections = ['contact', 'experience', 'education', 'skills']
        found_sections = [s.name for s in sections]
        
        score = 0.0
        for section in essential_sections:
            if section in found_sections:
                score += 25
        
        # Bonus for additional sections
        additional_sections = ['summary', 'projects', 'certifications', 'achievements']
        for section in additional_sections:
            if section in found_sections:
                score += 5
        
        return min(score, 100.0)

    def calculate_ats_score(self, text: str, job_role: str, sections: List[ResumeSection]) -> ATSMetrics:
        """Calculate overall ATS score"""
        keyword_score, missing_keywords = self.calculate_keyword_score(text, job_role)
        formatting_score = self.calculate_formatting_score(text)
        readability_score = self.calculate_readability_score(text)
        section_score = self.calculate_section_score(sections)
        
        # Weighted average
        overall_score = (
            keyword_score * 0.4 +
            formatting_score * 0.2 +
            readability_score * 0.2 +
            section_score * 0.2
        )
        
        # Generate suggestions
        suggestions = []
        if keyword_score < 60:
            suggestions.append("Add more relevant keywords for the target role")
        if formatting_score < 60:
            suggestions.append("Improve formatting with bullet points and consistent structure")
        if readability_score < 60:
            suggestions.append("Simplify language and improve readability")
        if section_score < 80:
            suggestions.append("Ensure all essential sections are present")
        
        return ATSMetrics(
            keyword_score=keyword_score,
            formatting_score=formatting_score,
            readability_score=readability_score,
            section_score=section_score,
            overall_score=overall_score,
            missing_keywords=missing_keywords,
            suggestions=suggestions
        )
    
    def comprehensive_ats_analysis(self, resume_text: str, job_role: str, job_description: str = "") -> ATSAnalysis:
        """Comprehensive ATS analysis similar to app.py"""
        # Extract sections
        sections = self.extract_sections(resume_text)
        
        # Initialize scoring components
        scores = {
            'keyword_match': 0,
            'formatting': 0,
            'section_completeness': 0,
            'readability': 0,
            'length_appropriateness': 0
        }
        
        # 1. Keyword Analysis (30% weight)
        keywords = self.role_keywords.get(job_role.lower().replace(' ', '_'), self.role_keywords['general'])
        if job_description:
            # Extract additional keywords from job description
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
        scores['readability'] = self.calculate_readability_score(resume_text)
        
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
        
        return ATSAnalysis(
            total_score=round(total_score, 1),
            component_scores=scores,
            missing_keywords=missing_keywords[:10],  # Top 10 missing
            sections_analysis=sections,
            formatting_checks=formatting_checks
        )
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common resume sections (from app.py)"""
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
    
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract keywords from job description (from app.py)"""
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
        """Check if formatting is consistent (from app.py)"""
        lines = text.split('\n')
        bullet_styles = set()
        
        for line in lines:
            if re.match(r'^\s*[•·▪▫◦‣⁃●○■□◆◇➔→⇒➤*]\s+', line):
                bullet = re.match(r'^\s*([•·▪▫◦‣⁃●○■□◆◇➔→⇒➤*])\s+', line).group(1)
                bullet_styles.add(bullet)
        
        # Consistent if using 0-1 bullet styles
        return len(bullet_styles) <= 1

    def process_resume(self, file_path: str = None, text: str = None, job_role: str = "software_engineer") -> Dict:
        """Main method to process resume"""
        try:
            # Extract text
            if file_path:
                if file_path.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.lower().endswith('.docx'):
                    text = self.extract_text_from_docx(file_path)
                else:
                    raise ValueError("Unsupported file format")
            elif text:
                text = text
            else:
                raise ValueError("Either file_path or text must be provided")
            
            # Clean text
            text = self.clean_text(text)
            
            # Identify sections
            sections = self.identify_sections(text)
            
            # Create chunks
            chunks = self.chunk_text(text)
            
            # Calculate ATS score
            ats_metrics = self.calculate_ats_score(text, job_role, sections)
            
            return {
                'text': text,
                'sections': sections,
                'chunks': chunks,
                'ats_metrics': ats_metrics,
                'word_count': len(text.split()),
                'line_count': len(text.split('\n'))
            }
            
        except Exception as e:
            raise Exception(f"Error processing resume: {str(e)}")

# Example usage
if __name__ == "__main__":
    processor = ResumeProcessor()
    
    # Test with sample text
    sample_text = """
    John Doe
    john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2020-2023
    • Developed web applications using Python and React
    • Led team of 3 developers
    • Implemented CI/CD pipelines
    
    EDUCATION
    Bachelor of Computer Science | University of Tech | 2018
    
    SKILLS
    Python, JavaScript, React, SQL, AWS, Docker
    """
    
    result = processor.process_resume(text=sample_text, job_role="software_engineer")
    print(f"ATS Score: {result['ats_metrics'].overall_score:.1f}")
    print(f"Missing Keywords: {result['ats_metrics'].missing_keywords}")
