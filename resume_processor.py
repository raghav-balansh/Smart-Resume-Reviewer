import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from functools import lru_cache
import time

@dataclass
class ATSAnalysis:
    """Streamlined ATS analysis results"""
    total_score: float
    component_scores: Dict[str, float]
    missing_keywords: List[str]
    recommendations: List[str]
    issues: List[str]

class OptimizedResumeProcessor:
    """Optimized resume processor focused on speed and efficiency"""
    
    def __init__(self):
        # Simplified keyword sets for faster processing
        self.role_keywords = {
            'data_scientist': [
                "python", "machine learning", "sql", "statistics", "tensorflow",
                "pytorch", "pandas", "numpy", "scikit-learn", "deep learning",
                "data visualization", "tableau", "power bi", "aws", "azure",
                "natural language processing", "big data", "hadoop", "spark"
            ],
            'software_engineer': [
                "java", "python", "javascript", "react", "node.js", "sql",
                "git", "docker", "kubernetes", "aws", "agile", "ci/cd",
                "rest api", "microservices", "cloud computing", "c++",
                "spring boot",'algorithms', 'data structures'
            ],
            'product_manager': [
                'product strategy', 'roadmap', 'agile', 'scrum', 'user stories', 
                'stakeholder', 'analytics', 'metrics', 'kpis', 'market research', 
                'user experience', 'a/b testing', 'wireframes', 'mvp'
            ],
            'marketing_manager': [
                'digital marketing', 'seo', 'sem', 'social media', 'content marketing', 
                'email marketing', 'analytics', 'campaign management', 'brand', 
                'conversion', 'lead generation', 'roi', 'crm'
            ],
            'business_analyst': [
                'business analysis', 'requirements', 'process improvement', 
                'stakeholder management', 'documentation', 'sql', 'excel', 
                'data analysis', 'reporting', 'workflow', 'testing'
            ]
        }
        
        # Common ATS-friendly formatting patterns
        self.formatting_patterns = {
            'bullet_points': r'[•\-\*]',
            'dates': r'\b(20\d{2}|19\d{2})\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'skills_separators': r'[,|;]',
            'section_headers': r'^[A-Z\s]{2,}$'
        }
        
        # Section identification patterns (optimized)
        self.section_patterns = {
            'summary': r'(?i)(summary|profile|objective|about|overview|highlights)',
            'experience': r'(?i)(experience|employment|work\s+history|professional|internship)',
            'education': r'(?i)(education|academic|qualification)',
            'skills': r'(?i)(skills|competencies|technical|technologies|expertise)',
            'projects': r'(?i)(projects|portfolio)',
            'certifications': r'(?i)(certifications|certificates|licenses|awards|certification)',
            'achievements': r'(?i)(achievements|awards|honors|recognitions)',
            'languages': r'(?i)(languages|language\s+skills)'
        }
    
    @lru_cache(maxsize=50)
    def get_role_keywords(self, role: str) -> List[str]:
        #Get cached keywords for a role
        role_key = role.lower().replace(' ', '_')
        return self.role_keywords.get(role_key, [])
    
    def extract_sections(self, resume_text: str) -> Dict[str, str]:
        #section extraction
        sections = {}
        lines = resume_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            section_name = None
            
            # Quick check for common section headers
            if len(line) < 50 and line.isupper():
                for section, pattern in self.section_patterns.items():
                    if re.search(pattern, line):
                        is_header = True
                        section_name = section
                        break
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_name
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # Content before any section
                    if 'header' not in sections:
                        sections['header'] = ''
                    sections['header'] += line + '\n'
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def calculate_keyword_match_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        #Fast keyword matching with caching
        if not keywords:
            return 50.0, []
        
        text_lower = text.lower()
        found_keywords = []
        missing_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Calculate score with bonus for multiple occurrences
        base_score = (len(found_keywords) / len(keywords)) * 100
        
        # Bonus for keyword density
        total_keyword_occurrences = sum(text_lower.count(kw.lower()) for kw in found_keywords)
        density_bonus = min(total_keyword_occurrences * 2, 20)        
        final_score = min(base_score + density_bonus, 100)
        
        return final_score, missing_keywords
    
    def analyze_formatting(self, text: str) -> Dict[str, float]:
        #formatting analysis
        scores = {}
        
        # 30 point assign for Bullet points section
        bullet_count = len(re.findall(self.formatting_patterns['bullet_points'], text))
        scores['bullet_points'] = min(bullet_count * 10, 30)
        
        # Date formatting 20 points for this section
        date_count = len(re.findall(self.formatting_patterns['dates'], text))
        scores['dates'] = min(date_count * 5, 20)
        
        # Contact information 20 points
        contact_score = 0
        if re.search(self.formatting_patterns['email'], text):
            contact_score += 10
        if re.search(self.formatting_patterns['phone'], text):
            contact_score += 10
        scores['contact'] = contact_score
        
        # Structure and length 30 points
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            scores['length'] = 30
        elif 200 <= word_count < 300 or 800 < word_count <= 1200:
            scores['length'] = 25
        else:
            scores['length'] = 15
        
        return scores
    
    def calculate_readability_score(self, text: str) -> float:
        #Simple readability calculation
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        
        if sentences == 0:
            return 50.0
        
        avg_sentence_length = words / sentences
        
        # Optimal range: 15-25 words per sentence
        if 15 <= avg_sentence_length <= 25:
            return 100.0
        elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 35:
            return 80.0
        else:
            return 60.0
    
    def analyze_section_completeness(self, sections: Dict[str, str]) -> float:
        #chechking about mendatory or non mendatory section details
        essential_sections = ['experience', 'education', 'skills']
        optional_sections = ['summary', 'projects', 'certifications']
        
        found_essential = sum(1 for section in essential_sections if section in sections and sections[section].strip())
        found_optional = sum(1 for section in optional_sections if section in sections and sections[section].strip())
        
        # Essential sections are percentage 70% and optional 30%
        essential_score = (found_essential / len(essential_sections)) * 70
        optional_score = min(found_optional / len(optional_sections), 1.0) * 30
        
        return essential_score + optional_score
    
    def comprehensive_ats_analysis(self, resume_text: str, job_role: str, job_description: str = "") -> ATSAnalysis:
        #comprehensive ATS analysis
        
        keywords = self.get_role_keywords(job_role)
        
        if job_description:
            # Simple keyword extraction from job description
            jd_words = re.findall(r'\b[a-zA-Z]{3,}\b', job_description.lower())
            # Add unique technical terms
            tech_terms = [word for word in jd_words if len(word) > 4 and word not in ['experience', 'required', 'preferred', 'skills']]
            keywords.extend(tech_terms[:10])  # Add top 10 additional areaas
        
        # Keyword analysis
        keyword_score, missing_keywords = self.calculate_keyword_match_score(resume_text, keywords)
        
        # Formatting analysis
        formatting_scores = self.analyze_formatting(resume_text)
        formatting_score = sum(formatting_scores.values())
        
        # Section analysis
        sections = self.extract_sections(resume_text)
        section_score = self.analyze_section_completeness(sections)
        
        # Readability analysis
        readability_score = self.calculate_readability_score(resume_text)
        
        # Calculate overall score (weighted average)
        total_score = (
            keyword_score * 0.35 + 
            formatting_score * 0.25 +   
            section_score * 0.25 +      
            readability_score * 0.15    
        )
        
        # Component scores for detailed feedback
        component_scores = {
            'keyword_match': keyword_score,
            'formatting': formatting_score,
            'section_completeness': section_score,
            'readability': readability_score,
            'length_appropriateness': formatting_scores.get('length', 50)
        }
        
        # recommendations system
        recommendations = []
        issues = []
        
        if keyword_score < 60:
            recommendations.append("Add more role-specific keywords throughout your resume")
            issues.append("Low keyword match with target role")
        
        if formatting_score < 70:
            recommendations.append("Improve formatting with bullet points and consistent structure")
            issues.append("Formatting could be more ATS-friendly")
        
        if section_score < 80:
            recommendations.append("Ensure all essential sections are present and complete")
            issues.append("Missing important resume sections")
        
        if readability_score < 70:
            recommendations.append("Simplify sentence structure for better readability")
            issues.append("Content readability needs improvement")
        
        # Specific formatting recommendations
        if formatting_scores['bullet_points'] < 20:
            recommendations.append("Use bullet points to highlight achievements")
        
        if formatting_scores['contact'] < 20:
            recommendations.append("Include complete contact information")
        
        return ATSAnalysis(
            total_score=total_score,
            component_scores=component_scores,
            missing_keywords=missing_keywords[:10],
            recommendations=recommendations[:7],
            issues=issues[:5]                  
        )
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        #contact information extraction
        contact_info = {}
        
        # email
        email_match = re.search(self.formatting_patterns['email'], text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # phone
        phone_match = re.search(self.formatting_patterns['phone'], text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # linkedIn
        linkedin_match = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # gitHub
        github_match = re.search(r'github\.com/([\w-]+)', text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = f"github.com/{github_match.group(1)}"
        
        return contact_info
    
    def extract_skills(self, text: str, skills_section: str = "") -> List[str]:
        skills = []
        
        if skills_section:
            text_to_analyze = skills_section
        else:
            sections = self.extract_sections(text)
            text_to_analyze = sections.get('skills', text)
        
        skill_candidates = re.split(r'[,;\n\|]', text_to_analyze)
        
        for skill in skill_candidates:
            skill = skill.strip()
            if skill and len(skill) > 2 and len(skill) < 50:
                # Remove bullet points and numbers
                skill = re.sub(r'^[•\-\*\d\.\)]+\s*', '', skill)
                if skill:
                    skills.append(skill)
        
        return skills[:]
    
    def calculate_experience_score(self, experience_text: str) -> float:
        if not experience_text.strip():
            return 0.0
        
        score = 0.0
        
        # if date is present 25 point will be add on
        if re.search(self.formatting_patterns['dates'], experience_text):
            score += 25
        
        #if bullet point is presentated add on 25 points
        bullet_count = len(re.findall(self.formatting_patterns['bullet_points'], experience_text))
        score += min(bullet_count * 5, 25)
        
        # Check for quantified achievement 30 points
        quantifiers = r'\b\d+%|\b\d+\+|\b\d+[kmbt]|\$\d+|increased|improved|reduced|saved|achieved'
        if re.search(quantifiers, experience_text.lower()):
            score += 30
        
        # action verbs- 20 points
        action_verbs = ['led', 'managed', 'developed', 'implemented', 'created', 'designed', 'built', 'achieved', 'improved']
        found_verbs = sum(1 for verb in action_verbs if verb in experience_text.lower())
        score += min(found_verbs * 3, 20)
        
        return min(score, 100)
    
    def quick_resume_summary(self, resume_text: str) -> Dict[str, any]:
        #Generate quick resume summary stats
        sections = self.extract_sections(resume_text)
        contact_info = self.extract_contact_info(resume_text)
        
        word_count = len(resume_text.split())
        
        # Count key elements
        bullet_points = len(re.findall(self.formatting_patterns['bullet_points'], resume_text))
        dates_mentioned = len(re.findall(self.formatting_patterns['dates'], resume_text))
        
        return {
            'word_count': word_count,
            'sections_found': list(sections.keys()),
            'contact_complete': len(contact_info) >= 2, 
            'has_bullets': bullet_points > 0,
            'has_dates': dates_mentioned > 0,
            'estimated_pages': max(1, word_count // 300) 
        }
    
    def identify_resume_gaps(self, sections: Dict[str, str], job_role: str) -> List[str]:
        #Identify common gaps in resume based on role
        gaps = []
        
        if 'experience' not in sections or not sections['experience'].strip():
            gaps.append("Missing or incomplete work experience section")
        
        if 'skills' not in sections or not sections['skills'].strip():
            gaps.append("Missing technical skills section")
        
        if 'education' not in sections or not sections['education'].strip():
            gaps.append("Missing education information")
        
        if job_role.lower() in ['data_scientist', 'software_engineer']:
            if 'projects' not in sections or not sections['projects'].strip():
                gaps.append("Consider adding a projects section to showcase technical work")
        
        if job_role.lower() in ['product_manager', 'marketing_manager']:
            if not any(metric in sections.get('experience', '').lower() for metric in ['%', 'increased', 'improved', 'roi']):
                gaps.append("Add quantified business metrics to experience descriptions")
        
        #quantified achievements
        if not re.search(r'\d+%|\d+\+|\$\d+', sections.get('experience', '')):
            gaps.append("Include specific numbers and metrics in achievements")
        
        return gaps[:10]
    
    def optimize_for_ats(self, resume_text: str, target_keywords: List[str]) -> str:
        #ATS optimization suggestions
        lines = resume_text.split('\n')
        suggestions = []
        
        text_lower = resume_text.lower()
        missing_keywords = [kw for kw in target_keywords if kw.lower() not in text_lower]
        
        if missing_keywords:
            suggestions.append(f"\nKEYWORD OPTIMIZATION:")
            suggestions.append(f"Consider naturally incorporating: {', '.join(missing_keywords[:5])}")
        
        # Check formatting
        if not re.search(self.formatting_patterns['bullet_points'], resume_text):
            suggestions.append(f"\nFORMATTING TIPS:")
            suggestions.append("• Use bullet points to organize information")
            suggestions.append("• Ensure consistent formatting throughout")
        
        # Check for quantified achievements
        if not re.search(r'\d+%|\d+\+|\$\d+', resume_text):
            suggestions.append(f"\nACHIEVEMENT ENHANCEMENT:")
            suggestions.append("• Add specific numbers and percentages")
            suggestions.append("• Quantify your accomplishments with metrics")
        
        return '\n'.join(lines + suggestions) if suggestions else resume_text

# Performance testing and validation
def test_performance():
    #Test the performance of optimized processor
    processor = OptimizedResumeProcessor()
    
    sample_resume = """
    John Doe
    john.doe@email.com | (555) 123-4567
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE  
    Senior Software Engineer | Tech Corp | 2020-2023
    • Developed web applications using Python and React
    • Improved system performance by 40%
    • Led team of 3 developers on critical projects
    
    Software Engineer | StartupXYZ | 2018-2020
    • Built REST APIs serving 100k+ requests daily
    • Implemented CI/CD pipelines reducing deployment time by 60%
    
    EDUCATION
    Bachelor of Computer Science | University of Tech | 2018
    
    SKILLS
    Python, JavaScript, React, Node.js, SQL, AWS, Docker, Git
    
    PROJECTS
    E-commerce Platform | 2023
    • Built full-stack application with React and Django
    • Integrated payment processing and inventory management
    """
    
    print("🔄 Testing OptimizedResumeProcessor performance...")
    
    # Test section extraction
    start_time = time.time()
    sections = processor.extract_sections(sample_resume)
    print(f"⏱️ Section extraction: {time.time() - start_time:.3f}s")
    print(f"Found sections: {list(sections.keys())}")
    
    # Test ATS analysis
    start_time = time.time()
    ats_analysis = processor.comprehensive_ats_analysis(sample_resume, "software_engineer")
    print(f"⏱️ ATS analysis: {time.time() - start_time:.3f}s")
    print(f"📊 ATS Score: {ats_analysis.total_score:.1f}/100")
    print(f"🎯 Missing keywords: {ats_analysis.missing_keywords[:5]}")
    
    # Test contact extraction
    start_time = time.time()
    contact_info = processor.extract_contact_info(sample_resume)
    print(f"⏱️ Contact extraction: {time.time() - start_time:.3f}s")
    print(f"📞 Contact info: {contact_info}")
    
    # Test skills extraction
    start_time = time.time()
    skills = processor.extract_skills(sample_resume)
    print(f"⏱️ Skills extraction: {time.time() - start_time:.3f}s")
    print(f"🛠️ Extracted skills: {skills}")
    
    # Test quick summary
    start_time = time.time()
    summary = processor.quick_resume_summary(sample_resume)
    print(f"⏱️ Quick summary: {time.time() - start_time:.3f}s")
    print(f"📋 Summary: {summary}")

if __name__ == "__main__":
    test_performance()