import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional, Tuple
import re
import json
from dataclasses import dataclass
import warnings
import threading
import time
from functools import lru_cache
warnings.filterwarnings("ignore")

@dataclass
class SectionFeedback:
    section: str
    score: float
    feedback: str
    suggestions: List[str]
    missing_keywords: List[str]

@dataclass
class OverallFeedback:
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    overall_score: float

class OptimizedHuggingFaceAnalyzer:
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_length: int = 256, batch_size: int = 1):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._cache = {}
        self._lock = threading.Lock()
        
        # Use rule-based analysis as primary with AI as secondary
        self.use_ai = False  # Start with rule-based for speed
        
        # Enhanced role-specific keywords for better analysis
        self.role_keywords = {
            'data_scientist': {
                'core_skills': ['python', 'r', 'sql', 'machine learning', 'statistics', 'data analysis'],
                'tools': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'jupyter'],
                'concepts': ['regression', 'classification', 'clustering', 'nlp', 'deep learning', 'visualization'],
                'certifications': ['aws certified', 'google cloud', 'coursera', 'kaggle']
            },
            'software_engineer': {
                'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust'],
                'frameworks': ['react', 'angular', 'vue', 'node.js', 'spring', 'django', 'flask'],
                'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'linux'],
                'concepts': ['api', 'microservices', 'agile', 'scrum', 'testing', 'ci/cd', 'devops']
            },
            'product_manager': {
                'skills': ['product strategy', 'roadmap', 'agile', 'scrum', 'user stories', 'stakeholder'],
                'tools': ['jira', 'confluence', 'figma', 'analytics', 'sql', 'tableau'],
                'concepts': ['market research', 'user experience', 'metrics', 'kpis', 'growth', 'mvp'],
                'soft_skills': ['leadership', 'communication', 'analytical', 'strategic thinking']
            },
            'marketing_manager': {
                'digital': ['seo', 'sem', 'ppc', 'social media', 'content marketing', 'email marketing'],
                'tools': ['google analytics', 'hubspot', 'salesforce', 'marketo', 'hootsuite'],
                'concepts': ['brand management', 'campaign', 'roi', 'conversion', 'lead generation'],
                'skills': ['creativity', 'analytics', 'communication', 'project management']
            },
            'business_analyst': {
                'skills': ['requirements analysis', 'process improvement', 'stakeholder management', 'documentation'],
                'tools': ['excel', 'sql', 'tableau', 'power bi', 'visio', 'jira'],
                'methodologies': ['agile', 'waterfall', 'lean', 'six sigma', 'business process modeling'],
                'concepts': ['gap analysis', 'use cases', 'user stories', 'testing', 'validation']
            }
        }
        
        # Initialize lightweight analyzer
        self._initialize_lightweight()
    
    def _initialize_lightweight(self):
        self.use_ai = False
    
    def _load_ai_model(self):
        """Load AI model only when needed (lazy loading)"""
        if self.pipeline is not None:
            return True
            
        try:
            print("Loading AI model for enhanced analysis...")
            
            # Use a very lightweight model or skip AI entirely for speed
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=100,  # Very limited for speed
                do_sample=True,
                temperature=0.7,
                device=0 if self.device == "cuda" else -1
            )
            
            self.use_ai = True
            print("AI model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Could not load AI model: {e}")
            print("Continuing with rule-based analysis for optimal speed")
            self.use_ai = False
            return False
    
    @lru_cache(maxsize=100)
    def _get_role_keywords(self, job_role: str) -> List[str]:
        """Get cached keywords for a role"""
        role_key = job_role.lower().replace(' ', '_')
        if role_key in self.role_keywords:
            keywords = []
            for category in self.role_keywords[role_key].values():
                keywords.extend(category)
            return keywords
        return []
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
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
        
        score = (len(found_keywords) / len(keywords)) * 100
        return score, missing_keywords[:10]  # Return top 10 missing
    
    def _analyze_structure(self, text: str) -> Dict[str, float]:
        scores = {}
        
        # Section completeness
        required_sections = ['experience', 'education', 'skills', 'summary']
        found_sections = 0
        
        for section in required_sections:
            if section in text.lower():
                found_sections += 1
        
        scores['section_completeness'] = (found_sections / len(required_sections)) * 100
        
        # Length appropriateness
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            scores['length_appropriateness'] = 100
        elif 200 <= word_count < 300 or 800 < word_count <= 1200:
            scores['length_appropriateness'] = 80
        else:
            scores['length_appropriateness'] = 60
        
        # Formatting score (basic checks)
        formatting_score = 0
        if '\n' in text:  # Has line breaks
            formatting_score += 30
        if '•' in text or '-' in text:  # Has bullet points
            formatting_score += 30
        if any(year in text for year in ['2020', '2021', '2022', '2023', '2024']):  # Has dates
            formatting_score += 40
        
        scores['formatting'] = min(formatting_score, 100)
        
        # Readability score
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentences, 1)
        
        if 10 <= avg_sentence_length <= 20:
            scores['readability'] = 100
        elif 8 <= avg_sentence_length < 10 or 20 < avg_sentence_length <= 25:
            scores['readability'] = 80
        else:
            scores['readability'] = 60
        
        return scores
    
    def _generate_rule_based_feedback(self, text: str, job_role: str) -> Tuple[OverallFeedback, Dict[str, float]]:
        keywords = self._get_role_keywords(job_role)
        keyword_score, missing_keywords = self._calculate_keyword_score(text, keywords)
        structure_scores = self._analyze_structure(text)
        
        # Calculate overall score
        overall_score = (
            keyword_score * 0.4 +
            structure_scores['section_completeness'] * 0.2 +
            structure_scores['formatting'] * 0.15 +
            structure_scores['readability'] * 0.15 +
            structure_scores['length_appropriateness'] * 0.1
        )
        
        # Generate strengths
        strengths = []
        if keyword_score >= 70:
            strengths.append("Strong keyword alignment with target role")
        if structure_scores['section_completeness'] >= 75:
            strengths.append("Well-structured resume with all essential sections")
        if structure_scores['formatting'] >= 80:
            strengths.append("Good formatting with clear organization")
        if structure_scores['readability'] >= 80:
            strengths.append("Clear and readable content structure")
        if structure_scores['length_appropriateness'] >= 80:
            strengths.append("Appropriate resume length for ATS scanning")
        
        # Generate weaknesses
        weaknesses = []
        if keyword_score < 50:
            weaknesses.append("Missing key industry-specific keywords")
        if structure_scores['section_completeness'] < 75:
            weaknesses.append("Missing important resume sections")
        if structure_scores['formatting'] < 60:
            weaknesses.append("Formatting could be improved for better ATS compatibility")
        if structure_scores['readability'] < 60:
            weaknesses.append("Content readability needs improvement")
        if len(text.split()) < 200:
            weaknesses.append("Resume might be too brief - consider adding more details")
        
        # Generate recommendations
        recommendations = []
        if missing_keywords:
            recommendations.append(f"Add these relevant keywords: {', '.join(missing_keywords[:5])}")
        if keyword_score < 60:
            recommendations.append("Include more role-specific technical terms and skills")
        if structure_scores['formatting'] < 70:
            recommendations.append("Use bullet points and consistent formatting")
        if '•' not in text and '-' not in text:
            recommendations.append("Use bullet points to highlight achievements")
        if not any(num in text for num in ['%','+']):
            recommendations.append("Quantify achievements with numbers and percentages")
        
        # Ensure we have at least some content
        if not strengths:
            strengths.append("Resume contains relevant professional experience")
        if not weaknesses:
            weaknesses.append("Consider optimizing for specific job requirements")
        if not recommendations:
            recommendations.append("Tailor content to match job description keywords")
        
        overall_feedback = OverallFeedback(
            strengths=strengths[:5],
            weaknesses=weaknesses[:5], 
            recommendations=recommendations[:5],
            overall_score=overall_score
        )
        
        component_scores = {
            'keyword_match': keyword_score,
            'formatting': structure_scores['formatting'],
            'section_completeness': structure_scores['section_completeness'],
            'readability': structure_scores['readability'],
            'length_appropriateness': structure_scores['length_appropriateness']
        }
        
        return overall_feedback, component_scores
    
    def analyze_overall(self, resume_text: str, job_role: str) -> OverallFeedback:
        """Perform overall resume analysis with caching"""
        cache_key = f"overall_{hash(resume_text[:500])}_{job_role}"
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Use rule-based analysis for speed
        overall_feedback, _ = self._generate_rule_based_feedback(resume_text, job_role)
        
        with self._lock:
            self._cache[cache_key] = overall_feedback
        
        return overall_feedback
    
    def analyze_section(self, resume_text: str, job_role: str, section: str) -> SectionFeedback:
        """Analyze a specific section with rule-based approach"""
        cache_key = f"section_{hash(resume_text[:300])}_{job_role}_{section}"
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Extract section content
        section_content = self._extract_section_content(resume_text, section)
        
        # Generate section-specific feedback
        keywords = self._get_role_keywords(job_role)
        keyword_score, missing_keywords = self._calculate_keyword_score(section_content, keywords)
        
        # Section-specific scoring
        if section.lower() in ['experience', 'work']:
            score = self._score_experience_section(section_content)
            feedback = self._get_experience_feedback(section_content)
            suggestions = self._get_experience_suggestions(section_content)
        elif section.lower() in ['skills', 'technical']:
            score = self._score_skills_section(section_content, keywords)
            feedback = self._get_skills_feedback(section_content)
            suggestions = self._get_skills_suggestions(section_content, job_role)
        elif section.lower() in ['education', 'academic']:
            score = self._score_education_section(section_content)
            feedback = self._get_education_feedback(section_content)
            suggestions = self._get_education_suggestions(section_content)
        else:
            score = keyword_score
            feedback = f"Section contains relevant content for {job_role} role"
            suggestions = ["Add more specific details", "Include relevant keywords"]
        
        section_feedback = SectionFeedback(
            section=section,
            score=score,
            feedback=feedback,
            suggestions=suggestions[:3],
            missing_keywords=missing_keywords[:5]
        )
        
        with self._lock:
            self._cache[cache_key] = section_feedback
        
        return section_feedback
    
    def _extract_section_content(self, text: str, section: str) -> str:
        """Extract content for a specific section"""
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        section_keywords = [section.lower(), section.lower().replace('_', ' ')]
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering the target section
            if any(keyword in line_lower for keyword in section_keywords) and len(line.strip()) < 50:
                in_section = True
                continue
            
            # Check if we're entering a different section
            if in_section and any(sec in line_lower for sec in ['experience', 'education', 'skills', 'projects', 'summary']) and len(line.strip()) < 50:
                if not any(keyword in line_lower for keyword in section_keywords):
                    break
            
            if in_section and line.strip():
                section_content.append(line)
        
        return '\n'.join(section_content)
    
    def _score_experience_section(self, content: str) -> float:
        """Score experience section"""
        score = 0
        
        # Check for dates
        if any(year in content for year in ['2020', '2021', '2022', '2023', '2024']):
            score += 25
        
        # Check for bullet points
        if '•' in content or '-' in content:
            score += 25
        
        # Check for quantified achievements
        if any(indicator in content for indicator in ['%', '+', 'increased', 'improved' , 'reduced']):
            score += 30
        
        # Check for action verbs
        action_verbs = ['led', 'managed', 'developed', 'implemented', 'created', 'designed', 'achieved']
        found_verbs = sum(1 for verb in action_verbs if verb in content.lower())
        score += min(found_verbs * 4, 20)
        
        return min(score, 100)
    
    def _score_skills_section(self, content: str, keywords: List[str]) -> float:
        """Score skills section"""
        if not content.strip():
            return 0
        
        # Count skills
        skills = [s.strip() for s in content.replace('\n', ',').split(',') if s.strip()]
        skill_count = len(skills)
        
        # Optimal range: 8-15 skills
        if 8 <= skill_count <= 15:
            count_score = 40
        elif 5 <= skill_count < 8 or 15 < skill_count <= 20:
            count_score = 30
        else:
            count_score = 20
        
        # Keyword relevance
        keyword_score, _ = self._calculate_keyword_score(content, keywords)
        
        return min(count_score + (keyword_score * 0.6), 100)
    
    def _score_education_section(self, content: str) -> float:
        """Score education section"""
        score = 0
        
        # Check for degree
        if any(degree in content.lower() for degree in ['bachelor', 'master', 'phd', 'degree']):
            score += 40
        
        # Check for institution
        if any(word in content.lower() for word in ['university', 'college', 'institute']):
            score += 30
        
        # Check for date
        if any(year in content for year in ['2020', '2021', '2022', '2023', '2024', '2019', '2018']):
            score += 30
        
        return min(score, 100)
    
    def _get_experience_feedback(self, content: str) -> str:
        """Get feedback for experience section"""
        if not any(indicator in content for indicator in ['%', 'increased', 'improved']):
            return "Add quantified achievements with specific metrics and numbers"
        elif not ('•' in content or '-' in content):
            return "Use bullet points to better organize accomplishments"
        else:
            return "Experience section shows good structure with quantified achievements"
    
    def _get_skills_feedback(self, content: str) -> str:
        """Get feedback for skills section"""
        skills = [s.strip() for s in content.replace('\n', ',').split(',') if s.strip()]
        skill_count = len(skills)
        
        if skill_count < 5:
            return "Add more relevant skills to strengthen your profile"
        elif skill_count > 20:
            return "Consider reducing to most relevant skills for better focus"
        else:
            return "Good variety of skills listed, well-organized"
    
    def _get_education_feedback(self, content: str) -> str:
        """Get feedback for education section"""
        if not any(degree in content.lower() for degree in ['bachelor', 'master', 'phd']):
            return "Include your degree type and major field of study"
        else:
            return "Education section contains essential information"
    
    def _get_experience_suggestions(self, content: str) -> List[str]:
        """Get suggestions for experience section"""
        suggestions = []
        
        if not any(indicator in content for indicator in ['%','increased']):
            suggestions.append("Add specific metrics and percentages to quantify achievements")
        
        if not ('•' in content or '-' in content):
            suggestions.append("Use bullet points to organize accomplishments clearly")
        
        action_verbs = ['led', 'managed', 'developed', 'implemented', 'created']
        if not any(verb in content.lower() for verb in action_verbs):
            suggestions.append("Start bullet points with strong action verbs")
        
        if not any(year in content for year in ['2020', '2021', '2022', '2023']):
            suggestions.append("Include employment dates for each position")
        
        return suggestions
    
    def _get_skills_suggestions(self, content: str, job_role: str) -> List[str]:
        """Get suggestions for skills section"""
        suggestions = []
        
        skills = [s.strip() for s in content.replace('\n', ',').split(',') if s.strip()]
        
        if len(skills) < 8:
            suggestions.append("Add more relevant technical and soft skills")
        
        if job_role.lower() in ['software engineer', 'data scientist']:
            if not any(tech in content.lower() for tech in ['python', 'java', 'javascript']):
                suggestions.append("Include relevant programming languages")
        
        suggestions.append("Consider grouping skills by category (Technical, Tools, Languages)")
        
        return suggestions
    
    def _get_education_suggestions(self, content: str) -> List[str]:
        """Get suggestions for education section"""
        suggestions = []
        
        if not any(degree in content.lower() for degree in ['bachelor', 'master']):
            suggestions.append("Include your degree type and field of study")
        
        if not any(year in content for year in ['2020', '2021', '2022', '2023', '2024']):
            suggestions.append("Add graduation year")
        
        suggestions.append("Consider adding relevant coursework or academic projects")
        
        return suggestions
    
    def optimize_resume(self, resume_text: str, job_role: str) -> str:
        """Generate optimized resume using rule-based improvements"""
        # For speed, return enhanced version with basic improvements
        lines = resume_text.split('\n')
        optimized_lines = []
        
        keywords = self._get_role_keywords(job_role)
        
        for line in lines:
            if line.strip():
                # Enhance bullet points
                if line.strip().startswith(('-', '*')):
                    line = line.replace('-', '•').replace('*', '•')
                
                # Add keywords where appropriate (simple version)
                optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        # Add missing keywords section if significantly missing
        keyword_score, missing = self._calculate_keyword_score(resume_text, keywords)
        
        if keyword_score < 40 and missing:
            optimized_lines.append("\n📝 SUGGESTED ADDITIONS:")
            optimized_lines.append(f"Consider adding these keywords: {', '.join(missing[:5])}")
        
        return '\n'.join(optimized_lines)

# Example usage and testing
if __name__ == "__main__":
    analyzer = OptimizedHuggingFaceAnalyzer()
    
    sample_resume = """
    Raghav maheshwari
    abc@email.com | 999-123-4537
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE
    Senior Software Engineer | medice | 2020-2023
    • Developed web applications using Python and React
    • Led team of 3 developers
    
    EDUCATION
    Bachelor of Computer Science | ABES Engineering college | 2024
    
    SKILLS
    Python, JavaScript, React, SQL, AWS
    """
    
    # Test overall analysis
    start_time = time.time()
    overall = analyzer.analyze_overall(sample_resume, "software_engineer")
    print(f"Overall analysis completed in {time.time() - start_time:.2f} seconds")
    print(f"Overall Score: {overall.overall_score:.1f}")
    print(f"Strengths: {overall.strengths}")
    print(f"Recommendations: {overall.recommendations}")
    
    # Test section analysis  
    start_time = time.time()
    section = analyzer.analyze_section(sample_resume, "software_engineer", "skills")
    print(f"Section analysis completed in {time.time() - start_time:.2f} seconds")
    print(f"Skills Score: {section.score:.1f}")
    print(f"Suggestions: {section.suggestions}")