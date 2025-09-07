from typing import Dict, List
from dataclasses import dataclass
import re
import time
from functools import lru_cache

# Import our optimized components
from hf_analyzer import OptimizedHuggingFaceAnalyzer, SectionFeedback, OverallFeedback
from resume_processor import OptimizedResumeProcessor, ATSAnalysis

@dataclass
class SectionAnalysis:
    section_name: str
    content: str
    score: float
    feedback: str
    suggestions: List[str]
    missing_keywords: List[str]
    improvements: List[str]

@dataclass
class ComprehensiveFeedback:
    overall_feedback: OverallFeedback
    section_analyses: List[SectionAnalysis]
    ats_analysis: ATSAnalysis
    priority_improvements: List[str]
    estimated_improvement: float
    processing_time: float

class OptimizedFeedbackGenerator:
    """Optimized feedback generator focused on speed and practical insights"""
    
    def __init__(self, hf_analyzer: OptimizedHuggingFaceAnalyzer = None):
        self.hf_analyzer = hf_analyzer or OptimizedHuggingFaceAnalyzer()
        self.resume_processor = OptimizedResumeProcessor()
        
        # Pre-defined feedback templates for common scenarios (for speed)
        self.feedback_templates = {
            'high_score': {
                'strengths': [
                    "Strong alignment with target role requirements",
                    "Well-structured resume with clear sections",
                    "Good use of quantified achievements",
                    "Professional formatting and presentation",
                    "Comprehensive skill set listed"
                ],
                'recommendations': [
                    "Continue to tailor keywords for specific job applications",
                    "Keep quantified achievements prominent",
                    "Maintain consistent formatting across all sections"
                ]
            },
            'medium_score': {
                'weaknesses': [
                    "Could benefit from more role-specific keywords",
                    "Some sections need more detailed information",
                    "Consider adding quantified achievements"
                ],
                'recommendations': [
                    "Add more specific metrics and numbers to achievements",
                    "Include additional relevant technical skills",
                    "Enhance job descriptions with stronger action verbs"
                ]
            },
            'low_score': {
                'weaknesses': [
                    "Missing key industry-specific keywords",
                    "Lacks quantified achievements and metrics",
                    "Resume structure could be improved",
                    "Some essential sections may be missing"
                ],
                'recommendations': [
                    "Add specific numbers and percentages to showcase impact",
                    "Include more relevant technical skills and keywords",
                    "Reorganize content with clear section headers",
                    "Use bullet points for better readability"
                ]
            }
        }
        
        # Section-specific quick analysis rules
        self.section_rules = {
            'summary': {
                'ideal_length': (50, 150),  # words
                'must_have': ['experience', 'skills'],
                'scoring_factors': ['length', 'keywords', 'career_focus']
            },
            'experience': {
                'must_have': ['dates', 'company', 'achievements'],
                'bonus_points': ['metrics', 'action_verbs', 'technologies'],
                'scoring_factors': ['quantification', 'relevance', 'recency']
            },
            'skills': {
                'ideal_count': (8, 15),
                'categories': ['technical', 'tools', 'languages', 'frameworks'],
                'scoring_factors': ['relevance', 'breadth', 'organization']
            },
            'education': {
                'must_have': ['degree', 'institution'],
                'bonus_points': ['gpa', 'honors', 'relevant_coursework'],
                'scoring_factors': ['relevance', 'recency', 'completeness']
            },
            'projects': {
                'must_have': ['description', 'technologies'],
                'bonus_points': ['outcomes', 'links', 'impact'],
                'scoring_factors': ['relevance', 'complexity', 'results']
            }
        }
    
    def generate_comprehensive_feedback(self, resume_text: str, job_role: str, job_description: str = "") -> ComprehensiveFeedback:
        """Generate comprehensive feedback optimized for speed"""
        start_time = time.time()
        
        # Step 1: Quick ATS analysis (most important)
        ats_analysis = self.resume_processor.comprehensive_ats_analysis(resume_text, job_role, job_description)
        
        # Step 2: Generate overall feedback based on score
        overall_feedback = self._generate_quick_overall_feedback(ats_analysis, resume_text, job_role)
        
        # Step 3: Section-wise analysis (optimized)
        sections = self.resume_processor.extract_sections(resume_text)
        section_analyses = self._generate_quick_section_analyses(sections, job_role, ats_analysis)
        
        # Step 4: Priority improvements
        priority_improvements = self._generate_priority_improvements(ats_analysis, section_analyses)
        
        # Step 5: Estimate improvement potential
        estimated_improvement = self._estimate_improvement_potential(ats_analysis)
        
        processing_time = time.time() - start_time
        
        return ComprehensiveFeedback(
            overall_feedback=overall_feedback,
            section_analyses=section_analyses,
            ats_analysis=ats_analysis,
            priority_improvements=priority_improvements,
            estimated_improvement=estimated_improvement,
            processing_time=processing_time
        )
    
    def _generate_quick_overall_feedback(self, ats_analysis: ATSAnalysis, resume_text: str, job_role: str) -> OverallFeedback:
        """Generate overall feedback quickly using templates and rules"""
        
        score = ats_analysis.total_score
        
        # Determine feedback category based on score
        if score >= 75:
            category = 'high_score'
        elif score >= 50:
            category = 'medium_score'
        else:
            category = 'low_score'
        
        # Start with template feedback
        template = self.feedback_templates[category]
        
        strengths = template.get('strengths', []).copy()
        weaknesses = template.get('weaknesses', []).copy()
        recommendations = template.get('recommendations', []).copy()
        
        # Add specific feedback based on component scores
        components = ats_analysis.component_scores
        
        # Keyword-specific feedback
        if components['keyword_match'] < 50:
            weaknesses.append(f"Low keyword match for {job_role} role")
            recommendations.append("Research job descriptions to identify key terms for your field")
        elif components['keyword_match'] >= 80:
            strengths.append("Excellent keyword alignment with target role")
        
        # Formatting-specific feedback
        if components['formatting'] < 60:
            weaknesses.append("Resume formatting needs improvement for ATS compatibility")
            recommendations.append("Use consistent bullet points and clear section headers")
        elif components['formatting'] >= 85:
            strengths.append("Professional formatting optimized for ATS systems")
        
        # Section completeness feedback
        if components['section_completeness'] < 70:
            weaknesses.append("Missing some important resume sections")
            recommendations.append("Ensure all essential sections are present and complete")
        
        # Readability feedback
        if components['readability'] < 70:
            weaknesses.append("Content readability could be enhanced")
            recommendations.append("Use shorter sentences and clearer language")
        
        # Add specific improvements from ATS analysis
        recommendations.extend(ats_analysis.recommendations[:2])  # Add top 2 ATS recommendations
        
        return OverallFeedback(
            strengths=strengths[:5],  # Limit to top 5
            weaknesses=weaknesses[:5],  # Limit to top 5
            recommendations=recommendations[:5],  # Limit to top 5
            overall_score=score
        )
    
    def _generate_quick_section_analyses(self, sections: Dict[str, str], job_role: str, ats_analysis: ATSAnalysis) -> List[SectionAnalysis]:
        """Generate section analyses quickly using rule-based approach"""
        
        section_analyses = []
        keywords = self.resume_processor.get_role_keywords(job_role)
        
        # Prioritize most important sections for analysis
        priority_sections = ['experience', 'skills', 'summary', 'education', 'projects']
        
        for section_name in priority_sections:
            if section_name in sections and sections[section_name].strip():
                section_content = sections[section_name]
                analysis = self._analyze_section_quickly(section_name, section_content, tuple(keywords), job_role)
                section_analyses.append(analysis)
                
                # Limit to 4 sections for performance
                if len(section_analyses) >= 4:
                    break
        
        return section_analyses
    
    @lru_cache(maxsize=50)
    def _analyze_section_quickly(self, section_name: str, content: str, keywords: tuple, job_role: str) -> SectionAnalysis:
        """Quick section analysis using cached results"""
        
        keywords = list(keywords)  # Convert back from tuple for caching
        
        # Calculate basic score
        score = self._calculate_section_score_fast(section_name, content, keywords)
        
        # Generate feedback
        feedback = self._generate_section_feedback_fast(section_name, content, score)
        
        # Generate suggestions
        suggestions = self._generate_section_suggestions_fast(section_name, content)
        
        # Find missing keywords
        content_lower = content.lower()
        missing_keywords = [kw for kw in keywords[:10] if kw.lower() not in content_lower]
        
        # Generate improvements
        improvements = self._generate_section_improvements_fast(section_name, score)
        
        return SectionAnalysis(
            section_name=section_name,
            content=content[:200] + "..." if len(content) > 200 else content,  # Truncate for memory
            score=score,
            feedback=feedback,
            suggestions=suggestions[:3],  # Limit suggestions
            missing_keywords=missing_keywords[:5],  # Limit keywords
            improvements=improvements[:3]  # Limit improvements
        )
    
    def _calculate_section_score_fast(self, section_name: str, content: str, keywords: List[str]) -> float:
        """Fast section scoring using simple rules"""
        
        if section_name == 'experience':
            return self._score_experience_fast(content)
        elif section_name == 'skills':
            return self._score_skills_fast(content, keywords)
        elif section_name == 'summary':
            return self._score_summary_fast(content, keywords)
        elif section_name == 'education':
            return self._score_education_fast(content)
        elif section_name == 'projects':
            return self._score_projects_fast(content, keywords)
        else:
            # Generic scoring
            return self._score_generic_section_fast(content, keywords)
    
    def _score_experience_fast(self, content: str) -> float:
        """Fast experience section scoring"""
        score = 0
        
        # Dates (25 points)
        if re.search(r'\b20\d{2}\b', content):
            score += 25
        
        # Bullet points (20 points)
        if re.search(r'[•\-\*]', content):
            score += 20
        
        # Quantified achievements (35 points)
        if re.search(r'\d+%|\d+\+|\$\d+|increased|improved|reduced', content.lower()):
            score += 35
        
        # Action verbs (20 points)
        action_verbs = ['led', 'managed', 'developed', 'created', 'implemented']
        found_verbs = sum(1 for verb in action_verbs if verb in content.lower())
        score += min(found_verbs * 5, 20)
        
        return min(score, 100)
    
    def _score_skills_fast(self, content: str, keywords: List[str]) -> float:
        """Fast skills section scoring"""
        # Count skills
        skills = [s.strip() for s in re.split(r'[,;\n]', content) if s.strip()]
        skill_count = len(skills)
        
        # Skill count score (40 points)
        if 8 <= skill_count <= 15:
            count_score = 40
        elif 5 <= skill_count <= 20:
            count_score = 30
        else:
            count_score = 20
        
        # Keyword relevance (60 points)
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
        keyword_score = min((keyword_matches / max(len(keywords), 1)) * 60, 60)
        
        return min(count_score + keyword_score, 100)
    
    def _score_summary_fast(self, content: str, keywords: List[str]) -> float:
        """Fast summary section scoring"""
        word_count = len(content.split())
        
        # Length score (40 points)
        if 50 <= word_count <= 150:
            length_score = 40
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            length_score = 30
        else:
            length_score = 20
        
        # Keyword score (40 points)
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in keywords[:10] if kw.lower() in content_lower)
        keyword_score = min((keyword_matches / 10) * 40, 40)
        
        # Professional tone score (20 points)
        professional_words = ['experienced', 'skilled', 'expertise', 'proven', 'successful']
        tone_score = 20 if any(word in content_lower for word in professional_words) else 10
        
        return min(length_score + keyword_score + tone_score, 100)
    
    def _score_education_fast(self, content: str) -> float:
        """Fast education section scoring"""
        score = 0
        
        # Degree (50 points)
        if any(degree in content.lower() for degree in ['bachelor', 'master', 'phd', 'degree']):
            score += 50
        
        # Institution (30 points)
        if any(word in content.lower() for word in ['university', 'college', 'institute']):
            score += 30
        
        # Date (20 points)
        if re.search(r'\b20\d{2}\b', content):
            score += 20
        
        return min(score, 100)
    
    def _score_projects_fast(self, content: str, keywords: List[str]) -> float:
        """Fast projects section scoring"""
        score = 0
        
        # Project count (30 points)
        project_count = content.lower().count('project') + len(re.findall(r'[•\-\*]', content))
        score += min(project_count * 10, 30)
        
        # Technology keywords (40 points)
        content_lower = content.lower()
        tech_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
        score += min((tech_matches / max(len(keywords), 1)) * 40, 40)
        
        # Impact/results (30 points)
        if any(word in content_lower for word in ['improved', 'increased', 'built', 'created']):
            score += 30
        
        return min(score, 100)
    
    def _score_generic_section_fast(self, content: str, keywords: List[str]) -> float:
        """Fast generic section scoring"""
        if not content.strip():
            return 0
        
        # Basic presence score
        score = 50
        
        # Keyword relevance
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in keywords[:5] if kw.lower() in content_lower)
        score += min(keyword_matches * 10, 30)
        
        # Structure score
        if re.search(r'[•\-\*]', content):
            score += 20
        
        return min(score, 100)
    
    def _generate_section_feedback_fast(self, section_name: str, content: str, score: float) -> str:
        """Generate quick feedback for sections"""
        
        if score >= 80:
            return f"{section_name.title()} section is well-developed with strong content and good structure."
        elif score >= 60:
            return f"{section_name.title()} section has good foundation but could be enhanced with more specific details."
        else:
            return f"{section_name.title()} section needs significant improvement to meet professional standards."
    
    def _generate_section_suggestions_fast(self, section_name: str, content: str) -> List[str]:
        """Generate quick suggestions for sections"""
        
        suggestions = []
        
        if section_name == 'experience':
            if not re.search(r'\d+%|\d+\+|\$\d+', content):
                suggestions.append("Add quantified achievements with specific numbers")
            if not re.search(r'[•\-\*]', content):
                suggestions.append("Use bullet points to organize accomplishments")
            suggestions.append("Start each point with a strong action verb")
        
        elif section_name == 'skills':
            suggestions.append("Organize skills by category (Programming, Tools, etc.)")
            suggestions.append("Include proficiency levels for key skills")
            suggestions.append("Add trending technologies relevant to your field")
        
        elif section_name == 'summary':
            suggestions.append("Include years of experience and key specializations")
            suggestions.append("Highlight your unique value proposition")
            suggestions.append("Keep it concise but impactful (2-3 sentences)")
        
        elif section_name == 'education':
            suggestions.append("Include graduation date if recent")
            suggestions.append("Add relevant coursework or academic projects")
            suggestions.append("Include GPA if 3.5 or higher")
        
        elif section_name == 'projects':
            suggestions.append("Include links to live projects or repositories")
            suggestions.append("Describe technologies used and your role")
            suggestions.append("Highlight measurable outcomes or impact")
        
        return suggestions[:3]
    
    def _generate_section_improvements_fast(self, section_name: str, score: float) -> List[str]:
        """Generate quick improvements for sections"""
        
        improvements = []
        
        if score < 60:
            improvements.append(f"Expand {section_name} section with more detailed information")
            improvements.append(f"Add relevant keywords specific to your target role")
            improvements.append(f"Improve formatting and structure for better readability")
        elif score < 80:
            improvements.append(f"Enhance {section_name} section with specific examples")
            improvements.append(f"Add more quantified achievements where possible")
        
        return improvements
    
    def _generate_priority_improvements(self, ats_analysis: ATSAnalysis, section_analyses: List[SectionAnalysis]) -> List[str]:
        """Generate prioritized improvement recommendations"""
        
        priorities = []
        
        # ATS-based priorities
        if ats_analysis.total_score < 60:
            priorities.append("🚨 URGENT: Overall ATS score needs significant improvement")
        
        if ats_analysis.component_scores['keyword_match'] < 50:
            priorities.append("🎯 HIGH: Add more role-specific keywords throughout resume")
        
        if ats_analysis.component_scores['formatting'] < 60:
            priorities.append("📝 HIGH: Improve formatting for better ATS compatibility")
        
        # Section-based priorities
        low_scoring_sections = [sa for sa in section_analyses if sa.score < 60]
        if low_scoring_sections:
            section_names = [sa.section_name for sa in low_scoring_sections]
            priorities.append(f"📊 MEDIUM: Strengthen {', '.join(section_names)} sections")
        
        # Specific improvements
        if ats_analysis.missing_keywords:
            priorities.append(f"🔍 MEDIUM: Include missing keywords: {', '.join(ats_analysis.missing_keywords[:3])}")
        
        return priorities[:5]  # Top 5 priorities
    
    def _estimate_improvement_potential(self, ats_analysis: ATSAnalysis) -> float:
        """Estimate potential score improvement"""
        
        current_score = ats_analysis.total_score
        
        # Calculate improvement potential based on current gaps
        improvement_potential = 0
        
        # Keyword improvements
        if ats_analysis.component_scores['keyword_match'] < 70:
            improvement_potential += 15
        
        # Formatting improvements
        if ats_analysis.component_scores['formatting'] < 70:
            improvement_potential += 10
        
        # Section completeness improvements
        if ats_analysis.component_scores['section_completeness'] < 80:
            improvement_potential += 12
        
        # Readability improvements
        if ats_analysis.component_scores['readability'] < 70:
            improvement_potential += 8
        
        # Cap the potential improvement
        estimated_new_score = min(current_score + improvement_potential, 95)
        
        return estimated_new_score

# Performance testing
def test_optimized_feedback_performance():
    """Test the performance of optimized feedback generator"""
    
    feedback_gen = OptimizedFeedbackGenerator()
    
    sample_resume = """
    Jane Smith
    jane.smith@email.com | (555) 987-6543 | linkedin.com/in/janesmith
    
    SUMMARY
    Data Scientist with 4+ years of experience in machine learning and statistical analysis.
    Expertise in Python, SQL, and data visualization. Proven track record of delivering
    actionable insights that drive business decisions.
    
    EXPERIENCE
    Senior Data Scientist | DataCorp Inc | 2021-2024
    • Developed machine learning models that improved customer retention by 25%
    • Led cross-functional team of 5 analysts on predictive analytics projects
    • Built automated reporting systems reducing manual work by 60%
    • Implemented A/B testing framework for product optimization
    
    Data Scientist | Analytics Solutions | 2020-2021
    • Created statistical models for sales forecasting with 95% accuracy
    • Analyzed customer behavior data to identify growth opportunities
    • Collaborated with product team to optimize user experience
    
    EDUCATION
    Master of Science in Data Science | Stanford University | 2020
    Bachelor of Science in Statistics | UC Berkeley | 2018
    
    SKILLS
    Programming: Python, R, SQL, Java
    Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch
    Tools: Jupyter, Git, Docker, AWS, Tableau, Power BI
    Databases: PostgreSQL, MongoDB, Redis
    
    PROJECTS
    Customer Segmentation Analysis | 2023
    • Implemented K-means clustering to identify customer segments
    • Increased marketing campaign effectiveness by 35%
    • Technologies: Python, Scikit-learn, Matplotlib
    
    Stock Price Prediction Model | 2023  
    • Built LSTM neural network for stock price forecasting
    • Achieved 12% improvement over baseline models
    • Technologies: Python, TensorFlow, Yahoo Finance API
    
    CERTIFICATIONS
    • AWS Certified Machine Learning - Specialty
    • Google Analytics Individual Qualification
    • Coursera Machine Learning Certificate
    """
    
    print("🔄 Testing OptimizedFeedbackGenerator performance...")
    
    # Test comprehensive feedback generation
    start_time = time.time()
    feedback = feedback_gen.generate_comprehensive_feedback(
        sample_resume, "data_scientist", "Looking for senior data scientist with Python and ML experience"
    )
    
    total_time = time.time() - start_time
    
    print(f"⏱️ Total feedback generation time: {total_time:.3f}s")
    print(f"📊 Overall ATS Score: {feedback.ats_analysis.total_score:.1f}/100")
    print(f"📈 Estimated improvement potential: {feedback.estimated_improvement:.1f}/100")
    print(f"🎯 Priority improvements: {len(feedback.priority_improvements)}")
    print(f"📋 Section analyses generated: {len(feedback.section_analyses)}")
    
    # Performance breakdown
    print(f"\n📈 Performance Metrics:")
    print(f"• Processing time: {feedback.processing_time:.3f}s")
    print(f"• Sections analyzed: {[sa.section_name for sa in feedback.section_analyses]}")
    print(f"• Missing keywords found: {len(feedback.ats_analysis.missing_keywords)}")
    print(f"• Component scores calculated: {len(feedback.ats_analysis.component_scores)}")
    
    # Test caching performance (second run should be faster)
    print(f"\n🔄 Testing cache performance...")
    start_time = time.time()
    feedback2 = feedback_gen.generate_comprehensive_feedback(
        sample_resume, "data_scientist", "Looking for senior data scientist with Python and ML experience"
    )
    cached_time = time.time() - start_time
    print(f"⏱️ Cached run time: {cached_time:.3f}s (should be faster)")
    
    # Display sample results
    print(f"\n📋 Sample Results:")
    print(f"Overall Score: {feedback.overall_feedback.overall_score:.1f}/100")
    print(f"Top Strengths: {feedback.overall_feedback.strengths[:2]}")
    print(f"Top Recommendations: {feedback.overall_feedback.recommendations[:2]}")
    print(f"Priority Improvements: {feedback.priority_improvements[:2]}")

if __name__ == "__main__":
    test_optimized_feedback_performance()