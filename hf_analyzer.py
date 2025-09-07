import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional, Tuple
import re
import json
from dataclasses import dataclass
import warnings
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

class HuggingFaceAnalyzer:
    """Resume analyzer using Hugging Face Llama model"""
    
    def __init__(self, model_name: str = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
        
        # Section-specific keywords for different roles
        self.role_keywords = {
            'data_scientist': {
                'technical': ['python', 'r', 'sql', 'machine learning', 'deep learning', 'statistics', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'],
                'tools': ['jupyter', 'tableau', 'power bi', 'matplotlib', 'seaborn', 'plotly', 'aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                'concepts': ['data analysis', 'data visualization', 'nlp', 'computer vision', 'big data', 'hadoop', 'spark']
            },
            'software_engineer': {
                'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust'],
                'frameworks': ['react', 'angular', 'vue', 'node.js', 'spring', 'django', 'flask', 'express'],
                'tools': ['git', 'github', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'ci/cd'],
                'concepts': ['rest api', 'graphql', 'microservices', 'agile', 'scrum', 'tdd', 'unit testing']
            },
            'product_manager': {
                'methodologies': ['agile', 'scrum', 'kanban', 'lean', 'design thinking'],
                'tools': ['jira', 'confluence', 'figma', 'sketch', 'analytics', 'sql', 'excel'],
                'concepts': ['product strategy', 'roadmap', 'user stories', 'backlog', 'stakeholder management', 'a/b testing'],
                'skills': ['leadership', 'communication', 'analytical', 'strategic thinking', 'market research']
            },
            'marketing_manager': {
                'digital': ['seo', 'sem', 'ppc', 'google ads', 'facebook ads', 'social media', 'content marketing'],
                'tools': ['google analytics', 'hubspot', 'salesforce', 'marketo', 'adobe creative suite'],
                'concepts': ['brand management', 'campaign management', 'roi', 'conversion optimization', 'customer acquisition'],
                'skills': ['analytics', 'creativity', 'communication', 'project management']
            }
        }
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to basic analysis...")
            self.pipeline = None
    
    def _create_prompt(self, resume_text: str, job_role: str, analysis_type: str, section: str = None) -> str:
        """Create prompts for different types of analysis"""
        
        if analysis_type == "overall":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert resume reviewer and ATS specialist. Analyze this resume for a {job_role} position and provide comprehensive feedback.

<|start_header_id|>user<|end_header_id|>

Resume Text:
{resume_text[:]}

Please provide:
1. Overall Assessment (2-3 sentences)
2. Top 5 Strengths
3. Top 5 Areas for Improvement
4. Top 5 Recommendations
5. Overall Score (0-100)

Format your response as:
**Overall Assessment:** [assessment]
**Strengths:** 
- [strength 1]
- [strength 2] 
- [strength 3]
- [strength 4]
- [strength 5]
**Areas for Improvement:**
- [improvement 1]
- [improvement 2]
- [improvement 3]
- [improvement 4]
- [improvement 5]
**Recommendations:**
- [recommendation 1]
- [recommendation 2]
- [recommendation 3]
- [recommendation 4]
- [recommendation 5]
**Overall Score:** [score]/100

<|start_header_id|>assistant<|end_header_id|>"""

        elif analysis_type == "section":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert resume reviewer. Analyze the {section} section of this resume for a {job_role} position.

<|start_header_id|>user<|end_header_id|>

Resume Text:
{resume_text[:]}

Focus on the {section} section and provide:
1. Section Assessment (1-5 sentences)
2. Score (0-100)
3. Specific suggestions for improvement
4. Missing keywords for this role

Format your response as:
**Section Assessment:** [assessment]
**Score:** [score]/100
**Suggestions:**
- [suggestion 1]
- [suggestion 2]
- [suggestion 3]
- [suggestion 4]
**Missing Keywords:** [keyword1, keyword2, keyword3, keyword4, keyword5]

<|start_header_id|>assistant<|end_header_id|>"""

        elif analysis_type == "optimization":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert resume writer. Create an optimized version of this resume for a {job_role} position.

<|start_header_id|>user<|end_header_id|>

Original Resume:
{resume_text}

Please rewrite this resume with:
1. All relevant keywords naturally integrated
2. Improved formatting for ATS scanning
3. Stronger action verbs and quantified achievements
4. Clear section headers
5. Optimized length (aim for 500-700 words)

Return ONLY the optimized resume text, properly formatted with clear sections.

<|start_header_id|>assistant<|end_header_id|>"""

        return prompt
    
    def _parse_response(self, response: str, analysis_type: str) -> Dict:
        
        try:
            if analysis_type == "overall":
                return self._parse_overall_response(response)
            elif analysis_type == "section":
                return self._parse_section_response(response)
            elif analysis_type == "optimization":
                return {"optimized_resume": response.strip()}
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"error": str(e)}
    
    def _parse_overall_response(self, response: str) -> Dict:
        #Parse overall analysis response
        result = {
            "assessment": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "overall_score": 0
        }
        
        # Extract assessment
        assessment_match = re.search(r'\*\*Overall Assessment:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if assessment_match:
            result["assessment"] = assessment_match.group(1).strip()
        
        # Extract strengths
        strengths_match = re.search(r'\*\*Strengths:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if strengths_match:
            strengths_text = strengths_match.group(1)
            result["strengths"] = [line.strip('- ').strip() for line in strengths_text.split('\n') if line.strip().startswith('-')]
        
        # Extract weaknesses
        weaknesses_match = re.search(r'\*\*Areas for Improvement:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if weaknesses_match:
            weaknesses_text = weaknesses_match.group(1)
            result["weaknesses"] = [line.strip('- ').strip() for line in weaknesses_text.split('\n') if line.strip().startswith('-')]
        
        # Extract recommendations
        recommendations_match = re.search(r'\*\*Recommendations:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if recommendations_match:
            recommendations_text = recommendations_match.group(1)
            result["recommendations"] = [line.strip('- ').strip() for line in recommendations_text.split('\n') if line.strip().startswith('-')]
        
        # Extract score
        score_match = re.search(r'\*\*Overall Score:\*\*\s*(\d+)', response)
        if score_match:
            result["overall_score"] = int(score_match.group(1))
        
        return result
    
    def _parse_section_response(self, response: str) -> Dict:
        """Parse section analysis response"""
        result = {
            "assessment": "",
            "score": 0,
            "suggestions": [],
            "missing_keywords": []
        }
        
        # Extract assessment
        assessment_match = re.search(r'\*\*Section Assessment:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if assessment_match:
            result["assessment"] = assessment_match.group(1).strip()
        
        # Extract score
        score_match = re.search(r'\*\*Score:\*\*\s*(\d+)', response)
        if score_match:
            result["score"] = int(score_match.group(1))
        
        # Extract suggestions
        suggestions_match = re.search(r'\*\*Suggestions:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if suggestions_match:
            suggestions_text = suggestions_match.group(1)
            result["suggestions"] = [line.strip('- ').strip() for line in suggestions_text.split('\n') if line.strip().startswith('-')]
        
        # Extract missing keywords
        keywords_match = re.search(r'\*\*Missing Keywords:\*\*\s*(.+?)(?=\*\*|$)', response, re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            result["missing_keywords"] = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        
        return result
    
    def analyze_overall(self, resume_text: str, job_role: str) -> OverallFeedback:
        """Perform overall resume analysis"""
        if not self.pipeline:
            return self._fallback_overall_analysis(resume_text, job_role)
        
        try:
            prompt = self._create_prompt(resume_text, job_role, "overall")
            response = self.pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]['generated_text']
            
            # Extract only the assistant's response
            if '<|start_header_id|>assistant<|end_header_id|>' in response:
                response = response.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
            
            parsed = self._parse_response(response, "overall")
            
            return OverallFeedback(
                strengths=parsed.get("strengths", []),
                weaknesses=parsed.get("weaknesses", []),
                recommendations=parsed.get("recommendations", []),
                overall_score=parsed.get("overall_score", 0)
            )
            
        except Exception as e:
            print(f"Error in overall analysis: {str(e)}")
            return self._fallback_overall_analysis(resume_text, job_role)
    
    def analyze_section(self, resume_text: str, job_role: str, section: str) -> SectionFeedback:
        """Analyze a specific section of the resume"""
        if not self.pipeline:
            return self._fallback_section_analysis(resume_text, job_role, section)
        
        try:
            prompt = self._create_prompt(resume_text, job_role, "section", section)
            response = self.pipeline(prompt, max_new_tokens=400, do_sample=True, temperature=0.7)[0]['generated_text']
            
            # Extract only the assistant's response
            if '<|start_header_id|>assistant<|end_header_id|>' in response:
                response = response.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
            
            parsed = self._parse_response(response, "section")
            
            return SectionFeedback(
                section=section,
                score=parsed.get("score", 0),
                feedback=parsed.get("assessment", ""),
                suggestions=parsed.get("suggestions", []),
                missing_keywords=parsed.get("missing_keywords", [])
            )
            
        except Exception as e:
            print(f"Error in section analysis: {str(e)}")
            return self._fallback_section_analysis(resume_text, job_role, section)
    
    def optimize_resume(self, resume_text: str, job_role: str) -> str:
        """Generate optimized resume"""
        if not self.pipeline:
            return resume_text
        
        try:
            prompt = self._create_prompt(resume_text, job_role, "optimization")
            response = self.pipeline(prompt, max_new_tokens=800, do_sample=True, temperature=0.6)[0]['generated_text']
            
            # Extract only the assistant's response
            if '<|start_header_id|>assistant<|end_header_id|>' in response:
                response = response.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error in resume optimization: {str(e)}")
            return resume_text
    
    def _fallback_overall_analysis(self, resume_text: str, job_role: str) -> OverallFeedback:
        """Fallback analysis when model is not available"""
        # Basic keyword analysis
        keywords = self._get_role_keywords(job_role)
        found_keywords = [kw for kw in keywords if kw.lower() in resume_text.lower()]
        score = min((len(found_keywords) / len(keywords)) * 100, 100) if keywords else 50
        
        return OverallFeedback(
            strengths=["Resume has good structure", "Contains relevant experience"],
            weaknesses=["Could use more specific keywords", "Needs quantified achievements"],
            recommendations=["Add more role-specific keywords", "Include metrics and numbers", "Improve formatting"],
            overall_score=score
        )
    
    def _fallback_section_analysis(self, resume_text: str, job_role: str, section: str) -> SectionFeedback:
        """Fallback section analysis when model is not available"""
        keywords = self._get_role_keywords(job_role)
        section_text = self._extract_section_text(resume_text, section)
        found_keywords = [kw for kw in keywords if kw.lower() in section_text.lower()]
        score = min((len(found_keywords) / len(keywords)) * 100, 100) if keywords else 50
        
        return SectionFeedback(
            section=section,
            score=score,
            feedback=f"The {section} section is present but could be improved.",
            suggestions=["Add more specific details", "Include relevant keywords", "Use action verbs"],
            missing_keywords=keywords[:5]  # Top 5 missing keywords
        )
    
    def _get_role_keywords(self, job_role: str) -> List[str]:
        """Get keywords for a specific role"""
        role_key = job_role.lower().replace(' ', '_')
        if role_key in self.role_keywords:
            keywords = []
            for category in self.role_keywords[role_key].values():
                keywords.extend(category)
            return keywords
        return []
    
    def _extract_section_text(self, resume_text: str, section: str) -> str:
        """Extract text for a specific section"""
        lines = resume_text.split('\n')
        section_text = []
        in_section = False
        
        for line in lines:
            if section.lower() in line.lower() and len(line.strip()) < 50:
                in_section = True
                continue
            elif in_section and any(keyword in line.lower() for keyword in ['experience', 'education', 'skills', 'projects']):
                break
            elif in_section:
                section_text.append(line)
        
        return '\n'.join(section_text)

# Example usage
if __name__ == "__main__":
    analyzer = HuggingFaceAnalyzer()
    
    sample_resume = """
    John Doe
    john.doe@email.com | (555) 123-4567
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2020-2023
    • Developed web applications using Python and React
    • Led team of 3 developers
    
    EDUCATION
    Bachelor of Computer Science | University of Tech | 2018
    
    SKILLS
    Python, JavaScript, React, SQL, AWS
    """
    
    # Test overall analysis
    overall = analyzer.analyze_overall(sample_resume, "software_engineer")
    print(f"Overall Score: {overall.overall_score}")
    print(f"Strengths: {overall.strengths}")
    
    # Test section analysis
    section = analyzer.analyze_section(sample_resume, "software_engineer", "skills")
    print(f"Skills Score: {section.score}")
    print(f"Suggestions: {section.suggestions}")