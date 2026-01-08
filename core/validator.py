"""
Response Validator
Validates that generated responses match the script and meet quality standards
"""

from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class ResponseValidator:
    """
    Validates responses to ensure accuracy and script adherence
    """
    
    def __init__(self, config):
        self.config = config
        self.min_confidence = config.MIN_RESPONSE_CONFIDENCE
        self.max_length = config.MAX_RESPONSE_LENGTH
        self.min_length = config.MIN_RESPONSE_LENGTH
    
    def validate(
        self,
        response: str,
        relevant_sections: List[Dict],
        user_input: str,
        intent_data: Dict
    ) -> Dict:
        """
        Validate generated response
        
        Returns:
            {
                'is_valid': bool,
                'confidence': float,
                'issues': List[str],
                'warnings': List[str]
            }
        """
        issues = []
        warnings = []
        confidence = 1.0
        
        # Check 1: Response length
        if len(response) < self.min_length:
            issues.append(f"Response too short ({len(response)} chars)")
            confidence -= 0.3
        
        if len(response) > self.max_length:
            warnings.append(f"Response too long ({len(response)} chars)")
            confidence -= 0.1
        
        # Check 2: Script overlap
        overlap_score = self._calculate_script_overlap(response, relevant_sections)
        if overlap_score < 0.2:
            issues.append(f"Low script overlap ({overlap_score:.2%})")
            confidence -= 0.4
        elif overlap_score < 0.4:
            warnings.append(f"Medium script overlap ({overlap_score:.2%})")
            confidence -= 0.1
        
        # Check 3: Hallucination detection
        has_hallucination = self._detect_hallucination(response)
        if has_hallucination:
            issues.append("Possible hallucination detected")
            confidence -= 0.5
        
        # Check 4: Inappropriate content
        has_inappropriate = self._check_inappropriate_content(response)
        if has_inappropriate:
            issues.append("Inappropriate content detected")
            confidence -= 0.6
        
        # Check 5: Repetition check
        is_repetitive = self._check_repetition(response)
        if is_repetitive:
            warnings.append("Response may be repetitive")
            confidence -= 0.1
        
        # Check 6: Intent alignment
        intent_aligned = self._check_intent_alignment(response, intent_data)
        if not intent_aligned:
            warnings.append("Response may not address user's intent")
            confidence -= 0.15
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine if valid
        is_valid = len(issues) == 0 and confidence >= self.min_confidence
        
        result = {
            'is_valid': is_valid,
            'confidence': confidence,
            'issues': issues,
            'warnings': warnings,
            'overlap_score': overlap_score
        }
        
        if not is_valid:
            logger.warning(f"Validation failed: {issues}")
        
        return result
    
    def _calculate_script_overlap(self, response: str, relevant_sections: List[Dict]) -> float:
        """Calculate word overlap between response and script"""
        if not relevant_sections:
            return 0.0
        
        # Get all script text
        script_text = " ".join([section['text'] for section in relevant_sections])
        
        # Tokenize
        response_words = set(self._tokenize(response.lower()))
        script_words = set(self._tokenize(script_text.lower()))
        
        if not response_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(response_words.intersection(script_words)) / len(response_words)
        
        return overlap
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _detect_hallucination(self, response: str) -> bool:
        """Detect common hallucination patterns"""
        response_lower = response.lower()
        
        # Hallucination indicators
        hallucination_phrases = [
            'i believe', 'i think', 'probably', 'might be',
            'could be', 'in my opinion', 'generally speaking',
            'usually', 'typically', 'as far as i know',
            'to the best of my knowledge', 'i assume'
        ]
        
        # Check for uncertainty phrases
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                return True
        
        # Check for made-up numbers/dates without context
        has_specific_numbers = bool(re.search(r'\$\d+\.\d{2}|\d{1,2}%', response))
        has_specific_dates = bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', response))
        
        if has_specific_numbers or has_specific_dates:
            # These should only appear if they're in the script
            # For now, just warn
            logger.debug("Response contains specific numbers/dates")
        
        return False
    
    def _check_inappropriate_content(self, response: str) -> bool:
        """Check for inappropriate content"""
        response_lower = response.lower()
        
        # Inappropriate patterns
        inappropriate = [
            'password', 'credit card', 'social security', 'ssn',
            'pin number', 'cvv', 'bank account', 'routing number'
        ]
        
        # Check if asking for sensitive info
        for term in inappropriate:
            if term in response_lower:
                logger.warning(f"Inappropriate content detected: {term}")
                return True
        
        return False
    
    def _check_repetition(self, response: str) -> bool:
        """Check if response is overly repetitive"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        # Check for repeated sentences
        unique_sentences = set(sentences)
        if len(unique_sentences) < len(sentences) * 0.8:
            return True
        
        # Check for repeated words
        words = self._tokenize(response)
        if len(words) < 5:
            return False
        
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 30% of the time
        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > len(words) * 0.3:
            return True
        
        return False
    
    def _check_intent_alignment(self, response: str, intent_data: Dict) -> bool:
        """Check if response addresses user's intent"""
        primary_intent = intent_data.get('primary_intent', 'UNKNOWN')
        
        # Intent-specific checks
        if primary_intent == 'QUESTION' and '?' not in response:
            # User asked a question, response should ideally answer it
            # This is a soft check
            pass
        
        if primary_intent == 'NEGATIVE' and 'no' not in response.lower():
            # User said no, response should acknowledge
            # Soft check
            pass
        
        # For now, always return True as this is a soft check
        # Can be enhanced based on specific requirements
        return True
    
    def suggest_improvements(self, validation_result: Dict, response: str) -> str:
        """Suggest improvements for invalid responses"""
        if validation_result['is_valid']:
            return response
        
        suggestions = []
        
        # If too short, suggest expanding
        if len(response) < self.min_length:
            suggestions.append("Expand response with more detail from script")
        
        # If low overlap, suggest using more script content
        if validation_result.get('overlap_score', 1.0) < 0.3:
            suggestions.append("Use more exact phrases from script")
        
        # If has issues, suggest reverting to exact script
        if validation_result['issues']:
            suggestions.append("Consider using exact script text")
        
        logger.info(f"Improvement suggestions: {suggestions}")
        
        return response  # Return original for now