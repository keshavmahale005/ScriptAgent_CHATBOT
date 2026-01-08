"""
Multi-Intent Detector
Detects user intent from complex sentences, handles:
- Multiple intents in one sentence
- Entity extraction (names, amounts, dates)
- Sentiment analysis
- Uncertainty detection
"""

import re
from typing import Dict, List, Tuple
from fuzzywuzzy import fuzz
import logging

logger = logging.getLogger(__name__)


class IntentDetector:
    """
    Detects user intent using rule-based + fuzzy matching
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        'POSITIVE': {
            'keywords': ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'definitely', 
                        'absolutely', 'correct', 'right', 'agreed', 'sounds good'],
            'phrases': ['i agree', 'that works', 'go ahead', 'lets do it']
        },
        'NEGATIVE': {
            'keywords': ['no', 'nope', 'nah', 'never', 'not interested', 'dont want',
                        'not now', 'not right now'],
            'phrases': ['no thanks', 'not interested', 'not for me']
        },
        'QUESTION': {
            'keywords': ['what', 'when', 'where', 'who', 'why', 'how', 'can you',
                        'could you', 'would you', 'explain', 'tell me'],
            'phrases': ['can you explain', 'what does', 'how does', 'tell me more']
        },
        'UNCERTAIN': {
            'keywords': ['maybe', 'not sure', 'dont know', 'unsure', 'uncertain',
                        'thinking', 'considering', 'might', 'could'],
            'phrases': ['not certain', 'let me think', 'need to think']
        },
        'OBJECTION': {
            'keywords': ['but', 'however', 'expensive', 'costly', 'worried', 'concern',
                        'problem', 'issue', 'uncomfortable'],
            'phrases': ['too expensive', 'too much', 'cant afford', 'not ready']
        },
        'REQUEST_INFO': {
            'keywords': ['details', 'information', 'more info', 'specifics', 'pricing',
                        'cost', 'rates', 'terms'],
            'phrases': ['more details', 'need information', 'want to know']
        },
        'BUSY': {
            'keywords': ['busy', 'bad time', 'not convenient', 'call back', 'later'],
            'phrases': ['bad time', 'not good time', 'call me back', 'call later']
        },
        'FRUSTRATED': {
            'keywords': ['frustrated', 'annoyed', 'upset', 'angry', 'third time',
                        'again', 'still', 'always'],
            'phrases': ['third time calling', 'nobody helped', 'still waiting']
        }
    }
    
    # Entity patterns
    ENTITY_PATTERNS = {
        'amount': r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars|usd|pounds)?',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'date': r'\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        'time': r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b'
    }
    
    def __init__(self, fuzzy_threshold: int = 80):
        self.fuzzy_threshold = fuzzy_threshold
    
    def detect(self, user_input: str) -> Dict:
        """
        Detect all intents and entities from user input
        
        Returns:
            {
                'primary_intent': str,
                'all_intents': List[str],
                'confidence': float,
                'entities': Dict,
                'sentiment': str,
                'is_question': bool
            }
        """
        user_lower = user_input.lower().strip()
        
        # Detect all intents
        intent_scores = self._score_all_intents(user_lower)
        
        # Extract entities
        entities = self._extract_entities(user_input)
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'NEUTRAL'
        primary_confidence = intent_scores.get(primary_intent, 0.0)
        
        # Get all intents above threshold
        all_intents = [
            intent for intent, score in intent_scores.items()
            if score >= 0.5
        ]
        
        # Detect sentiment
        sentiment = self._detect_sentiment(user_lower, intent_scores)
        
        # Check if it's a question
        is_question = self._is_question(user_input)
        
        result = {
            'primary_intent': primary_intent,
            'all_intents': all_intents,
            'confidence': primary_confidence,
            'entities': entities,
            'sentiment': sentiment,
            'is_question': is_question,
            'has_multiple_intents': len(all_intents) > 1
        }
        
        logger.info(f"Intent detection: {primary_intent} (confidence: {primary_confidence:.2f})")
        
        return result
    
    def _score_all_intents(self, user_input: str) -> Dict[str, float]:
        """Calculate score for each intent"""
        scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            matches = 0
            
            # Check exact keyword matches
            for keyword in patterns['keywords']:
                if keyword in user_input:
                    score += 1.0
                    matches += 1
            
            # Check phrase matches
            for phrase in patterns['phrases']:
                if phrase in user_input:
                    score += 1.5
                    matches += 1
            
            # Fuzzy matching for keywords
            words = user_input.split()
            for keyword in patterns['keywords']:
                for word in words:
                    similarity = fuzz.ratio(keyword, word)
                    if similarity >= self.fuzzy_threshold:
                        score += 0.5
                        matches += 1
                        break
            
            # Normalize score
            if matches > 0:
                max_possible = len(patterns['keywords']) + len(patterns['phrases'])
                scores[intent] = min(score / max_possible, 1.0)
        
        return scores
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities like amounts, dates, emails"""
        entities = {}
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        # Extract potential names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+\b', text)
        if names:
            entities['potential_names'] = names
        
        return entities
    
    def _detect_sentiment(self, user_input: str, intent_scores: Dict) -> str:
        """Detect overall sentiment"""
        # Check for negative sentiment
        if intent_scores.get('FRUSTRATED', 0) > 0.5 or intent_scores.get('OBJECTION', 0) > 0.5:
            return 'NEGATIVE'
        
        # Check for positive sentiment
        if intent_scores.get('POSITIVE', 0) > 0.5:
            return 'POSITIVE'
        
        # Check for uncertainty
        if intent_scores.get('UNCERTAIN', 0) > 0.5:
            return 'UNCERTAIN'
        
        # Neutral
        return 'NEUTRAL'
    
    def _is_question(self, text: str) -> bool:
        """Check if input is a question"""
        # Has question mark
        if '?' in text:
            return True
        
        # Starts with question word
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would']
        first_word = text.lower().split()[0] if text.split() else ''
        
        return first_word in question_words
    
    def get_keywords(self, user_input: str) -> List[str]:
        """Extract important keywords from user input"""
        # Remove stop words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again'
        }
        
        # Tokenize
        words = re.findall(r'\b\w+\b', user_input.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords