"""
Semantic Matcher - Find relevant script sections (Minimal version)
Uses TF-IDF instead of sentence transformers for lighter dependencies
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticMatcher:
    def __init__(self):
        """Initialize semantic matcher with TF-IDF"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.script_embeddings = None
        self.script_sections = []
        
    def encode_sections(self, sections: List[Dict]) -> None:
        """
        Encode script sections using TF-IDF
        
        Args:
            sections: List of script sections with text
        """
        try:
            self.script_sections = sections
            
            # Combine section content for embedding
            texts = []
            for section in sections:
                text_parts = []
                
                # Add section name
                if 'name' in section:
                    text_parts.append(section['name'])
                
                # Add dialogue
                if 'dialogue' in section:
                    text_parts.append(section['dialogue'])
                
                # Add transitions
                if 'transitions' in section:
                    for trans in section['transitions']:
                        if isinstance(trans, dict) and 'condition' in trans:
                            text_parts.append(trans['condition'])
                
                texts.append(' '.join(text_parts))
            
            # Fit and transform
            self.script_embeddings = self.vectorizer.fit_transform(texts)
            
        except Exception as e:
            print(f"Error encoding sections: {e}")
            raise
    
    def find_relevant_sections(
        self,
        user_input: str,
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Find relevant script sections for user input
        
        Args:
            user_input: User's message
            top_k: Number of sections to return
            
        Returns:
            List of (section, similarity_score) tuples
        """
        try:
            if self.script_embeddings is None:
                return []
            
            # Encode user input
            user_embedding = self.vectorizer.transform([user_input])
            
            # Calculate similarities
            similarities = cosine_similarity(user_embedding, self.script_embeddings)[0]
            
            # Get top k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                
                # Only include if above threshold (0.1 for minimal matching)
                if score >= 0.1:
                    results.append((self.script_sections[idx], score))
            
            return results
            
        except Exception as e:
            print(f"Error finding sections: {e}")
            return []
    
    def find_section_by_name(self, section_name: str) -> Tuple[Dict, float]:
        """
        Find a specific section by name
        
        Args:
            section_name: Name of section to find
            
        Returns:
            (section, similarity_score) or (None, 0.0)
        """
        try:
            matches = self.find_relevant_sections(section_name, top_k=1)
            return matches[0] if matches else (None, 0.0)
        except Exception as e:
            print(f"Error finding section by name: {e}")
            return (None, 0.0)