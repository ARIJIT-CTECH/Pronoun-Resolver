"""
Heuristic-based Pronoun Resolution Module
Implements rule-based logic for pronoun coreference resolution
"""

import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class HeuristicPronounResolver:
    """
    Heuristic-based pronoun resolver using rule-based approaches:
    - Nearest noun matching
    - Gender matching
    - Number agreement
    - Subject preference
    """
    
    def __init__(self):
        # Load spaCy model for better NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using NLTK only.")
            self.nlp = None
        
        # Define pronoun categories
        self.pronouns = {
            'masculine': ['he', 'him', 'his', 'himself'],
            'feminine': ['she', 'her', 'hers', 'herself'],
            'neutral': ['it', 'its', 'itself'],
            'plural': ['they', 'them', 'their', 'theirs', 'themselves'],
            'first_person': ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'],
            'second_person': ['you', 'your', 'yours', 'yourself', 'yourselves']
        }
        
        # Gender indicators for names
        self.male_indicators = ['john', 'michael', 'david', 'james', 'robert', 'william', 'richard', 
                               'joseph', 'thomas', 'charles', 'christopher', 'daniel', 'matthew']
        self.female_indicators = ['mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara',
                                 'susan', 'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty']
    
    def get_pronoun_category(self, pronoun: str) -> str:
        """Determine the category of a pronoun."""
        pronoun_lower = pronoun.lower()
        for category, pron_list in self.pronouns.items():
            if pronoun_lower in pron_list:
                return category
        return 'unknown'
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract potential entities (nouns and named entities) from text."""
        entities = []
        
        if self.nlp:
            # Use spaCy for better entity extraction
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'type': 'named_entity'
                    })
            
            # Also extract noun chunks
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short noun phrases
                    entities.append({
                        'text': chunk.text,
                        'label': 'NOUN_CHUNK',
                        'start': chunk.start_char,
                        'end': chunk.end_char,
                        'type': 'noun_chunk'
                    })
        else:
            # Fallback to NLTK
            sentences = sent_tokenize(text)
            for sent in sentences:
                tokens = word_tokenize(sent)
                tagged = pos_tag(tokens)
                
                # Extract named entities
                tree = ne_chunk(tagged)
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity_text = ' '.join([token for token, pos in subtree.leaves()])
                        entities.append({
                            'text': entity_text,
                            'label': subtree.label(),
                            'start': text.find(entity_text),
                            'end': text.find(entity_text) + len(entity_text),
                            'type': 'named_entity'
                        })
                
                # Extract noun phrases
                for i, (word, pos) in enumerate(tagged):
                    if pos.startswith('NN') and word.lower() not in ['time', 'people', 'way', 'day', 'man', 'thing']:
                        entities.append({
                            'text': word,
                            'label': 'NOUN',
                            'start': text.find(word),
                            'end': text.find(word) + len(word),
                            'type': 'noun'
                        })
        
        return entities
    
    def get_gender(self, entity: str) -> str:
        """Determine gender of an entity based on name indicators."""
        entity_lower = entity.lower()
        if entity_lower in self.male_indicators:
            return 'masculine'
        elif entity_lower in self.female_indicators:
            return 'feminine'
        return 'unknown'
    
    def is_plural(self, entity: str) -> bool:
        """Check if an entity is plural."""
        # Simple heuristic: ends with 's' but not common singular words ending with 's'
        if entity.endswith('s') and len(entity) > 3:
            return True
        return False
    
    def find_antecedents(self, text: str) -> Dict[str, str]:
        """Find antecedents for pronouns using heuristic rules."""
        sentences = sent_tokenize(text)
        entities = self.extract_entities(text)
        pronoun_resolutions = {}
        
        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            # Find pronouns in this sentence
            for i, (word, pos) in enumerate(tagged):
                if pos == 'PRP' or pos == 'PRP$':  # Personal pronouns
                    pronoun_category = self.get_pronoun_category(word)
                    
                    if pronoun_category == 'unknown':
                        continue
                    
                    # Find potential antecedents
                    candidates = []
                    
                    # Look in current sentence (before pronoun)
                    for j in range(i):
                        if tagged[j][1].startswith('NN'):
                            candidate_word, candidate_pos = tagged[j]
                            candidates.append({
                                'text': candidate_word,
                                'sentence_distance': 0,
                                'word_distance': i - j,
                                'sentence_idx': sent_idx
                            })
                    
                    # Look in previous sentences
                    for prev_sent_idx in range(max(0, sent_idx - 2), sent_idx):
                        prev_sentence = sentences[prev_sent_idx]
                        prev_tokens = word_tokenize(prev_sentence)
                        prev_tagged = pos_tag(prev_tokens)
                        
                        for prev_word, prev_pos in prev_tagged:
                            if prev_pos.startswith('NN'):
                                candidates.append({
                                    'text': prev_word,
                                    'sentence_distance': sent_idx - prev_sent_idx,
                                    'word_distance': float('inf'),  # Different sentence
                                    'sentence_idx': prev_sent_idx
                                })
                    
                    # Score candidates based on heuristic rules
                    best_candidate = self.score_candidates(word, candidates, pronoun_category)
                    
                    if best_candidate:
                        pronoun_resolutions[word.lower()] = best_candidate['text']
        
        return pronoun_resolutions
    
    def score_candidates(self, pronoun: str, candidates: List[Dict], pronoun_category: str) -> Optional[Dict]:
        """Score and select the best antecedent candidate."""
        if not candidates:
            return None
        
        scored_candidates = []
        
        for candidate in candidates:
            score = 0
            candidate_text = candidate['text']
            
            # Rule 1: Prefer closer candidates (recency)
            if candidate['word_distance'] != float('inf'):
                score += max(0, 10 - candidate['word_distance'])
            
            # Rule 2: Sentence distance penalty
            score -= candidate['sentence_distance'] * 2
            
            # Rule 3: Gender agreement
            if pronoun_category in ['masculine', 'feminine']:
                candidate_gender = self.get_gender(candidate_text)
                if candidate_gender == pronoun_category:
                    score += 15
                elif candidate_gender == 'unknown':
                    score += 5
                else:
                    score -= 10
            
            # Rule 4: Number agreement
            if pronoun_category == 'plural':
                if self.is_plural(candidate_text):
                    score += 10
                else:
                    score -= 5
            elif pronoun_category in ['masculine', 'feminine', 'neutral']:
                if not self.is_plural(candidate_text):
                    score += 5
                else:
                    score -= 10
            
            # Rule 5: Subject preference (simplified - prefer first words)
            if candidate['word_distance'] < 3:  # Likely subject position
                score += 3
            
            # Rule 6: Named entity preference
            if any(char.isupper() for char in candidate_text):
                score += 5
            
            scored_candidates.append((score, candidate))
        
        # Return the highest scoring candidate
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return scored_candidates[0][1]
        
        return None
    
    def resolve_pronouns(self, text: str) -> Dict:
        """Main method to resolve pronouns in text."""
        resolutions = self.find_antecedents(text)
        
        # Create highlighted text
        highlighted_text = text
        for pronoun, antecedent in resolutions.items():
            # Replace pronouns with highlighted versions
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            highlighted_text = re.sub(
                pattern, 
                f"**{pronoun}** → [{antecedent}]", 
                highlighted_text,
                flags=re.IGNORECASE
            )
        
        return {
            'resolutions': resolutions,
            'highlighted_text': highlighted_text,
            'method': 'heuristic'
        }


def test_heuristic_resolver():
    """Test the heuristic pronoun resolver."""
    resolver = HeuristicPronounResolver()
    
    test_text = "John went to the store. He bought milk. Then he met Sarah and she greeted him."
    
    result = resolver.resolve_pronouns(test_text)
    
    print("Heuristic Pronoun Resolution Results:")
    print(f"Input: {test_text}")
    print(f"Resolutions: {result['resolutions']}")
    print(f"Highlighted: {result['highlighted_text']}")


if __name__ == "__main__":
    test_heuristic_resolver()
