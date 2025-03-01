import spacy
import re

class JewelryPromptParser:
    """
    A class to parse jewelry editing prompts and extract structured information
    about the requested edits.
    """
    
    def __init__(self, nlp=None):
        """
        Initialize the parser with optional spaCy model
        
        Args:
            nlp: Optional pre-loaded spaCy model
        """
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp
            
        # Define action keywords
        self.action_patterns = {
            "change": ["change", "convert", "transform", "switch", "make this", "make the"],
            "make_bigger": ["make bigger", "increase size", "enlarge", "bigger"],
            "make_smaller": ["make smaller", "decrease size", "reduce", "smaller"],
            "add": ["add", "insert", "include", "put", "place"],
            "remove": ["remove", "delete", "eliminate", "take out", "take away"]
        }
        
        # Define common jewelry terms for better target identification
        self.jewelry_terms = [
            "gold", "silver", "platinum", "white gold", "rose gold", 
            "diamond", "diamonds", "ruby", "emerald", "sapphire", "pearl", "gems", "gemstone",
            "ring", "necklace", "bracelet", "earring", "pendant", "chain",
            "setting", "prong", "clasp", "bail", "bezel", "stone"
        ]
    
    def parse(self, prompt):
        """
        Parse a jewelry editing prompt and extract structured information
        
        Args:
            prompt: The natural language prompt describing the edit
            
        Returns:
            dict: A dictionary containing action, target, and new_state (if applicable)
        """
        # Initialize variables
        result = {
            "action": None,
            "target": None,
            "new_state": None,
            "location": None,
            "raw_prompt": prompt
        }
        
        # Convert to lowercase for easier matching
        text = prompt.lower()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Identify action
        result["action"] = self._identify_action(text)
        
        # Extract targets and new states based on action
        if result["action"] == "change":
            # Handle change actions like "Change gold to white gold"
            target, new_state = self._extract_change_components(text)
            result["target"] = target
            result["new_state"] = new_state
            
        elif result["action"] in ["make_bigger", "make_smaller"]:
            # Handle size changes like "Make the diamonds bigger"
            result["target"] = self._extract_size_target(text, result["action"])
            
        elif result["action"] == "add":
            # Handle additions like "Add a diamond to the center"
            what_to_add, location = self._extract_add_components(text)
            result["target"] = what_to_add
            result["location"] = location
            
        elif result["action"] == "remove":
            # Handle removals like "Remove the small diamonds"
            result["target"] = self._extract_remove_target(text)
        
        # If no target was found but we have jewelry terms in the prompt, try to extract them
        if not result["target"]:
            result["target"] = self._extract_jewelry_terms(text)
            
        return result
    
    def _identify_action(self, text):
        """Identify the primary action in the prompt"""
        for action, patterns in self.action_patterns.items():
            if any(pattern in text for pattern in patterns):
                return action
        # Default to change if no action is detected
        return "change"
    
    def _extract_change_components(self, text):
        """Extract target and new state from a change prompt"""
        # Check for patterns like "Change X to Y"
        to_pattern = re.compile(r"(?:change|convert|transform|switch|make)\s+(?:the|this)?\s*([\w\s]+?)\s+(?:to|into)\s+([\w\s]+)")
        match = to_pattern.search(text)
        
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        # Check for patterns like "Make this ring in gold"
        in_pattern = re.compile(r"make\s+(?:the|this)\s+([\w\s]+?)\s+in\s+([\w\s]+)")
        match = in_pattern.search(text)
        
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        # Default: try to extract jewelry terms
        for term in self.jewelry_terms:
            if term in text:
                # Find what comes after the term
                parts = text.split(term)
                if len(parts) > 1 and parts[1]:
                    words = parts[1].strip().split()
                    if words:
                        return term, " ".join(words[:2]).strip()
        
        return None, None
    
    def _extract_size_target(self, text, action):
        """Extract target for size change prompts"""
        # Pattern for "Make X bigger/smaller"
        pattern = re.compile(r"make\s+(?:the)?\s*([\w\s]+?)\s+(?:bigger|larger|smaller|tinier)")
        match = pattern.search(text)
        
        if match:
            return match.group(1).strip()
        
        # If not found, check for jewelry terms
        for term in self.jewelry_terms:
            if term in text:
                return term
                
        return None
    
    def _extract_add_components(self, text):
        """Extract what to add and where for add prompts"""
        # Pattern for "Add X to/at Y"
        pattern = re.compile(r"add\s+(?:a|an)?\s*([\w\s]+?)\s+(?:to|at|on|in)\s+([\w\s]+)")
        match = pattern.search(text)
        
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        # Simpler pattern just to get what to add
        simple_pattern = re.compile(r"add\s+(?:a|an)?\s*([\w\s]+)")
        match = simple_pattern.search(text)
        
        if match:
            return match.group(1).strip(), None
                
        return None, None
    
    def _extract_remove_target(self, text):
        """Extract target for remove prompts"""
        # Pattern for "Remove X"
        pattern = re.compile(r"remove\s+(?:the)?\s*([\w\s]+)")
        match = pattern.search(text)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_jewelry_terms(self, text):
        """Extract known jewelry terms from text"""
        for term in self.jewelry_terms:
            if term in text:
                return term
        return None


# Example usage
if __name__ == "__main__":
    parser = JewelryPromptParser()
    
    # Test various prompts
    prompts = [
        "Change gold to white gold",
        "Make the diamonds bigger",
        "Add a ruby to the center",
        "Remove the small diamonds",
        "Make this ring in platinum",
        "Change the gemstone to sapphire"
    ]
    
    for prompt in prompts:
        result = parser.parse(prompt)
        print(f"Prompt: {prompt}")
        print(f"Parsed: {result}")
        print("---")