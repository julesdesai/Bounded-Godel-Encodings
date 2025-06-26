import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

# Large prime for modular arithmetic (keeps numbers bounded)
MODULUS = 2**31 - 1  # Mersenne prime

@dataclass
class Theorem:
    """Represents a mathematical theorem with its logical structure"""
    id: str
    statement: str
    dependencies: List[str]  # IDs of theorems/axioms this depends on
    proof_method: str  # Type of proof used
    mathematical_objects: List[str]  # Key mathematical concepts involved
    
class BoundedGodelEncoder:
    """
    Implements bounded Gödel-like encoding using modular arithmetic.
    Each theorem gets a unique encoding based on its logical dependencies.
    """
    
    def __init__(self, modulus: int = MODULUS):
        self.modulus = modulus
        self.theorems: Dict[str, Theorem] = {}
        self.base_encodings: Dict[str, int] = {}  # Axioms and basic definitions
        self.theorem_encodings: Dict[str, int] = {}
        self.encoding_to_theorem: Dict[int, str] = {}
        
        # Proof method encodings
        self.proof_methods = {
            'direct': 2,
            'contradiction': 3, 
            'induction': 5,
            'construction': 7,
            'axiom': 11
        }
        
    def add_axiom(self, axiom_id: str, statement: str):
        """Add a base axiom with a prime encoding"""
        prime = self._generate_safe_prime()
        self.base_encodings[axiom_id] = prime
        self.theorems[axiom_id] = Theorem(
            id=axiom_id,
            statement=statement,
            dependencies=[],
            proof_method='axiom',
            mathematical_objects=[]
        )
        
    def add_theorem(self, theorem: Theorem):
        """Add a theorem and compute its Gödel-like encoding"""
        self.theorems[theorem.id] = theorem
        encoding = self._compute_encoding(theorem)
        self.theorem_encodings[theorem.id] = encoding
        self.encoding_to_theorem[encoding] = theorem.id
        
    def _generate_safe_prime(self) -> int:
        """Generate a random prime less than modulus"""
        # Simple prime generation - in practice you'd use a more robust method
        candidate = random.randint(13, 1000)
        while not self._is_prime(candidate):
            candidate = random.randint(13, 1000)
        return candidate
        
    def _is_prime(self, n: int) -> bool:
        """Simple primality test"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    def _compute_encoding(self, theorem: Theorem) -> int:
        """Compute bounded Gödel encoding for a theorem using enhanced composition"""
        if not theorem.dependencies:
            # Base case - should have been added as axiom
            return self.base_encodings.get(theorem.id, self._generate_safe_prime())
            
        # Enhanced composition to avoid collisions
        encoding = 1
        proof_multiplier = self.proof_methods.get(theorem.proof_method, 1)
        
        # Position-sensitive dependency encoding
        for i, dep_id in enumerate(theorem.dependencies):
            if dep_id in self.base_encodings:
                dep_encoding = self.base_encodings[dep_id]
            elif dep_id in self.theorem_encodings:
                dep_encoding = self.theorem_encodings[dep_id]
            else:
                raise ValueError(f"Dependency {dep_id} not found")
                
            # Position matters: use different powers for different positions
            position_factor = (i + 1) * 13  # 13 is arbitrary prime
            encoding = (encoding * dep_encoding * position_factor) % self.modulus
            
        # Include proof method
        encoding = (encoding * proof_multiplier) % self.modulus
        
        # Include mathematical objects to distinguish similar theorems
        concept_hash = self._hash_concepts(theorem.mathematical_objects)
        encoding = (encoding * concept_hash) % self.modulus
        
        # Include statement structure hash to make truly unique
        statement_hash = self._hash_statement(theorem.statement)
        encoding = (encoding + statement_hash) % self.modulus  # Addition to avoid zeros
        
        return encoding
        
    def _hash_concepts(self, concepts: List[str]) -> int:
        """Create a hash from mathematical concepts"""
        if not concepts:
            return 1
        concept_sum = sum(hash(concept) % 1000 for concept in concepts)
        return (concept_sum % 997) + 1  # Ensure non-zero, 997 is prime
        
    def _hash_statement(self, statement: str) -> int:
        """Create a hash from theorem statement"""
        # Simple hash of statement to ensure uniqueness
        return (hash(statement) % 991) + 1  # 991 is prime
        
    def decompose_encoding(self, encoding: int) -> Optional[Dict]:
        """Attempt to decompose an encoding back to its components"""
        if encoding in self.encoding_to_theorem:
            theorem_id = self.encoding_to_theorem[encoding]
            theorem = self.theorems[theorem_id]
            
            return {
                'theorem_id': theorem_id,
                'statement': theorem.statement,
                'dependencies': theorem.dependencies,
                'proof_method': theorem.proof_method,
                'encoding': encoding
            }
        return None
        
    def find_similar_theorems(self, theorem_id: str, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Find theorems with similar encodings (arithmetic similarity)"""
        if theorem_id not in self.theorem_encodings:
            return []
            
        target_encoding = self.theorem_encodings[theorem_id]
        similarities = []
        
        for other_id, other_encoding in self.theorem_encodings.items():
            if other_id != theorem_id:
                # Use GCD as a measure of shared structure
                gcd_val = np.gcd(target_encoding, other_encoding)
                similarity = gcd_val / max(target_encoding, other_encoding)
                
                if similarity > threshold:
                    similarities.append((other_id, similarity))
                    
        return sorted(similarities, key=lambda x: x[1], reverse=True)
        
    def get_stats(self) -> Dict:
        """Get statistics about the encoding scheme"""
        all_encodings = list(self.base_encodings.values()) + list(self.theorem_encodings.values())
        unique_encodings = len(set(all_encodings))
        total_encodings = len(all_encodings)
        
        return {
            'num_axioms': len(self.base_encodings),
            'num_theorems': len(self.theorem_encodings),
            'total_encodings': total_encodings,
            'unique_encodings': unique_encodings,
            'uniqueness_ratio': unique_encodings / total_encodings if total_encodings > 0 else 0,
            'encoding_range': {
                'min': min(all_encodings) if all_encodings else 0,
                'max': max(all_encodings) if all_encodings else 0
            },
            'modulus': self.modulus,
            'collisions': total_encodings - unique_encodings
        }

class TheoremEvaluator:
    """Evaluation framework for testing the Gödel encoding scheme"""
    
    def __init__(self, encoder: BoundedGodelEncoder):
        self.encoder = encoder
        
    def test_reconstruction_accuracy(self) -> float:
        """Test if we can reconstruct theorems from their encodings"""
        correct = 0
        total = len(self.encoder.theorem_encodings)
        
        print("  Detailed reconstruction test:")
        for theorem_id, encoding in self.encoder.theorem_encodings.items():
            reconstructed = self.encoder.decompose_encoding(encoding)
            if reconstructed and reconstructed['theorem_id'] == theorem_id:
                correct += 1
                print(f"    ✓ {theorem_id}: Success")
            else:
                print(f"    ✗ {theorem_id}: Failed - got {reconstructed}")
                
        return correct / total if total > 0 else 0
        
    def test_dependency_preservation(self) -> Dict:
        """Test if theorems that share dependencies have related encodings"""
        shared_deps = defaultdict(list)
        
        # Group theorems by shared dependencies
        for theorem_id, theorem in self.encoder.theorems.items():
            if theorem.dependencies:
                dep_key = tuple(sorted(theorem.dependencies))
                shared_deps[dep_key].append(theorem_id)
                
        results = {
            'groups_with_shared_deps': len([g for g in shared_deps.values() if len(g) > 1]),
            'total_dependency_groups': len(shared_deps),
            'avg_similarity_within_groups': 0
        }
        
        # Calculate average similarity within groups
        similarities = []
        for group in shared_deps.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        similar = self.encoder.find_similar_theorems(group[i])
                        for sim_id, sim_score in similar:
                            if sim_id == group[j]:
                                similarities.append(sim_score)
                                
        if similarities:
            results['avg_similarity_within_groups'] = np.mean(similarities)
            
        return results
        
    def test_compositionality(self) -> float:
        """Test if we can predict encodings by composition"""
        # Simple test: if A→B and B→C, can we predict something about A→C?
        implications = []
        for theorem in self.encoder.theorems.values():
            if len(theorem.dependencies) == 1:
                implications.append((theorem.dependencies[0], theorem.id))
                
        # Find chains A→B→C
        chains = 0
        successful_predictions = 0
        
        for a, b in implications:
            for b2, c in implications:
                if b == b2:  # Found a chain A→B→C
                    chains += 1
                    # Simple predictability test: do shared factors indicate relationship?
                    if a in self.encoder.theorem_encodings and c in self.encoder.theorem_encodings:
                        enc_a = self.encoder.theorem_encodings[a]
                        enc_c = self.encoder.theorem_encodings[c]
                        if np.gcd(enc_a, enc_c) > 1:
                            successful_predictions += 1
                            
        return successful_predictions / chains if chains > 0 else 0

# Sample mathematical theorems for testing
def create_sample_theorems(encoder: BoundedGodelEncoder):
    """Create a sample database of mathematical theorems"""
    
    # Axioms
    encoder.add_axiom('peano_1', 'Zero is a natural number')
    encoder.add_axiom('peano_2', 'Every natural number has a successor')
    encoder.add_axiom('even_def', 'n is even iff n = 2k for some integer k')
    encoder.add_axiom('divisibility', 'a divides b iff b = ac for some integer c')
    
    # Basic theorems
    theorems = [
        Theorem(
            id='even_square',
            statement='If n is even, then n² is even',
            dependencies=['even_def'],
            proof_method='direct',
            mathematical_objects=['integers', 'squares', 'even_numbers']
        ),
        Theorem(
            id='odd_square', 
            statement='If n is odd, then n² is odd',
            dependencies=['even_def'],
            proof_method='direct',
            mathematical_objects=['integers', 'squares', 'odd_numbers']
        ),
        Theorem(
            id='sum_even_even',
            statement='The sum of two even numbers is even',
            dependencies=['even_def'],
            proof_method='direct', 
            mathematical_objects=['integers', 'addition', 'even_numbers']
        ),
        Theorem(
            id='divisibility_transitivity',
            statement='If a|b and b|c, then a|c',
            dependencies=['divisibility'],
            proof_method='direct',
            mathematical_objects=['integers', 'divisibility']
        ),
        Theorem(
            id='even_divisible_by_2',
            statement='n is even iff 2|n',
            dependencies=['even_def', 'divisibility'],
            proof_method='direct',
            mathematical_objects=['integers', 'even_numbers', 'divisibility']
        )
    ]
    
    for theorem in theorems:
        encoder.add_theorem(theorem)

# Demo usage
if __name__ == "__main__":
    # Create encoder and add sample theorems
    encoder = BoundedGodelEncoder()
    create_sample_theorems(encoder)
    
    print("=== Bounded Gödel Encoding for Mathematical Theorems ===\n")
    
    # Show encodings with detailed breakdown
    print("Axiom Encodings:")
    for axiom_id, encoding in encoder.base_encodings.items():
        theorem = encoder.theorems[axiom_id]
        print(f"  {axiom_id}: {encoding}")
        print(f"    Statement: {theorem.statement}")
    
    print("\nTheorem Encodings (with breakdown):")
    for theorem_id, encoding in encoder.theorem_encodings.items():
        theorem = encoder.theorems[theorem_id]
        print(f"  {theorem_id}: {encoding}")
        print(f"    Statement: {theorem.statement}")
        print(f"    Dependencies: {theorem.dependencies}")
        print(f"    Proof method: {theorem.proof_method}")
        print(f"    Math objects: {theorem.mathematical_objects}")
        print()
        
    # Test reconstruction
    print("\n=== Testing Reconstruction ===")
    for theorem_id, encoding in encoder.theorem_encodings.items():
        reconstructed = encoder.decompose_encoding(encoding)
        if reconstructed:
            print(f"✓ {theorem_id}: Successfully reconstructed")
        else:
            print(f"✗ {theorem_id}: Failed to reconstruct")
            
    # Test similarity
    print("\n=== Testing Similarity ===")
    similar = encoder.find_similar_theorems('even_square')
    print(f"Theorems similar to 'even_square': {similar}")
    
    # Run evaluations
    print("\n=== Evaluation Results ===")
    evaluator = TheoremEvaluator(encoder)
    
    reconstruction_acc = evaluator.test_reconstruction_accuracy()
    print(f"Reconstruction Accuracy: {reconstruction_acc:.2%}")
    
    dependency_results = evaluator.test_dependency_preservation()
    print(f"Dependency Preservation: {dependency_results}")
    
    compositionality = evaluator.test_compositionality()
    print(f"Compositionality Score: {compositionality:.2%}")
    
    stats = encoder.get_stats()
    print(f"\nSystem Stats: {stats}")