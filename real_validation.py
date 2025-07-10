#!/usr/bin/env python3
"""
Real validation framework for Bounded Gödel Encoding system
Tests actual performance with synthetic mathematical data
"""

import time
import psutil
import random
import string
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import memory_profiler
from pathlib import Path
import json

# Import our system components
from enhanced_bounded_godel_encoding import EnhancedGodelEncoder, EncodingMetrics
from mathematical_library_integration import FormalTheorem

logger = logging.getLogger(__name__)

@dataclass
class RealValidationResult:
    """Real validation results with actual measurements"""
    test_name: str
    actual_performance: float
    theoretical_performance: float
    validation_passed: bool
    measurement_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class SyntheticTheoremGenerator:
    """Generate synthetic theorems for testing"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.domains = ['algebra', 'analysis', 'topology', 'geometry', 'number_theory', 'combinatorics']
        self.types = ['theorem', 'lemma', 'definition', 'axiom', 'corollary', 'proposition']
        self.libraries = ['lean4', 'coq', 'isabelle']
        
    def generate_theorem(self, theorem_id: str, 
                        complexity: int = None,
                        domain: str = None,
                        dependencies: List[str] = None) -> FormalTheorem:
        """Generate a synthetic theorem with realistic properties"""
        
        if complexity is None:
            complexity = random.randint(1, 10)
        if domain is None:
            domain = random.choice(self.domains)
        if dependencies is None:
            dependencies = []
            
        # Generate realistic theorem statement
        statement = self._generate_statement(domain, complexity)
        
        return FormalTheorem(
            id=theorem_id,
            name=f"synthetic_{theorem_id}",
            statement=statement,
            dependencies=dependencies,
            proof_text=self._generate_proof_text(complexity),
            library_source=random.choice(self.libraries),
            mathematical_domain=domain,
            theorem_type=random.choice(self.types),
            complexity_score=complexity,
            namespace=f"synthetic.{domain}",
            variables=self._generate_variables(complexity),
            proof_complexity=min(complexity + random.randint(0, 3), 10)
        )
    
    def _generate_statement(self, domain: str, complexity: int) -> str:
        """Generate a realistic theorem statement"""
        templates = {
            'algebra': [
                "For all x, y in Group G, (x * y)^n = x^n * y^n for some n = {}",
                "If R is a ring and I is an ideal, then R/I is a field iff I is maximal",
                "Every finite group of order {} is isomorphic to a subgroup of S_{}",
                "The order of element x in group G divides the order of G when |G| = {}"
            ],
            'analysis': [
                "If f: [a,b] -> R is continuous, then f attains its maximum",
                "For f differentiable at x, the derivative exists and equals the limit",
                "If sum a_n converges absolutely, then sum a_n converges",
                "Every bounded sequence in R^{} has a convergent subsequence"
            ],
            'topology': [
                "Every compact subset of a metric space is closed and bounded",
                "The continuous image of a compact set is compact",
                "A topological space is compact iff every open cover has a finite subcover",
                "Every metric space with diameter {} can be embedded in R^{}"
            ],
            'geometry': [
                "In Euclidean space of dimension {}, the volume of a unit ball is well-defined",
                "The sum of angles in a triangle equals {} degrees in Euclidean geometry",
                "Every convex polytope in R^{} has at least {} vertices"
            ],
            'number_theory': [
                "Every integer n > {} can be expressed as a sum of {} primes",
                "The number of primes less than {} is approximately {}/ln({})",
                "Every even integer greater than {} is the sum of two primes"
            ],
            'combinatorics': [
                "The number of ways to choose {} objects from {} is C({}, {})",
                "A graph on {} vertices has at most {} edges",
                "The chromatic number of a planar graph is at most {}"
            ]
        }
        
        template = random.choice(templates.get(domain, templates['algebra']))
        
        # Fill in numbers based on complexity, ensuring we have enough numbers
        placeholder_count = template.count('{}')
        if placeholder_count > 0:
            numbers = [str(random.randint(1, complexity + 1)) for _ in range(placeholder_count)]
            return template.format(*numbers)
        else:
            return template
    
    def _generate_proof_text(self, complexity: int) -> str:
        """Generate synthetic proof text"""
        proof_steps = [
            "By definition of the operation",
            "Using the associativity property",
            "Applying the fundamental theorem",
            "By contradiction, assume the opposite",
            "Using mathematical induction",
            "By the intermediate value theorem",
            "This follows from the previous lemma",
            "By direct computation"
        ]
        
        num_steps = min(complexity + random.randint(1, 3), 10)
        selected_steps = random.sample(proof_steps, min(num_steps, len(proof_steps)))
        
        return ". ".join(selected_steps) + ". QED."
    
    def _generate_variables(self, complexity: int) -> List[str]:
        """Generate variable names based on complexity"""
        variables = ['x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'i', 'j', 'k']
        return random.sample(variables, min(complexity, len(variables)))
    
    def generate_theorem_batch(self, count: int, 
                             max_complexity: int = 10,
                             create_dependencies: bool = True) -> List[FormalTheorem]:
        """Generate a batch of related theorems"""
        theorems = []
        
        for i in range(count):
            theorem_id = f"synth_{i:05d}"
            complexity = random.randint(1, max_complexity)
            domain = random.choice(self.domains)
            
            # Create dependencies on previous theorems
            dependencies = []
            if create_dependencies and i > 0:
                num_deps = min(random.randint(0, 3), i)
                dep_indices = random.sample(range(i), num_deps)
                dependencies = [f"synth_{j:05d}" for j in dep_indices]
            
            theorem = self.generate_theorem(theorem_id, complexity, domain, dependencies)
            theorems.append(theorem)
        
        return theorems

class RealValidationFramework:
    """Real validation framework with actual measurements"""
    
    def __init__(self):
        self.encoder = EnhancedGodelEncoder()
        self.theorem_generator = SyntheticTheoremGenerator()
        self.results = []
        
    def run_real_validation(self, test_sizes: List[int] = None) -> List[RealValidationResult]:
        """Run comprehensive real validation"""
        
        if test_sizes is None:
            test_sizes = [100, 500, 1000, 5000]
        
        print("🔍 REAL VALIDATION FRAMEWORK - ACTUAL MEASUREMENTS")
        print("=" * 60)
        
        for size in test_sizes:
            print(f"\n📊 Testing with {size:,} theorems...")
            
            # Generate test data
            theorems = self.theorem_generator.generate_theorem_batch(size)
            
            # Run all validation tests
            self._test_encoding_performance(theorems)
            self._test_memory_usage(theorems)
            self._test_uniqueness_validation(theorems)
            self._test_retrieval_performance(theorems)
            self._test_collision_resolution(theorems)
            
        return self.results
    
    def _test_encoding_performance(self, theorems: List[FormalTheorem]):
        """Test actual encoding performance"""
        
        print(f"  ⚡ Testing encoding performance...")
        
        # Sequential encoding
        start_time = time.time()
        sequential_results = {}
        for theorem in theorems:
            encoding = self.encoder.encode_theorem(theorem)
            sequential_results[theorem.id] = encoding
        sequential_time = time.time() - start_time
        
        # Reset encoder for parallel test
        self.encoder = EnhancedGodelEncoder()
        
        # Parallel encoding
        start_time = time.time()
        parallel_results = self.encoder.encode_theorem_batch(theorems, parallel=True)
        parallel_time = time.time() - start_time
        
        # Calculate metrics
        sequential_throughput = len(theorems) / sequential_time
        parallel_throughput = len(theorems) / parallel_time
        
        result = RealValidationResult(
            test_name=f"Encoding Performance ({len(theorems)} theorems)",
            actual_performance=parallel_throughput,
            theoretical_performance=5000,  # From benchmark
            validation_passed=parallel_throughput > 1000,  # Reasonable threshold
            measurement_details={
                'sequential_time': sequential_time,
                'parallel_time': parallel_time,
                'sequential_throughput': sequential_throughput,
                'parallel_throughput': parallel_throughput,
                'speedup_factor': parallel_throughput / sequential_throughput,
                'theorem_count': len(theorems)
            }
        )
        
        self.results.append(result)
        
        print(f"    • Sequential: {sequential_throughput:.0f} theorems/sec")
        print(f"    • Parallel: {parallel_throughput:.0f} theorems/sec") 
        print(f"    • Speedup: {parallel_throughput/sequential_throughput:.1f}x")
        print(f"    • Validation: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
    
    @memory_profiler.profile
    def _test_memory_usage(self, theorems: List[FormalTheorem]):
        """Test actual memory usage"""
        
        print(f"  💾 Testing memory usage...")
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Encode all theorems
        encodings = self.encoder.encode_theorem_batch(theorems)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Calculate efficiency
        memory_per_theorem = memory_used / len(theorems)
        memory_per_1000 = memory_per_theorem * 1000
        
        result = RealValidationResult(
            test_name=f"Memory Usage ({len(theorems)} theorems)",
            actual_performance=memory_per_1000,
            theoretical_performance=15.0,  # From benchmark (15MB per 1000 theorems)
            validation_passed=memory_per_1000 < 30.0,  # Reasonable threshold
            measurement_details={
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_used_mb': memory_used,
                'memory_per_theorem_kb': memory_per_theorem * 1024,
                'memory_per_1000_theorems_mb': memory_per_1000,
                'theorem_count': len(theorems)
            }
        )
        
        self.results.append(result)
        
        print(f"    • Memory used: {memory_used:.1f} MB")
        print(f"    • Per theorem: {memory_per_theorem*1024:.1f} KB")
        print(f"    • Per 1000 theorems: {memory_per_1000:.1f} MB")
        print(f"    • Validation: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
    
    def _test_uniqueness_validation(self, theorems: List[FormalTheorem]):
        """Test actual encoding uniqueness"""
        
        print(f"  🔢 Testing encoding uniqueness...")
        
        # Get all encodings
        encodings = list(self.encoder.theorem_encodings.values())
        
        # Check uniqueness
        unique_encodings = len(set(encodings))
        total_encodings = len(encodings)
        collision_count = total_encodings - unique_encodings
        
        uniqueness_ratio = unique_encodings / total_encodings if total_encodings > 0 else 0
        collision_rate = collision_count / total_encodings if total_encodings > 0 else 0
        
        result = RealValidationResult(
            test_name=f"Encoding Uniqueness ({len(theorems)} theorems)",
            actual_performance=uniqueness_ratio,
            theoretical_performance=1.0,  # Perfect uniqueness
            validation_passed=uniqueness_ratio > 0.99,
            measurement_details={
                'total_encodings': total_encodings,
                'unique_encodings': unique_encodings,
                'collision_count': collision_count,
                'uniqueness_ratio': uniqueness_ratio,
                'collision_rate': collision_rate,
                'collision_methods_used': [r.resolution_method for r in self.encoder.collision_history]
            }
        )
        
        self.results.append(result)
        
        print(f"    • Total encodings: {total_encodings:,}")
        print(f"    • Unique encodings: {unique_encodings:,}")
        print(f"    • Collisions: {collision_count}")
        print(f"    • Uniqueness: {uniqueness_ratio:.1%}")
        print(f"    • Validation: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
    
    def _test_retrieval_performance(self, theorems: List[FormalTheorem]):
        """Test actual retrieval performance"""
        
        print(f"  🔍 Testing retrieval performance...")
        
        # Test domain-based retrieval
        domain_queries = 100
        domain_times = []
        
        for _ in range(domain_queries):
            domain = random.choice(['algebra', 'analysis', 'topology'])
            domain_factor = self.encoder.mathematical_domains.get(domain, 127)
            
            start_time = time.time()
            matching_theorems = [
                tid for tid, encoding in self.encoder.theorem_encodings.items()
                if encoding % domain_factor == 0
            ]
            query_time = time.time() - start_time
            domain_times.append(query_time * 1000)  # Convert to milliseconds
        
        avg_domain_query_time = np.mean(domain_times)
        
        # Test complexity-based retrieval
        complexity_queries = 100
        complexity_times = []
        
        for _ in range(complexity_queries):
            complexity_threshold = random.randint(5, 8)
            complexity_factor = self.encoder.complexity_factors.get(complexity_threshold, 167)
            
            start_time = time.time()
            complex_theorems = [
                tid for tid, encoding in self.encoder.theorem_encodings.items()
                if encoding % complexity_factor == 0
            ]
            query_time = time.time() - start_time
            complexity_times.append(query_time * 1000)
        
        avg_complexity_query_time = np.mean(complexity_times)
        avg_query_time = (avg_domain_query_time + avg_complexity_query_time) / 2
        
        result = RealValidationResult(
            test_name=f"Retrieval Performance ({len(theorems)} theorems)",
            actual_performance=avg_query_time,
            theoretical_performance=0.5,  # From benchmark (0.5ms)
            validation_passed=avg_query_time < 5.0,  # Reasonable threshold
            measurement_details={
                'domain_query_time_ms': avg_domain_query_time,
                'complexity_query_time_ms': avg_complexity_query_time,
                'avg_query_time_ms': avg_query_time,
                'queries_tested': domain_queries + complexity_queries,
                'theorem_count': len(theorems)
            }
        )
        
        self.results.append(result)
        
        print(f"    • Domain queries: {avg_domain_query_time:.2f}ms avg")
        print(f"    • Complexity queries: {avg_complexity_query_time:.2f}ms avg")
        print(f"    • Overall avg: {avg_query_time:.2f}ms")
        print(f"    • Validation: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
    
    def _test_collision_resolution(self, theorems: List[FormalTheorem]):
        """Test collision resolution effectiveness"""
        
        print(f"  🔧 Testing collision resolution...")
        
        collision_history = self.encoder.collision_history
        total_collisions = len(collision_history)
        resolved_collisions = sum(1 for r in collision_history if r.success)
        
        resolution_rate = resolved_collisions / total_collisions if total_collisions > 0 else 1.0
        
        # Count resolution methods used
        method_counts = Counter(r.resolution_method for r in collision_history)
        
        result = RealValidationResult(
            test_name=f"Collision Resolution ({len(theorems)} theorems)",
            actual_performance=resolution_rate,
            theoretical_performance=0.98,  # From benchmark
            validation_passed=resolution_rate > 0.95,
            measurement_details={
                'total_collisions': total_collisions,
                'resolved_collisions': resolved_collisions,
                'resolution_rate': resolution_rate,
                'method_counts': dict(method_counts),
                'avg_iterations': np.mean([r.iterations for r in collision_history]) if collision_history else 0
            }
        )
        
        self.results.append(result)
        
        print(f"    • Total collisions: {total_collisions}")
        print(f"    • Resolved successfully: {resolved_collisions}")
        print(f"    • Resolution rate: {resolution_rate:.1%}")
        print(f"    • Methods used: {dict(method_counts)}")
        print(f"    • Validation: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'validation_summary': {
                'total_tests': len(self.results),
                'passed_tests': sum(1 for r in self.results if r.validation_passed),
                'failed_tests': sum(1 for r in self.results if not r.validation_passed),
                'overall_success_rate': sum(1 for r in self.results if r.validation_passed) / len(self.results) if self.results else 0
            },
            'performance_analysis': {},
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'actual_performance': r.actual_performance,
                    'theoretical_performance': r.theoretical_performance,
                    'validation_passed': r.validation_passed,
                    'measurement_details': r.measurement_details
                } for r in self.results
            ],
            'recommendations': []
        }
        
        # Analyze performance gaps
        for result in self.results:
            if 'Performance' in result.test_name:
                gap = abs(result.actual_performance - result.theoretical_performance)
                relative_gap = gap / result.theoretical_performance if result.theoretical_performance > 0 else 0
                report['performance_analysis'][result.test_name] = {
                    'actual': result.actual_performance,
                    'theoretical': result.theoretical_performance,
                    'absolute_gap': gap,
                    'relative_gap': relative_gap
                }
        
        # Generate recommendations
        failed_tests = [r for r in self.results if not r.validation_passed]
        if failed_tests:
            report['recommendations'].append(f"Address {len(failed_tests)} failed validation tests")
        
        if report['validation_summary']['overall_success_rate'] < 0.8:
            report['recommendations'].append("Overall validation success rate below 80% - significant improvements needed")
        
        return report

def run_real_validation():
    """Run the real validation framework"""
    
    print("🚀 BOUNDED GÖDEL ENCODING - REAL VALIDATION")
    print("=" * 60)
    print("Running actual performance measurements with synthetic data...")
    
    # Initialize framework
    validator = RealValidationFramework()
    
    # Run validation with different dataset sizes
    test_sizes = [100, 500, 1000]
    results = validator.run_real_validation(test_sizes)
    
    # Generate report
    report = validator.generate_validation_report()
    
    print("\n📊 REAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {report['validation_summary']['total_tests']}")
    print(f"Passed: {report['validation_summary']['passed_tests']}")
    print(f"Failed: {report['validation_summary']['failed_tests']}")
    print(f"Success rate: {report['validation_summary']['overall_success_rate']:.1%}")
    
    print("\n🎯 PERFORMANCE ANALYSIS")
    print("-" * 30)
    for test_name, analysis in report['performance_analysis'].items():
        print(f"{test_name}:")
        print(f"  Actual: {analysis['actual']:.2f}")
        print(f"  Theoretical: {analysis['theoretical']:.2f}")
        print(f"  Gap: {analysis['relative_gap']:.1%}")
    
    if report['recommendations']:
        print("\n📋 RECOMMENDATIONS")
        print("-" * 30)
        for rec in report['recommendations']:
            print(f"• {rec}")
    
    print("\n✨ REAL VALIDATION COMPLETE!")
    print("This validation used synthetic data to test actual system performance.")
    print("For production validation, replace with real theorem libraries.")
    
    return validator, report

if __name__ == "__main__":
    validator, report = run_real_validation()