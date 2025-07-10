#!/usr/bin/env python3
"""
Real Lean 4 mathlib parser for extracting theorems, definitions, and proofs
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

# Import our existing theorem structure
from mathematical_library_integration import FormalTheorem

logger = logging.getLogger(__name__)

@dataclass
class LeanDeclaration:
    """Represents a Lean declaration (theorem, lemma, definition, etc.)"""
    name: str
    declaration_type: str  # theorem, lemma, def, instance, etc.
    statement: str
    proof: Optional[str] = None
    namespace: str = ""
    dependencies: List[str] = field(default_factory=list)
    line_number: int = 0
    file_path: str = ""
    docstring: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)

class Lean4Parser:
    """Parser for Lean 4 .lean files"""
    
    def __init__(self, mathlib_path: str):
        self.mathlib_path = Path(mathlib_path)
        self.declarations = []
        self.namespace_stack = []
        self.current_namespace = ""
        
        # Lean 4 keywords for declarations
        self.declaration_keywords = {
            'theorem', 'lemma', 'def', 'instance', 'axiom', 'constant',
            'example', 'noncomputable', 'structure', 'class', 'inductive'
        }
        
        # Lean 4 proof keywords
        self.proof_keywords = {
            'by', 'exact', 'apply', 'intro', 'cases', 'induction',
            'simp', 'rw', 'refl', 'sorry', 'trivial', 'assumption'
        }
        
    def parse_file(self, file_path: Path) -> List[LeanDeclaration]:
        """Parse a single Lean file"""
        
        logger.debug(f"Parsing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        declarations = []
        lines = content.split('\n')
        
        i = 0
        current_namespace = ""
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('--') or line.startswith('/-'):
                if line.startswith('/-'):
                    # Skip multi-line comments
                    while i < len(lines) and not lines[i].strip().endswith('-/'):
                        i += 1
                i += 1
                continue
            
            # Handle namespace declarations
            if line.startswith('namespace '):
                namespace_match = re.match(r'namespace\s+([A-Za-z_][A-Za-z0-9_\.]*)', line)
                if namespace_match:
                    current_namespace = namespace_match.group(1)
                i += 1
                continue
                
            if line == 'end' or line.startswith('end '):
                current_namespace = ""
                i += 1
                continue
            
            # Check for declarations
            declaration = self._parse_declaration(lines, i, current_namespace, str(file_path))
            if declaration:
                declarations.append(declaration)
                # Skip to end of declaration
                i = self._find_declaration_end(lines, i)
            else:
                i += 1
        
        return declarations
    
    def _parse_declaration(self, lines: List[str], start_idx: int, namespace: str, file_path: str) -> Optional[LeanDeclaration]:
        """Parse a declaration starting at the given line"""
        
        line = lines[start_idx].strip()
        
        # Match declaration patterns
        # Pattern: (attributes)? (noncomputable)? (private)? keyword name (params)? : type (where)? := proof
        decl_pattern = r'(?:@\[[^\]]+\]\s*)?(?:noncomputable\s+)?(?:private\s+)?(theorem|lemma|def|instance|axiom|constant|example)\s+([A-Za-z_][A-Za-z0-9_\.]*)'
        
        match = re.match(decl_pattern, line)
        if not match:
            return None
        
        decl_type = match.group(1)
        decl_name = match.group(2)
        
        # Extract the full declaration text
        declaration_text, proof_text = self._extract_declaration_text(lines, start_idx)
        
        # Parse statement (everything between name and :=)
        statement = self._extract_statement(declaration_text)
        
        # Extract dependencies from imports and statement
        dependencies = self._extract_dependencies(declaration_text)
        
        # Extract attributes
        attributes = self._extract_attributes(lines[start_idx])
        
        # Extract docstring if present
        docstring = self._extract_docstring(lines, start_idx - 1)
        
        return LeanDeclaration(
            name=decl_name,
            declaration_type=decl_type,
            statement=statement,
            proof=proof_text,
            namespace=namespace,
            dependencies=dependencies,
            line_number=start_idx + 1,
            file_path=file_path,
            docstring=docstring,
            attributes=attributes
        )
    
    def _extract_declaration_text(self, lines: List[str], start_idx: int) -> Tuple[str, Optional[str]]:
        """Extract the full declaration text and separate proof"""
        
        declaration_lines = []
        proof_lines = []
        in_proof = False
        brace_count = 0
        paren_count = 0
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            if not in_proof:
                declaration_lines.append(line)
                
                # Check if we've reached the proof
                if ':=' in line:
                    in_proof = True
                    # Split at := to separate declaration and proof
                    parts = line.split(':=', 1)
                    if len(parts) > 1 and parts[1].strip():
                        proof_lines.append(parts[1].strip())
                elif 'by' in line and (line.endswith('by') or 'by ' in line):
                    in_proof = True
                    proof_lines.append(line)
            else:
                proof_lines.append(line)
            
            # Track braces and parentheses to find end of declaration
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')
            
            # Simple heuristic: declaration ends when we reach balanced braces/parens
            # and encounter certain keywords or patterns
            if (brace_count == 0 and paren_count == 0 and 
                (line == '' or 
                 any(line.startswith(kw) for kw in self.declaration_keywords) or
                 line.startswith('namespace') or
                 line.startswith('end'))):
                break
                
            i += 1
        
        declaration_text = ' '.join(declaration_lines)
        proof_text = ' '.join(proof_lines) if proof_lines else None
        
        return declaration_text, proof_text
    
    def _find_declaration_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a declaration"""
        
        brace_count = 0
        paren_count = 0
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')
            
            # Declaration ends when balanced and we see a new declaration or namespace
            if (brace_count == 0 and paren_count == 0 and i > start_idx and
                (any(line.startswith(kw) for kw in self.declaration_keywords) or
                 line.startswith('namespace') or
                 line.startswith('end') or
                 line == '')):
                return i
                
            i += 1
        
        return len(lines)
    
    def _extract_statement(self, declaration_text: str) -> str:
        """Extract the statement/type part of a declaration"""
        
        # Remove attributes and keywords
        text = re.sub(r'@\[[^\]]+\]', '', declaration_text)
        text = re.sub(r'(noncomputable|private)\s+', '', text)
        
        # Find the pattern: keyword name (params)? : statement := proof
        pattern = r'(?:theorem|lemma|def|instance|axiom|constant|example)\s+[A-Za-z_][A-Za-z0-9_\.]*[^:]*:\s*([^:=]+?)(?::=|$)'
        match = re.search(pattern, text)
        
        if match:
            return match.group(1).strip()
        else:
            # Fallback: everything after the first colon
            colon_idx = text.find(':')
            if colon_idx != -1:
                equals_idx = text.find(':=', colon_idx)
                if equals_idx != -1:
                    return text[colon_idx + 1:equals_idx].strip()
                else:
                    return text[colon_idx + 1:].strip()
        
        return ""
    
    def _extract_dependencies(self, declaration_text: str) -> List[str]:
        """Extract dependencies from a declaration"""
        
        dependencies = []
        
        # Look for identifiers that might be dependencies
        # This is a simplified approach - in practice, we'd need semantic analysis
        identifier_pattern = r'\b([A-Z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\b'
        matches = re.findall(identifier_pattern, declaration_text)
        
        # Filter out common types and keywords
        common_types = {'Type', 'Prop', 'Sort', 'Nat', 'Int', 'Real', 'Complex', 'String', 'Bool'}
        
        for match in matches:
            if match not in common_types and '.' in match:
                dependencies.append(match)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_attributes(self, line: str) -> List[str]:
        """Extract attributes from a declaration line"""
        
        attributes = []
        attr_pattern = r'@\[([^\]]+)\]'
        matches = re.findall(attr_pattern, line)
        
        for match in matches:
            # Split multiple attributes
            attrs = [attr.strip() for attr in match.split(',')]
            attributes.extend(attrs)
        
        return attributes
    
    def _extract_docstring(self, lines: List[str], before_idx: int) -> Optional[str]:
        """Extract docstring before a declaration"""
        
        if before_idx < 0:
            return None
        
        # Look for /-- ... -/ pattern before the declaration
        i = before_idx
        while i >= 0 and (not lines[i].strip() or lines[i].strip().startswith('--')):
            i -= 1
        
        if i >= 0 and lines[i].strip().endswith('-/'):
            # Found end of doc comment, now find start
            end_idx = i
            while i >= 0 and not lines[i].strip().startswith('/-'):
                i -= 1
            
            if i >= 0:
                doc_lines = []
                for j in range(i, end_idx + 1):
                    line = lines[j].strip()
                    if line.startswith('/-'):
                        line = line[2:].strip()
                    if line.endswith('-/'):
                        line = line[:-2].strip()
                    if line.startswith('-'):
                        line = line[1:].strip()
                    if line:
                        doc_lines.append(line)
                
                return '\n'.join(doc_lines) if doc_lines else None
        
        return None
    
    def parse_mathlib_subset(self, max_files: int = 100) -> List[LeanDeclaration]:
        """Parse a subset of mathlib files"""
        
        logger.info(f"Parsing subset of mathlib (max {max_files} files)")
        
        all_declarations = []
        files_processed = 0
        
        # Start with core algebra files
        algebra_dir = self.mathlib_path / "Mathlib" / "Algebra"
        if algebra_dir.exists():
            for lean_file in algebra_dir.rglob("*.lean"):
                if files_processed >= max_files:
                    break
                    
                declarations = self.parse_file(lean_file)
                all_declarations.extend(declarations)
                files_processed += 1
                
                if files_processed % 10 == 0:
                    logger.info(f"Processed {files_processed} files, found {len(all_declarations)} declarations")
        
        # Add some number theory files
        if files_processed < max_files:
            number_theory_dir = self.mathlib_path / "Mathlib" / "NumberTheory"
            if number_theory_dir.exists():
                for lean_file in number_theory_dir.rglob("*.lean"):
                    if files_processed >= max_files:
                        break
                    
                    declarations = self.parse_file(lean_file)
                    all_declarations.extend(declarations)
                    files_processed += 1
        
        logger.info(f"Completed parsing: {files_processed} files, {len(all_declarations)} declarations")
        return all_declarations
    
    def convert_to_formal_theorems(self, declarations: List[LeanDeclaration]) -> List[FormalTheorem]:
        """Convert LeanDeclarations to FormalTheorem objects"""
        
        formal_theorems = []
        
        for decl in declarations:
            # Only convert theorem-like declarations
            if decl.declaration_type not in ['theorem', 'lemma', 'axiom']:
                continue
            
            # Map mathematical domains based on file path and namespace
            domain = self._infer_domain(decl.file_path, decl.namespace)
            
            # Estimate complexity based on statement length and dependencies
            complexity = min(max(1, len(decl.dependencies) + len(decl.statement) // 100), 10)
            
            formal_theorem = FormalTheorem(
                id=f"lean4_{decl.name}",
                name=decl.name,
                statement=decl.statement,
                dependencies=decl.dependencies,
                proof_text=decl.proof,
                library_source="lean4",
                namespace=decl.namespace,
                mathematical_domain=domain,
                theorem_type=decl.declaration_type,
                complexity_score=complexity,
                type_signature=decl.statement,  # In Lean, statement is the type
                variables=decl.variables,
                proof_complexity=min(complexity + 1, 10) if decl.proof else 1,
                file_path=decl.file_path  # Preserve file path for domain validation
            )
            
            formal_theorems.append(formal_theorem)
        
        return formal_theorems
    
    def _infer_domain(self, file_path: str, namespace: str) -> str:
        """Infer mathematical domain from file path and namespace"""
        
        # Convert to Path object for better analysis
        file_path_obj = Path(file_path)
        path_parts = [part.lower() for part in file_path_obj.parts]
        path_lower = str(file_path_obj).lower()
        namespace_lower = namespace.lower()
        
        # Enhanced domain mappings with more keywords
        domain_mappings = {
            # Primary domains
            'algebra': 'algebra',
            'number': 'number_theory',
            'numbertheory': 'number_theory',
            'topology': 'topology',
            'analysis': 'analysis',
            'geometry': 'geometry',
            'combinatorics': 'combinatorics',
            'logic': 'logic',
            'category': 'category_theory',
            'categorytheory': 'category_theory',
            
            # Algebra subfields
            'ring': 'algebra',
            'group': 'algebra',
            'field': 'algebra',
            'linear': 'algebra',
            'module': 'algebra',
            'ideal': 'algebra',
            'polynomial': 'algebra',
            'galois': 'algebra',
            'commutative': 'algebra',
            'associative': 'algebra',
            'homomorphism': 'algebra',
            'isomorphism': 'algebra',
            
            # Analysis subfields
            'measure': 'analysis',
            'integration': 'analysis',
            'differential': 'analysis',
            'calculus': 'analysis',
            'functional': 'analysis',
            'harmonic': 'analysis',
            'fourier': 'analysis',
            'real': 'analysis',
            'complex': 'analysis',
            
            # Topology subfields
            'continuous': 'topology',
            'compact': 'topology',
            'connected': 'topology',
            'metric': 'topology',
            'uniform': 'topology',
            'homeomorphism': 'topology',
            
            # Number theory subfields
            'prime': 'number_theory',
            'divisibility': 'number_theory',
            'congruence': 'number_theory',
            'diophantine': 'number_theory',
            'arithmetic': 'number_theory',
            
            # Geometry subfields
            'euclidean': 'geometry',
            'manifold': 'geometry',
            'algebraic_geometry': 'geometry',
            'differential_geometry': 'geometry',
            
            # Combinatorics subfields
            'graph': 'combinatorics',
            'partition': 'combinatorics',
            'enumeration': 'combinatorics',
            
            # Logic subfields
            'set_theory': 'logic',
            'settheory': 'logic',
            'model_theory': 'logic',
            'modeltheory': 'logic',
            'proof': 'logic'
        }
        
        # Check path parts first (more specific)
        for part in path_parts:
            for keyword, domain in domain_mappings.items():
                if keyword in part:
                    return domain
        
        # Check full path
        for keyword, domain in domain_mappings.items():
            if keyword in path_lower:
                return domain
        
        # Check namespace
        for keyword, domain in domain_mappings.items():
            if keyword in namespace_lower:
                return domain
        
        # Special cases for common patterns
        if any('alg' in part for part in path_parts):
            return 'algebra'
        if any('num' in part for part in path_parts):
            return 'number_theory'
        if any('top' in part for part in path_parts):
            return 'topology'
        if any('geom' in part for part in path_parts):
            return 'geometry'
        
        return 'general'

def test_lean4_parser():
    """Test the Lean 4 parser with a small subset"""
    
    mathlib_path = "/Users/julesdesai/Documents/HAI Lab Code/Bounded Godel Encoding/mathlib4"
    
    parser = Lean4Parser(mathlib_path)
    
    print("🔍 Testing Lean 4 Parser...")
    
    # Parse a small subset
    declarations = parser.parse_mathlib_subset(max_files=20)
    
    print(f"📊 Found {len(declarations)} declarations")
    
    # Show some examples
    print("\n📝 Example declarations:")
    for i, decl in enumerate(declarations[:5]):
        print(f"\n{i+1}. {decl.declaration_type.upper()}: {decl.name}")
        print(f"   Namespace: {decl.namespace}")
        print(f"   Statement: {decl.statement[:100]}...")
        print(f"   Dependencies: {len(decl.dependencies)}")
        print(f"   File: {Path(decl.file_path).name}")
    
    # Convert to formal theorems
    formal_theorems = parser.convert_to_formal_theorems(declarations)
    print(f"\n🎯 Converted {len(formal_theorems)} formal theorems")
    
    # Show domain distribution
    domains = {}
    for theorem in formal_theorems:
        domains[theorem.mathematical_domain] = domains.get(theorem.mathematical_domain, 0) + 1
    
    print("\n📈 Domain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"   {domain}: {count}")
    
    return parser, declarations, formal_theorems

if __name__ == "__main__":
    parser, declarations, formal_theorems = test_lean4_parser()