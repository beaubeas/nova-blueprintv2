from rdkit import Chem, DataStructs
import bittensor as bt
from rdkit.Chem import Descriptors, MACCSkeys, rdFMCS
from rdkit.Chem import rdSynthonSpaceSearch, rdFingerprintGenerator
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import random
from datetime import datetime
import os
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import math
from collections import defaultdict
from itertools import chain, combinations

load_dotenv(override=True)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from nova_ph2.utils.molecules import get_heavy_atom_count

# Global synthon searcher reference
synthon_searcher = None



def update_missing_smarts(db_path: str) -> bool:
    """Update reactions with N/A SMARTS to valid patterns"""
    
    smarts_updates = {
        1: '[C:1]#[C].[N:2][N+]#[N-]>>[C:1]1[N:2]N=NN1',  # Triazole click
        3: '[C:1]#[C].[N:2][N+]#[N-].[C:3](=O)[OH]>>[c:1]1[n:2]nnn1[C:3]=O',  # Click-amide cascade
        5: '[c:1][Br,Cl].[c:2][B]([OH])[OH]>>[c:1][c:2]'  # Sequential Suzuki
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            for rxn_id, smarts in smarts_updates.items():
                # Check current value
                cursor.execute("SELECT name, smarts FROM reactions WHERE rxn_id = ?", (rxn_id,))
                result = cursor.fetchone()
                
                if result:
                    name, current_smarts = result
                    if not current_smarts or current_smarts.upper() in ['N/A', 'NULL', '']:
                        cursor.execute(
                            "UPDATE reactions SET smarts = ? WHERE rxn_id = ?",
                            (smarts, rxn_id)
                        )
            
            conn.commit()
            return True
            
    except Exception as e:
        return False

def get_reaction_info(reaction_id: int, db_path: str):
    """Get reaction information from database"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT smarts, roleA, roleB, roleC 
                FROM reactions 
                WHERE rxn_id = ?
            """, (reaction_id,))
            result = cursor.fetchone()
            
            if result:
                smarts, roleA, roleB, roleC = result
                return smarts, roleA, roleB, roleC
            return None
    except Exception as e:
        return None

def get_molecules_by_role(role_id: int, db_path: str):
    """Get molecules for a specific role"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # The role_mask seems to be a bitmask or category identifier
            cursor.execute("""
                SELECT mol_id, smiles, role_mask 
                FROM molecules 
                WHERE role_mask = ? OR role_mask & ? != 0
                ORDER BY mol_id
            """, (role_id, role_id))
            return cursor.fetchall()
    except Exception as e:
        return []

def convert_db_to_synthon_format(sqlite_db_path: str, synthon_output_path: str) -> bool:
    """Convert database to synthon format by building SynthonSpace programmatically"""
    
    if not os.path.exists(sqlite_db_path):
        return False
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdChemReactions
        from datetime import datetime
        
        # We'll build a simple pickle-based synthon database instead
        # since RDKit's SynthonSpace text format is finicky
        
        synthon_data = {
            'reactions': [],
            'synthons': {},
            'metadata': {
                'version': '1.0',
                'created': str(datetime.now())
            }
        }
        
        with sqlite3.connect(sqlite_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT rxn_id, name, smarts, roleA, roleB, roleC 
                FROM reactions 
                ORDER BY rxn_id
            """)
            reactions = cursor.fetchall()
            
            if not reactions:
                return False
                
            valid_reactions = 0
            total_synthons = 0
            
            for rxn_id, name, smarts, roleA, roleB, roleC in reactions:
                try:
                    # Skip reactions with invalid SMARTS
                    if not smarts or smarts.strip() == "" or smarts.upper() in ["N/A", "NULL"]:
                        continue
                    
                    if ">>" not in smarts:
                        continue
                    
                    # Try to parse the reaction SMARTS
                    try:
                        rxn = AllChem.ReactionFromSmarts(smarts)
                        if not rxn:
                            continue
                    except Exception as e:
                        continue
                    
                    # Get molecules for each role
                    molecules_A = get_molecules_by_role(roleA, sqlite_db_path) if roleA else []
                    molecules_B = get_molecules_by_role(roleB, sqlite_db_path) if roleB else []
                    molecules_C = get_molecules_by_role(roleC, sqlite_db_path) if roleC else []
                    
                    if not molecules_A or not molecules_B:
                        continue
                    
                    # Collect valid synthons
                    valid_A = []
                    valid_B = []
                    valid_C = []
                    
                    for mol_id, smiles, _ in molecules_A[:1000]:
                        if smiles and smiles.strip():
                            try:
                                mol = Chem.MolFromSmiles(smiles.strip())
                                if mol:
                                    valid_A.append({
                                        'id': f"A_{rxn_id}_{mol_id}",
                                        'smiles': smiles.strip()
                                    })
                            except:
                                pass
                    
                    for mol_id, smiles, _ in molecules_B[:1000]:
                        if smiles and smiles.strip():
                            try:
                                mol = Chem.MolFromSmiles(smiles.strip())
                                if mol:
                                    valid_B.append({
                                        'id': f"B_{rxn_id}_{mol_id}",
                                        'smiles': smiles.strip()
                                    })
                            except:
                                pass
                    
                    if molecules_C:
                        for mol_id, smiles, _ in molecules_C[:1000]:
                            if smiles and smiles.strip():
                                try:
                                    mol = Chem.MolFromSmiles(smiles.strip())
                                    if mol:
                                        valid_C.append({
                                            'id': f"C_{rxn_id}_{mol_id}",
                                            'smiles': smiles.strip()
                                        })
                                except:
                                    pass
                    
                    if not valid_A or not valid_B:
                        continue
                    
                    # Store reaction data (don't store the mol objects, just SMILES)
                    reaction_data = {
                        'rxn_id': rxn_id,
                        'name': name,
                        'smarts': smarts,
                        'num_components': 3 if valid_C else 2,
                        'synthons_A': [s['smiles'] for s in valid_A],
                        'synthons_B': [s['smiles'] for s in valid_B],
                        'synthons_C': [s['smiles'] for s in valid_C] if valid_C else []
                    }
                    
                    synthon_data['reactions'].append(reaction_data)
                    
                    # Store synthons
                    for s in valid_A:
                        synthon_data['synthons'][s['id']] = s['smiles']
                    for s in valid_B:
                        synthon_data['synthons'][s['id']] = s['smiles']
                    for s in valid_C:
                        synthon_data['synthons'][s['id']] = s['smiles']
                    
                    valid_reactions += 1
                    total_synthons += len(valid_A) + len(valid_B) + len(valid_C)
                    
                except Exception as e:
                    continue
            
            if valid_reactions == 0:
                return False
        
        # Save as pickle file
        import pickle
        with open(synthon_output_path, 'wb') as f:
            pickle.dump(synthon_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify
        if not os.path.exists(synthon_output_path):
            return False
        
        output_size = os.path.getsize(synthon_output_path)
        
        if output_size < 1000:
            return False
        
        # Calculate potential products
        total_products = 0
        for rxn in synthon_data['reactions']:
            if rxn['num_components'] == 2:
                products = len(rxn['synthons_A']) * len(rxn['synthons_B'])
            else:
                products = len(rxn['synthons_A']) * len(rxn['synthons_B']) * len(rxn['synthons_C'])
            total_products += products
        
        return True
        
    except Exception as e:
        return False


class ImprovedMolecularSearch:
    """Enhanced molecular search using custom synthon database"""
    
    def __init__(self, synthon_db_path: str):
        """Initialize with custom synthon database"""
        import pickle
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        
        with open(synthon_db_path, 'rb') as f:
            self.synthon_data = pickle.load(f)
        
        self.reactions = self.synthon_data['reactions']
        self.synthons = self.synthon_data['synthons']
        
        # Use new MorganGenerator API (no deprecation warnings)
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        # Pre-compute fingerprints for faster searching
        self.synthon_fps = {}
        computed = 0
        for synthon_id, smiles in self.synthons.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    self.synthon_fps[synthon_id] = self.fp_gen.GetFingerprint(mol)
                    computed += 1
            except:
                pass
    
    def get_num_products(self) -> int:
        """Get total number of potential products"""
        total = 0
        for rxn in self.reactions:
            if rxn['num_components'] == 2:
                total += len(rxn['synthons_A']) * len(rxn['synthons_B'])
            else:
                total += len(rxn['synthons_A']) * len(rxn['synthons_B']) * len(rxn['synthons_C'])
        return total
    
    def search_by_fingerprint(self, query_mol, max_results: int = 100) -> List[str]:
        """Search for similar molecules using fingerprint similarity"""
        from rdkit import DataStructs
        
        if not query_mol:
            return []
        
        try:
            query_fp = self.fp_gen.GetFingerprint(query_mol)
        except:
            return []
        
        # Find similar synthons
        similarities = []
        for synthon_id, fp in self.synthon_fps.items():
            try:
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                if sim > 0.3:  # Threshold
                    similarities.append((synthon_id, sim, self.synthons[synthon_id]))
            except:
                pass
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for synthon_id, sim, smiles in similarities[:max_results]:
            results.append(smiles)
        
        return results
    
    def enumerate_products(self, max_products: int = 1000) -> List[str]:
        """Enumerate random products from the synthon space"""
        import random
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        products = []
        products_set = set()  # Track unique products
        attempts = 0
        max_attempts = max_products * 10
        
        while len(products) < max_products and attempts < max_attempts:
            attempts += 1
            
            # Pick a random reaction
            rxn_data = random.choice(self.reactions)
            
            try:
                # Pick random synthons
                smiles_A = random.choice(rxn_data['synthons_A'])
                smiles_B = random.choice(rxn_data['synthons_B'])
                
                mol_A = Chem.MolFromSmiles(smiles_A)
                mol_B = Chem.MolFromSmiles(smiles_B)
                
                if not mol_A or not mol_B:
                    continue
                
                # Parse reaction
                rxn = AllChem.ReactionFromSmarts(rxn_data['smarts'])
                if not rxn:
                    continue
                
                # Run reaction
                if rxn_data['num_components'] == 2:
                    reactants = (mol_A, mol_B)
                else:
                    smiles_C = random.choice(rxn_data['synthons_C'])
                    mol_C = Chem.MolFromSmiles(smiles_C)
                    if not mol_C:
                        continue
                    reactants = (mol_A, mol_B, mol_C)
                
                products_tuple = rxn.RunReactants(reactants)
                
                if products_tuple:
                    for product_set in products_tuple:
                        for product in product_set:
                            try:
                                Chem.SanitizeMol(product)
                                product_smiles = Chem.MolToSmiles(product)
                                if product_smiles and product_smiles not in products_set:
                                    products.append(product_smiles)
                                    products_set.add(product_smiles)
                                    if len(products) >= max_products:
                                        return products
                            except:
                                pass
            except Exception as e:
                continue
        
        return products
    
    def search_similar(self, target_smiles: str, max_results: int = 100) -> List[str]:
        """Search for molecules similar to target"""
        from rdkit import Chem
        
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            if not target_mol:
                return []
            
            return self.search_by_fingerprint(target_mol, max_results)
        except:
            return []
    
    def generate_from_synthons(self, synthon_ids: List[str]) -> Optional[str]:
        """Generate a product from specific synthon IDs"""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if len(synthon_ids) < 2:
            return None
        
        try:
            # Get synthon SMILES
            synthon_smiles = [self.synthons.get(sid) for sid in synthon_ids]
            if not all(synthon_smiles):
                return None
            
            # Convert to molecules
            mols = [Chem.MolFromSmiles(s) for s in synthon_smiles]
            if not all(mols):
                return None
            
            # Find compatible reaction
            for rxn_data in self.reactions:
                if rxn_data['num_components'] != len(mols):
                    continue
                
                try:
                    rxn = AllChem.ReactionFromSmarts(rxn_data['smarts'])
                    if not rxn:
                        continue
                    
                    products = rxn.RunReactants(tuple(mols))
                    if products:
                        for product_set in products:
                            for product in product_set:
                                try:
                                    Chem.SanitizeMol(product)
                                    return Chem.MolToSmiles(product)
                                except:
                                    pass
                except:
                    continue
            
            return None
        except:
            return None

    def similarity_based_exploration(self, reference_smiles_list: List[str], max_results: int = 200) -> List[str]:
        """
        Generate molecules similar to high-scoring references using synthon combinations
        
        Args:
            reference_smiles_list: List of reference SMILES to use as templates
            max_results: Maximum number of molecules to generate
            
        Returns:
            List of generated SMILES strings
        """
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        import random
        
        if not reference_smiles_list:
            return []
        
        results = []
        results_set = set()
        
        # Get fingerprints for reference molecules
        reference_fps = []
        for smiles in reference_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = self.fp_gen.GetFingerprint(mol)
                    reference_fps.append((smiles, fp))
            except:
                pass
        
        if not reference_fps:
            return []
        
        # Strategy: For each reaction, find synthons with ANY similarity to references
        # and combine them systematically
        
        for rxn_idx, rxn_data in enumerate(self.reactions):
            if len(results) >= max_results:
                break
            
            # Find synthons for this reaction that have similarity to references
            similar_A = []
            similar_B = []
            similar_C = []
            
            # Check synthons_A
            for synthon_smiles in rxn_data['synthons_A'][:100]:  # Limit to top 100 for speed
                try:
                    mol = Chem.MolFromSmiles(synthon_smiles)
                    if mol:
                        fp = self.fp_gen.GetFingerprint(mol)
                        max_sim = max(DataStructs.TanimotoSimilarity(fp, ref_fp) for _, ref_fp in reference_fps)
                        if max_sim > 0.2:  # Very low threshold
                            similar_A.append((synthon_smiles, max_sim))
                except:
                    pass
            
            # Check synthons_B
            for synthon_smiles in rxn_data['synthons_B'][:100]:
                try:
                    mol = Chem.MolFromSmiles(synthon_smiles)
                    if mol:
                        fp = self.fp_gen.GetFingerprint(mol)
                        max_sim = max(DataStructs.TanimotoSimilarity(fp, ref_fp) for _, ref_fp in reference_fps)
                        if max_sim > 0.2:
                            similar_B.append((synthon_smiles, max_sim))
                except:
                    pass
            
            # Check synthons_C if 3-component
            if rxn_data['num_components'] == 3:
                for synthon_smiles in rxn_data['synthons_C'][:100]:
                    try:
                        mol = Chem.MolFromSmiles(synthon_smiles)
                        if mol:
                            fp = self.fp_gen.GetFingerprint(mol)
                            max_sim = max(DataStructs.TanimotoSimilarity(fp, ref_fp) for _, ref_fp in reference_fps)
                            if max_sim > 0.2:
                                similar_C.append((synthon_smiles, max_sim))
                    except:
                        pass
            
            # If we don't have enough similar synthons, use random ones
            if len(similar_A) < 10:
                for synthon_smiles in random.sample(rxn_data['synthons_A'], min(20, len(rxn_data['synthons_A']))):
                    if synthon_smiles not in [s[0] for s in similar_A]:
                        similar_A.append((synthon_smiles, 0.1))
            
            if len(similar_B) < 10:
                for synthon_smiles in random.sample(rxn_data['synthons_B'], min(20, len(rxn_data['synthons_B']))):
                    if synthon_smiles not in [s[0] for s in similar_B]:
                        similar_B.append((synthon_smiles, 0.1))
            
            if rxn_data['num_components'] == 3 and len(similar_C) < 10:
                for synthon_smiles in random.sample(rxn_data['synthons_C'], min(20, len(rxn_data['synthons_C']))):
                    if synthon_smiles not in [s[0] for s in similar_C]:
                        similar_C.append((synthon_smiles, 0.1))
            
            if not similar_A or not similar_B:
                continue
            
            # Sort by similarity
            similar_A.sort(key=lambda x: x[1], reverse=True)
            similar_B.sort(key=lambda x: x[1], reverse=True)
            if rxn_data['num_components'] == 3:
                similar_C.sort(key=lambda x: x[1], reverse=True)
            
            # Parse reaction
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_data['smarts'])
                if not rxn:
                    continue
            except:
                continue
            
            # Generate products
            attempts = 0
            max_attempts_per_rxn = max_results * 10
            
            if rxn_data['num_components'] == 2:
                # 2-component: try top combinations
                for i in range(min(len(similar_A), 30)):
                    for j in range(min(len(similar_B), 30)):
                        if len(results) >= max_results or attempts >= max_attempts_per_rxn:
                            break
                        
                        attempts += 1
                        
                        try:
                            mol_A = Chem.MolFromSmiles(similar_A[i][0])
                            mol_B = Chem.MolFromSmiles(similar_B[j][0])
                            
                            if not mol_A or not mol_B:
                                continue
                            
                            products_tuple = rxn.RunReactants((mol_A, mol_B))
                            
                            if products_tuple:
                                for product_set in products_tuple:
                                    for product in product_set:
                                        try:
                                            Chem.SanitizeMol(product)
                                            product_smiles = Chem.MolToSmiles(product)
                                            
                                            if product_smiles and product_smiles not in results_set:
                                                results.append(product_smiles)
                                                results_set.add(product_smiles)
                                                
                                                if len(results) >= max_results:
                                                    return results
                                        except:
                                            pass
                        except:
                            pass
            
            else:
                # 3-component
                if not similar_C:
                    continue
                
                for i in range(min(len(similar_A), 15)):
                    for j in range(min(len(similar_B), 15)):
                        for k in range(min(len(similar_C), 15)):
                            if len(results) >= max_results or attempts >= max_attempts_per_rxn:
                                break
                            
                            attempts += 1
                            
                            try:
                                mol_A = Chem.MolFromSmiles(similar_A[i][0])
                                mol_B = Chem.MolFromSmiles(similar_B[j][0])
                                mol_C = Chem.MolFromSmiles(similar_C[k][0])
                                
                                if not mol_A or not mol_B or not mol_C:
                                    continue
                                
                                products_tuple = rxn.RunReactants((mol_A, mol_B, mol_C))
                                
                                if products_tuple:
                                    for product_set in products_tuple:
                                        for product in product_set:
                                            try:
                                                Chem.SanitizeMol(product)
                                                product_smiles = Chem.MolToSmiles(product)
                                                
                                                if product_smiles and product_smiles not in results_set:
                                                    results.append(product_smiles)
                                                    results_set.add(product_smiles)
                                                    
                                                    if len(results) >= max_results:
                                                        return results
                                            except:
                                                pass
                            except:
                                pass
        
        return results

    def pharmacophore_guided_search(self, reference_smiles_list: List[str], max_results: int = 150) -> List[str]:
        """
        Generate molecules with similar pharmacophoric features to references
        
        Args:
            reference_smiles_list: List of reference SMILES
            max_results: Maximum number of molecules to generate
            
        Returns:
            List of generated SMILES strings
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        import random
        
        if not reference_smiles_list:
            return []
        
        # Extract pharmacophoric features from references
        target_features = {
            'hbd': [],  # H-bond donors
            'hba': [],  # H-bond acceptors
            'mw': [],   # Molecular weight
            'logp': [], # LogP
            'rings': [] # Ring count
        }
        
        for smiles in reference_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    target_features['hbd'].append(Lipinski.NumHDonors(mol))
                    target_features['hba'].append(Lipinski.NumHAcceptors(mol))
                    target_features['mw'].append(Descriptors.MolWt(mol))
                    target_features['logp'].append(Descriptors.MolLogP(mol))
                    target_features['rings'].append(Lipinski.RingCount(mol))
            except:
                pass
        
        if not target_features['hbd']:
            return []
        
        # Calculate average features
        avg_features = {
            'hbd': sum(target_features['hbd']) / len(target_features['hbd']),
            'hba': sum(target_features['hba']) / len(target_features['hba']),
            'mw': sum(target_features['mw']) / len(target_features['mw']),
            'logp': sum(target_features['logp']) / len(target_features['logp']),
            'rings': sum(target_features['rings']) / len(target_features['rings'])
        }
        
        # Generate molecules and filter by pharmacophoric similarity
        results = []
        results_set = set()
        
        # Generate candidates
        candidates = self.enumerate_products(max_products=max_results * 5)
        
        # Score candidates by pharmacophoric similarity
        scored_candidates = []
        for smiles in candidates:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue
                
                # Calculate features
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                rings = Lipinski.RingCount(mol)
                
                # Calculate similarity score (lower is better)
                score = (
                    abs(hbd - avg_features['hbd']) +
                    abs(hba - avg_features['hba']) +
                    abs(mw - avg_features['mw']) / 100.0 +
                    abs(logp - avg_features['logp']) +
                    abs(rings - avg_features['rings'])
                )
                
                scored_candidates.append((smiles, score))
            except:
                pass
        
        # Sort by score and take top results
        scored_candidates.sort(key=lambda x: x[1])
        
        for smiles, score in scored_candidates[:max_results]:
            if smiles not in results_set:
                results.append(smiles)
                results_set.add(smiles)
        
        return results

def initialize_synthon_searcher_safe(config: dict) -> tuple[bool, Optional['ImprovedMolecularSearch']]:
    """Safely initialize synthon searcher with fallbacks"""
    # Remove the global declaration - don't use global here
    
    synthon_db_path = config.get("synthon_db_path")
    if not synthon_db_path:
        from pathlib import Path
        import nova_ph2
        DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
        
        synthon_db_path = os.path.join(os.path.dirname(__file__), "synthon_space.spc")
        
        if not os.path.exists(synthon_db_path):
            update_missing_smarts(DB_PATH)
            
            success = convert_db_to_synthon_format(DB_PATH, synthon_db_path)
            if not success:
                return False, None
    
    try:
        if os.path.exists(synthon_db_path) and os.path.getsize(synthon_db_path) > 1000:
            from molecules_new import ImprovedMolecularSearch
            
            searcher = ImprovedMolecularSearch(synthon_db_path)
            
            num_products = searcher.get_num_products()
            
            try:
                test_products = searcher.enumerate_products(max_products=10)
                if test_products and len(test_products) > 0:
                    return True, searcher  # Return both status and the searcher object
                else:
                    return False, None
            except Exception as e:
                return False, None
        else:
            return False, None
            
    except Exception as e:
        return False, None


def extract_pharmacophores_from_molecules(molecules_df: pd.DataFrame, 
                                        min_frequency: int = 2,
                                        max_patterns: int = 10) -> List[str]:
    """Extract common pharmacophoric patterns from high-scoring molecules"""
    
    if molecules_df.empty or len(molecules_df) < min_frequency:
        return []
    
    try:
        # Convert SMILES to molecules
        mols = []
        for smiles in molecules_df['smiles'].dropna():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
        
        if len(mols) < min_frequency:
            return []
        
        pharmacophores = set()
        
        # Find maximum common substructures (limited to avoid timeout)
        for i in range(min(len(mols), 5)):
            for j in range(i + 1, min(len(mols), 5)):
                try:
                    # Find MCS between pairs of molecules
                    mcs_result = rdFMCS.FindMCS([mols[i], mols[j]], 
                                              bondCompare=rdFMCS.BondCompare.CompareAny,
                                              atomCompare=rdFMCS.AtomCompare.CompareAny,
                                              timeout=2)  # Reduced timeout
                    
                    if mcs_result.numAtoms >= 5:
                        mcs_smarts = mcs_result.smartsString
                        if mcs_smarts and len(mcs_smarts) > 10:
                            pharmacophores.add(mcs_smarts)
                            
                            if len(pharmacophores) >= max_patterns:
                                break
                except Exception:
                    continue
            
            if len(pharmacophores) >= max_patterns:
                break
        
        # Add some common drug-like pharmacophores
        common_pharmacophores = [
            "c1ccc(cc1)C(=O)N",  # Benzamide
            "c1ccc2c(c1)cccc2",  # Naphthalene
            "c1ccc(cc1)N",       # Aniline
            "c1ccc(cc1)O",       # Phenol
            "C(=O)N",            # Amide
            "c1ccccc1",          # Benzene ring
        ]
        
        pharmacophores.update(common_pharmacophores)
        
        return list(pharmacophores)[:max_patterns]
        
    except Exception as e:
        return []

# Cached functions for performance
@lru_cache(maxsize=1000_000)
def _get_smiles_from_reaction_cached(name: str):
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _mol_from_smiles_cached(smiles: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _maccs_fp_from_smiles_cached(smiles: str):
    """Cache MACCS fingerprints for SMILES strings for fast Tanimoto similarity."""
    if not smiles:
        return None
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _inchikey_from_name_cached(name: str) -> str:
    """Cache InChIKey generation from molecule name to avoid repeated computation."""
    try:
        s = _get_smiles_from_reaction_cached(name)
        if not s:
            return ""
        return generate_inchikey(s)
    except Exception:
        return ""

def compute_maccs_entropy(smiles_list: list[str]) -> float:
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)
    return avg_entropy

def compute_enhanced_diversity(molecules_df: pd.DataFrame) -> float:
    """Enhanced diversity calculation combining MACCS and synthon-based metrics"""
    global synthon_searcher
    
    if molecules_df.empty:
        return 0.0
    
    try:
        # Original MACCS entropy
        maccs_entropy = compute_maccs_entropy(molecules_df['smiles'].tolist())
    except Exception as e:
        maccs_entropy = 0.0
    
    # Synthon-based diversity if available
    synthon_diversity = 0.0
    if synthon_searcher:
        try:
            synthon_diversity = synthon_searcher.compute_synthon_based_diversity(molecules_df)
        except Exception as e:
            pass
    
    # Combine metrics (weighted average)
    combined_diversity = 0.7 * maccs_entropy + 0.3 * synthon_diversity
    return combined_diversity

def num_rotatable_bonds(smiles: str) -> int:
    """Get number of rotatable bonds from SMILES string."""
    if not smiles:
        return 0
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return 0

def generate_inchikey(smiles: str) -> str:
    """Generate InChIKey from SMILES string."""
    if not smiles:
        return ""
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        return ""

def validate_molecules(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate molecules by checking heavy atom count and rotatable bonds."""
    if data.empty:
        return data
    
    data = data.copy()
    data['smiles'] = data["name"].apply(_get_smiles_from_reaction_cached)
    
    data = data[data['smiles'].notna()]
    if data.empty:
        return data
    
    data['heavy_atoms'] = data["smiles"].apply(get_heavy_atom_count)
    data['bonds'] = data["smiles"].apply(num_rotatable_bonds)
    
    mask = (
        (data['heavy_atoms'] >= config['min_heavy_atoms']) &
        (data['bonds'] >= config['min_rotatable_bonds']) &
        (data['bonds'] <= config['max_rotatable_bonds'])
    )
    data = data[mask]
    
    if not data.empty:
        data['InChIKey'] = data["smiles"].apply(generate_inchikey)
    
    return data

@lru_cache(maxsize=None)
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    try:
        abs_db_path = os.path.abspath(db_path)
        with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
                (role_mask, role_mask)
            )
            results = cursor.fetchall()
        return results
    except Exception as e:
        return []

def generate_valid_random_molecules_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    batch_size: int = 200,
    seed: int = None,
    elite_names: list[str] | None = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    avoid_inchikeys: set[str] | None = None,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []

    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    if elite_names:
        for name in elite_names:
            A, B, C = _parse_components(name)
            if A is not None: 
                elite_As.add(A)
            if B is not None: 
                elite_Bs.add(B)
            if C is not None and is_three_component: 
                elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []

    valid_dfs = []
    seen_keys = set()
    total_valid = 0

    while total_valid < n_samples:
        needed = n_samples - total_valid
        batch_size_actual = min(max(batch_size, 300), needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                pool_A_ids=pool_A_ids,
                pool_B_ids=pool_B_ids,
                pool_C_ids=pool_C_ids,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10,
                elite_As=elite_As,
                elite_Bs=elite_Bs,
                elite_Cs=elite_Cs,
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch
        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
        
        if not batch_molecules:
            continue
            
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]
        if batch_df.empty:
            continue
            
        batch_df = validate_molecules(batch_df, subnet_config)
        
        if batch_df.empty:
            continue

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        
        mask = ~batch_df["InChIKey"].isin(seen_keys)
        if avoid_inchikeys:
            mask = mask & ~batch_df["InChIKey"].isin(avoid_inchikeys)
        batch_df = batch_df[mask]
        
        if batch_df.empty:
            continue
        
        seen_keys.update(batch_df["InChIKey"].values)
        valid_dfs.append(batch_df[["name", "smiles", "InChIKey"]].copy())
        total_valid += len(batch_df)
        
        if total_valid >= n_samples:
            break
        
    if not valid_dfs:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    result_df = pd.concat(valid_dfs, ignore_index=True)
    return result_df.head(n_samples).copy()

def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None,
                                component_weights: dict = None) -> List[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None
    
    # Use weighted sampling if component weights are provided
    if component_weights:
        # Build weights for each component pool
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None
        
        # Normalize weights
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = rng.choices(A_ids, weights=weights_A, k=n) if weights_A else rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, weights=weights_B, k=n) if weights_B else rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, weights=weights_C, k=n) if weights_C else rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    # Remove duplicates while preserving order
    names = list(dict.fromkeys(names))
    return names

def _parse_components(name: str) -> tuple[int, int, int | None]:
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int,
                                   is_three_component: bool,
                                   pool_A_ids: list,
                                   pool_B_ids: list,
                                   pool_C_ids: list,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10,
                                   elite_As: set[int] = None,
                                   elite_Bs: set[int] = None,
                                   elite_Cs: set[int] = None) -> list[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []

    out = []
    local_names = set()
    check_inchikeys = avoid_inchikeys is not None and len(avoid_inchikeys) > 0
    
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(elite_Bs_list)
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            # Fast checks first (set membership is O(1))
            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if check_inchikeys:
                try:
                    key = _inchikey_from_name_cached(name)
                    if key and key in avoid_inchikeys:
                        continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            if name is None:
                A = rng.choice(pool_A_ids)
                B = rng.choice(pool_B_ids)
                if is_three_component:
                    C = rng.choice(pool_C_ids) if pool_C_ids else 0
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out

def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """Select diverse elite molecules: top by score, but ensure diversity in component space."""
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take top candidates (more than needed for diversity filtering)
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: at least min_score_ratio of max score
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
    if not candidates.empty:
        top_idx = candidates.index[0]
        top_row = candidates.iloc[0]
        selected.append(top_idx)
        parts = top_row['name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass
    
    # Then add diverse molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue
        
        parts = row['name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                
                # Prefer molecules with new components
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.5:  # Always take some top ones
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots with best remaining molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)

def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores
    for _, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)  # Only positive contributions
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1  # Smoothing
    
    return weights
