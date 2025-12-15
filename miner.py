import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import rdSynthonSpaceSearch, rdFingerprintGenerator
from typing import List, Optional, Callable

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_maccs_entropy,
    ImprovedMolecularSearch,
    convert_db_to_synthon_format,
    extract_pharmacophores_from_molecules,
    initialize_synthon_searcher_safe,
    compute_enhanced_diversity
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

target_models = []
antitarget_models = []
synthon_searcher = None

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models, synthon_searcher
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)
    
    # Initialize synthon searcher - get the object back
    synthon_available, synthon_searcher = initialize_synthon_searcher_safe(config)
    
    if not synthon_available or synthon_searcher is None:
        synthon_searcher = None

def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        return pd.Series(dtype=float)

def antitarget_scores():
    """Score molecules against all antitarget models."""
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df, 
                                            antitarget_model.model, 
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict, 
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        
        if not antitarget_scores:
            return pd.Series(dtype=float)
        
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        return pd.Series(dtype=float)

def enhanced_molecular_generation(config: dict, top_pool: pd.DataFrame, 
                                 iteration: int, n_samples: int, 
                                 avoid_inchikeys: set = None,
                                 elapsed_time: float = 0.0,
                                 total_time_budget: float = 1800.0) -> pd.DataFrame:
    """Enhanced multi-stage generation with time-based adaptive ratios (1800s budget)"""
    global synthon_searcher
    
    candidates = []
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    
    # 🎯 Time-based adaptive ratios with exact specifications
    if elapsed_time < 100:  # 0-100s
        synthon_ratio = 0.0
        pharma_ratio = 0.0
        traditional_ratio = 1.0
        elite_frac = 0.1
        mutation_prob = 0.7
        phase = "BOOTSTRAP (0:0:10)"
        
    elif elapsed_time < 200:  # 100-200s
        synthon_ratio = 0.05
        pharma_ratio = 0.05
        traditional_ratio = 0.9
        elite_frac = 0.2
        mutation_prob = 0.65
        phase = "EARLY EXPLORATION (0.5:0.5:9)"
        
    elif elapsed_time < 300:  # 200-300s
        synthon_ratio = 0.1
        pharma_ratio = 0.1
        traditional_ratio = 0.8
        elite_frac = 0.25
        mutation_prob = 0.6
        phase = "LIGHT EXPLORATION (1:1:8)"
        
    elif elapsed_time < 400:  # 300-400s
        synthon_ratio = 0.15
        pharma_ratio = 0.15
        traditional_ratio = 0.7
        elite_frac = 0.3
        mutation_prob = 0.55
        phase = "BROAD EXPLORATION (1.5:1.5:7)"
        
    elif elapsed_time < 500:  # 400-500s
        synthon_ratio = 0.2
        pharma_ratio = 0.2
        traditional_ratio = 0.6
        elite_frac = 0.35
        mutation_prob = 0.5
        phase = "BALANCED EXPLORATION (2:2:6)"
        
    elif elapsed_time < 600:  # 500-600s
        synthon_ratio = 0.25
        pharma_ratio = 0.25
        traditional_ratio = 0.5
        elite_frac = 0.4
        mutation_prob = 0.45
        phase = "TRANSITION (2.5:2.5:5)"
        
    elif elapsed_time < 900:  # 600-900s
        synthon_ratio = 0.4
        pharma_ratio = 0.3
        traditional_ratio = 0.3
        elite_frac = 0.5
        mutation_prob = 0.4
        phase = "EARLY EXPLOITATION (4:3:3)"
        
    elif elapsed_time < 1200:  # 900-1200s
        synthon_ratio = 0.5
        pharma_ratio = 0.3
        traditional_ratio = 0.2
        elite_frac = 0.6
        mutation_prob = 0.35
        phase = "MID EXPLOITATION (5:3:2)"
        
    elif elapsed_time < 1500:  # 1200-1500s
        synthon_ratio = 0.6
        pharma_ratio = 0.3
        traditional_ratio = 0.1
        elite_frac = 0.7
        mutation_prob = 0.25
        phase = "HEAVY EXPLOITATION (6:3:1)"
        
    else:  # 1500-1800s
        synthon_ratio = 0.7
        pharma_ratio = 0.25
        traditional_ratio = 0.05
        elite_frac = 0.8
        mutation_prob = 0.15
        phase = "FINAL OPTIMIZATION (7:2.5:0.5)"
    
    progress_pct = (elapsed_time / total_time_budget) * 100
    remaining_time = total_time_budget - elapsed_time
    
    # Stage 1: Synthon similarity search (SKIP when ratio is 0)
    if synthon_ratio > 0 and synthon_searcher and not top_pool.empty and iteration > 1:
        try:
            n_similarity = int(n_samples * synthon_ratio)
            if n_similarity > 0:
                # Adaptive reference pool size based on time
                if elapsed_time < 300:
                    n_refs = min(10, len(top_pool))
                elif elapsed_time < 900:
                    n_refs = min(20, len(top_pool))
                else:
                    n_refs = min(30, len(top_pool))
                
                reference_smiles = top_pool.head(n_refs)['smiles'].tolist()
                similarity_results = synthon_searcher.similarity_based_exploration(
                    reference_smiles, 
                    max_results=n_similarity
                )
                
                if similarity_results:
                    similarity_candidates = pd.DataFrame({
                        'smiles': similarity_results,
                        'name': [f'synthon_sim_{i}' for i in range(len(similarity_results))]
                    })
                    from rdkit import Chem
                    similarity_candidates['InChIKey'] = similarity_candidates['smiles'].apply(
                        lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None
                    )
                    similarity_candidates = similarity_candidates.dropna(subset=['InChIKey'])
                    
                    candidates.append(similarity_candidates)
        except Exception as e:
            pass
    
    # Stage 2: Pharmacophore-based search (SKIP when ratio is 0)
    if pharma_ratio > 0 and synthon_searcher and iteration > 2 and not top_pool.empty:
        try:
            n_pharma = int(n_samples * pharma_ratio)
            if n_pharma > 0:
                # Adaptive reference pool
                if elapsed_time < 300:
                    n_refs = min(15, len(top_pool))
                elif elapsed_time < 900:
                    n_refs = min(30, len(top_pool))
                else:
                    n_refs = min(50, len(top_pool))
                
                reference_smiles = top_pool.head(n_refs)['smiles'].tolist()
                pharma_results = synthon_searcher.pharmacophore_guided_search(
                    reference_smiles,
                    max_results=n_pharma
                )
                
                if pharma_results:
                    pharma_candidates = pd.DataFrame({
                        'smiles': pharma_results,
                        'name': [f'synthon_pharma_{i}' for i in range(len(pharma_results))]
                    })
                    from rdkit import Chem
                    pharma_candidates['InChIKey'] = pharma_candidates['smiles'].apply(
                        lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None
                    )
                    pharma_candidates = pharma_candidates.dropna(subset=['InChIKey'])
                    
                    candidates.append(pharma_candidates)
        except Exception as e:
            pass
    
    # Stage 3: Traditional combinatorial generation
    current_count = sum(len(df) for df in candidates)
    n_traditional = max(n_samples - current_count, int(n_samples * traditional_ratio))
    
    if n_traditional > 0:
        component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
        elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
        
        traditional_candidates = generate_valid_random_molecules_batch(
            rxn_id,
            n_samples=n_traditional,
            db_path=DB_PATH,
            subnet_config=config,
            batch_size=300,
            elite_names=elite_names,
            elite_frac=elite_frac,
            mutation_prob=mutation_prob,
            avoid_inchikeys=avoid_inchikeys,
            component_weights=component_weights,
        )
        
        if not traditional_candidates.empty:
            candidates.append(traditional_candidates)
    
    # Combine and deduplicate
    if candidates:
        combined = pd.concat(candidates, ignore_index=True)
        combined = combined.drop_duplicates(subset=['InChIKey'], keep='first')
        result = combined.head(n_samples)
        return result
    
    return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset_enhanced(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    """Enhanced diverse subset selection using both MACCS and synthon diversity"""
    global synthon_searcher
    
    smiles_list = pool["smiles"].tolist()
    
    best_combination = None
    best_score = -1
    
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        
        # Enhanced diversity score
        maccs_entropy = compute_maccs_entropy(test_subset)
        
        # Add synthon diversity if available
        synthon_score = 0.0
        if synthon_searcher:
            try:
                test_df = pd.DataFrame({'smiles': test_subset})
                test_df['name'] = ['temp'] * len(test_df)  # Placeholder names
                synthon_score = synthon_searcher.compute_synthon_based_diversity(test_df)
            except Exception:
                pass
        
        combined_score = 0.7 * maccs_entropy + 0.3 * synthon_score
        
        if combined_score >= entropy_threshold:
            print(f"Enhanced Diversity Threshold Met: MACCS={maccs_entropy:.4f}, Synthon={synthon_score:.4f}, Combined={combined_score:.4f}")
            return pool[pool["smiles"].isin(combination)]
        
        if combined_score > best_score:
            best_score = combined_score
            best_combination = combination
    
    print(f"No combination exceeded threshold. Best score: {best_score:.4f}")
    if best_combination:
        return pool[pool["smiles"].isin(best_combination)]
    return pd.DataFrame()

def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    seen_inchikeys = set()
    start_time = time.time()
    total_time_budget = 1800.0  # 30 minutes
    prev_avg_score = None
    current_avg_score = None
    score_improvement_rate = 0.0
    total_time = 0.0
    total_requested = 0
    total_unique = 0
    
    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples * 4
    
    with ProcessPoolExecutor(max_workers=1) as cpu_executor:
        while True:
            iteration += 1
            iter_start_time = time.time()
            
            # Calculate elapsed time and progress
            elapsed_time = time.time() - start_time
            remaining_time = total_time_budget - elapsed_time
            
            # Check if time budget exceeded
            if elapsed_time >= total_time_budget:
                break

            adjust_for_entropy = False
            if remaining_time <= 60:
                adjust_for_entropy = True

            # Enhanced molecular generation with time-based adaptation
            data = enhanced_molecular_generation(
                config=config,
                top_pool=top_pool,
                iteration=iteration,
                n_samples=n_samples_first_iteration if iteration == 1 else n_samples,
                avoid_inchikeys=seen_inchikeys,
                elapsed_time=elapsed_time,
                total_time_budget=total_time_budget
            )

            gen_time = time.time() - iter_start_time

            if data.empty:
                continue
            
            total_requested += len(data)
            
            # Filter out previously seen molecules
            try:
                filtered_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                total_unique += len(filtered_data)
                
                data = filtered_data

            except Exception as e:
                total_unique += len(data)

            if data.empty:
                continue

            data = data.reset_index(drop=True)

            # GPU scoring
            gpu_start_time = time.time()
            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])
            
            gpu_time = time.time() - gpu_start_time
            
            # Update seen molecules
            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            
            # Store previous iteration's final average BEFORE updating pool
            prev_iter_avg_score = top_pool['score'].mean() if not top_pool.empty else None
            
            # Update pool
            top_pool = pd.concat([top_pool, total_data])
            top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.sort_values(by="score", ascending=False)

            # Enhanced diversity adjustment
            if adjust_for_entropy:
                try:
                    top_95 = top_pool.iloc[:95]
                    remaining_pool = top_pool.iloc[95:]
                    additional_5 = select_diverse_subset_enhanced(
                        remaining_pool, 
                        top_95["smiles"].tolist(), 
                        subset_size=5, 
                        entropy_threshold=config['entropy_min_threshold']
                    )
                    if not additional_5.empty:
                        top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                        entropy = compute_enhanced_diversity(top_pool)
                    else:
                        top_pool = top_pool.head(config["num_molecules"])
                        entropy = compute_enhanced_diversity(top_pool)
                
                except Exception as e:
                    pass
            else:
                top_pool = top_pool.head(config["num_molecules"])
            
            # Calculate improvement metrics
            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None:
                if prev_iter_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_iter_avg_score) / max(abs(prev_iter_avg_score), 1e-6)
                else:
                    score_improvement_rate = 0.0
            else:
                score_improvement_rate = 0.0
            
            iter_total_time = time.time() - iter_start_time
            total_time += iter_total_time
            
            # Calculate improvement percentage for logging
            improvement_pct = 0.0
            if prev_iter_avg_score is not None and current_avg_score is not None and prev_iter_avg_score != 0:
                improvement_pct = (current_avg_score - prev_iter_avg_score) / abs(prev_iter_avg_score)
            
            prev_avg_score = current_avg_score
            
            # Calculate statistics
            avg_score = float(top_pool['score'].mean()) if not top_pool.empty else 0.0
            max_score = float(top_pool['score'].max()) if not top_pool.empty else 0.0
            min_score = float(top_pool['score'].min()) if not top_pool.empty else 0.0
            
            top_entries = {"molecules": top_pool["name"].tolist()}
            
            # Enhanced logging with time progress
            synthon_info = ""
            if synthon_searcher:
                synthon_info = f" | Synthon: ✓"
            
            progress_pct = (elapsed_time / total_time_budget) * 100

            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
    
    # Final summary
    final_elapsed = time.time() - start_time

if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    main(config)
