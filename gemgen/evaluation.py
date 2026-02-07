import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# Progress bar settings for non-interactive environments
TQDM_KWARGS = {'disable': not sys.stdout.isatty()}


def read_json_results(json_path):
    """
    Read the generator output JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
    
    Returns:
        List of [prompt, [smiles_list]] pairs.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def is_valid_smiles(smiles):
    """
    Check if a SMILES string is valid.
    
    Args:
        smiles (str): SMILES string.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def deduplicate_smiles_for_prompt(prompt_smiles_list):
    """
    Deduplicate SMILES for a single prompt with standardization.

    Args:
        prompt_smiles_list (list): List of SMILES strings for a prompt.

    Returns:
        list: Deduplicated list of valid standardized SMILES.
    """
    seen = set()
    unique_valid = []

    for smiles in prompt_smiles_list:
        # Convert SMILES to molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # Skip invalid SMILES

        # Convert molecule back to canonical SMILES for standardization
        std_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Check if this standardized SMILES has been seen
        if std_smiles not in seen:
            seen.add(std_smiles)
            # Store the standardized SMILES for consistency
            unique_valid.append(std_smiles)

    return unique_valid


def process_generator_results(json_path, output_csv):
    """
    Process generator results: read, deduplicate, filter invalid molecules.
    
    Args:
        json_path (str): Path to input JSON.
        output_csv (str): Path for output CSV.
    
    Returns:
        pandas.DataFrame: Processed dataframe with columns: description, smiles.
    """
    print("Reading JSON results...")
    data = read_json_results(json_path)
    
    processed_rows = []
    
    print("Processing prompts...")
    for entry in tqdm(data, desc="Processing prompts", **TQDM_KWARGS):
        prompt = entry[0]
        generated_smiles = entry[1]
        
        if not generated_smiles:
            continue
        
        unique_valid_smiles = deduplicate_smiles_for_prompt(generated_smiles)
        
        for smiles in unique_valid_smiles:
            processed_rows.append({"description": prompt, "smiles": smiles})
    
    df = pd.DataFrame(processed_rows, columns=["description", "smiles"])
    
    print(f"Processed {len(df)} valid, unique molecules from {len(data)} prompts.")
    return df


def compute_properties(smiles):
    """
    Compute QED, SAS, and Ro5 for a single SMILES.
    
    Args:
        smiles (str): SMILES string.
    
    Returns:
        tuple: (smiles, qed, sas, ro5) with None for invalid molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, None, None
    
    try:
        qed = QED.qed(mol)
    except:
        qed = None
    
    try:
        sas = sascorer.calculateScore(mol)
    except:
        sas = None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        ro5 = int(all([
            mw <= 500,
            logp <= 5,
            h_donors <= 5,
            h_acceptors <= 10
        ]))
    except:
        ro5 = None
    
    return smiles, qed, sas, ro5


def add_property_columns(df, n_jobs=None):
    """
    Add QED, SAS, and Ro5 columns to dataframe.
    
    Args:
        df (pandas.DataFrame): Dataframe with 'smiles' column.
        n_jobs (int, optional): Number of parallel jobs. Default: cpu_count.
    
    Returns:
        pandas.DataFrame: Dataframe with added property columns.
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    unique_smiles = df['smiles'].unique().tolist()
    
    print(f"Computing properties for {len(unique_smiles)} unique molecules...")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(compute_properties, unique_smiles, chunksize=100),
            total=len(unique_smiles),
            desc="Computing properties",
            **TQDM_KWARGS
        ))
    
    prop_dict = {}
    for smiles, qed, sas, ro5 in results:
        prop_dict[smiles] = {'qed': qed, 'sas': sas, 'ro5': ro5}
    
    df['QED'] = df['smiles'].map(lambda x: prop_dict.get(x, {}).get('qed'))
    df['SAS'] = df['smiles'].map(lambda x: prop_dict.get(x, {}).get('sas'))
    df['Ro5'] = df['smiles'].map(lambda x: prop_dict.get(x, {}).get('ro5'))
    
    return df


def smiles_to_fp(smiles, radius=2, nBits=2048):
    """
    Convert SMILES to Morgan fingerprint.
    
    Args:
        smiles (str): SMILES string.
        radius (int): Morgan fingerprint radius.
        nBits (int): Fingerprint length.
    
    Returns:
        RDKit fingerprint or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)


def tanimoto_similarity(fp1, fp2):
    """
    Calculate Tanimoto similarity between two fingerprints.
    
    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.
    
    Returns:
        float: Similarity or None if invalid.
    """
    if fp1 is None or fp2 is None:
        return None
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_max_similarity_for_molecule(gen_smiles, gt_smiles_list, gt_fp_dict):
    """
    Calculate maximum similarity between a generated molecule and ground truth molecules.
    
    Args:
        gen_smiles (str): Generated SMILES.
        gt_smiles_list (list): List of ground truth SMILES.
        gt_fp_dict (dict): Precomputed fingerprint dictionary for ground truth.
    
    Returns:
        tuple: (max_similarity, most_similar_gt_smiles)
    """
    gen_fp = smiles_to_fp(gen_smiles)
    if gen_fp is None or not gt_smiles_list:
        return None, None
    
    max_sim = 0.0
    most_similar = None
    
    for gt_smiles in gt_smiles_list:
        gt_fp = gt_fp_dict.get(gt_smiles)
        if gt_fp is None:
            continue
        
        sim = tanimoto_similarity(gen_fp, gt_fp)
        if sim is not None and sim > max_sim:
            max_sim = sim
            most_similar = gt_smiles
    
    return max_sim if max_sim > 0 else None, most_similar


def add_similarity_to_ground_truth(df, gt_csv_path, n_jobs=None):
    """
    Add maximum similarity to ground truth molecules.
    
    Args:
        df (pandas.DataFrame): Dataframe with generated molecules.
        gt_csv_path (str): Path to ground truth CSV with 'description' and 'smiles' columns.
        n_jobs (int, optional): Number of parallel jobs. Default: cpu_count.
    
    Returns:
        pandas.DataFrame: Dataframe with added similarity columns.
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Loading ground truth from {gt_csv_path}...")
    gt_df = pd.read_csv(gt_csv_path)
    
    # Create dictionary: description -> list of ground truth SMILES
    gt_dict = {}
    all_gt_smiles = set()
    
    for _, row in tqdm(gt_df.iterrows(), desc="Processing ground truth", **TQDM_KWARGS):
        desc = str(row['description']).strip()
        smiles_str = str(row['smiles']).strip()
        
        if pd.isna(smiles_str) or smiles_str == "":
            gt_smiles_list = []
        else:
            # Split by semicolon and clean
            gt_smiles_list = [s.strip() for s in smiles_str.split(';') if s.strip()]
        
        gt_dict[desc] = gt_smiles_list
        all_gt_smiles.update(gt_smiles_list)
    
    # Precompute fingerprints for all ground truth molecules
    print(f"Precomputing fingerprints for {len(all_gt_smiles)} ground truth molecules...")
    gt_fp_dict = {}
    for smiles in tqdm(all_gt_smiles, desc="Fingerprinting", **TQDM_KWARGS):
        fp = smiles_to_fp(smiles)
        if fp is not None:
            gt_fp_dict[smiles] = fp
    
    # Process each generated molecule
    print("Calculating similarities to ground truth...")
    results = []
    
    for _, row in tqdm(df.iterrows(), desc="Similarity calculation", **TQDM_KWARGS):
        desc = str(row['description']).strip()
        gen_smiles = row['smiles']
        
        gt_smiles_list = gt_dict.get(desc, [])
        max_sim, most_similar = calculate_max_similarity_for_molecule(
            gen_smiles, gt_smiles_list, gt_fp_dict
        )
        
        results.append((max_sim, most_similar))
    
    df['max_similarity_to_gt'] = [r[0] for r in results]
    df['most_similar_gt_smiles'] = [r[1] for r in results]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Post-process and evaluate small molecule generator results.'
    )
    parser.add_argument('--input_json', type=str, required=True, help='Path to generator output JSON file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path for output CSV file')
    parser.add_argument('--evaluate_properties', action='store_true', help='Calculate QED, SAS, and Ro5 properties')
    parser.add_argument('--evaluate_hit', action='store_true', help='Calculate similarity to ground truth molecules')
    parser.add_argument('--gt_csv', type=str, help='Path to ground truth CSV file (required if --evaluate_hit is set)')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel jobs (default: all CPUs)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.evaluate_hit and not args.gt_csv:
        parser.error("--gt_csv is required when --evaluate_hit is set")
    
    if args.gt_csv and not os.path.exists(args.gt_csv):
        parser.error(f"Ground truth CSV file not found: {args.gt_csv}")
    
    # Step 1: Basic processing
    print("=" * 60)
    print("Step 1: Basic processing (deduplication and validity check)")
    print("=" * 60)
    df = process_generator_results(args.input_json, args.output_csv)
    
    if df.empty:
        print("No valid molecules found. Exiting.")
        return
    
    # Step 2: Optional property evaluation
    if args.evaluate_properties:
        print("\n" + "=" * 60)
        print("Step 2: Calculating molecular properties (QED, SAS, Ro5)")
        print("=" * 60)
        df = add_property_columns(df, args.n_jobs)
    
    # Step 3: Optional hit evaluation
    if args.evaluate_hit:
        print("\n" + "=" * 60)
        print("Step 3: Calculating similarity to ground truth molecules")
        print("=" * 60)
        df = add_similarity_to_ground_truth(df, args.gt_csv, args.n_jobs)
    
    # Save results
    print(f"\nSaving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} molecules to {args.output_csv}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total molecules: {len(df)}")
    print(f"Unique prompts: {df['description'].nunique()}")
    
    if 'QED' in df.columns:
        print(f"QED (mean ± std): {df['QED'].mean():.3f} ± {df['QED'].std():.3f}")
        print(f"SAS (mean ± std): {df['SAS'].mean():.3f} ± {df['SAS'].std():.3f}")
        print(f"Ro5 compliant: {df['Ro5'].sum()} ({df['Ro5'].mean()*100:.1f}%)")
    
    print("Processing finished successfully.")


if __name__ == "__main__":
    main()