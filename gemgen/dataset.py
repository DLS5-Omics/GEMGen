import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nlm_tokenizer import NatureLM1BTokenizer
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScorerDataset(Dataset):
    """Molecule-text dataset, processing raw TSV/CSV files directly"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_len: int = 8192,
        padding_idx: int = 0,
        sep: str = "\t",
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Data file path
            tokenizer_path: Tokenizer path
            max_len: Maximum sequence length
            padding_idx: Padding token id
            sep: File separator
        """
        super().__init__()
        
        self.max_len = max_len
        self.padding_idx = padding_idx
        
        # Load tokenizer
        logger.info("Loading tokenizer ...")
        self.tokenizer = NatureLM1BTokenizer.from_pretrained(tokenizer_path)
        
        # Load data
        logger.info(f"Reading data from {data_path} ...")
        self.df = pd.read_csv(data_path, sep=sep)
        
        # Check required columns
        required_cols = ['description', 'smiles']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Preprocess and cache data
        logger.info("Preprocessing data...")
        self.cached_data = self._preprocess_data()
        logger.info(f"Valid samples: {len(self.cached_data)}")
    
    def _preprocess_single_sample(self, row):
        """Process single sample"""
        try:
            cur_txt = f"{row['description']} <mol>{row['smiles']}</mol> Yes"
            # Encode text
            inputs_id = self.tokenizer.encode(cur_txt, add_special_tokens=True)
            
            # Filter out too long sequences
            if len(inputs_id) > self.max_len:
                logger.warning(f"[Warning] Skip sample: sequence length {len(inputs_id)} > max_len {self.max_len}")
                return None
            
            # Create num_idx_list
            num_idx_list = np.zeros(len(inputs_id), dtype=int)
            num_idx_list[-2] = 1
            
            return {
                'input_ids': inputs_id,
                'num_idx_list': num_idx_list,
                'length': len(inputs_id)
            }
            
        except Exception as e:
            logger.warning(f"[Warning] Skip sample: {e}")
            return None
    
    def _preprocess_data(self):
        """Preprocess all data"""
        data_list = []
        skipped_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), 
                            total=len(self.df), 
                            desc="Preprocessing data"):
            processed = self._preprocess_single_sample(row)
            if processed is not None:
                data_list.append(processed)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid samples")
        
        return data_list
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        """Get single sample"""
        sample = self.cached_data[idx]
        # Convert to tensor
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        num_idx_list = torch.tensor(sample['num_idx_list'], dtype=torch.long)
        
        return input_ids, num_idx_list
    
    def collate_fn(self, batch):
        """
        Batch processing function
        
        Args:
            batch: Batch data list
            
        Returns:
            dict: Batched tensors
        """
        # Get maximum length in current batch
        max_len = max(len(input_ids) for input_ids, _ in batch)
        batch_size = len(batch)
        
        # Initialize padded tensors
        padded_input_ids = torch.full((batch_size, max_len), self.padding_idx, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_num_idx_list = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Pad data
        for i, (input_ids, num_idx_list) in enumerate(batch):
            seq_len = len(input_ids)
            padded_input_ids[i, :seq_len] = input_ids
            attention_mask[i, :seq_len] = 1
            padded_num_idx_list[i, :seq_len] = num_idx_list
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'num_idxs': padded_num_idx_list
        }


def get_dataloader(data_path, tokenizer_path, batch_size=8, shuffle=False, 
                   max_len=8192, padding_idx=0, sep="\t", 
                   num_workers=4, pin_memory=True, persistent_workers=True):
    """
    Convenience function to create data loader
    
    Args:
        data_path: Data file path
        tokenizer_path: Tokenizer path
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_len: Maximum sequence length
        padding_idx: Padding token id
        sep: File separator
        num_workers: Number of data loading threads
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        
    Returns:
        DataLoader: PyTorch data loader
    """
    dataset = ScorerDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_len=max_len,
        padding_idx=padding_idx,
        sep=sep,
    )
    
    # Set tokenizers parallelism if using multiple workers
    if num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    return dataloader


# Usage example
if __name__ == "__main__":
    data_path = "/home/lizhen/jiangqun/druggen/GEMGen/data/scorer_test_demo.tsv"
    tokenizer_path = "/home/lizhen/jiangqun/druggen/models/tokenizer"
    
    # Create dataset
    dataset = ScorerDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_len=8192,
        padding_idx=0,
        sep="\t"
    )
    
    # Get single sample
    input_ids, num_idx_list = dataset[0]
    logger.info(f"Sample 0 - input_ids shape: {input_ids.shape}")
    logger.info(f"Sample 0 - num_idx_list shape: {num_idx_list.shape}")
    logger.info(f"Sample 0 - num_idx_list sum: {num_idx_list.sum().item()}")
    logger.info(f"Sample 0 - num_idx_list positions: {torch.where(num_idx_list == 1)[0].tolist()}")
    
    # Test collate_fn directly
    test_batch = [dataset[i] for i in range(4)]
    batched = dataset.collate_fn(test_batch)
    logger.info(f"Test batch - input_ids: {batched['input_ids'].shape}")
    logger.info(f"Test batch - attention_mask: {batched['attention_mask'].shape}")
    logger.info(f"Test batch - num_idxs: {batched['num_idxs'].shape}")
    logger.info(f"Test batch - num_idxs sum per sample: {batched['num_idxs'].sum(dim=1)}")

    # Create data loader
    dataloader = get_dataloader(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        batch_size=8,
        shuffle=False,
        sep="\t",
        num_workers=0  # Start with 0 workers to debug
    )
    
    # Iterate one batch
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Batch {batch_idx}:")
        logger.info(f"  input_ids: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask: {batch['attention_mask'].shape}")
        logger.info(f"  num_idxs: {batch['num_idxs'].shape}")
        logger.info(f"  num_idxs sum per sample: {batch['num_idxs'].sum(dim=1)}")
        
        # Only check first batch
        if batch_idx == 0:
            break