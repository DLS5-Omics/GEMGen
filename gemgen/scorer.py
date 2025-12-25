import os
from typing import Optional, Tuple, List
import argparse
import pandas as pd
import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel
)
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataset import get_dataloader

class NumMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(NumMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GEMGenScorer(LlamaPreTrainedModel):
    """
    A causal LM that uses HuggingFace LlamaModel directly, and adds an extra head (num_head).
    Returns:
        lm_logits: (B, T, vocab)
        num_logits: (B, T, 1)
    """
    def __init__(self, args):
        self.args = args
        config = LlamaConfig.from_pretrained(args.dict_path)
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_head = NumMLP(config.hidden_size, 4 * config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,                  
        llm_mask: Optional[torch.Tensor] = None,             
        position_ids: Optional[torch.LongTensor] = None,   
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if llm_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)
        else:
            attention_mask = llm_mask.to(device)
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.long()
            else:
                attention_mask = (attention_mask != 0).long()

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=kwargs.get("use_cache", False),
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  

        lm_logits = self.lm_head(hidden_states)   
        num_logits = self.num_head(hidden_states) 

        return lm_logits, num_logits
    
    def get_score_result(self, input_ids, llm_mask, idx_list, position_ids=None, **kwargs):
        """
        Get score results
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            llm_mask: Attention mask [batch_size, seq_len]
            idx_list: Numeric token indices [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            Score list for each sample
        """
        _, num_logits = self.forward(
            input_ids=input_ids,
            llm_mask=llm_mask,
            position_ids=position_ids,
            **kwargs,
        )

        # Squeeze num_logits to 2D: [batch_size, seq_len]
        num_logits = num_logits.squeeze(-1)
        
        scores = []
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # Get marked positions for current sample
            marked_positions = torch.where(idx_list[i] == 1)[0]
            # Get scores at marked positions
            marked_score = num_logits[i, marked_positions]
            sigmoid = nn.Sigmoid()
            scores.append(sigmoid(marked_score))
        
        return torch.tensor(scores, device=input_ids.device)
    
    def predict_batch(self, batch, device):
        """
        Predict for a single batch
        
        Args:
            batch: Dictionary containing input_ids, attention_mask, num_idxs
            device: Computing device
            
        Returns:
            Score tensor
        """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        num_idxs = batch['num_idxs'].to(device)
        
        with torch.no_grad():
            scores = self.get_score_result(
                input_ids=input_ids,
                llm_mask=attention_mask,
                idx_list=num_idxs
            )
        
        return scores
    
    def score_samples(self, descriptions: List[str], smiles: List[str]) -> List[float]:
        """
        Score a list of description and SMILES pairs
        
        Args:
            descriptions: List of description strings
            smiles: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            List of scores
        """
        from dataset import ScorerDataset
        import tempfile
        import numpy as np
        
        # Create temporary dataframe
        temp_data = pd.DataFrame({
            'description': descriptions,
            'smiles': smiles
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            temp_data.to_csv(f.name, sep='\t', index=False)
            temp_file_path = f.name
        
        try:
            # Create dataset
            dataset = ScorerDataset(
                data_path=temp_file_path,
                tokenizer_path=self.args.tokenizer_path,
                max_len=self.args.max_len,
                padding_idx=0,
                sep="\t"
            )
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=dataset.collate_fn
            )
            
            # Perform inference
            all_scores = []
            self.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    batch_scores = self.predict_batch(batch, self.args.device)
                    all_scores.extend(batch_scores.cpu().numpy())
                    
            return all_scores.tolist()
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)


def main():
    parser = argparse.ArgumentParser(description="GEMGen Scorer Inference")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to model dictionary")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--save_input_columns", action="store_true", help="Whether to save input columns (description and smiles) in the output.")
    
    args = parser.parse_args()
    
    try:
        from dataset import get_dataloader
    except ImportError as e:
        logger.error(f"Failed to import dataset module: {e}")
        logger.info("Please ensure dataset.py is in the same directory or in PYTHONPATH")
        return
    
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_csv(args.data_path, sep='\t')
    
    # Create data loader
    logger.info("Creating data loader...")
    dataloader = get_dataloader(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        shuffle=False,
        max_len=args.max_len,
        padding_idx=0,
        sep="\t",
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = GEMGenScorer(args)
    
    # Load model weights
    logger.info(f"Loading model weights from {args.model_path}...")
    try:
        if args.model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(args.model_path)
        else:
            # Original torch loading method
            state_dict = torch.load(args.model_path, map_location='cpu')
            
        # Check if this is a complete model state dict
        if 'model_state_dict' in state_dict:
            # This is a checkpoint containing multiple components
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            # This is direct model weights
            model.load_state_dict(state_dict)
            
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
    
    model.to(args.device)
    model.eval()
    
    # Perform inference
    logger.info("Performing inference...")
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Predict for batch
            batch_scores = model.predict_batch(batch, args.device)
            
            # Save scores
            all_scores.extend(batch_scores.cpu().numpy())
    
    # Ensure number of scores matches number of data rows
    if len(all_scores) != len(test_df):
        logger.warning(f"Number of scores ({len(all_scores)}) doesn't match number of rows ({len(test_df)})")
        # If counts don't match, only keep scores matching data rows count
        all_scores = all_scores[:len(test_df)]
    
    # Add scores to dataframe
    test_df['score'] = all_scores
    
    # If not saving input columns, remove description and smiles columns
    if not args.save_input_columns:
        columns_to_keep = [col for col in test_df.columns if col not in ['description', 'smiles']]
        test_df = test_df[columns_to_keep]
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save as TSV file
    test_df.to_csv(args.output_path, sep='\t', index=False)
    
    # Output statistics
    logger.info("Inference completed successfully!")
    logger.info(f"Total samples: {len(test_df)}")
    logger.info(f"Score statistics:")
    logger.info(f"  Mean: {test_df['score'].mean():.4f}")
    logger.info(f"  Std: {test_df['score'].std():.4f}")
    logger.info(f"  Min: {test_df['score'].min():.4f}")
    logger.info(f"  Max: {test_df['score'].max():.4f}")
    logger.info(f"  Median: {test_df['score'].median():.4f}")


if __name__ == "__main__":
    main()