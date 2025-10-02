import torch
import torch.nn as nn
import math
import os
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class TranslationDataset(Dataset):
    def __init__(self, dataset_path, en_tokenizer_path, ar_tokenizer_path, max_seq_len= 512):
        """
        Args:
            dataset_path: Path to saved Hugging Face dataset
            en_tokenizer_path: Path to English tokenizer
            ar_tokenizer_path: Path to Arabic tokenizer
            max_seq_length: Maximum sequence length
        """
        # Load dataset
        self.dataset= load_from_disk(dataset_path)

        # Load tokenizers
        self.en_tokenizer= Tokenizer.from_file(en_tokenizer_path)
        self.ar_tokenizer= Tokenizer.from_file(ar_tokenizer_path)

        # Get special tokens
        self.pad_id= self.en_tokenizer.token_to_id("<PAD>")
        self.sos_id= self.en_tokenizer.token_to_id("<SOS>")
        self.eos_id= self.en_tokenizer.token_to_id("<EOS>")

        self.max_seq_length= max_seq_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item= self.dataset[idx]
        en_text= item['en']
        ar_text= item['ar']

        # Tokenize English (source)
        en_encoding= self.en_tokenizer.encode(en_text)
        en_ids= en_encoding.ids[:self.max_seq_length]

        # Tokenize Arabic (target) - add special tokens
        ar_encoding = self.ar_tokenizer.encode(ar_text)
        ar_ids = [self.sos_id] + ar_encoding.ids[:self.max_seq_length-2] + [self.eos_id]

        return {
            'en_ids': torch.tensor(en_ids, dtype=torch.long),
            'ar_ids': torch.tensor(ar_ids, dtype= torch.long),
            'en_text': en_text,
            'ar_text': ar_text
        }

def collate_fn(batch, pad_id):
    """Custom collate function to pad sequences and create masks"""
    en_ids= [item['en_ids'] for item in batch]
    ar_ids= [item['ar_ids'] for item in batch]

    # Pad Sequences
    en_ids_padded= torch.nn.utils.rnn.pad_sequence(
        en_ids, batch_first= True, padding_value= pad_id
    )
    ar_ids_padded= torch.nn.utils.rnn.pad_sequence(
        ar_ids, batch_first= True, padding_value= pad_id
    )

    # Create masks
    en_mask= (en_ids_padded != pad_id).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, src_len)
    ar_mask= (ar_ids_padded != pad_id).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, tgt_len)

    # Create causal mask for decoder
    tgt_len= ar_ids_padded.size(1)
    causal_mask= torch.tril(torch.ones(tgt_len, tgt_len)).bool().unsqueeze(0).unsqueeze(0) # (1, 1, tgt_len, tgt_len)

    return {
        'en_ids': en_ids_padded,
        'ar_ids': ar_ids_padded,
        'en_mask': en_mask,
        'ar_mask': ar_mask,
        'causal_mask': causal_mask,
        'en_texts': [item['en_text'] for item in batch],
        'ar_texts': [item['ar_text'] for item in batch]
    }

def create_data_loaders(data_dir_path, tokenizer_path, batch_size= 32, max_seq_length= 512):
    """
    Create Train and validation data loaders
    """
    # Dataset paths
    train_ds_path= os.path.join(data_dir_path, 'train_ds')
    val_ds_path= os.path.join(data_dir_path, 'val_ds') 
    # Tokenizer paths
    en_tokenizer_path= os.path.join(tokenizer_path, 'bpe_tokenizer_en.json')
    ar_tokenizer_path= os.path.join(tokenizer_path, 'bpe_tokenizer_ar.json')
    
    # Create datasets
    train_dataset= TranslationDataset(train_ds_path, en_tokenizer_path, ar_tokenizer_path, max_seq_length)
    val_dataset= TranslationDataset(val_ds_path, en_tokenizer_path, ar_tokenizer_path, max_seq_length)

    # Get pad_id from tokenizer
    en_tokenizer= Tokenizer.from_file(en_tokenizer_path)
    pad_id= en_tokenizer.token_to_id("<PAD>")

    # Create data loaders
    train_loader= DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle= True,
        collate_fn= lambda batch: collate_fn(batch, pad_id),
        num_workers= 4,
        pin_memory= True
    )
    val_loader= DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle= False,
        collate_fn= lambda batch: collate_fn(batch, pad_id),
        num_workers= 4,
        pin_memory= True
    )
    return train_loader, val_loader
    