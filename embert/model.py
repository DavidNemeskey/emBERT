import torch
from torch import nn
from transformers import AutoModelForTokenClassification

class TokenClassifier(nn.Module):
    def __init__(self, model_name_or_path, config, **kwargs):
        super().__init__()
        self.num_labels = config.num_labels
        
        # 1. Load the underlying model dynamically (BERT, RoBERTa, etc.)
        self.hf_model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=config, **kwargs
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, **kwargs):
        # 2. Allows your script to initialize this just like a Hugging Face model
        return cls(model_name_or_path, config, **kwargs)

    def save_pretrained(self, save_directory):
        # 3. Routes the save command to the underlying Hugging Face model
        # so your script's save/load logic works flawlessly.
        self.hf_model.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, valid_ids=None, attention_mask_label=None):
        
        # 4. DYNAMIC BASE MODEL EXTRACTION
        # This asks the model what its base prefix is.
        # If it's BERT, it grabs self.hf_model.bert. 
        # If it's RoBERTa, it grabs self.hf_model.roberta!
        base_model = getattr(self.hf_model, self.hf_model.base_model_prefix)
        
        # 5. Safely pass inputs (RoBERTa sometimes doesn't expect token_type_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
            
        sequence_output = base_model(**inputs)[0]
        
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32,
            device=next(self.parameters()).device
        )
        
        # 6. Your vectorized left-aligned packing
        for i in range(batch_size):
            valid_mask = valid_ids[i] == 1
            valid_tokens = sequence_output[i][valid_mask]
            num_valid = valid_tokens.size(0)
            if num_valid > 0:
                valid_output[i, :num_valid, :] = valid_tokens
                
        # 7. Use the dropout and classifier from the loaded HF model
        sequence_output = self.hf_model.dropout(valid_output)
        logits = self.hf_model.classifier(sequence_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            attention_mask_label = None 
            
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), 
                    labels.view(-1)
                )
            return (loss, logits)
        else:
            return (logits,)