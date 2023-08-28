import pytorch_lightning as pl
from transformers import BertModel, AdamW, AutoModelForQuestionAnswering
import torch.nn as nn
import torchmetrics 
import torch.optim as optim
import torchmetrics 
import logging
import torch


class QuestionAnswering(pl.LightningModule):
    
    def __init__(self, plm, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.model = AutoModelForQuestionAnswering.from_pretrained(plm)
        # self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.num_labels)
        
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids = input_ids, attention_mask = attention_mask
                            )
        
        # sequence_ouput = output[0]
        
        # logits = self.qa_outputs (output[0])
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        
        start_logits = output.start_logits
        end_logits = output.end_logits
        
        return start_logits, end_logits
    
    def compute_loss(self, start_logits, end_logits, start_positions, end_positions):
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        total_loss = (start_loss + end_loss) / 2
        return total_loss


    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
      
        start_positions = batch['start_position']
        end_positions = batch['end_position']

        start_logits, end_logits = self(input_ids, attention_mask)
        loss = self.compute_loss(start_logits, end_logits, start_positions, end_positions)
        acc = (self.accuracy(start_logits, start_positions) + self.accuracy(end_logits, end_positions)) / 2

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def accuracy(self, logits, positions):
        preds = torch.argmax(logits, dim=1)
        correct = (preds == positions).float()
        acc = correct.sum() / len(correct)
        
        return acc
        

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        start_positions = batch['start_position']
        end_positions = batch['end_position']

        start_logits, end_logits = self(input_ids, attention_mask)
        loss = self.compute_loss(start_logits, end_logits, start_positions, end_positions)
        acc = (self.accuracy(start_logits, start_positions) + self.accuracy(end_logits, end_positions)) / 2

        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log the training accuracy as well
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)
    
