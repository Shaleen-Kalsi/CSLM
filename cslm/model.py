import torch
from pytorch_lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BertModel,
)
from torch.optim import AdamW
import numpy as np

class LightningModel(LightningModule):
    def __init__(self, config, num_labels):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        self.auto_config = AutoConfig.from_pretrained(self.config.upstream_model, num_labels=num_labels)
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.config.upstream_model, config=self.auto_config)
        self.model = BertModel.from_pretrained(self.config.upstream_model, config=self.auto_config)
        #print(self.model)
        #print("*******************************************************************************************")
        #bert_params = list(self.model.named_parameters())
        #x = self.model(0, 0)[0][-1]
        #print("x:", x)
        #print(bert_params[17][1])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        input_dim = 768
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_labels)
        )
        #print(self.classifier)
        self.mixup_type = self.config.mixup_type
        # freeze the all weights except the classifier weights at the end
        for name, param in self.model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
        
    def basic_forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def mixup_forward(self, input_ids_x, input_ids_mixup_x, attention_mask_x, attention_mask_mixup_x, lam):
        x = self.model(input_ids=input_ids_x, attention_mask=attention_mask_x)[1]
        mixup_x = self.model(input_ids=input_ids_mixup_x, attention_mask=attention_mask_mixup_x)[1]
        
        x = lam*x + (1-lam)*mixup_x

        logits = self.classifier(x)
        return logits
    
    def forward(self, input_ids_x, input_ids_mixup_x, attention_mask_x, attention_mask_mixup_x, labels_x, labels_mixup_x, lam):
        return self.mixup_forward(input_ids_x, input_ids_mixup_x, attention_mask_x, attention_mask_mixup_x, lam)

    def training_step(self, batch, batch_idx):
        # batch
        print("start training")
        input_ids_x = batch['input_ids_x']
        input_ids_mixup_x = batch['input_ids_mixup_x']
        labels_x = batch['labels_x']
        labels_mixup_x = batch['labels_mixup_x']
        attention_mask_x = batch['attention_mask_x']
        attention_mask_mixup_x = batch['attention_mask_mixup_x']

        alpha = 0.5
        lam = np.random.beta(alpha, alpha)

        #construct the mixup label
        #labels_x = torch.stack(labels_x)
        #labels_mixup_x = torch.stack(labels_mixup_x)
        labels_x = lam*labels_x + (1-lam)*labels_mixup_x

        #token_type_ids = batch['token_type_ids']
        # fwd
        print("***start fwd***")
        logits = self(input_ids_x, input_ids_mixup_x, attention_mask_x, attention_mask_mixup_x, labels_x, labels_mixup_x, lam)
        #logits = outputs.last_hidden_state#[0] # Tuple containing loss[in this case loss is not calculated], logits, hidden states and attentions, since loss is None, only one element in tuple
        
        # loss
        loss = self.loss_fn(logits.view(-1, self.config.num_classes), labels_x)

        return {'loss': loss} # For backprop

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        input_ids_x = batch['input_ids_x']
        labels_x = batch['labels_x']
        attention_mask_x = batch['attention_mask_x']

        outputs = self.basic_forward(input_ids_x, attention_mask_x, labels_x)

        logits = self.classifier(outputs[1])

        if self.config.num_classes > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.config.num_classes == 1:
            preds = logits.squeeze()

        labels = labels_x

        print("labels shape", labels_x.shape)

        val_loss = self.loss_fn(logits.view(-1, self.config.num_classes), labels)

        return {"val-loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["val-loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        #self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        adam_epsilon = 1e-8
        warmup_steps = 0
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]