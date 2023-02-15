import torch
from pytorch_lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

class LightningModel(LightningModule):
    def __init__(self, config, num_labels):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        self.auto_config = AutoConfig.from_pretrained(self.config.upstream_model, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.upstream_model, config=self.auto_config)

    def forward(self, **inputs):
        return self.model(**inputs) 

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.config.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.config.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
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