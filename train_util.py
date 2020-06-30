import time
import torch
import numpy as np
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
import pytorch_lightning as pl
from pytorch_lightning import Trainer

def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)

def load_model():
    pass

def run_train(config, model, train_loader, eval_loader, writer):
    init_seed()
    class MyLightninModule(pl.LightningModule):
        
        def __init__(self, model,train_loder,eval_loder,config):
            super(MyLightninModule, self).__init__()
            self.model = model
            self.train_loder=train_loder
            self.eval_loder=eval_loder
            self.config=config
            self.trloss=[]
            self.vlloss=[]

        def forward(self,x,y):
            return self.model(x,y)
        
        def training_step(self, batch, batch_idx, optimizer_idx=2):
            x, y = batch
            loss, logits = self.forward(x,y)
            logs = {'train_loss': loss}
            return {'loss': loss, 'log': logs}
    
        def validation_step(self,batch, batch_idx):
            x, y = batch
            loss,logits=self.forward(x,y)
            pred_flat = np.argmax(logits, axis=2).flatten()
            labels_flat = y.flatten()
            logs = {'val_loss': loss}
            return {'val_loss': loss,'correct': (pred_flat ==labels_flat).float(), 'log':logs}
        
        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            acc = torch.cat([x['correct'] for x in outputs]).mean()
            logs = {'val_loss': avg_loss, 'val_acc': acc}
            return {'val_loss': avg_loss,'val_acc': acc, 'log': logs}
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_loder), eta_min=self.config.lr)
            return [optimizer],[scheduler]
        
        def train_dataloader(self):
            return self.train_loder

        @pl.data_loader
        def val_dataloader(self):
            return self.eval_loder

    mode=MyLightninModule(model,train_loader,eval_loader,config)
    trainer = Trainer(max_epochs=config.epochs)
    out=trainer.fit(mode)




