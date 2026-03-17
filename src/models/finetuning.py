import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from peft import LoraConfig, get_peft_model

import numpy as np

# ENVIRONMENT SETUP FOR ESM MODEL COMPATIBILITY
# ============================================
# This code supports both ESM2 and ESM3/ESMC models, but they require different environments
# due to conflicting dependencies in their respective 'esm' libraries.
#
# SETUP INSTRUCTIONS:
# 1. Create two separate conda environments:
#    - 'workspace': For ESM3/ESMC models (newer ESM library)
#    - 'workspace-esm': For ESM2 models (original ESM library)
#
# 2. Configure your environment based on which models you want to use:
#    - If using ESM3/ESMC only: activate 'workspace' environment
#    - If using ESM2 only: activate 'workspace-esm' environment
#    - If using both: switch environments as needed
#
# 3. To customize for your setup:
#    - Modify the environment names below to match your conda environments
#    - Comment out unused model classes (ESM2* or ESMC*) if you only need one type
#    - Update the exception message with your specific environment names

env = os.environ['CONDA_DEFAULT_ENV']
if env == 'esm3':
    # ESM3/ESMC environment - newer ESM library with multimodal capabilities
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
elif env == 'workspace-esm':
    # ESM2 environment - original ESM library for protein language models
    import esm
else:
    raise EnvironmentError(
        f"Unsupported conda environment: '{env}'. "
        "This module requires either 'workspace' (for ESM3/ESMC) or 'workspace-esm' (for ESM2). "
        "Please activate the appropriate environment or modify the environment names in this file "
        "to match your setup."
    )


class ProteinFunDatasetLora(Dataset):
    """
    PyTorch Dataset for protein fitness prediction using LoRA finetuning.
    
    Loads protein sequences and fitness values for regression training.
    """
    
    def __init__(self, df):
        # self.seq, self.y = df['seq'].to_numpy(), df['fitness_log'].to_numpy().astype(np.float32)
        self.seq, self.y = df['seq'].to_numpy(), df['fitness_raw'].to_numpy().astype(np.float32)
    
    def __len__(self):
        return self.seq.shape[0]
    
    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx]


class ProteinFunDatasetContrast(Dataset):
    """
    PyTorch Dataset for contrastive protein fitness prediction.
    
    Prepares data for contrastive learning by computing mutation positions
    relative to wildtype sequence.
    """
    
    def __init__(self, df, wt):
        self.seq, self.y = df['seq'].to_numpy(), df['fitness_raw'].to_numpy()
        self.wt = np.array([wt]*self.seq.shape[0], dtype='object')
        self.n_mut = df['n_mut'].to_numpy()

        self.positions = []
        for _, row in df.iterrows():
            mt_sequence = row['seq']
            pos = []
            for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt)):
                if aa_wt != aa_mt:
                    ## mutation pos
                    pos.append(i)

            assert len(pos) == row['n_mut']

            self.positions.append(np.array(pos))

        assert len(self.positions) == self.seq.shape[0]
    
    def __len__(self):
        return self.seq.shape[0]
    
    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx], self.wt[idx], self.positions[idx], self.n_mut[idx]
    
    @staticmethod
    def collate_fn(data):
        """Custom collate function for handling variable-length mutation position arrays."""
        seq = np.array([x[0] for x in data], dtype='object')
        y = torch.tensor([x[1] for x in data])
        wt = np.array([x[2] for x in data], dtype='object')
        pos = [x[3] for x in data]
        n_mut = np.array([x[4] for x in data])
        return seq, y, wt, pos, n_mut


class ESM2ConFit(pl.LightningModule):
    """
    ESM2-based contrastive finetuning model for protein fitness prediction.
    
    Uses LoRA (Low-Rank Adaptation) to finetune ESM2 with Bradley-Terry loss
    and KL divergence regularization.
    """
    
    def __init__(self, model_path, config) -> None:
        super().__init__()
        self.config = config

        self.basemodel, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.model_reg, _ = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias='all'
        )
        
        self.model = get_peft_model(self.basemodel, peft_config)

        if self.config['use_seq_head']:
            for name, pm in self.model.named_parameters():
                if 'lm_head' in name or 'emb_layer_norm_after' in name:
                    pm.requires_grad = True
        
        if config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.accumulate_batch_bt_loss_train = []
        self.accumulate_batch_bt_loss_val = []
        self.accumulate_batch_kl_div_train = []
        self.accumulate_batch_kl_div_val = []
        self.debug=True

    def forward(self, batch, batch_tokens_masked, batch_tokens, batch_tokens_wt):
        mt_seq, _, wt_seq, pos, n_mut = batch
        
        logits = self.model(batch_tokens_masked)['logits']
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = torch.zeros(log_probs.shape[0])
        if self.config['device'] == 'gpu':
            scores = scores.cuda()

        for i in range(log_probs.shape[0]):
            scores[i] = torch.sum(log_probs[i, pos[i]+1, batch_tokens[i][pos[i]+1]] - log_probs[i, pos[i]+1, batch_tokens_wt[i][pos[i]+1]])
        
        return scores, logits
    
    def BT_loss(self, scores, y):
        loss = torch.tensor(0.)
        if self.config['device'] == 'gpu':
            loss = loss.cuda()

        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                if y[i] > y[j]:
                    loss += torch.log(1 + torch.exp(scores[j]-scores[i]))
                else:
                    loss += torch.log(1 + torch.exp(scores[i]-scores[j]))
        return loss

    def training_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        data = [
            (f'P{i}', wt_i) for i, wt_i in enumerate(wt_seq)
            ]
        _, _, batch_tokens_wt = self.batch_converter(data)

        data = [
            (f'P{i}', s) for i, s in enumerate(mt_seq)
            ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.alphabet.mask_idx
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        with torch.no_grad():
            logits_reg = self.model_reg(batch_tokens_masked)['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        self.accumulate_batch_bt_loss_train.append(bt_loss.item())
        self.accumulate_batch_kl_div_train.append(l_reg.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        data = [
            (f'P{i}', wt_i) for i, wt_i in enumerate(wt_seq)
            ]
        _, _, batch_tokens_wt = self.batch_converter(data)

        data = [
            (f'P{i}', s) for i, s in enumerate(mt_seq)
            ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.alphabet.mask_idx
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        with torch.no_grad():
            logits_reg = self.model_reg(batch_tokens_masked)['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())
        self.accumulate_batch_bt_loss_val.append(bt_loss.item())
        self.accumulate_batch_kl_div_val.append(l_reg.item())

    def trainmodel(self, df, wt, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetContrast(df, wt)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetContrast(val, wt)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=True)
        
        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)

        if self.config['model_checkpoint']:
            if callbacks is None:
                callbacks = []
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                filename="best-checkpoint-{epoch:02d}",
                save_top_k=2,
                mode="min",
            )
            callbacks.append(checkpoint_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                # precision="16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size'],
                                gradient_clip_val=1.0
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
           
    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()

        self.accumulate_batch_bt_loss_train.clear()
        self.accumulate_batch_bt_loss_val.clear()

        self.accumulate_batch_kl_div_train.clear()
        self.accumulate_batch_kl_div_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} bt loss: {np.mean(self.accumulate_batch_bt_loss_train)} kl div {np.mean(self.accumulate_batch_kl_div_train)} val loss: {np.mean(self.accumulate_batch_loss_val)} bt loss: {np.mean(self.accumulate_batch_bt_loss_val)} kl div {np.mean(self.accumulate_batch_kl_div_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def get_log_prob(self, sequence):
        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()
            self.model = self.model.cuda()

        with torch.no_grad():
            logits = self.model(batch_tokens)['logits']

        log_prob = torch.log_softmax(logits, dim=-1)[0,1:-1,:]

        return log_prob.cpu().numpy()
    
    def get_masked_marginal(self, mt_sequence, wt_sequence, mask_token = '<mask>'):

        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        mask_positions = []
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1
                mask_positions.append(i)

        assert len(mask_positions) == n_muts
        masked_query = list(wt_sequence)
        for _pos in mask_positions:
            masked_query[_pos] = mask_token
        masked_sequence = ''.join(masked_query)

        masked_log_prob = self.get_log_prob(sequence=masked_sequence)
        
        score = 0
        _idx = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos

                assert mask_positions[_idx] == i
                _idx += 1

                idx_mt = self.alphabet.get_idx(aa_mt)
                idx_wt = self.alphabet.get_idx(aa_wt)
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]

        return score, n_muts
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def predict(self, sequences, wt_sequence):
        pred = []
        for seq in tqdm(sequences):
            score, _ = self.get_masked_marginal(seq, wt_sequence)
            pred.append(score)

        return np.array(pred)


class ESMCConFit(pl.LightningModule):
    """
    ESMC-based contrastive finetuning model for protein fitness prediction.
    
    Uses LoRA adaptation to finetune ESMC with Bradley-Terry loss
    and KL divergence regularization.
    """
    
    def __init__(self, name, config) -> None:
        super().__init__()
        self.config = config

        if name == 'esmc_300m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 960
        elif name == 'esmc_600m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 1152
        else:
            raise Exception('Check ESMC name')
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["out_proj"],
        )
        
        self.model = get_peft_model(self.basemodel, peft_config)
        for name, pm in self.model.named_parameters():
            # if 'q_ln' in name or 'k_ln' in name or 'norm.weight' in name:
            #     pm.requires_grad = True
            if 'q_ln' in name or 'k_ln' in name:
                pm.requires_grad = True

        if self.config['use_seq_head']:
            for name, pm in self.model.named_parameters():
                if 'sequence_head' in name:
                    pm.requires_grad = True
        
        if config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.accumulate_batch_bt_loss_train = []
        self.accumulate_batch_bt_loss_val = []
        self.accumulate_batch_kl_div_train = []
        self.accumulate_batch_kl_div_val = []
        self.debug=True

    def forward(self, batch_tokens_masked, batch_tokens, batch_tokens_wt, pos):
        
        output = self.model(batch_tokens_masked)
        logits = output.sequence_logits
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = torch.zeros(log_probs.shape[0])
        if self.config['device'] == 'gpu':
            scores = scores.cuda()

        for i in range(log_probs.shape[0]):
            scores[i] = torch.sum(log_probs[i, pos[i]+1, batch_tokens[i][pos[i]+1]] - log_probs[i, pos[i]+1, batch_tokens_wt[i][pos[i]+1]])
        
        return scores, logits
    
    def BT_loss(self, scores, y):
        loss = torch.tensor(0.)
        if self.config['device'] == 'gpu':
            loss = loss.cuda()

        for i in range(len(scores)):
            for j in range(i, len(scores)):
                if y[i] > y[j]:
                    if torch.abs(scores[j]-scores[i]) < 80:
                        loss += torch.log(1 + torch.exp(scores[j]-scores[i]))
                else:
                    if torch.abs(scores[i]-scores[j]) < 80:
                        loss += torch.log(1 + torch.exp(scores[i]-scores[j]))
        return loss

    def training_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        batch_tokens_wt = self.model._tokenize(wt_seq)
        batch_tokens = self.model._tokenize(mt_seq)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.model.tokenizer.mask_token_id

        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        assert len(pos) == len(batch_tokens_masked)

        y_hat, logits = self(batch_tokens_masked, batch_tokens, batch_tokens_wt, pos)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        bt_loss = self.BT_loss(y_hat, y)

        with torch.no_grad():
            output = self.model_reg(batch_tokens_masked)
            logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        # print(f'contrast loss: {bt_loss.item()} | reg loss: {l_reg.item()} | loss: {loss.item()}')

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        self.accumulate_batch_bt_loss_train.append(bt_loss.item())
        self.accumulate_batch_kl_div_train.append(l_reg.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        batch_tokens_wt = self.model._tokenize(wt_seq)
        batch_tokens = self.model._tokenize(mt_seq)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.model.tokenizer.mask_token_id
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch_tokens_masked, batch_tokens, batch_tokens_wt, pos)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        with torch.no_grad():
            output = self.model_reg(batch_tokens_masked)
            logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss

        # print(f'contrast loss: {bt_loss.item()} | reg loss: {l_reg.item()} | loss: {loss.item()}')

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())
        self.accumulate_batch_bt_loss_val.append(bt_loss.item())
        self.accumulate_batch_kl_div_val.append(l_reg.item())

    def trainmodel(self, df, wt, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetContrast(df, wt)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetContrast(val, wt)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=True)

        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)

        if self.config['model_checkpoint']:
            if callbacks is None:
                callbacks = []
            checkpoint_callback = ModelCheckpoint(
                monitor="train/loss",
                filename="best-checkpoint-{epoch:02d}",
                save_top_k=3,
                mode="min",
            )
            callbacks.append(checkpoint_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                precision="bf16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size'],
                                gradient_clip_val=1.0
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
    def on_train_epoch_start(self):
        """Clear loss accumulators at the start of each epoch."""
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()

        self.accumulate_batch_bt_loss_train.clear()
        self.accumulate_batch_bt_loss_val.clear()

        self.accumulate_batch_kl_div_train.clear()
        self.accumulate_batch_kl_div_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} bt loss: {np.mean(self.accumulate_batch_bt_loss_train)} kl div {np.mean(self.accumulate_batch_kl_div_train)} val loss: {np.mean(self.accumulate_batch_loss_val)} bt loss: {np.mean(self.accumulate_batch_bt_loss_val)} kl div {np.mean(self.accumulate_batch_kl_div_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def get_log_prob(self, sequence):
        esm_protein = ESMProtein(sequence=sequence)

        if self.config['device'] == 'gpu':
            self.model = self.model.cuda()

        esm_tensor = self.model.encode(esm_protein)

        with torch.no_grad():
            results = self.model.logits(
                esm_tensor, LogitsConfig(sequence=True, return_embeddings=False)
            )

        logits = results.logits.sequence

        log_prob = torch.log_softmax(logits[0, 1:-1, :33], dim=-1)

        return log_prob.to(torch.float32).cpu().numpy()
    
    def get_masked_marginal(self, mt_sequence, wt_sequence, mask_token = '_'):

        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        mask_positions = []
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1
                mask_positions.append(i)

        assert len(mask_positions) == n_muts
        masked_query = list(wt_sequence)
        for _pos in mask_positions:
            masked_query[_pos] = mask_token
        masked_sequence = ''.join(masked_query)

        masked_log_prob = self.get_log_prob(sequence=masked_sequence)
        
        score = 0
        _idx = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos

                assert mask_positions[_idx] == i
                _idx += 1

                idx_mt = self.model.tokenizer.convert_tokens_to_ids(aa_mt)
                idx_wt = self.model.tokenizer.convert_tokens_to_ids(aa_wt)
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]


        return score, n_muts
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def predict(self, sequences, wt_sequence):
        pred = []
        for seq in tqdm(sequences):
            score, _ = self.get_masked_marginal(seq, wt_sequence)
            pred.append(score)

        return np.array(pred)


class ESM2LoraRegression(pl.LightningModule):
    """
    ESM2-based LoRA regression model for direct fitness prediction.
    
    Uses LoRA adaptation on ESM2 with MSE regression loss and KL regularization.
    """
    
    def __init__(self, model_path, config) -> None:
        super().__init__()
        
        self.basemodel, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.model_reg, _ = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()

        self.config = config

        if 't6_8M' in model_path:
            self.rep_layer = 6
            self.emb_dim = 320
        elif 't30_150M' in model_path:
            self.rep_layer = 30
            self.emb_dim = 640
        elif 't33_650M' in model_path:
            self.rep_layer = 33
            self.emb_dim = 1280
        else:
            raise Exception('I need to work on this. Feel free to extend :)')
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()

        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias='all'
        )

        self.model = get_peft_model(self.basemodel, peft_config)

        if self.config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.mlp = nn.Linear(self.emb_dim, 1)

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.accumulate_batch_mse_loss_train = []
        self.accumulate_batch_mse_loss_val = []
        self.accumulate_batch_kl_div_train = []
        self.accumulate_batch_kl_div_val = []
        self.debug=True

    def forward(self, batch_tokens):
        
        output = self.model(batch_tokens, repr_layers=[self.rep_layer], return_contacts=False)
        logits = output['logits']
        embeddings = output['representations'][self.rep_layer]

        cls_embedding = embeddings[:, 0, :]
        
        y_hat = self.mlp(cls_embedding)

        return y_hat, logits
    
    def training_step(self, batch, batch_idx):
        seq, y = batch

        data = [
            (f"P{i+1}", _seq) for i, _seq in enumerate(seq)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()

        y_hat, logits = self(batch_tokens)

        mse_loss = nn.functional.mse_loss(y_hat.flatten(), y)

        with torch.no_grad():
            output = self.model_reg(batch_tokens, repr_layers=[self.rep_layer], return_contacts=False)
        logits_reg = output['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = mse_loss + self.lambda_reg*l_reg

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        self.accumulate_batch_mse_loss_train.append(mse_loss.item())
        self.accumulate_batch_kl_div_train.append(l_reg.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, y = batch

        data = [
            (f"P{i+1}", _seq) for i, _seq in enumerate(seq)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()

        y_hat, logits = self(batch_tokens)

        mse_loss = nn.functional.mse_loss(y_hat.flatten(), y)

        with torch.no_grad():
            output = self.model_reg(batch_tokens, repr_layers=[self.rep_layer], return_contacts=False)
        logits_reg = output['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = mse_loss + self.lambda_reg*l_reg

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())
        self.accumulate_batch_mse_loss_val.append(mse_loss.item())
        self.accumulate_batch_kl_div_val.append(l_reg.item())

    def trainmodel(self, df, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetLora(df)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetLora(val)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                # precision="bf16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size'],
                                gradient_clip_val=1.0
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['lr'])

    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()

        self.accumulate_batch_mse_loss_train.clear()
        self.accumulate_batch_mse_loss_val.clear()

        self.accumulate_batch_kl_div_train.clear()
        self.accumulate_batch_kl_div_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} mse loss: {np.mean(self.accumulate_batch_mse_loss_train)} kl div {np.mean(self.accumulate_batch_kl_div_train)} val loss: {np.mean(self.accumulate_batch_loss_val)} mse loss: {np.mean(self.accumulate_batch_mse_loss_val)} kl div {np.mean(self.accumulate_batch_kl_div_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def predict(self, sequences):
        """
        Generate fitness predictions from protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Numpy array of fitness predictions
        """
        pred = []
        for seq in tqdm(sequences):
            data = [
                (f"P{i+1}", _seq) for i, _seq in enumerate([seq])
            ]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            if self.config['device'] == 'gpu':
                batch_tokens = batch_tokens.cuda()
                self.cuda()

                assert next(self.parameters()).is_cuda == True

            with torch.no_grad():
                output = self.model(batch_tokens, repr_layers=[self.rep_layer], return_contacts=False)
                embeddings = output['representations'][self.rep_layer]

                cls_embedding = embeddings[:, 0, :]
                
                y_hat = self.mlp(cls_embedding)

                assert y_hat.shape[0] == 1
                assert y_hat.shape[1] == 1
                pred.append(y_hat[0][0].cpu().item())

        return np.array(pred)
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


class ESMCLoraRegression(pl.LightningModule):
    """
    ESMC-based LoRA regression model for direct fitness prediction.
    
    Uses LoRA adaptation on ESMC with MSE regression loss and KL regularization.
    """
    
    def __init__(self, name, config) -> None:
        super().__init__()
        
        if name == 'esmc_300m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 960
        elif name == 'esmc_600m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 1152
        else:
            raise Exception('Check ESMC name')

        self.config = config
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()

        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            # target_modules=["out_proj", "ffn.1", "ffn.3", "layernorm_qkv.1"],
            target_modules=["out_proj"],
        )

        self.model = get_peft_model(self.basemodel, peft_config)
        for name, pm in self.model.named_parameters():
            if 'q_ln' in name or 'k_ln' in name or 'transformer.norm.weight' in name or 'layernorm_qkv.0' in name or 'ffn.0' in name:
                pm.requires_grad = True

        if self.config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.mlp = nn.Linear(self.emb_dim, 1)

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.accumulate_batch_mse_loss_train = []
        self.accumulate_batch_mse_loss_val = []
        self.accumulate_batch_kl_div_train = []
        self.accumulate_batch_kl_div_val = []
        self.debug=True

    def forward(self, batch_tokens):
        
        output = self.model(batch_tokens)
        logits = output.sequence_logits
        embeddings = output.embeddings

        cls_embedding = embeddings[:, 0, :]
        
        y_hat = self.mlp(cls_embedding)

        return y_hat, logits
    
    def training_step(self, batch, batch_idx):
        seq, y = batch

        batch_tokens = self.model._tokenize(seq)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()

        y_hat, logits = self(batch_tokens)

        mse_loss = nn.functional.mse_loss(y_hat.flatten(), y)

        with torch.no_grad():
            output = self.model_reg(batch_tokens)
            logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = mse_loss + self.lambda_reg*l_reg

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        self.accumulate_batch_mse_loss_train.append(mse_loss.item())
        self.accumulate_batch_kl_div_train.append(l_reg.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, y = batch

        batch_tokens = self.model._tokenize(seq)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()

        y_hat, logits = self(batch_tokens)

        mse_loss = nn.functional.mse_loss(y_hat.flatten(), y)

        with torch.no_grad():
            output = self.model_reg(batch_tokens)
            logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = mse_loss + self.lambda_reg*l_reg

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())
        self.accumulate_batch_mse_loss_val.append(mse_loss.item())
        self.accumulate_batch_kl_div_val.append(l_reg.item())

    def trainmodel(self, df, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetLora(df)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetLora(val)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                precision="bf16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size'],
                                gradient_clip_val=1.0
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['lr'])

    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()

        self.accumulate_batch_mse_loss_train.clear()
        self.accumulate_batch_mse_loss_val.clear()

        self.accumulate_batch_kl_div_train.clear()
        self.accumulate_batch_kl_div_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} mse loss: {np.mean(self.accumulate_batch_mse_loss_train)} kl div {np.mean(self.accumulate_batch_kl_div_train)} val loss: {np.mean(self.accumulate_batch_loss_val)} mse loss: {np.mean(self.accumulate_batch_mse_loss_val)} kl div {np.mean(self.accumulate_batch_kl_div_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def predict(self, sequences):
        """
        Generate fitness predictions from protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Numpy array of fitness predictions
        """
        pred = []
        for seq in tqdm(sequences):
            batch_tokens = self.model._tokenize([seq])

            if self.config['device'] == 'gpu':
                batch_tokens = batch_tokens.cuda()
                self.cuda()

                assert next(self.parameters()).is_cuda == True

            with torch.no_grad():
                output = self.model(batch_tokens)
                embeddings = output.embeddings

                cls_embedding = embeddings[:, 0, :]
                
                y_hat = self.mlp(cls_embedding.to(torch.float32))

                assert y_hat.shape[0] == 1
                assert y_hat.shape[1] == 1
                pred.append(y_hat[0][0].cpu().item())

        return np.array(pred)
    
    def print_trainable_parameters(self, model):
        """Print the number and percentage of trainable ESMCLoraRegression parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )