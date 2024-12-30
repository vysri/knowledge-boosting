import os

import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch import Callback
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import wandb


from src.losses.LossFn import LossFn

import src.utils as utils

class TSHearPLModule(pl.LightningModule):
    def __init__(self, joint_model, joint_model_params, 
                 small_model, small_model_params, 
                 big_model, big_model_params,
                 sr, freeze_bm = False,
                 init_ckpt = None,
                 optimizer=None, optimizer_params=None,
                 scheduler=None, scheduler_params = None,
                 big_model_init_ckpt = None, loss_params=None):
        super(TSHearPLModule, self).__init__()

        _small_model = utils.import_attr(small_model)(**small_model_params)
        _big_model = utils.import_attr(big_model)(**big_model_params) 
        
        if big_model_init_ckpt is not None:
            bm_ckpt = torch.load(big_model_init_ckpt)
            _big_model.load_state_dict(bm_ckpt)
            # state_dict = torch.load(big_model_init_ckpt)['state_dict']
            # state_dict = {k[6:] : v for k, v in state_dict.items() if k.startswith('model.') }

        if freeze_bm:
            for param in _big_model.parameters():
                param.requires_grad = False

        # print("PL MODULE SMALL", self.small_model)
        # print("PL MODULE BIG", self.big_model)

        self.joint_model = utils.import_attr(joint_model)(small_model = _small_model,
                                                          big_model = _big_model,
                                                          **joint_model_params)

        self.sr = sr

        # Values to log
        self.val_samples = []
        self.train_samples = []

        # Metric to monitor
        self.monitor = 'val/si_snr_i_sm'
        self.monitor_mode = 'max'

        # Initialize loss function
        loss_args = {}
        if loss_params is not None:
            loss_args = loss_params
        self.loss_fn = LossFn(**loss_args)

        # Initialize optimizer
        self.optimizer = utils.import_attr(optimizer)(self.parameters(), **optimizer_params)

        # Initialize scheduler
        if scheduler is not None:
            if scheduler == 'sequential':
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = utils.import_attr(scheduler_param['name'])(self.optimizer, **scheduler_param['params'])
                    schedulers.append(sched)
                    milestones.append(scheduler_param['epochs'])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i-1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones)
            else:
                self.scheduler = utils.import_attr(scheduler)(self.optimizer, **scheduler_params)
        else:
            self.scheduler = scheduler

    def forward(self, x):
        return self.joint_model(x)['output']

    def _step(self, batch, step='train'):
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]

        # Forward pass
        y_bm, y_sm = self.joint_model(inputs)
        output1_sm = y_sm['output1']
        output2_sm = y_sm['output2']

        output1_bm = y_bm['output1']
        output2_bm = y_bm['output2']

        # Compute loss and reorder outputs
        loss_bm, output1_bm, output2_bm = self._loss(None, None, est1=output1_bm, est2=output2_bm, gt1=targets['target1'], gt2 = targets['target2'])
        loss_sm, output1_sm, output2_sm = self._loss(None, None, est1=output1_sm, est2=output2_sm, gt1=targets['target1'], gt2 = targets['target2'])

        # Log metrics for large model
        snr_i_bm = torch.mean(self._metric_i(snr, inputs['mixture'], output1_bm, targets['target1']))
        snr_i_bm += torch.mean(self._metric_i(snr, inputs['mixture'], output2_bm, targets['target2']))
        snr_i_bm /= 2

        si_snr_i_bm = torch.mean(self._metric_i(si_snr, inputs['mixture'], output1_bm, targets['target1']))
        si_snr_i_bm += torch.mean(self._metric_i(si_snr, inputs['mixture'], output2_bm, targets['target2']))
        si_snr_i_bm /= 2

        # Log metrics for small model
        snr_i_sm = torch.mean(self._metric_i(snr, inputs['mixture'], output1_sm, targets['target1']))
        snr_i_sm += torch.mean(self._metric_i(snr, inputs['mixture'], output2_sm, targets['target2']))
        snr_i_sm /= 2

        si_snr_i_sm = torch.mean(self._metric_i(si_snr, inputs['mixture'], output1_sm, targets['target1']))
        si_snr_i_sm += torch.mean(self._metric_i(si_snr, inputs['mixture'], output2_sm, targets['target2']))
        si_snr_i_sm /= 2

        # Log small model metrics
        on_step = step == 'train'
        self.log(
            f'{step}/loss_sm', loss_sm, batch_size=batch_size, on_step=on_step,
            on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f'{step}/snr_i_sm', snr_i_sm.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False,
            sync_dist=True)
        # TODO: REMOVE EVENTUALLY
        self.log(
            f'{step}/si_snr_i_sm', si_snr_i_sm.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True)
        self.log(
            f'{step}/si_snr_i', si_snr_i_sm.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True)
        

        self.log(
            f'{step}/loss_bm', loss_bm, batch_size=batch_size, on_step=on_step,
            on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f'{step}/snr_i_bm', snr_i_bm.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False,
            sync_dist=True)
        self.log(
            f'{step}/si_snr_i_bm', si_snr_i_bm.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True)

        # Log additional metrics for validation and test
        if step in ['val', 'test']:
            pass

        output1_bm = output1_bm / torch.abs(output1_bm).max() * torch.abs(targets['target1']).max()
        output2_bm = output2_bm / torch.abs(output2_bm).max() * torch.abs(targets['target2']).max()

        output1_sm = output1_sm / torch.abs(output1_sm).max() * torch.abs(targets['target1']).max()
        output2_sm = output2_sm / torch.abs(output2_sm).max() * torch.abs(targets['target2']).max()

        sample = {
            'mixture': inputs['mixture'],
            'target1': targets['target1'],
            'output1_sm': output1_sm.detach(),
            'output1_bm': output1_bm.detach(),
            'target2': targets['target2'],
            'output2_sm': output2_sm.detach(),
            'output2_bm': output2_bm.detach(),
        }

        return loss_sm, loss_bm, sample

    def get_torch_model(self):
        return self.joint_model

    def _loss(self, pred, tgt, **kwargs):
        return self.loss_fn(pred, tgt, **kwargs)

    def _metric_i(self, metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).mean())
        return torch.stack(_vals)

    def training_step(self, batch, batch_idx):
        loss_sm, loss_bm, sample = self._step(batch, step='train')

        # Save some outputs for visualization
        if batch_idx % 200 == 0:
            self.train_samples.append(sample)

        return loss_sm

    def validation_step(self, batch, batch_idx):
        _, _, sample = self._step(batch, step='val')

        # Save some outputs for visualization
        if batch_idx % 10 == 0:
            self.val_samples.append(sample)

        return sample['output1_sm'], sample['output1_bm']

    def test_step(self, batch, batch_idx):
        _, _, sample = self._step(batch, step='test')

        # Save some outputs for visualization
        if batch_idx % 10 == 0:
            self.val_samples.append(sample)

        return sample['output1_sm'], sample['output1_bm']

    def configure_optimizers(self):
        if self.scheduler is not None:
            # For reduce LR on plateau, we need to provide more information
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_cfg = {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.monitor,
                    "strict": False
                }
            else:
                scheduler_cfg = self.scheduler
            return [self.optimizer], [scheduler_cfg]
        else:
            return self.optimizer

class TSHearLogger(Callback):
    def _log_audio(self, logger, key, samples, sr):
        # columns = ['mixture', 'target1', 'output1_bm', 'output1_sm', 
        #            'target2', 'output2_bm', 'output2_sm']
        columns = ['mixture', 'target1', 'output1_sm', 
                   'target2', 'output2_sm']
        wandb_samples = []
        for i, sample in enumerate(samples):
            for k in columns:
                if k in ['output1_bm', 'output1_sm', 'output2_bm', 'output2_sm']:
                    #print("SHAPE", sample[k][:,0:1,:].shape)
                    wandb_samples.append(wandb.Audio(
                        sample[k][0].permute(1, 0).cpu().numpy(),
                        sample_rate=sr, caption=f'{i}/{k}'))
                else:
                    wandb_samples.append(wandb.Audio(
                        sample[k][0].permute(1, 0).cpu().numpy(),
                        sample_rate=sr, caption=f'{i}/{k}'))
        logger.experiment.log({key: wandb_samples})

    def on_epoch_start(self):
        print('\n')

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "train/audio_samples", pl_module.train_samples,
            sr=pl_module.sr)
        pl_module.train_samples.clear()

    def on_validation_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "val/audio_samples", pl_module.val_samples,
            sr=pl_module.sr)
        pl_module.val_samples.clear()

    def on_test_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "test/audio_samples", pl_module.val_samples,
            sr=pl_module.sr)
        pl_module.val_samples.clear()