import os
os.environ['CXX'] = 'g++-8'
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
import numpy as np
from data.load_data.data_tusz import *
from args import get_args_tusz

from model.TSADC import *
from utils.schedulers import *
from utils.model_utils import *
from tqdm import tqdm
from dotted_dict import DottedDict

from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')
args = get_args_tusz()


#############################################################


class PLModel(pl.LightningModule):
    def __init__(self, args, lr=1e-3, weight_decay=1e-3, optimizer_name="adamw", scheduler_name="cosine",
            steps_per_epoch=None, scaler=None, log_prefix="", **scheduler_kwargs):
        super().__init__()
        self.args = args
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.steps_per_epoch = steps_per_epoch
        self.scaler = scaler
        self.scheduler_kwargs = scheduler_kwargs
        self.log_prefix = log_prefix

        self._build_model()

    def _build_model(self):
        args = self.args
        undirected_graph = True

        self.model = TSADC(
            input_dim=args.input_dim,
            num_nodes=args.num_nodes,
            dropout=args.dropout,
            num_temporal_layers=args.num_temporal_layers,
            g_conv=args.g_conv,
            num_gnn_layers=args.num_gcn_layers,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            interval=args.interval,
            state_dim=args.state_dim,
            channels=args.channels,
            bidirectional=args.bidirectional,
            temporal_pool=args.temporal_pool,
            prenorm=args.prenorm,
            postact=args.postact,
            metric=args.graph_learn_metric,
            adj_embed_dim=args.adj_embed_dim,
            gin_mlp=args.gin_mlp,
            train_eps=args.train_eps,
            prune_method=args.prune_method,
            edge_top_perc=args.edge_top_perc,
            thresh=args.thresh,
            graph_pool=args.graph_pool,
            activation_fn=args.activation_fn,
            num_classes=args.output_dim,
            undirected_graph=undirected_graph,
            K=args.knn,
            regularizations=args.regularizations,
            residual_weight=args.residual_weight,
            masking = args.masking,
            masking_r = args.masking_r,
            diffuse_T = args.diffuse_T,
            diffuse_beta_0 = args.diffuse_beta_0,
            diffuse_beta_T = args.diffuse_beta_T,
            only_generat_missing = args.only_generate_missing,
            in_channels = args.in_channels,
            out_channels = args.out_channels,
            num_res_layers = args.num_res_layers,
            res_channels = args.res_channels,
            skip_channels = args.skip_channels,
            diffusion_step_embed_dim_in = args.diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid = args.diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out = args.diffusion_step_embed_dim_out,
            s4_max = args.s4_max,
            s4_d_state = args.s4_d_state,
            s4_dropout = args.s4_dropout,
            s4_bidirectional = args.s4_bidirectional,
            s4_layernorm = args.s4_layernorm,
            feature_smoothing_weight=args.feature_smoothing_weight,
            degree_weight=args.degree_weight,
            sparse_weight=args.sparse_weight,
            step_in_seq = args.step_in_seq,
            step_in_seq_test = args.step_in_seq_test,
            test_samples = args.test_samples,
            lambda_1 = args.lambda_1,
            lambda_2 = args.lambda_2,
            tau = args.tau,

        )


    def training_step(self, batch, batch_idx):

        y, rec_mask_loss, reg_loss, rec_loss = self._shared_step(batch, mode="train")
        log_dict = {}
        loss = rec_mask_loss + reg_loss + rec_loss
        log_dict["{}train/rec_mask_loss".format(self.log_prefix)] = rec_mask_loss.item()
        log_dict["{}train/reg_loss".format(self.log_prefix)] = reg_loss.item()
        log_dict["{}train/rec_loss".format(self.log_prefix)] = rec_loss.item()

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True,
                      add_dataloader_idx=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        (y, rec_mask_loss, reg_loss, rec_loss) = self._shared_step(batch, mode="train")
        return {"rec_mask_loss":rec_mask_loss, "reg_loss": reg_loss, "rec_loss": rec_loss}

    def validation_epoch_end(self, outputs):
        log_dict = {}
        for curr_outputs in outputs:
            rec_mask_loss = curr_outputs["rec_mask_loss"]
            rec_mask_loss = rec_mask_loss.item()
            reg_loss = curr_outputs["reg_loss"]
            reg_loss = reg_loss.item()
            rec_loss = curr_outputs["rec_loss"]
            rec_loss = rec_loss.item()

            loss = rec_mask_loss + reg_loss + rec_loss

            log_dict["{}val/loss".format(self.log_prefix)] = loss

            self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True,
                          add_dataloader_idx=False, sync_dist=True)


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        (orig_x, sampled_x, mask_sampled, pred_x, logits, y) = self._shared_step(batch, mode="test")
        return {"orig_x": orig_x, "labels": y, "sampled_x": sampled_x, "mask_sampled": mask_sampled,
                "pred_x": pred_x, "logits": logits}

    def test_epoch_end(self, outputs):

        rec1_score = []
        rec2_score = []
        labels = []

        for curr_outputs in outputs:
            orig_x = torch.cat([curr_outputs["orig_x"]]).squeeze()
            sampled_x = torch.cat([curr_outputs["sampled_x"]]).squeeze()
            mask_sampled = torch.cat([curr_outputs["mask_sampled"]]).squeeze()

            pred_x = torch.cat([curr_outputs["pred_x"]]).squeeze()
            logits = torch.cat([curr_outputs["logits"]]).squeeze()
            y = torch.cat([curr_outputs["labels"]]).squeeze()
            labels.append(int(y.item()))
            for idx in range(0, orig_x.shape[1], self.args.step_in_seq_test):

                ori_x = orig_x[:, idx:idx + self.args.step_in_seq_test]
                sam_x = sampled_x[:, idx:idx + self.args.step_in_seq_test]
                mask = mask_sampled[:, idx:idx + self.args.step_in_seq_test]
                pre_x = pred_x[:, idx:idx + self.args.step_in_seq_test]
                log = logits[:, idx:idx + self.args.step_in_seq_test]

                ori_x = ori_x.detach().cpu().numpy()
                sam_x = sam_x.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                pre_x = pre_x.detach().cpu().numpy()
                log = log.detach().cpu().numpy()

                score1 = mean_squared_error(ori_x[~mask.astype(bool)], sam_x[~mask.astype(bool)])
                score2 = mean_squared_error(ori_x, log)

                score1 = np.sqrt(score1)
                score2 = np.sqrt(score2)

                rec1_score.append(score1.item())
                rec2_score.append(score2.item())


        rec1_score = [rec1_score[x::self.args.test_samples] for x in range(0, int(len(rec1_score)/
                                                int(self.args.max_seq_len/self.args.step_in_seq_test)))]
        rec2_score = [rec2_score[x::self.args.test_samples] for x in range(0, int(len(rec2_score) /
                                                             int(self.args.max_seq_len / self.args.step_in_seq_test)))]

        score1 = np.average(rec1_score, axis=1)
        score2 = np.average(rec2_score, axis=1)
        score = (self.args.lambda_1*score1 + self.args.lambda_2*score2)

        y_pred = (score1 > self.args.tau).astype(int) #tau must be selected by X_valid
        score_dict = dict_metrics(y_pred=y_pred, y=labels, y_prob=score, file_names=None, average="binary")

        print("=========================================")
        print("Final Scores:")
        print(score_dict)


    def _shared_step(self, batch, mode = "train"):
        y = batch.y
        if y.shape[-1] == 1:
            y = y.view(-1)


        if mode == "test":
            sampled_x, mask_sampled, pred_x, logits= self.model(batch, return_attention=True, mode=mode)
            return (batch.x, sampled_x, mask_sampled,  pred_x, logits, y)
        else:
            rec_mask_loss, reg_losses, rec_loss = self.model(batch, return_attention=False, mode=mode)
            return (y, rec_mask_loss, reg_losses, rec_loss)




    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        elif self.scheduler_name == "one_cycle":
            print("steps_per_epoch:", self.steps_per_epoch)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.args.num_epochs,
            )
        elif self.scheduler_name == "timm_cosine":
            scheduler = TimmCosineLRScheduler(optimizer, **self.scheduler_kwargs)
        else:
            raise NotImplementedError

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main(args):

    # random seed
    pl.seed_everything(args.rand_seed, workers=True)

    scaler = None


    datamodule = TUSZ_DataModule(
        raw_data_path=args.raw_data_dir,
        dataset_name=args.dataset,
        freq=args.sampling_freq,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        standardize=True, #True
        balanced_sampling=args.balanced_sampling,
        pin_memory=True,
    )

    if args.load_model_path is not None:
        pl_model = PLModel.load_from_checkpoint(
            args.load_model_path,
            args=args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )
    else:
        pl_model = PLModel(
            args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )

    if args.do_train:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val/loss", mode="min", patience=args.patience
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        if not (args.gpus > 1):
            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=args.num_epochs,
                max_steps=-1,
                enable_progress_bar=True,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=args.gpu_id,  # default to 1 GPU
                accumulate_grad_batches=args.accumulate_grad_batches,
            )
        else:
            # distributed data parallel
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy=pl.strategies.DDPSpawnStrategy(
                    find_unused_parameters=False
                ),
                replace_sampler_ddp=False,
                max_epochs=args.num_epochs,
                max_steps=-1,
                enable_progress_bar=True,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=torch.cuda.device_count(),
                accumulate_grad_batches=args.accumulate_grad_batches,
            )

        trainer.fit(pl_model, datamodule=datamodule)
        print("Training DONE!")

        trainer.test(
            model=pl_model,
            ckpt_path= "best",
            dataloaders=datamodule.test_dataloader(),
        )

    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.gpu_id,
        )

        trainer.test(
            model=pl_model,
            ckpt_path=args.load_model_path,
            dataloaders=datamodule.test_dataloader(),
        )

if __name__ == "__main__":
    main(get_args_tusz())

