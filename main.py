
import os
import argparse
import torch
import warnings


import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger

from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler

from model.geomclip_train import GeomCLIP_PLModule
from model.unimol_simple import SimpleUniMolModel
from data_provider.geomclip_dm import GeomClipDM

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
#torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = GeomCLIP_PLModule.load_from_checkpoint(args.init_checkpoint, device=args.devices, args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = GeomCLIP_PLModule(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))
    print(args.match_batch_size)

    dm = GeomClipDM(args.num_workers, args.batch_size, args.root3d, args.text_max_len, model.geomclip.dictionary_mol, None, model.geomclip.tokenizer, args)
    dm.train_dataset.tokenizer = model.geomclip.tokenizer
    dm.val_dataset.tokenizer = model.geomclip.tokenizer
    dm.val_match_loader_3dtext.tokenizer = model.geomclip.tokenizer
    model.val_match_loader_3dtext = dm.val_match_loader_3dtext
    model.test_match_loader_3dtext = dm.test_match_loader_3dtext


    callbacks = []    
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    
    find_unused_parameters = True
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPSpawnStrategy(find_unused_parameters=find_unused_parameters)
    else:
        strategy = None
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'{args.store_path}/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger
                                         )
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1_test")
    parser.add_argument('--store_path', type=str, default="./all_checkpoints")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)
    parser.add_argument('--mode', type=str, default='train')

    parser = Trainer.add_argparse_args(parser)
    parser = GeomCLIP_PLModule.add_model_specific_args(parser)  # add model args
    parser = GeomClipDM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)

    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision='16',
                        max_epochs=100,
                        accumulate_grad_batches=4,
                        val_check_interval=0.01)

    args = parser.parse_args()
 
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

