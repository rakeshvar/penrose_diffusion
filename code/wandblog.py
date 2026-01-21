import warnings
from pathlib import Path
from typing import Optional, Dict
import wandb

try:
    import wandb
    WANDB_AVAILABLE = True
except ModuleNotFoundError:
    WANDB_AVAILABLE = False

class WandBLog:
    def __init__(self, rank:int, wbconfig: Dict):
        self.config = wbconfig
        self.rank = rank
        self.is_master = (rank == 0)
        self.enabled = False
        self.run = None

        if not WANDB_AVAILABLE:
            print("WandB logging disabled: wandb module not found.")
            return
        
        if not self.is_master:
            return
            
        if not wbconfig['enabled']:
            print("WandB logging disabled by config.")
            return

        init_args = {
            'project': wbconfig['project'],
            'name': wbconfig['run_name'],
            'id': wbconfig['run_id'],
            'resume': wbconfig['run_id'] is not None,
        }
        
        self.run = wandb.init(**init_args) # type: ignore            
        self.enabled = True
        print(f"✓ WandB initialized: {self.run.url}")                   
    
    #---------------------------------------
    # Log once by updating config
    #---------------------------------------
    def update_run_config(self, new_dict):
         if not self.enabled:
            return
         try:
             self.run.config.update(new_dict, allow_val_change=True) # type: ignore
         except Exception as e:
             warnings.warn(f"Failed to log: {e}")
    
    def info(self, tr_conf, compat, dataset, denoiser, loss_functor):
        if not self.enabled:
            return
        
        self.update_run_config({
            **tr_conf.conf_dict,
            'timestamp': tr_conf.timestamp,
            'checkpoint_path': str(tr_conf.checkpoint_path) if tr_conf.checkpoint_path else None,
            'data_path': str(tr_conf.data_path_orig),
            'resume_epoch': tr_conf.resume_epoch,
            'device_type': 'TPU' if compat.IS_TPU else ('GPU' if compat.IS_GPU else 'CPU'),
            'locale': 'GCP' if 'is_gcp' else ('COLAB' if compat.IS_COLAB else 'LOCAL'),
            'num_tiles': dataset.num_tiles,
            'symmetry': dataset.symmetry,
            'side': dataset.side,
            'num_classes': dataset.num_classes,
            'dataset_size': len(dataset),
        })
    
        num_params = sum(p.numel() for p in denoiser.parameters())
        num_trainable = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        self.update_run_config({
            'loss_function': type(loss_functor).__qualname__,
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
        })
            
        if self.config['watch_freq'] > 0:
            watch_freq = self.config['watch_freq']
            self.run.watch(denoiser, log='all', log_freq=watch_freq) #type: ignore
            print(f"✓ WandB watching model (freq={watch_freq})")
    
    #--------------------------------------
    # Log every epoch
    #--------------------------------------
    def log_step(self, new_dict, step):
        if not self.enabled:
            return
        try:
            self.run.log(new_dict, step=step) # type: ignore
        except Exception as e:
            warnings.warn(f"Failed to log: {e}")
    
    def lsepoch_metrics(self, epoch: int, metrics: Dict):
        metrics['epoch'] = epoch
        self.log_step(metrics, step=epoch)
    
    def lsgradient_norm(self, epoch: int, model):
        if not self.enabled:
            return
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log_step({'gradient_norm': total_norm}, step=epoch)

    
    def lsvg(self, epoch: int, svg_content: str, class_label: int, class_name: str):
        if not self.enabled:
            return
        
        self.log_step({
            f'samples/epoch_{epoch}': wandb.Html(svg_content),
            f'samples/class_label': class_label,
            f'samples/class_name': class_name,
        }, step=epoch)


    #--------------------------------------
    # Wrap up
    #--------------------------------------
    def get_run_id(self) -> Optional[str]:
        """Get current WandB run ID for checkpoint saving."""
        if self.enabled and self.run:
            return self.run.id
        return None
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            try:
                print("✓ Finishing WandB run...")
                self.run.finish()
            except Exception as e:
                warnings.warn(f"Failed to finish WandB run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False