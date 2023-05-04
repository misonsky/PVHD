#coding=utf-8
import tensorflow as tf
from dataclasses import dataclass
from utils.file_utils import cached_property,tf_required


@dataclass
class StrategyObject(object):
    def __init__(self,config):
        super(StrategyObject,self).__init__()
        self.config = config
    @cached_property
    @tf_required
    def _setup_strategy(self):
        gpus = tf.config.list_physical_devices("GPU")
        if self.config.fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy) 
        if self.config.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError("Cannot find the proper strategy please check your environment properties.")    
        return strategy 
    @property
    @tf_required
    def strategy(self) -> "tf.distribute.Strategy":
        """
        The strategy used for distributed training.
        """
        return self._setup_strategy
    @property
    @tf_required
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        return self._setup_strategy.num_replicas_in_sync
       
    @property
    def train_batch_size(self) -> int:
        return self.config.train_batch_size * self.n_replicas 
    @property
    def eval_batch_size(self) -> int:
        return self.config.eval_batch_size * self.n_replicas
      
    @property
    @tf_required
    def n_gpu(self) -> int:
        return self._setup_strategy.num_replicas_in_sync    
        
        
