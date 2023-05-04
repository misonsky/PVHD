#coding=utf-8
import fnmatch
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import numpy as np
from tqdm.auto import tqdm

import requests
from filelock import FileLock
import tensorflow as tf

assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
_tf_available = True


def is_tf_available():
    return _tf_available

def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_tf_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires TF.")

    return wrapper
class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached
def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    if avg:
        return tf.reduce_mean(embedding, axis=1)
    else:
        return tf.reduce_sum(embedding, axis=1)
def bow_inputs(latent_sample,last_encoder):
    gen_inputs = tf.concat([last_encoder, latent_sample], 1)
    return gen_inputs







