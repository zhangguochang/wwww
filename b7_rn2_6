#!/usr/bin/env python
# coding: utf-8

# ## Configuration

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from keras import backend as K
try:
    del model
    gc.collect()
except:
    pass
try:
    del test_preds
    gc.collect()
except:
    pass
# In[10]:
def binary_focal_loss_fixed(y_true, y_pred):
    """
    y_true shape need be (None,1)
    y_pred need be compute after sigmoid
    """
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    return K.mean(focal_loss)
def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy


def build_decoder(with_labels=True, target_size=(300, 300), ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)

        return img
    
    def decode_with_labels(path, label):
        return decode(path), label
    
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    
    return dset


# In[11]:


COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()


# In[12]:


IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600,750)

load_dir = f"/kaggle/input/{COMPETITION_NAME}/"
sub_df = pd.read_csv(load_dir + 'sample_submission.csv')
test_paths = load_dir + "test/" + sub_df['StudyInstanceUID'] + '.jpg'

# Get the multi-labels
label_cols = sub_df.columns[1:]


# #efnb7_fold_0

# In[13]:


BATCH_SIZE = strategy.num_replicas_in_sync * 80
test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[7], IMSIZE[7]))
dtest_norm = build_dataset(
        test_paths, bsize=BATCH_SIZE, repeat=False, 
        shuffle=False, augment=False, cache=False,
        decode_fn=test_decoder
        )
model = tf.keras.models.load_model(
        '../input/md-fc0/model_fold_0.h5',custom_objects={'binary_focal_loss': binary_focal_loss,'binary_focal_loss_fixed':binary_focal_loss_fixed}
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=tppp*0.06
try:
    del tppp
    gc.collect()
except:
    pass
dtest_aug = build_dataset(
        test_paths, bsize=BATCH_SIZE, repeat=False, 
        shuffle=False, augment=True, cache=False,
        decode_fn=test_decoder
        )
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del model,tppp
    gc.collect()
except:
    pass


# #efnb7_fold_1

# In[14]:


model = tf.keras.models.load_model(
        '../input/md-fc1/model_fold_1.h5',custom_objects={'binary_focal_loss': binary_focal_loss,'binary_focal_loss_fixed':binary_focal_loss_fixed}
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp,model
    gc.collect()
except:
    pass


# #efnb7_fold_2

# In[15]:


model = tf.keras.models.load_model(
        '../input/md-fc2/model_fold_2.h5',custom_objects={'binary_focal_loss': binary_focal_loss,'binary_focal_loss_fixed':binary_focal_loss_fixed}
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp,model
    gc.collect()
except:
    pass


# #efnb7_fold_3

# In[16]:


model = tf.keras.models.load_model(
        '../input/md-fc3/model_fold_3.h5',custom_objects={'binary_focal_loss': binary_focal_loss,'binary_focal_loss_fixed':binary_focal_loss_fixed}
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp,model
    gc.collect()
except:
    pass


# #efnb7_fold_4

# In[17]:


model = tf.keras.models.load_model(
        '../input/md-fc4/model_fold_4.h5',custom_objects={'binary_focal_loss': binary_focal_loss,'binary_focal_loss_fixed':binary_focal_loss_fixed}
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.06
try:
    del tppp,model,test_decoder,dtest_norm,dtest_aug
    gc.collect()
except:
    pass


# #rsnt_fold_0

# In[18]:

BATCH_SIZE = strategy.num_replicas_in_sync * 90
test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[8], IMSIZE[8]))
dtest_norm = build_dataset(
        test_paths, bsize=BATCH_SIZE, repeat=False, 
        shuffle=False, augment=False, cache=False,
        decode_fn=test_decoder
        )
model = tf.keras.models.load_model(
        '../input/rsnt-mm0/model_fold_0.h5'
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp
    gc.collect()
except:
    pass
dtest_aug = build_dataset(
        test_paths, bsize=BATCH_SIZE, repeat=False, 
        shuffle=False, augment=True, cache=False,
        decode_fn=test_decoder
        )
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp,model
    gc.collect()
except:
    pass


# #rsnt_fold_1

# In[19]:


model = tf.keras.models.load_model(
        '../input/rsnt-m1/model_fold_1.h5'
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp,model
    gc.collect()
except:
    pass


# #rsnt_fold_2

# In[20]:


model = tf.keras.models.load_model(
        '../input/rsnt-m2/model_fold_2.h5'
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp,model
    gc.collect()
except:
    pass


# #rsnt_fold_3

# In[21]:


model = tf.keras.models.load_model(
        '../input/rsnt-mm3/model_fold_3.h5'
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp,model
    gc.collect()
except:
    pass


# #rsnt_fold_4

# In[22]:


model = tf.keras.models.load_model(
        '../input/rsnt-m4/model_fold_4.h5'
        )
tppp = model.predict(dtest_norm, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp
    gc.collect()
except:
    pass
tppp = model.predict(dtest_aug, verbose=1)
sub_df[label_cols]=sub_df[label_cols]+tppp*0.04
try:
    del tppp,model,test_decoder,dtest_norm,dtest_aug
    gc.collect()
except:
    pass



# #combine

# In[41]:

submission=pd.read_csv("./submission969.csv")
submission2=pd.read_csv("./seresnet152d_962.csv")
sub_df[label_cols]=(sub_df[label_cols]*0.15+submission[label_cols]*0.85)*0.9+submission2[label_cols]*0.03


# In[42]:


sub_df.to_csv('./submission.csv', index=False)
sub_df.head()
