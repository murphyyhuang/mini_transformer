# Mini Transformer
This project is a realization of Universal Transformer[1][3]. It is base on the `Tensor2Tensor` framwork[2] but simplifies some of the structures.

This mini Tensor2Tensor framework is succinct but powerful enough for the development of deep learning network in the laboratory level. It enables you to regard data preprocessing, model, layer as separate components and provides great flexibility to explore different network structure in some sort of plug-and-play style.

More information about Tensor2Tensor framework, please refer to: [Tensor2Tensor Documentation](https://tensorflow.github.io/tensor2tensor/)

Besides, this project is based on [TensorFlow Eager framework](https://www.tensorflow.org/guide/eager), which is an imperative programming environment that evaluates operations immediately, without building graphs. It is eaiser to debug the project but only supports single GPU for the time being.


## Project Structure
### Data Reader
Under the `pct/utils/data_reader.py`, currently do not support different data reader class.

### Models
The main part for the model structure. To add more models into this framework, refer to the following steps:

1. Create class that extends T2TModel in this example it will be a copy of existing basic fully connected network:
```
from pct.utils import registry
from pct.utils import base_model

@registry.register_model
class MyFC(base_model.BaseModel):
    pass
```

2. Implement body methods:
```
class MyFC(t2t_model.T2TModel):
  def body(self, features):
    hparams = self.hparams
    x = features["inputs"]
    shape = common_layers.shape_list(x)
    x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # Flatten input as in T2T they are all 4D vectors
    for i in range(hparams.num_hidden_layers): # create layers
      x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
      x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
      x = tf.nn.relu(x)
    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.
```

3. Declare in the `pct/models/__init__.py` , take the universal transformer for example:
```
from pct.models import universal_transformer
```

4. Modify the config file to utilize the model:
```
# under a .yml config file
model: universal_transformer
```


### Layers
Define the general substructure that could be utilized and reused by different models.

## Universal Transformer Model Structure

![](./fig/universaltransformer.gif)
GIF taken from: https://twitter.com/OriolVinyalsML/status/1017523208059260929


## Usage
```
# Training:
python3 train.py --config_dir=pct/test_data/chunking_pretrain_hparams.yml --random_seed=123 # or other path to your config file
# Decoding:
python3 decode.py --config_dir=pct/test_data/chunking_pretrain_hparams.yml
```


## Reference
[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

[2] Vaswani, Ashish, et al. "Tensor2tensor for neural machine translation." arXiv preprint arXiv:1803.07416 (2018).

[3] Dehghani, Mostafa, et al. "Universal transformers." arXiv preprint arXiv:1807.03819 (2018).
