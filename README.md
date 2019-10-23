# Mini Transformer
This project is a realization of Transformer[1]. It is base on the Tensor2Tensor framwork[2] but simplifies some of the structures. This mini Tensor2Tensor framework is succinct but powerful enough for the development of deep learning network in the laboratory level. It enables you to regard data preprocessing, model, layer as separate components and provides great flexibility to explore different network structure in some sort of plug-and-play style.  
  
  Besides, this project is based on [TensorFlow Eager framework](https://www.tensorflow.org/guide/eager), which is an imperative programming environment that evaluates operations immediately, without building graphs. It is eaiser to debug the project but only supports single GPU for the time being.


# Reference
[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

[2] Vaswani, Ashish, et al. "Tensor2tensor for neural machine translation." arXiv preprint arXiv:1803.07416 (2018).
