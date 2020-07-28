# awesome-transformers

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

This repo is a collection of AWESOME things about attention mechanism, Transformer networks, pre-trained models, including papers, code, etc. Feel free to star and fork.
Please feel free to pull requests or report issues.

# Contents
- [awsome-domain-adaptation](#awsome-domain-adaptation)
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory](#theory)
  - [Sparse Transformers](#sparse-transformers)
- [Library](#library)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Papers
## Survey
* Pre-trained Models for Natural Language Processing: A Survey [Invited Review of Science China Technological Sciences] [[__pdf__](https://arxiv.org/pdf/2003.08271.pdf)]

## Theory

## Sparse Transformers
* Deep High-Resolution Representation Learning for Visual Recognition [TPAMI 2019] [[__pdf__](https://arxiv.org/pdf/1908.07919.pdf)]  [[__code__](https://github.com/HRNet)] 
* Deep High-Resolution Representation Learning for Human Pose Estimation [CVPR 2019] [[__pdf__](https://arxiv.org/pdf/1902.09212.pdf)]  [[__code__](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)] 
Gradually add high-to-low resolution subnetworks one by one to form more stages, and connect the mutliresolution subnetworks in parallel.
* Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks [NIPS 2015] [[__pdf__](https://arxiv.org/pdf/1506.05751.pdf)]  [[__code__](https://github.com/witnessai/LAPGAN)]  
Use a cascade of convolutional networks within a Laplacian pyramid framework to generate images in a coarse-to-fine fashion.
* Feature Pyramid Networks for Object Detection [CVPR 2017] [[__pdf__](https://arxiv.org/pdf/1612.03144.pdf)] [[__code__](https://github.com/jwyang/fpn.pytorch)]  
Exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.
* U-Net: Convolutional Networks for Biomedical Image Segmentation  [MICCAI 2015] [[__pdf__](https://arxiv.org/pdf/1505.04597.pdf)] [[__code__](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)]  
The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
* Fully Convolutional Networks for Semantic Segmentation [CVPR 2015] [[__pdf__](https://arxiv.org/pdf/1411.4038.pdf)] [[__code__](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)]  
Build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning.
* Multiresolution Transformer Networks: Recurrence is Not Essential for Modeling Hierarchical Structure  [arXiv Aug 2019] [[__pdf__](https://arxiv.org/pdf/1908.10408.pdf)]
Establish connections between the dynamics in Transformer and recurrent networks to argue that several factors including gradient flow along an ensemble of multiple weakly dependent paths play a paramount role in the success of Transformer. Then leverage the dynamics to introduce Multiresolution Transformer Networks as the first architecture that exploits hierarchical structure in data via self-attention.
* MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning [arXiv Nov 2019] [[__pdf__](https://arxiv.org/pdf/1911.09483.pdf)]  [[__code__](https://github.com/lancopku/Prime)]  
Explore parallel multi-scale representation learning on sequence data, striving to capture both long-range and short-range language structures.
* Multi-scale Transformer Language Models [arXiv 1st May 2020] [[__pdf__](https://arxiv.org/pdf/2005.00581.pdf)]
Learn representations of text at multiple scales and present three different architectures that have an inductive bias to handle the hierarchical nature of language.
---------------------------------------

* REFORMER: The Efficient Transformer [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2001.04451.pdf)] [[__code__](https://github.com/google/trax/tree/master/trax/models/reformer)]  
Replace dot-product attention by one that uses locality-sensitive
hashing, changing its complexity from O(L^2) to O(LlogL), where L is the length of the sequence.
* Longformer: The Long-Document Transformer
 [arXiv Apr. 2020] [[__pdf__](https://arxiv.org/pdf/2004.05150.pdf)] [[__code__](https://github.com/allenai/longformer)]  
Propose an attention mechanism with a drop-in replacement
for the standard self-attention and combines
a local windowed attention with a task motivated global attention.
* Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing [arXiv Jun. 2020] [[__pdf__](https://arxiv.org/pdf/2006.03236.pdf)] [[__code__](https://github.com/laiguokun/Funnel-Transformer)]  
* Lite Transformer with Long-Short Range Attention [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2004.11886.pdf)] [[__code__](https://github.com/mit-han-lab/lite-transformer)]  
Present an efficient mobile NLP architecture, Lite Transformer, to
facilitate deploying mobile NLP applications on edge devices.
* Tree-structured Attention with Hierarchical Accumulation [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2002.08046.pdf)] [[__code__](https://github.com/nxphi47/tree_transformer)]  
Present an attention-based hierarchical encoding method.
* Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention [ICLR 2020] [[__pdf__](https://openreview.net/pdf?id=r1eIiCNYwS)]
Present Transformer-XH, which uses extra hop attention to enable intrinsic modeling of structured texts in a fully data-driven way.
* Linformer: Self-Attention with Linear Complexity [arXiv Jun. 2020] [[__pdf__](https://arxiv.org/pdf/2006.04768.pdf)] 
Demonstrate that the self-attention mechanism can be approximated by a low-rank matrix and propose a new self-attention mechanism, which reduces the overall self-attention complexity from O(n^2) to O(n) in both time and space.
* Adaptive Attention Span in Transformers [ACL 2019] [[__pdf__](https://arxiv.org/pdf/1905.07799.pdf)] [[__code__](https://github.com/facebookresearch/adaptive-span)] 
Propose a novel self-attention mechanism
that can learn its optimal attention span.
* BP-Transformer: Modelling Long-Range Context via Binary Partitioning
 [arXiv Nov. 2019] [[__pdf__](https://arxiv.org/pdf/1911.04070.pdf)] [[__code__](https://github.com/yzh119/BPT)]  
Adopt a fine-to-coarse attention mechanism on multi-scale spans via binary partitioning.
* Adaptively Sparse Transformers [EMNLP 2019] [[__pdf__](https://arxiv.org/pdf/1909.00015.pdf)] [[__code__](https://github.com/deep-spin/entmax)]  
Introduce the adaptively sparse Transformer,
wherein attention heads have flexible, contextdependent sparsity patterns. This sparsity is accomplished by replacing softmax with α-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight.



# Library

# Lectures and Tutorials

# Other Resources
