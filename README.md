# lsbi: linear simulation based inference

## Abstract

Simulation-based inference is undergoing a renaissance in statistics and machine learning. With several packages implementing the state-of-the-art in expressive AI [[mackelab/sbi](https://github.com/mackelab/sbi)] [[undark-lab/swyft](https://github.com/undark-lab/sywft], it is now being effectively applied to a wide range of problems in the physical sciences, biology, and beyond.

Given the rapid pace of AI/ML, there is little expectation that the implementations of the future will resemble these current first generation neural network-based approaches. This talk will present a new framework for simulation-based inference, linear simulation-based inference (lsbi), which abstracts the core principles of SBI from the specific details of machine learning, implementing a plug-and-play framework of linear and mixture models. lsbi has several use-cases:

1. It is pedagogically helpful to separate out the general principles of SBI from the specific details of neural networks (particularly for ML skeptics).
2. It is practically useful for producing expressive examples with known ground truths.
3. It is pragmatically useful, since in many cases, lsbi is competitive with neural approaches in terms of accuracy, whilst being faster and more interpretable.

An evolving code-driven PyPI/conda research package is available at: [handley-lab/lsbi](https://github.com/handley-lab/lsbi)



[[PDF](https://github.com/williamjameshandley/talks/raw/imperial_2024/will_handley_imperial_2024.pdf)] 
