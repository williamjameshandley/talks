# Next-Generation Model Comparison for Primordial Cosmology

## Abstract

A new paradigm is emerging in computational cosmology, shifting from legacy CPU-bound codes to unified, GPU-native ecosystems. Our work is at the forefront of this transition, aimed at one of the key challenges in the field: performing robust Bayesian model selection to test the limits of the ΛCDM model with CMB data.

We have developed a complete inference pipeline in `jax`, integrating a novel Boltzmann solver with a vectorized Nested Sampling algorithm. The construction of this pipeline, a task of significant numerical complexity, was accelerated through the use of AI agentic systems for code generation in our group. By unifying the physics simulation and statistical inference on the GPU, we can efficiently explore high-dimensional parameter spaces and accurately compute the Bayesian evidence.

I will present the first results from this framework, focusing on its application to key questions in primordial cosmology, such as the physics of inflation and the nature of dark energy. The performance gains demonstrated will be crucial for fully exploiting upcoming datasets from next-generation CMB experiments, and I look forward to exploring opportunities for joint projects.


 
[[PDF](https://github.com/williamjameshandley/talks/raw/lmu_2025/will_handley_lmu_2025.pdf)] 
