# Frontiers in nested sampling

Invited talk given at [MaxEnt 2022](https://maxent22.see.asso.fr/) in Paris, 20th July 2022

## Abstract

In this talk I will give an overview of the state of the art and future research directions in the field of nested sampling. Nested sampling is a numerical method for (a) scanning and optimising a priori unknown functions (b) generating samples from probability distributions, and is uniquely capable of (c) computing high-dimensional numerical integrals and partition functions using probabilistic integration. Implementations of John Skilling's nested sampling meta-algorithm include MultiNest, PolyChord, NeuralNest, DNest, UltraNest & dynesty. Whilst most of this headline literature implements variations on the mechanism for sampling from a likelihood-constrained prior, in the past two decades there have also been several advances in the content of the meta-algorithm and surrounding ecosystem of theory and numerical techniques. These include; dynamic nested sampling, diffusive nested sampling, quantifying parameter estimation errors, post-processing cross-checks using nestcheck and anesthetic,  computing $p$-values in frequentist analyses and using normalising flows to specify priors from previous posterior samples. In my overview of future frontiers, I will discuss reversible nested sampling, using insertion indices or likelihood values rather than likelihood ordering for improved accuracy, applying nested sampling in the field of likelihood-free and simulation-based inference, estimating the end of nested sampling, importance nested sampling, transdimensional nested sampling & multi-objective nested sampling. Interest in collaborations for developing further any of the above will be warmly welcomed.


[[PDF](https://github.com/williamjameshandley/talks/raw/paris_maxent_2022/will_handley_paris_maxent_2022.pdf)] 
