# PolyChord: next generation nested sampling

Talk given in December 2015 at the [MPA](https://www.mpa-garching.mpg.de/) Max Planck Institute, Garching, Germany.

## Abstract

PolyChord is a novel Bayesian inference tool for high-dimensional parameter
estimation and model comparison. It represents the latest advance in nested
sampling technology, and is the natural successor to MultiNest. The algorithm
uses John Skilling's slice sampling, utilising a slice-sampling
Markov-Chain-Monte-Carlo approach for the generation of new live points. It
has cubic scaling with dimensionality, and is capable of exploring highly
degenerate multi-modal distributions. Further, it is capable of exploiting a
hierarchy of parameter speeds present in many cosmological likelihoods.

In this talk I will give a brief account of nested sampling, and the workings
of PolyChord. I will then demonstrate its efficacy by application to
challenging toy likelihoods and real-world cosmology problems.

[[PDF](https://github.com/williamjameshandley/talks/raw/garching_2015/will_handley_garching_2015.pdf)] 
