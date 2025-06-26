I need to prepare a fresh 30m talk on our new blackjax nested sampler.
The talk should cover the background in nested sampling, how it relates to SMC, finishing with an example of us putting it on to GPUs. Leave space for slides on applications to gravitational wave parameter estimation and CMB cosmology.

Title: GPU Accelerated Nested Sampling

Abstract: Nested Sampling is a Monte Carlo method that performs parameter estimation and model comparison robustly for a variety of high dimensional and complicated distributions. It has seen widespread usage in the physical sciences, however in recent years increasingly it is viewed as part of a legacy code base, with GPU native paradigms such as neural simulation based inference coming to the fore. In this work we demonstrate that we can effectively reformulate Nested Sampling to a form that is highly amenable to modern GPU hardware, taking unique advantage of vectorization opportunities to accelerate numerical inference to state of the art levels. We provide a public implementation of this code, distributed via the blackjax statistical framework, which allows direct comparison with other well-established statistical methods such as Hamiltonian Monte Carlo and Sequential Monte Carlo, and in this contribution will explore its application to a number of inference problems such as Gravitational Wave parameter estimation and CMB cosmology.

Contributors:
Will Handley (Speaker)
David Yallup (Co-author)

Important material:
- Original nested sampling paper:
  - https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2F06-BA127
    - no source, just pdf 
- Review paper: [Nested sampling for physical scientists](https://arxiv.org/abs/2205.15570)
  - in particular the source tex and figures 
- Review paper: [Nested Sampling Methods](https://arxiv.org/abs/2101.09675)
  - in particular the source tex and figures 
- BlackJAX pre-print: [Nested Slice Sampling](https://openreview.net/forum?id=ekbkMSuPo4)
  - openreview version 
- BlackJax paper source: git@github.com:yallup/nss.git
  - submitted to NeurIPS 2025 
  - in particular the source tex and figures are in the directory paper/icml2025
- Workshop: git@github.com/handley-lab/workshop-blackjax-nested-sampling
- blackjax source code: git@github.com/handley-lab/blackjax/
  - in particular the files blackjax/ns/ and blackjax/mcmc/ss 

I have a talk framework I have been giving for a while in will_handley.tex. 
You can see a recent talk on the branch bristol_2025, but there would also be an up-to-date nested sampling review in sydney_2024, with relevant figures.

My notes:

In general, SMC is a more rigorous statistical field, whilst nested sampling is preferred in the physical sciences as a practical tool (with fewer theorems).
SMC people will often characterise this as a type of nested sampling, which is strictly true, although it is a singular type.
Our motivation for putting this in the blackjax framework was to allow a like-for-like comparison with other methods, such as HMC and SMC, and to allow us to use the same GPU code base for all of these methods.
