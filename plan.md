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
- GW analysis paper:
  - git@github.com:yallup/blackjaxgw.git
    - in particular paper/ 
- CMB/shear paper:
  - https://git@git.overleaf.com/683ee8c279a2889270baec94
I have a talk framework I have been giving for a while in will_handley.tex. 
You can see a recent talk on the branch bristol_2025, but there would also be an up-to-date nested sampling review in sydney_2024, with relevant figures.



My notes:

In general, SMC is a more rigorous statistical field, whilst nested sampling is preferred in the physical sciences as a practical tool (with fewer theorems).
SMC people will often characterise this as a type of nested sampling, which is strictly true, although it is a singular type.
Our motivation for putting this in the blackjax framework was to allow a like-for-like comparison with other methods, such as HMC and SMC, and to allow us to use the same GPU code base for all of these methods.


Workshop:

Workshop theme
This workshop focuses on leveraging modern machine learning to accelerate statistical inference, experimental design, and scientific discovery. It features research talks, tutorials, and a poster session, with ample time reserved for informal discussions. The tutorials cover important background, e.g. on experimental design and diffusion models.


The overall aim is that, by the end of the workshop, attendees will be well-acquainted with both the foundations and the latest trends and advances in the field.

Speakers include:

Eviatar Bach (University of Reading)
Andrew Gelman (Columbia University)
Daolang Huang (Aalto University)
Desi Ivanova (University of Oxford)
Jacopo Iollo (INRIA)
Jeremias Knoblauch (University College London)
Louis Sharrock (Lancaster University)
Masha Naslidnyk (University College London)
Nikolay Malkin (University of Edinburgh)
Youssef Marzouk (Massachusetts Institute of Technology)
Call for posters 
The organisers would like to encourage those applying to submit a poster to present at the workshop. If you wish to do so, please indicate this on the relevant section of the application form and provide a title and abstract. Kindly note this is for physical participants only, we are unable to accommodate virtual posters. 

Poster acceptance will be determined by the workshop organisers. The deadline for submissions is the 16 February 2025. Notifications of acceptance will be sent shortly thereafter. 

Applications for this workshop are now closed. If you are interested in attending, please contact workshops@newton.ac.uk to enquire about availability.
 

Fees
Accommodation
Meals
The Institute kindly requests that any papers published as a result of this programme’s activities are credited as such. Please acknowledge the support of the Institute in your paper using the following text:

The author(s) would like to thank the Isaac Newton Institute for Mathematical Sciences, Cambridge, for support and hospitality during the programme Accelerating statistical inference and experimental design with machine learning, where work on this paper was undertaken. This work was supported by EPSRC grant EP/Z000580/1.

 

Timetable
Monday 23rd June 2025
09:30 to 10:00	
Registration

No Room Required	
10:00 to 10:05	
Ulrike Tillmann Isaac Newton Institute; University of Oxford
Director's Briefing

Room 1	
10:05 to 10:15	
Michael Gutmann University of Edinburgh
Organiser's welcome

Room 1	
10:15 to 11:15	
Jeremias Knoblauch University College London
Chair: Michael Gutmann
Bayesian Inference Tutorial

Room 1	
11:15 to 11:45	
Morning Coffee

No Room Required	
11:45 to 12:45	
Desi Ivanova University of Oxford
Chair: Michael Gutmann
Bayesian Experimental Design: Principles and Computation

Room 1	
13:00 to 14:00	
Lunch at Churchill College

External	
14:00 to 15:00	
Nikolay Malkin University of Edinburgh
Chair: Michael Gutmann
Diffusion model tutorial

Room 1	
15:00 to 16:00	
Afternoon Tea

No Room Required	
16:00 to 17:00	
Christopher Bishop Microsoft (UK)
Chair: Michael Gutmann
Accelerating Scientific Discovery with Deep Learning

Room 1	
17:00 to 18:00	
Welcome Reception

No Room Required	
Tuesday 24th June 2025
09:15 to 10:15	
Desi Ivanova University of Oxford
Chair: Florence d'Alché-Buc
Secure active causal dataset acquisition

Room 1	
10:15 to 10:45	
Sahel Mohammad Iqbal Aalto University
Chair: Florence d'Alché-Buc
Amortized Bayesian experimental design with sequential Monte Carlo

Room 1	
10:45 to 10:50	
Group Photo

No Room Required	
10:50 to 11:15	
Morning Coffee

No Room Required	
11:15 to 12:15	
Daolang Huang Aalto University; University of Helsinki
Chair: Rich Wilkinson
Accelerating Bayesian Inference and Data Acquisition via Amortization

Room 1	
12:15 to 12:45	
Poster spotlights

Room 1	
13:00 to 14:00	
Lunch at Churchill College

External	
14:00 to 15:00	
Youssef Marzouk Massachusetts Institute of Technology
Chair: Rich Wilkinson
Mean-shift interacting particle systems for optimal quantization and beyond

Room 1	
15:00 to 15:30	
Afternoon Tea

No Room Required	
15:30 to 17:00	
Poster Session

Discussion Room	
Wednesday 25th June 2025
09:15 to 10:15	
Jacopo Iollo INRIA
Chair: Florence d'Alché-Buc
Diffusion-based Bayesian Experimental Design

Room 1	
10:15 to 10:45	
Yufei Wu KU Leuven
Chair: Florence d'Alché-Buc
Testing and improving the robustness of amortized Bayesian inference for cognitive models

Room 1	
10:45 to 11:15	
Morning Coffee

No Room Required	
11:15 to 12:15	
Claudia Strauch Universität Heidelberg
Chair: Rich Wilkinson
On the statistical analysis of denoising diffusion models

Room 1	
12:15 to 12:45	
Poster spotlights

Room 1	
13:00 to 14:00	
Lunch at Churchill College

External	
14:00 to 15:00	
Nikolay Malkin University of Edinburgh
Chair: Rich Wilkinson
Diffusion modelling for amortised inference

Room 1	
15:00 to 15:30	
Afternoon Tea

No Room Required	
15:30 to 17:00	
Poster Session

Discussion Room	
Thursday 26th June 2025
09:15 to 10:15	
Jeremias Knoblauch University College London
Chair: David Ginsbourger
Title TBC

Room 1	
10:15 to 10:45	
Pietro Vischia Universidad de Oviedo
Chair: David Ginsbourger
Machine Learning Optimized Design of Experiments at the frontiers of computation: methods and new perspectives

Room 1	
10:45 to 11:15	
Morning Coffee

No Room Required	
11:15 to 12:15	
Masha Naslidnyk University College London
Chair: David Ginsbourger
On Bayesian Quadrature Estimators

Room 1	
12:15 to 12:45	
Duc-Lam Duong LUT University
Chair: David Ginsbourger
A scalable diffusion posterior sampling in nonparametric Bayesian linear inverse problems

Room 1	
13:00 to 14:00	
Lunch at Churchill College

External	
14:00 to 15:00	
Louis Sharrock University College London; University of Bristol
Chair: David Ginsbourger
Learning-Rate-Free Optimisation on the Space of Probability Measures

Room 1	
15:00 to 15:30	
Afternoon Tea

No Room Required	
15:30 to 16:30	
Andrew Gelman Columbia University
Chair: David Ginsbourger
Bayesian Workflow

Room 1	
19:30 to 21:30	
Formal Dinner at Sidney Sussex College

External	
Friday 27th June 2025
10:15 to 11:15	
Eviatar Bach University of Reading
Chair: Michael Gutmann
Learning probabilistic filters for data assimilation

Room 1	
11:15 to 11:45	
Morning Coffee

No Room Required	
11:45 to 12:15	
Will Handley University of Cambridge
Chair: Michael Gutmann
GPU Accelerated Nested Sampling

Room 1	
12:15 to 12:45	
Dasha Semochkina University of Southampton
Chair: Michael Gutmann
AI-Enhanced Bayesian Experimental Design for Reaction Kinetics: Bridging Statistics and Chemistry

Room 1	
12:45 to 14:00	
Lunch at Churchill College

External	
14:00 to 15:00	
Informal discussion

No Room Required	
15:00 to 15:30	
Afternoon Tea

No Room Required	
