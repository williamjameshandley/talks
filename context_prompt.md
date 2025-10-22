Project Path: context

Source Tree:

```txt
context
├── bristol_2025.tex
├── caltech_2025.tex
├── ini_2025.tex
├── kicc_ai_tools_2025.tex
├── sed_2025.tex
└── suhail_notes.md

```

`bristol_2025.tex`:

```tex
\documentclass[aspectratio=169]{beamer}
\usepackage{will_handley_beamer}
\usepackage{title_page}
\usetikzlibrary{positioning, calc, arrows.meta, shapes}

% Commands
% --------
% - \arxiv{arxiv number}
% - \arxiv{<number>}            arxiv.org/abs/<number>
% - \oldarxiv{<arxiv number>}   arxiv.org/<number>
% - \doi{<doi>}                 doi.org/<doi>
% - \xkcd{<number>}             xkcd.com/<number>
% - \email{<email>}             <<email>>
% - \tthref{<website>}          <website>
% - \av[dist]{<quantity>}       <quantity>_{dist}
% - \student{<name>}{<detail>}{<photo>}

% Talk details
% ------------
\title{GPU-native nested sampling in BlackJAX}
\subtitle{For simulation-based inference at scale}
\date{29\textsuperscript{th} May 2025}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}
    \frametitle{What is Nested Sampling?}
    \begin{itemize}
        \item Nested sampling is a radical, multi-purpose numerical tool.
        \item Given a (scalar) function $f$ with a vector of parameters $\theta$, it can be used for:
    \end{itemize}
    \vspace{-10pt}
    \begin{columns}[t]
        \column{0.3\textwidth}
        \begin{block}{Optimisation}
            \[\theta_\text{max} = \max_\theta{f(\theta)}\]
        \end{block}
        \column{0.3\textwidth}
        \begin{block}{Exploration}
            \vspace{-10pt}
            \[\text{draw/sample}\quad \theta\sim f\]
            \vspace{-15pt}
        \end{block}
        \column{0.3\textwidth}
        \begin{block}{Integration}
            \[\int f(\theta) dV \]
        \end{block}
    \end{columns}
    \begin{columns}[t]
        \column{0.33\textwidth}
        \centerline{\includegraphics[width=0.8\textwidth,page=13]{figures/himmelblau}}
        \column{0.33\textwidth}
        \centerline{\includegraphics[width=0.8\textwidth,page=15]{figures/himmelblau}}
        \column{0.33\textwidth}
        \centerline{\includegraphics[width=0.8\textwidth,page=14]{figures/himmelblau}}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Why nested sampling for SBI?}
    \begin{columns}
        \column{0.4\textwidth}
        \begin{itemize}
            \item Other than NPE, all SBI methods (NLE, NRE, NJE etc) need a sampler
            \item Most existing samplers are limited:
                \begin{itemize}
                    \item \textbf{Legacy implementations}: MultiNest, PolyChord (Fortran)
                    \item \textbf{Slow Python codes}: dynesty, ultranest, nautilus
                \end{itemize}
            \item BlackJAX solution:
                \begin{itemize}
                    \item GPU-native implementation
                    \item Open source \& community-driven 
                    \item Dissociated from original authors
                \end{itemize}
        \end{itemize}
        \column{0.6\textwidth}
        \centerline{%
            \begin{tikzpicture}[
                    every node/.style={align=center, font=\bfseries},
                    box/.style={draw, rounded corners, minimum width=2cm, minimum height=1cm, fill=C2!30},
                    trapezoid/.style={draw, trapezium, minimum width=2cm, minimum height=1cm, trapezium stretches body, fill=C1!50},
                    ellipsoid/.style={draw, ellipse, minimum width=2cm, minimum height=1cm, fill=C0!50},
                    cloudoid/.style={draw, cloud, cloud puffs=10, aspect=2, inner ysep=1em, fill=C3!30},
                    arrow/.style={thick, -{Stealth[length=3mm, width=2mm]}}
                ]
                \node[ellipsoid] (theory) {\Large Theory};
                \node[trapezoid] (experiment) at (6,0) {\Large Experiment};
                \node[box] (forward) at (3,1.5) {Forward Model};
                \node[cloudoid] (inference) at (3,-1.5) {Inference};
                \draw[arrow] (theory) to[bend left=25]  (forward);
                \draw[arrow] (forward) to[bend left=25] (experiment);
                \draw[arrow] (experiment) to[bend left=25] (inference);
                \draw[arrow] (inference) to[bend left=25] (theory);
            \end{tikzpicture}
        }
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The GPU imperative}
    \begin{columns}
        \column{0.6\textwidth}
        \begin{itemize}
            \item The future is GPU, whether we like it or not:
                \begin{itemize}
                    \item All future HPC heavily weighted toward GPUs
                    \item Driven by machine learning adoption
                \end{itemize}
            \item JAX does two \emph{separate} things:
                \begin{enumerate}
                    \item \textbf{Automatic differentiation}
                    \item \textbf{Just-in-time compilation} for GPUs
                \end{enumerate}
            \item People often conflate these - they are separate and glorious!
            \item Alternative tools (harmonic, floz) get strength from GPU, not sampling method
        \end{itemize}
        \column{0.4\textwidth}
        \begin{block}{GPU ecosystem growth}
            \begin{description}
                \item[CMB] \texttt{cosmopower}, \texttt{candl}
                \item[SNe] \texttt{BayesSN}
                \item[GW] \texttt{ripple}, \texttt{jim}
                \item[EP] \texttt{ExoJAX}
            \end{description}
        \end{block}
        \github{JAXtronomy}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{BlackJAX nested sampling}
    \student{david_yallup}{David Yallup}{PDRA}
    \begin{columns}
        \column{0.75\textwidth}
        \begin{itemize}
            \item Very recent development (December 2024!)
            \item GPU-native nested slice sampler
            \item Modern alternative to MultiNest/PolyChord
            \item Slice sampling with adaptive live point allocation
            \item Compatible with existing BlackJAX ecosystem
        \end{itemize}
        \begin{block}{Learn more}
            \begin{itemize}
                \item Theory: \texttt{handley-lab.co.uk/nested-sampling-book}
                \item Implementation: David Yallup's PDRA work
                \item Workshop: Hands-on tutorial today
            \end{itemize}
        \end{block}
        \begin{block}{Installation}
            \texttt{pip install git+https://github.com/handley-lab/blackjax}
        \end{block}
        \column{0.25\textwidth}
        \includegraphics[width=\textwidth]{people/david_yallup.jpg}
        \vspace{10pt}
        \centerline{\includegraphics[width=0.8\textwidth]{figures/jax_logo.png}}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{AI-enabled research workflows}
    \begin{columns}
        \column{0.57\textwidth}
        \begin{itemize}
            \item Modern scientific computing benefits from AI integration:
                \begin{itemize}
                    \item GPU-accelerated inference
                    \item Neural networks in SBI methods
                    \item Automatic differentiation for sampling
                \end{itemize}
            \item But the real coming disruption is from using AI to do day-to-day scientific tasks.
                \begin{itemize}
                    \item I wrote this talk with AI assistance [claude code]. This morning. 
                    \item I have written and been awarded several grants that were majority written by AI.
                    \item My most recent papers are drafted with AI.
                \end{itemize}
        \end{itemize}
        \column{0.43\textwidth}
        \includegraphics<1>[width=\textwidth]{figures/2025-05-29_10-34-55.png}%
        \includegraphics<2>[width=\textwidth]{figures/2025-05-29_10-37-37.png}%
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Workshop goals}
    \begin{columns}
        \column{0.5\textwidth}
        \begin{itemize}
            \item Today we'll explore:
                \begin{enumerate}
                    \item Running nested sampling with BlackJAX
                    \item Visualization with Anesthetic
                    \item Performance comparison: nested sampling vs AIES
                    \item Integration with your JAX workflows
                \end{enumerate}
            \item Hands-on notebook environment
            \item Google Colab compatible
            \item Build on Viraj's JAX/SciML workshop
        \end{itemize}
        \column{0.5\textwidth}
        \begin{block}{Links}
            \begin{itemize}
                \item BlackJAX: \github{handley-lab/blackjax}
                \item Anesthetic: \tthref{anesthetic.readthedocs.io}
                \item Workshop materials: \github{handley-lab}
            \end{itemize}
        \end{block}
        \vspace{10pt}
        \begin{alertblock}{Performance promise}
            Compare BlackJAX nested sampling performance with traditional tools and see the GPU advantage firsthand!
        \end{alertblock}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Conclusions}
    \framesubtitle{\github{handley-lab}}
    \tikz[overlay,remember picture]
        \node[anchor=north east] (A) at ($(current page.north east)+(0,0)$) {
        \includegraphics[width=0.09\textheight]{people/adam_ormondroyd.jpg}%
        \includegraphics[width=0.09\textheight]{people/charlotte_priestley.jpg}%
        \includegraphics[width=0.09\textheight]{people/david_yallup.jpg}%
        \includegraphics[width=0.09\textheight]{people/dily_ong.jpg}%
        \includegraphics[width=0.09\textheight]{people/harry_bevins.jpg}%
        \includegraphics[width=0.09\textheight]{people/harvey_williams.jpg}%
        \includegraphics[width=0.09\textheight]{people/krish_nanavati.jpg}%
        \includegraphics[width=0.09\textheight]{people/metha_prathaban.jpg}%
        \includegraphics[width=0.09\textheight]{people/ming_yang.jpg}%
        \includegraphics[width=0.09\textheight]{people/namu_kroupa.jpg}%
        \includegraphics[width=0.09\textheight]{people/sam_leeney.jpg}%
        \includegraphics[width=0.09\textheight]{people/sinah_legner.jpg}%
        \includegraphics[width=0.09\textheight]{people/toby_lovick.jpg}%
        \includegraphics[width=0.09\textheight]{people/wei-ning_deng.jpg}%
        \includegraphics[width=0.09\textheight]{people/will_handley.jpg}%
        \includegraphics[width=0.09\textheight]{people/will_templeton.jpg}%
    };
    \vspace{-0.1\textheight}
    \begin{itemize}
        \item \textbf{Nested sampling is essential} for most SBI methods (except NPE)
        \item \textbf{BlackJAX} provides GPU-native, community-driven implementation
        \item \textbf{JAX's dual power}: autodiff + JIT compilation for unprecedented performance
        \item \textbf{The future is GPU-accelerated} scientific computing with AI integration
    \end{itemize}
\end{frame}

\appendix

\begin{frame}
    \frametitle{Integration in Physics}
    \begin{itemize}
        \item Integration is a fundamental concept in physics, statistics and data science:
    \end{itemize}
    \begin{columns}
        \column{0.3\textwidth}
        \begin{block}{Partition functions}
            \vspace{-11pt}
            \[ Z(\beta) = \int e^{-\beta H(q,p)} dq dp \]
        \end{block}
        \column{0.3\textwidth}
        \begin{block}{Path integrals}
            \[ \Psi = \int e^{i S} \mathcal{D}x \]
        \end{block}
        \column{0.3\textwidth}
        \begin{block}{Bayesian marginals}
            \vspace{-11pt}
            \[ \mathcal{Z}(D) = \int \mathcal{L}(D|\theta) \pi(\theta) d\theta \]
        \end{block}
    \end{columns}
    \begin{columns}
        \column{0.6\textwidth}
        \begin{itemize}
            \item Need numerical tools if analytic solution unavailable.
            \item High-dimensional numerical integration is hard.
            \item Riemannian strategy estimates volumes geometrically:
                \[ \int f(x) d^nx \approx \sum_i f(x_i) \Delta V_i \sim \mathcal{O}(e^n) \]
            \item Curse of dimensionality $\Rightarrow$ exponential scaling.
        \end{itemize}
        \column{0.4\textwidth}
        \includegraphics[width=\textwidth]{figures/integration}
    \end{columns}
\end{frame}

\begin{frame}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{\textbf{MCMC}}
            \only<16>{
                \begin{itemize}
                    \item Single ``walker''
                    \item Explores posterior
                    \item Fast, if proposal matrix is tuned
                    \item Parameter estimation, suspiciousness calculation
                    \item Channel capacity optimised for generating posterior samples
                \end{itemize}
            }
        \end{block}
            \includegraphics<1>[width=\textwidth,page=16]{figures/himmelblau}%
            \includegraphics<2>[width=\textwidth,page=17]{figures/himmelblau}%
            \includegraphics<3>[width=\textwidth,page=18]{figures/himmelblau}%
            \includegraphics<4>[width=\textwidth,page=19]{figures/himmelblau}%
            \includegraphics<5>[width=\textwidth,page=20]{figures/himmelblau}%
            \includegraphics<6-15>[width=\textwidth,page=21]{figures/himmelblau}%
        \centerline{\includegraphics<16>[width=0.5\textwidth,page=19]{figures/himmelblau}}
        \column{0.48\textwidth}
        \begin{block}<7->{\textbf{Nested sampling}}
            \only<16>{
                \begin{itemize}
                    \item Ensemble of ``live points''
                    \item Scans from prior to peak of likelihood
                    \item Slower, no tuning required
                    \item Parameter estimation, model comparison, tension quantification
                    \item Channel capacity optimised for computing partition function
                \end{itemize}
            }
        \end{block}
            \includegraphics<7|handout:0>[width=\textwidth,page=1]{figures/himmelblau}%
            \includegraphics<8|handout:0>[width=\textwidth,page=2]{figures/himmelblau}%
            \includegraphics<9|handout:0>[width=\textwidth,page=3]{figures/himmelblau}%
            \includegraphics<10          >[width=\textwidth,page=4]{figures/himmelblau}%
            \includegraphics<11|handout:0>[width=\textwidth,page=5]{figures/himmelblau}%
            \includegraphics<12|handout:0>[width=\textwidth,page=6]{figures/himmelblau}%
            \includegraphics<13|handout:0>[width=\textwidth,page=7]{figures/himmelblau}%
            \includegraphics<14|handout:0>[width=\textwidth,page=8]{figures/himmelblau}%
            \includegraphics<15|handout:0>[width=\textwidth,page=15]{figures/himmelblau}%
        \centerline{\includegraphics<16>[width=0.5\textwidth,page=4]{figures/himmelblau}} 
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The nested sampling meta-algorithm: live points}
    \begin{columns}
        \column{0.5\textwidth}
        \begin{itemize}
            \item Start with $n$ random samples over the space.
            \item Delete outermost sample, and replace with a new random one at higher integrand value.
            \item The ``live points'' steadily contract around the peak(s) of the function.
            \item We can use this evolution to estimate volume \emph{probabilistically}.
            \item At each iteration, the contours contract by $\sim\frac{1}{n}\only<5->{\pm \frac{1}{n}}$ of their volume.
            \item This is an exponential contraction, so
                \[  \int f(x) dV \approx \sum_i f(x_i) \Delta V_i, \quad V_i = V_0 e^{-\only<5->{(}i\only<5->{\pm\sqrt{i})}/n} \]
        \end{itemize}
        \column{0.5\textwidth}
        \includegraphics<1|handout:0>[width=\textwidth,page=1]{figures/himmelblau}%
        \includegraphics<2|handout:0>[width=\textwidth,page=2]{figures/himmelblau}%
        \includegraphics<3|handout:0>[width=\textwidth,page=3]{figures/himmelblau}%
        \includegraphics<4-         >[width=\textwidth,page=4]{figures/himmelblau}%
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Types of nested sampler}
    \begin{itemize}
        \item Broadly, most nested samplers can be split into how they create new live points.
        \item i.e. how they sample from the hard likelihood constraint $\{\theta\sim \pi : \mathcal{L}(\theta)>\mathcal{L}_* \}$.
    \end{itemize}
    \vspace{-10pt}
    \begin{columns}[t]
        \column{0.48\textwidth}
        \begin{block}{Rejection samplers}
            \begin{itemize}
                \item e.g. \texttt{MultiNest}, \texttt{UltraNest}.
                \item Constructs bounding region and draws many invalid points until $\mathcal{L}(\theta)>\mathcal{L}_*$.
                \item Efficient in low dimensions, exponentially inefficient $\sim\mathcal{O}(e^{d/d_0})$ in high $d>d_0\sim10$.
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Chain-based samplers}
            \begin{itemize}
                \item e.g. \texttt{PolyChord}, \texttt{ProxNest}.
                \item Run Markov chain starting at a live point, generating many valid (correlated) points.
                \item Linear $\sim\mathcal{O}(d)$ penalty in decorrelating new live point from the original seed point.
            \end{itemize}
        \end{block}
    \end{columns}
    \vspace{5pt}
    \begin{itemize}
        \item Nested samplers usually come with:
            \begin{itemize}
                \item \emph{resolution} parameter $n_\mathrm{live}$ (which improve results as $\sim\mathcal{O}(n_\mathrm{live}^{-1/2})$).
                \item set of \emph{reliability} parameters, which don't improve results if set arbitrarily high, but introduce systematic errors if set too low.
                \item e.g. \texttt{Multinest} efficiency \texttt{eff} or \texttt{PolyChord} chain length $n_\mathrm{repeats}$.
            \end{itemize}
    \end{itemize}
\end{frame}

\end{document}

```

`caltech_2025.tex`:

```tex
\documentclass[aspectratio=169]{beamer}
\usepackage{will_handley_beamer}
\usepackage{title_page}
\usepackage[normalem]{ulem}

% Commands
% --------
% - \arxiv{arxiv number}
% - \arxiv{<number>}            arxiv.org/abs/<number>
% - \oldarxiv{<arxiv number>}   arxiv.org/<number>
% - \doi{<doi>}                 doi.org/<doi>
% - \xkcd{<number>}             xkcd.com/<number>
% - \email{<email>}             <<email>>
% - \tthref{<website>}          <website>
% - \av[dist]{<quantity>}       <quantity>_{dist}
% - \student{<name>}{<detail>}{<photo>}

% Talk details
% ------------
\title{GPU Accelerated Nested Sampling for 21-cm Cosmology}
\date{21\textsuperscript{st} January 2025}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}
    \frametitle{Bayesian Inference Challenges in 21cm Cosmology}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{Computational Challenges}
            \begin{itemize}
                \item High-dimensional parameter spaces (20-100+ parameters)
                \item Complex, multimodal posteriors
                \item Expensive likelihood evaluations
                \item Months of computation time
                \item Model comparison bottlenecks
            \end{itemize}
        \end{block}
        \column{0.49\textwidth}
        \begin{block}{Physical Challenges}
            \begin{itemize}
                \item Foreground contamination 10$^5$× stronger than signal
                \item Instrumental systematics
                \item Ionospheric effects
                \item Degeneracies between astrophysics and cosmology
                \item Need for robust uncertainty quantification
            \end{itemize}
        \end{block}
    \end{columns}
    \vspace{10pt}
    \begin{center}
        \Large
        \textbf{These challenges demand new computational approaches}
    \end{center}
\end{frame}

\begin{frame}
    \frametitle{GPU Computing: Beyond Machine Learning}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{GPU vs CPU for Scientific Computing}
            \begin{itemize}
                \item \textbf{CPU}: Few powerful cores (10s), complex control.
                \item \textbf{GPU}: Many simple cores (1000s), simple control.
                \item \textbf{Memory bandwidth}: GPU ~10× faster than CPU.
                \item \textbf{Perfect for}: Independent parallel tasks.
                \item \textbf{Scientific algorithms}: MCMC chains, likelihood evaluations, simulations.
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{HPC Landscape Evolution}
            \begin{itemize}
                \item HPC transitioning to GPU-based architectures.
                \item ML adoption accelerating hardware development.
                \item Legacy CPU codes require modernization.
            \end{itemize}
        \end{block}
        \begin{block}{Key Point}
            \begin{center}
                \textbf{GPU $\neq$ Machine Learning}\\
                GPUs accelerate any parallel algorithm
            \end{center}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Modern Languages: Two Independent Capabilities}
    \begin{center}
        \textbf{Differentiable programming languages}: JAX, PyTorch, TensorFlow, Julia, Stan, \ldots
    \end{center}
    \vspace{-5pt}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{Capability 1: Free Gradients}
            \begin{itemize}
                \item \textbf{Automatic differentiation}: $\nabla_\theta \log \mathcal{L}(\theta)$.
                \item Enables gradient-based MCMC (HMC, NUTS).
                \item Essential for modern optimization.
            \end{itemize}
        \end{block}
        \begin{block}{Traditional Physics Benefits}
            \begin{itemize}
                \item \textbf{Nested sampling}: Massive parallelization.
                \item \textbf{21cm signals}: Vectorized across frequency/time/angle.
                \item \textbf{N-body sims}: GPU acceleration.
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Capability 2: GPU Parallelization}
            \begin{itemize}
                \item \textbf{Vectorization across ensembles}.
                \item Run 1000s of parallel chains/particles.
                \item Evaluate likelihoods simultaneously.
            \end{itemize}
        \end{block}
        \begin{block}{Key Insight: Often Confused}
            \begin{center}
                \textbf{These are completely independent.}\\
                \textbf{People mistake one for the other.}\\
                You can use gradients on CPU.\\
                You can GPU parallelize without gradients.\\
                \textbf{They serve different purposes.}
            \end{center}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{BlackJAX: GPU Native Sampling}
    \framesubtitle<1-10>{Gradient descent: inference at speed}
    \framesubtitle<11-19>{Metropolis-Hastings: error bars}
    \framesubtitle<20-28>{emcee: adaptive ensemble algorithms}
    \framesubtitle<29-36>{Nested sampling: model comparison}
    \framesubtitle<37-44>{Hamiltonian Monte Carlo: inference with gradients}
    \student{david_yallup}{David Yallup}{Postdoc}
    \begin{columns}
        \column{0.48\textwidth}
        \vspace{-10pt}
        \begin{itemize}
            \item Sampling traditionally CPU-bound.
            \item Different algorithms, same GPU challenge.
            \item Need unified GPU-native framework.
            \item From optimization to model comparison.
        \end{itemize}
        \vspace{10pt}
        \begin{itemize}
            \item BlackJAX: Full JAX ecosystem.
            \item All algorithms GPU-accelerated.
            \item Gradient descent through nested sampling.
            \item Unified interface, maximum performance.
        \end{itemize}
        \vspace{5pt}
        \begin{itemize}
            \item Framework: more like \texttt{numpy} or \texttt{scipy} than \texttt{cobaya} or \texttt{cosmosis}.
        \end{itemize}
        \column{0.48\textwidth}
        \vspace{10pt}
        \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_gradient_ascent}%
        \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_gradient_ascent}%
        \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_gradient_ascent}%
        \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_gradient_ascent}%
        \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_gradient_ascent}%
        \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_gradient_ascent}%
        \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_gradient_ascent}%
        \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_gradient_ascent}%
        \includegraphics<9>[width=\textwidth,page=9]{figures/himmelblau_gradient_ascent}%
        \includegraphics<10>[width=\textwidth,page=10]{figures/himmelblau_gradient_ascent}%
        \includegraphics<11>[width=\textwidth,page=1]{figures/himmelblau_mcmc}%
        \includegraphics<12>[width=\textwidth,page=2]{figures/himmelblau_mcmc}%
        \includegraphics<13>[width=\textwidth,page=3]{figures/himmelblau_mcmc}%
        \includegraphics<14>[width=\textwidth,page=4]{figures/himmelblau_mcmc}%
        \includegraphics<15>[width=\textwidth,page=5]{figures/himmelblau_mcmc}%
        \includegraphics<16>[width=\textwidth,page=6]{figures/himmelblau_mcmc}%
        \includegraphics<17>[width=\textwidth,page=7]{figures/himmelblau_mcmc}%
        \includegraphics<18>[width=\textwidth,page=8]{figures/himmelblau_mcmc}%
        \includegraphics<19>[width=\textwidth,page=9]{figures/himmelblau_mcmc}%
        \includegraphics<20>[width=\textwidth,page=1]{figures/himmelblau_emcee}%
        \includegraphics<21>[width=\textwidth,page=2]{figures/himmelblau_emcee}%
        \includegraphics<22>[width=\textwidth,page=3]{figures/himmelblau_emcee}%
        \includegraphics<23>[width=\textwidth,page=4]{figures/himmelblau_emcee}%
        \includegraphics<24>[width=\textwidth,page=5]{figures/himmelblau_emcee}%
        \includegraphics<25>[width=\textwidth,page=6]{figures/himmelblau_emcee}%
        \includegraphics<26>[width=\textwidth,page=7]{figures/himmelblau_emcee}%
        \includegraphics<27>[width=\textwidth,page=8]{figures/himmelblau_emcee}%
        \includegraphics<28>[width=\textwidth,page=9]{figures/himmelblau_emcee}%
        \includegraphics<29>[width=\textwidth,page=1]{figures/himmelblau_ns}%
        \includegraphics<30>[width=\textwidth,page=2]{figures/himmelblau_ns}%
        \includegraphics<31>[width=\textwidth,page=3]{figures/himmelblau_ns}%
        \includegraphics<32>[width=\textwidth,page=4]{figures/himmelblau_ns}%
        \includegraphics<33>[width=\textwidth,page=5]{figures/himmelblau_ns}%
        \includegraphics<34>[width=\textwidth,page=6]{figures/himmelblau_ns}%
        \includegraphics<35>[width=\textwidth,page=7]{figures/himmelblau_ns}%
        \includegraphics<36>[width=\textwidth,page=8]{figures/himmelblau_ns}%
        \includegraphics<37>[width=\textwidth,page=1]{figures/himmelblau_hmc}%
        \includegraphics<38>[width=\textwidth,page=2]{figures/himmelblau_hmc}%
        \includegraphics<39>[width=\textwidth,page=3]{figures/himmelblau_hmc}%
        \includegraphics<40>[width=\textwidth,page=4]{figures/himmelblau_hmc}%
        \includegraphics<41>[width=\textwidth,page=5]{figures/himmelblau_hmc}%
        \includegraphics<42>[width=\textwidth,page=6]{figures/himmelblau_hmc}%
        \includegraphics<43>[width=\textwidth,page=7]{figures/himmelblau_hmc}%
        \includegraphics<44>[width=\textwidth,page=8]{figures/himmelblau_hmc}%
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Recent GPU-Accelerated Applications}
    \framesubtitle{Case study 1/4: CMB and Cosmic Shear \arxiv{2509.13307}}
    \student{toby_lovick}{Toby Lovick}{PhD}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{itemize}
            \item \textbf{CMB (6 params)}: 300× speedup vs CPU PolyChord
            \item \textbf{Cosmic Shear (37 params)}: Days vs months
            \item \textbf{Method}: JAX neural emulators + GPU NS
            \item \textbf{Evidence}: Direct calculation with error bars
            \item \textbf{Models}: $\Lambda$CDM vs $w_0w_a$ comparison
            \item \textbf{Impact}: NS competitive with MCMC+evidence methods
        \end{itemize}
        \column{0.48\textwidth}
        \includegraphics<1>[width=\textwidth]{figures/cmbscaling.pdf}%
        \vspace{5pt}
        \includegraphics<2>[width=\textwidth]{figures/jaxSHEARfull.png}
        %\vspace{10pt}
        %\begin{block}{Key Insight: Classical > ML Gold Rush}
        %    \begin{itemize}
        %        \item \textbf{No machine learning}: Pure nested sampling
        %        \item \textbf{Statistical guarantees}: Proper uncertainties + evidence
        %        \item \textbf{Factor in training costs}: ML often slower end-to-end
        %        \item \textbf{Transparent algorithms}: No black boxes
        %    \end{itemize}
        %\end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Recent GPU-Accelerated Applications}
    \framesubtitle{Case study 2/4: Bayesian Anomaly Detection for Type Ia Supernovae \arxiv{2509.13394}}
    \student{sam_leeney}{Sam Leeney}{PhD}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{itemize}
            \item \textbf{Problem}: Manual photometric rejection not scalable for LSST
            \item \textbf{Solution}: Bayesian anomaly detection integrated into SALT3 fitting
            \item \textbf{Method}: Model contamination probability per measurement
            \item \textbf{Result}: Automatic outlier/corrupted band rejection
            \item \textbf{Finding}: Contaminants systematically brighter/bluer
            \item \textbf{Impact}: Essential for unbiased cosmology at scale
        \end{itemize}
        \column{0.48\textwidth}
        \vspace{10pt}
        \includegraphics<1>[width=\textwidth]{2509.13394/images/19ekb_light_curves_all_paper.png}%
        \includegraphics<2>[width=\textwidth]{2509.13394/images/recreated_contamination_plot.png}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Recent GPU-Accelerated Applications}
    \framesubtitle{Case study 3/4: Dark Energy vs Supernova Systematics \arxiv{2509.13220}}
    \student{adam_ormondroyd}{Adam Ormondroyd}{PhD}
    \begin{columns}
        \column{0.65\textwidth}
        \begin{itemize}
            \item \textbf{Question}: DESI+DES $w_0w_a$ preference - new physics or systematics?
            \item \textbf{Method}: Bayesian model comparison
            \item \textbf{Models}: Dynamic DE vs redshift-dependent SN bias
            \item \textbf{Result}: Systematics fit equally well with lower complexity
            \item \textbf{Evidence}: Favors systematic explanation
            \item \textbf{Lesson}: Test mundane before claiming exotic
        \end{itemize}
        \column{0.35\textwidth}
        \vspace{10pt}
        \includegraphics<1>[width=\textwidth]{2509.13220/plots/desidr2_des5yoffset_20_wa.pdf}%
        \includegraphics<2>[width=\textwidth]{2509.13220/plots/des5yoffset_20_wa.pdf}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Recent GPU-Accelerated Applications}
    \framesubtitle{Case study 4/4: Gravitational Wave Inference \arxiv{2509.04336}}
    \student{metha_prathaban}{Metha Prathaban}{PhD}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{itemize}
            \item \textbf{Goal}: GPU-accelerate bilby's acceptance-walk NS
            \item \textbf{Implementation}: Faithful port to blackjax-ns
            \item \textbf{Performance}: 20-40× speedup for BBH
            \item \textbf{Validation}: Identical posteriors/evidences
            \item \textbf{Hardware}: Single GPU vs CPU clusters
            \item \textbf{Impact}: Clean baseline for future methods
        \end{itemize}
        \column{0.48\textwidth}
        \includegraphics<1>[width=\textwidth]{prathaban_handley_2509.04336/figures/8s_corner_comparison.pdf}%
        \includegraphics<2>[width=\textwidth]{prathaban_handley_2509.04336/figures/walltime_speedup.pdf}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The Future: AI in Scientific Code Development}
    \student{claude}{Claude Code}{AI Assistant}
    \vspace{-1em}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{The Real AI Revolution: LLMs}
            The biggest impact of AI will not be in analyzing data, but in helping us write the code to do it.
            \begin{itemize}
                \item \textbf{Automated code translation}: LLMs can help port legacy Fortran/C++ models to modern, GPU-friendly \& differentiable frameworks like JAX or PyTorch.
            \end{itemize}
        \end{block}
        \column{0.49\textwidth}
        \begin{block}{The 80/20 Rule of Scientific Work}
            \begin{itemize}
                \item \textbf{80\% ``boring'' tasks}: Writing code, debugging, drafting \& reviewing papers, munging data, organising meetings...
                \item \textbf{20\% ``hard thinking''}: The actual scientific insight.
            \end{itemize}
            AI's biggest immediate impact is automating and accelerating the 80\%, freeing up human time for the 20\%.
        \end{block}
    \end{columns}
    \begin{alertblock}{Key Message}
        AI is not just a tool for analysis; it's about to fundamentally change how we develop, optimize, and deploy our science
    \end{alertblock}
\end{frame}


\begin{frame}
    \frametitle{Conclusions}
    \framesubtitle{\tthref{github.com/handley-lab/group}}
        \begin{enumerate}
            \item \textbf{GPU $\neq$ Machine Learning: Two Independent Capabilities}
                \begin{itemize}
                    \item GPUs accelerate any parallel algorithm.
                    \item Automatic differentiation + massive parallelization.
                    \item Often confused, serve different purposes.
                \end{itemize}
            \item \textbf{Classical Algorithms on GPU Competitive with ML State of the Art}
                \begin{itemize}
                    \item Traditional physics methods + GPU = superior performance.
                \end{itemize}
            \item \textbf{AI Accelerates Development as well as Computation}
                \begin{itemize}
                    \item LLMs solve the GPU porting challenge at scale.
                    \item 10× development speedup enables widespread adoption.
                \end{itemize}
        \end{enumerate}
        \vfill
        \begin{alertblock}{Get Started with GPU-Accelerated Sampling}
            \centering
            \Large
            \tthref{handley-lab.co.uk/nested-sampling-book}
        \end{alertblock}
    \tikz[overlay,remember picture]
        \node[anchor=north east] (A) at ($(current page.north east)+(0,0)$) {
        \includegraphics[width=0.06\textheight]{people/adam_ormondroyd.jpg}%
        \includegraphics[width=0.06\textheight]{people/charlotte_priestley.jpg}%
        \includegraphics[width=0.06\textheight]{people/claude.jpg}%
        \includegraphics[width=0.06\textheight]{people/david_yallup.jpg}%
        \includegraphics[width=0.06\textheight]{people/dily_ong.jpg}%
        \includegraphics[width=0.06\textheight]{people/gemini.jpg}%
        \includegraphics[width=0.06\textheight]{people/harry_bevins.jpg}%
        \includegraphics[width=0.06\textheight]{people/metha_prathaban.jpg}%
        \includegraphics[width=0.06\textheight]{people/ming_yang.jpg}%
        \includegraphics[width=0.06\textheight]{people/namu_kroupa.jpg}%
        \includegraphics[width=0.06\textheight]{people/openai.jpg}%
        \includegraphics[width=0.06\textheight]{people/sam_leeney.jpg}%
        \includegraphics[width=0.06\textheight]{people/sinah_legner.jpg}%
        \includegraphics[width=0.06\textheight]{people/toby_lovick.jpg}%
        \includegraphics[width=0.06\textheight]{people/wei-ning_deng.jpg}%
        \includegraphics[width=0.06\textheight]{people/will_handley.jpg}%
        \includegraphics[width=0.06\textheight]{people/will_templeton.jpg}%
    };
\end{frame}

\end{document}

```

`ini_2025.tex`:

```tex
\documentclass[aspectratio=169]{beamer}
\usepackage{will_handley_beamer}
\usepackage{title_page}

% Commands
% --------
% - \arxiv{arxiv number}
% - \arxiv{<number>}            arxiv.org/abs/<number>
% - \oldarxiv{<arxiv number>}   arxiv.org/<number>
% - \doi{<doi>}                 doi.org/<doi>
% - \xkcd{<number>}             xkcd.com/<number>
% - \email{<email>}             <<email>>
% - \tthref{<website>}          <website>
% - \av[dist]{<quantity>}       <quantity>_{dist}
% - \student{<name>}{<detail>}{<photo>}

% Talk details
% ------------
\title{GPU Accelerated Nested Sampling}
\date{27$^{\text{th}}$ June 2025}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}
    \frametitle{A Case Study in Astrostatistics}
    \begin{columns}
        \column{0.48\textwidth}
        \vspace{-10pt}
        \begin{block}{The Challenge: GW170817}
            \begin{itemize}
                \item \textbf{Gravitational wave detected}: Binary neutron star merger
                \item \textbf{Real-time parameter estimation}: 15+ dimensional space
                \item \textbf{Sky localization}: $\sim 30$ deg$^2$ uncertainty
                \item \textbf{EM counterpart follow-up}: Telescopes need targets within seconds to minutes
            \end{itemize}
        \end{block}
        \begin{block}{Statistical Requirements}
            \begin{itemize}
                \item \textbf{Parameter estimation}: Masses, spins, distance, sky position
                \item \textbf{Model comparison}: Signal vs noise, waveform models
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \includegraphics[height=0.49\textwidth]{figures/ligo_m1_m2}
        \includegraphics[height=0.49\textwidth]{figures/ligo_lambert-skymap}
        \begin{block}{The Broader Astrostatistics Context}
            \begin{itemize}
                \item \textbf{High-dimensional}: $d \sim 10^2$--$10^3$
                \item \textbf{Multimodal}: Competing physical models
                \item \textbf{Expensive likelihoods}: Complex sims
                \item \textbf{Model selection critical}: Which physics?
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The Bayesian Inference Challenge}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{Parameter Estimation $P(\theta|D,M)$}
            \begin{itemize}
                \item Posterior: $\mathcal{P}(\theta|D) \propto \mathcal{L}(D|\theta) \pi(\theta)$
                \item Need samples from $\mathcal{P}(\theta|D)$
                \item Standard approach: MCMC methods
                \item Well-solved problem in many cases
            \end{itemize}
        \end{block}
        \begin{block}{Model Comparison $P(M|D)$}
            \begin{itemize}
                \item $\mathcal{Z} = \mathcal{P}(D|M) = \int \mathcal{L}(D|\theta) \pi(\theta) d\theta$
                \item Evidence/marginal likelihood/Bayes factor
                \item \textbf{Much harder to compute}
                \item MCMC doesn't estimate $\mathcal{Z}$ directly
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Challenges for Modern Science}
            \begin{itemize}
                \item \textbf{High dimensions}: $d \sim 10^2$--$10^3$
                \item \textbf{Natural, relevant multimodality}: Multiple acceptable answers need investigation
                \item \textbf{Computational cost}: Complex forward models
                \item \textbf{Model selection}: Which physics to include?
            \end{itemize}
        \end{block}
        \begin{center}
            \textbf{Key Insight:}\\
            Need methods that compute \emph{both} \\
            posterior samples \emph{and} evidence
        \end{center}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Sampling Methods for Bayesian Inference}
    \begin{columns}
        \column{0.65\textwidth}
        \begin{block}{Single-Chain MCMC}
            \begin{itemize}
                \item \textbf{Metropolis-Hastings}: Simple, widely used (PyMC)
                \item \textbf{HMC/NUTS}: Gradient-based, efficient (Stan, BlackJAX)
                \item Fast for unimodal well-conditioned problems, no evidence
            \end{itemize}
        \end{block}
        \begin{block}{Ensemble Methods}
            \begin{itemize}
                \item \textbf{Affine-invariant}: emcee, zeus
                \item \textbf{Sequential Monte Carlo}: Tempering, annealing
                \item can struggle with multimodality, some estimate evidence
            \end{itemize}
        \end{block}
        \column{0.35\textwidth}
        \begin{columns}
            \column{0.5\textwidth}
        \includegraphics[width=\textwidth]{figures/emcee}
        \includegraphics[width=\textwidth]{figures/metropolis-hastings}
            \column{0.5\textwidth}
        \includegraphics[width=\textwidth]{figures/zeus}
        \end{columns}
        \includegraphics[width=\textwidth]{figures/hmc_explained}
        \vspace{-20pt}
        \begin{center}
            \textbf{Nested sampling}: \\
            Unique in targeting 
            evidence computation directly
        \end{center}
    \end{columns}
\end{frame}

\begin{frame}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{\textbf{MCMC}}
            \only<16>{
                \begin{itemize}
                    \item Single ``walker''
                    \item Explores posterior
                    \item Fast, if proposal matrix is tuned
                    \item Parameter estimation
                    \item Channel capacity optimised for generating posterior samples
                \end{itemize}
            }
        \end{block}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_mcmc}%
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_mcmc}%
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_mcmc}%
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_mcmc}%
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_mcmc}%
            \includegraphics<6-15>[width=\textwidth,page=9]{figures/himmelblau_mcmc}%
        \centerline{\includegraphics<16>[width=0.5\textwidth,page=9]{figures/himmelblau_mcmc}}%
        \column{0.48\textwidth}
        \begin{block}<7->{\textbf{Nested sampling}}
            \only<16>{
                \begin{itemize}
                    \item Ensemble of ``live points''
                    \item Scans from prior to peak of likelihood
                    \item Slower, no tuning required
                    \item Parameter estimation, model comparison
                    \item Channel capacity optimised for computing partition function
                \end{itemize}
            }
        \end{block}
            \includegraphics<7|handout:0>[width=\textwidth,page=1]{figures/himmelblau_ns}%
            \includegraphics<8|handout:0>[width=\textwidth,page=2]{figures/himmelblau_ns}%
            \includegraphics<9|handout:0>[width=\textwidth,page=3]{figures/himmelblau_ns}%
            \includegraphics<10          >[width=\textwidth,page=4]{figures/himmelblau_ns}%
            \includegraphics<11|handout:0>[width=\textwidth,page=5]{figures/himmelblau_ns}%
            \includegraphics<12|handout:0>[width=\textwidth,page=6]{figures/himmelblau_ns}%
            \includegraphics<13|handout:0>[width=\textwidth,page=7]{figures/himmelblau_ns}%
            \includegraphics<14|handout:0>[width=\textwidth,page=8]{figures/himmelblau_ns}%
            \includegraphics<15|handout:0>[width=\textwidth,page=8]{figures/himmelblau_ns}%
        \centerline{\includegraphics<16>[width=0.5\textwidth,page=4]{figures/himmelblau_ns}} %
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The nested sampling meta-algorithm: live points}
    \begin{columns}
        \column{0.5\textwidth}
        \begin{itemize}
            \item Start with $n$ random samples over the space.
            \item Delete outermost sample, and replace with a new random one at higher integrand value.
            \item The ``live points'' steadily contract around the peak(s) of the function.
            \item We can use this evolution to estimate volume \emph{probabilistically}.
            \item At each iteration, the contours contract by $\sim\frac{1}{n}\only<5->{\pm \frac{1}{n}}$ of their volume.
            \item This is an exponential contraction, so
                \[  \int f(x) dV \approx \sum_i f(x_i) \Delta V_i, \quad V_i = V_0 e^{-\only<5->{(}i\only<5->{\pm\sqrt{i})}/n} \]
        \end{itemize}
        \column{0.5\textwidth}
        \includegraphics<1|handout:0>[width=\textwidth,page=1]{figures/himmelblau_ns}%
        \includegraphics<2|handout:0>[width=\textwidth,page=2]{figures/himmelblau_ns}%
        \includegraphics<3|handout:0>[width=\textwidth,page=3]{figures/himmelblau_ns}%
        \includegraphics<4-         >[width=\textwidth,page=4]{figures/himmelblau_ns}%
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The nested sampling meta-algorithm: dead points}
    \includegraphics[width=\textwidth]{figures/dead_measure}
    \begin{columns}
        \column{0.7\textwidth}
        \begin{itemize}
            \item At the end, left with a set of discarded ``dead points''.
            \item Dead points have a unique scale-invariant distribution $\propto\: \tfrac{dV}{V}$.
            \item Each dead point gets a \textbf{posterior weight}: $w_i = \mathcal{L}_i \Delta V_i$
        \end{itemize}
        \column{0.3\textwidth}
        \begin{block}{Key Output}
        \begin{itemize}
            \item \textbf{Posterior samples} $\theta_i$ weight~$w_i=\mathcal{L}_i \Delta V_i$
            \item \textbf{Evidence} $\mathcal{Z} = \sum_i w_i$
        \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Nested Sampling as Partition Function Calculator $\log \mathcal{Z}(\beta)$}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{The Key Insight}
            \begin{itemize}
                \item Nested sampling directly estimates the \textbf{density of states}:
                    \[ g(\mathcal{L}) = \int \delta(\mathcal{L}(\theta) - \mathcal{L}) \pi(\theta) d\theta \]
                \item This is the \textbf{partition function} at inverse temperature $\beta$:
                    \[ \mathcal{Z}(\beta) = \int g(\mathcal{L}) \mathcal{L}^{\beta} d\mathcal{L} \]
                \item Evidence is special case: $\mathcal{Z} = \mathcal{Z}(\beta=1)$
                \item \textbf{In practice}: $\mathcal{Z}(\beta) \approx \sum_i \mathcal{L}_i^\beta \Delta V_i$
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Statistical Physics Connection}
            \begin{itemize}
                \item \textbf{Canonical ensemble}: $p(\theta|\beta) \propto \mathcal{L}(\theta)^\beta \pi(\theta)$
                \item \textbf{Free energy}: $\beta F = -\log \mathcal{Z}$
                \item \textbf{Internal energy}: $U = -\frac{\partial \log \mathcal{Z}}{\partial \beta}$
                \item \textbf{Heat capacity}: $C = \frac{\partial U}{\partial \beta}$
            \end{itemize}
        \end{block}
        \vspace{10pt}
        \begin{center}
            \textbf{Nested sampling provides the fundamental thermodynamic quantities}\\
            for any probabilistic model
        \end{center}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Why GPUs? The Future of High-Performance Computing}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{GPU Advantages (Often Confused!)}
            \begin{itemize}
                \item \textbf{Massive Parallelization}:
                    \begin{itemize}
                        \item 1000s of cores vs 10s on CPU
                        \item Perfect for ensemble algorithms
                        \item Vectorization across particles/chains
                        \item Independent likelihood evaluations
                    \end{itemize}
                \item \textbf{Automatic Differentiation}:
                    \begin{itemize}
                        \item GPU-accelerated gradients ``for free''
                        \item JAX/PyTorch ecosystem make this possible
                        \item Essential for modern optimization
                    \end{itemize}
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{The HPC Reality}
            \begin{itemize}
                \item \textbf{Future HPC is GPU dominated}:
                \item \textbf{Legacy CPU codes becoming obsolete}
            \end{itemize}
        \end{block}
        \begin{block}{Apples-to-Apples comparison}
            \begin{itemize}
                \item Quantifying GPU advantage
                    \begin{itemize}
                        \item GPUs ~40× more expensive to rent
                        \item GPUs ~100× rarer in HPC allocations
                    \end{itemize}
                \item Sometimes you don't care about walltime.
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Why BlackJAX? Unified GPU Framework for Bayesian Inference}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{The Fragmentation Problem}
            \begin{itemize}
                \item \textbf{Scattered ecosystem}:
                    \texttt{MultiNest}, \texttt{PolyChord}, \texttt{dynesty}, \texttt{UltraNest}, \texttt{nautilus}, \texttt{nessai}, \ldots
            \end{itemize}
        \end{block}
        \begin{block}{BlackJAX Solution}
            \begin{itemize}
                \item \textbf{Community JAX codebase}
                \item \textbf{Fair benchmarking} with identical GPU infrastructure
                \item \textbf{Composable algorithms} with shared components
                \item \textbf{Modern ML ecosystem integration}
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Algorithm-Hardware Matching}
            \begin{itemize}
                \item \textbf{Ensemble methods $\leftrightarrow$ GPU parallelization}:
                    \begin{itemize}
                        \item Nested sampling: 100-1000 live points
                        \item SMC: 1000s of particles
                        \item Embarrassingly parallel operations
                    \end{itemize}
                \item \textbf{Scientific problems are compute-bound}:
                    \begin{itemize}
                        \item Unlike ultra-large DL models
                        \item GPU memory rarely limiting
                        \item Perfect match for vectorization
                    \end{itemize}
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{BlackJAX Nested Sampling: Kernel Choice}
    \student{david_yallup}{David Yallup}{PDRA}
    \begin{columns}
        \column{0.5\textwidth}
        \begin{block}{Nested Sampling Meta-Algorithm}
            \begin{itemize}
                \item \textbf{Framework is kernel-agnostic}:
                    \begin{itemize}
                        \item Original: Metropolis Hastings (Skilling 2006)
                        \item MultiNest: rejection ellipsoids
                        \item PolyChord: slice sampling
                        \item nessai/nautilus: ML techniques
                    \end{itemize}
            \end{itemize}
        \end{block}
        \begin{block}{Our Choice: Slice Sampling}
            \begin{itemize}
                \item \textbf{First scalable generic solution} in BlackJAX
                \item \textbf{No tuning required} (unlike MCMC proposal matrices)
                \item \textbf{Robust across dimensions} \& problem types
            \end{itemize}
        \end{block}
        \column{0.5\textwidth}
        \vspace{7pt}   
        \includegraphics[width=\textwidth,page=4]{figures/himmelblau_ns}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{GW150914 Binary Black Hole Merger}
    \student{metha_prathaban}{Metha Prathaban}{PhD}
    \begin{columns}
        \column{0.5\textwidth}
        \begin{block}{Performance on Real Data}
            \begin{itemize}
                \item \textbf{BlackJAX GPU-NS}: 207 seconds (1 GPU)
                \item \textbf{FlowMC (GPU MCMC)}: 742 seconds (1 GPU)
                \item \textbf{Bilby/Dynesty}: ~2 hours (400 CPUs)
            \end{itemize}
            \begin{center}
                \textbf{Orders of magnitude speedup over CPU}\\
                \textbf{Comparable to other GPU-native methods}
            \end{center}
            \vspace{5pt}
        \end{block}
        \begin{block}{Key Achievement}
            \begin{itemize}
                \item Nested sampling now competitive on GPUs
                \item Direct evidence computation included
            \end{itemize}
        \end{block}
        \column{0.5\textwidth}
        \includegraphics[width=\textwidth]{figures/gw_corner.pdf}
        \vspace{5pt}
        \begin{center}
            \small{Good agreement between BlackJAX\\and FlowMC posteriors}
        \end{center}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{CMB \& Cosmic Shear Analysis}
    \student{toby_lovick}{Toby Lovick}{PhD}
    \begin{columns}
        \column{0.55\textwidth}
        \begin{block}{CMB Power Spectrum (6 params)}
            \begin{itemize}
                \item \textbf{PolyChord (CPU)}: ~1 hour
                \item \textbf{BlackJAX (GPU)}: 12 seconds
            \end{itemize}
            \begin{center}
                \textbf{300× speedup}
            \end{center}
        \end{block}
        \begin{block}{Cosmic Shear (37 params)}
            \begin{itemize}
                \item \textbf{PolyChord (48 CPUs)}: ~8 months
                \item \textbf{NUTS (12 A100 GPUs)}: 2 days
                \item \textbf{BlackJAX (1 A100 GPU)}: 4.5 hours
            \end{itemize}
            \begin{center}
                \textbf{$>$1000× speedup vs CPU}\\
                \textbf{10× speedup vs existing GPU approach}\arxiv{2405.12965}
            \end{center}
        \end{block}
        \column{0.45\textwidth}
        \includegraphics<1>[width=\textwidth]{figures/wl_corner.pdf}%
        \includegraphics<2>[width=\textwidth]{figures/wl_scaling.pdf}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The Real AI Revolution: LLMs as the Missing Piece}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{LLMs: The GPU Code Translator}
            \begin{itemize}
                \item Automated translation: Fortran/C++ $\rightarrow$ JAX/PyTorch
                \item Bridges the gap between legacy science and modern hardware
            \end{itemize}
        \end{block}
        \begin{block}{The 80/20 Rule of Scientific Work}
            \begin{itemize}
                \item \textbf{80\% ``boring'' tasks}: forms, papers, grants, reviews, grading, code writing\ldots
                \item \textbf{20\% hard thinking}: Novel insights, experimental design, theory
                \item \textbf{AI's biggest impact}: Automating the 80\%, not the 20\%
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Beyond Scientific Analysis}
            \begin{itemize}
                \item \textbf{Common focus}: Using LLMs for analysis
                \item \textbf{Real transformation}: Automating workflow
                \item \textbf{Already happening}:
                    \begin{itemize}
                        \item Grant writing assistance
                        \item Paper drafting and review
                        \item Code generation and debugging
                        \item Literature review automation
                    \end{itemize}
            \end{itemize}
        \end{block}
        \begin{block}{The Productivity Explosion}
            \begin{itemize}
                \item \textbf{Quality control}: Becomes the limiting factor
                \item \textbf{Focus shift}: writing $\rightarrow$ critical thinking
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Resources}
    \begin{itemize}
        \item \textbf{Installation}: \texttt{pip install git+https://github.com/handley-lab/blackjax}
        \item \textbf{Documentation}: \tthref{handley-lab.co.uk/nested-sampling-book}
    \end{itemize}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{block}{BlackJAX Implementation}
            \begin{itemize}
                \item \textbf{BlackJAX}: \github{handley-lab/blackjax}
                \item \textbf{Nested sampling}: In PR to \github{blackjax-devs/blackjax} \#755
            \end{itemize}
        \end{block}
        \begin{block}{Theory \& Background}
            \begin{itemize}
                \item \textbf{Review papers}: \arxiv{2205.15570}, \arxiv{2101.09675}
                \item \textbf{Original paper}: \citehref{https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full}{}{Skilling (2006)}
            \end{itemize}
        \end{block}
        \column{0.48\textwidth}
        \begin{block}{Workshop \& Learning}
            \begin{itemize}
                \item \textbf{GPU Nested Sampling Workshop}: \tthref{github.com/handley-lab/workshop}
                \item \textbf{Interactive tutorials}: JAX, BlackJAX, GPU acceleration
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Conclusions}
    \framesubtitle{\tthref{github.com/handley-lab/group}}
    \tikz[overlay,remember picture]
        \node[anchor=north east] (A) at ($(current page.north east)+(0,0)$) {
        \includegraphics[width=0.09\textheight]{people/adam_ormondroyd.jpg}%
        \includegraphics[width=0.09\textheight]{people/charlotte_priestley.jpg}%
        \includegraphics[width=0.09\textheight]{people/david_yallup.jpg}%
        \includegraphics[width=0.09\textheight]{people/dily_ong.jpg}%
        \includegraphics[width=0.09\textheight]{people/harry_bevins.jpg}%
        \includegraphics[width=0.09\textheight]{people/harvey_williams.jpg}%
        \includegraphics[width=0.09\textheight]{people/krish_nanavati.jpg}%
        \includegraphics[width=0.09\textheight]{people/metha_prathaban.jpg}%
        \includegraphics[width=0.09\textheight]{people/ming_yang.jpg}%
        \includegraphics[width=0.09\textheight]{people/namu_kroupa.jpg}%
        \includegraphics[width=0.09\textheight]{people/sam_leeney.jpg}%
        \includegraphics[width=0.09\textheight]{people/sinah_legner.jpg}%
        \includegraphics[width=0.09\textheight]{people/toby_lovick.jpg}%
        \includegraphics[width=0.09\textheight]{people/wei-ning_deng.jpg}%
        \includegraphics[width=0.09\textheight]{people/will_handley.jpg}%
        \includegraphics[width=0.09\textheight]{people/will_templeton.jpg}%
    };
    \vspace{-0.1\textheight}
    \begin{itemize}
        \item \textbf{Nested sampling is widely used} across physical sciences for parameter estimation and model comparison
        \item \textbf{BlackJAX provides GPU-native implementation} with $10\times$-$100\times$ speedups
        \item \textbf{JAX ecosystem integration} enables modern scientific workflows
        \item \textbf{Real applications} from gravitational waves to cosmology benefit immediately
        \item \textbf{The future is GPU-accelerated} scientific computing with AI integration
    \end{itemize}
\end{frame}

\end{document}

```

`kicc_ai_tools_2025.tex`:

```tex
\documentclass[aspectratio=169]{beamer}
\usepackage{will_handley_beamer}
\usepackage{title_page}

% Commands
% --------
% - \arxiv{arxiv number}
% - \arxiv{<number>}            arxiv.org/abs/<number>
% - \oldarxiv{<arxiv number>}   arxiv.org/<number>
% - \doi{<doi>}                 doi.org/<doi>
% - \xkcd{<number>}             xkcd.com/<number>
% - \email{<email>}             <<email>>
% - \tthref{<website>}          <website>
% - \av[dist]{<quantity>}       <quantity>_{dist}
% - \student{<name>}{<detail>}{<photo>}

% Talk details
% ------------
\title{AI/ML Tools for Research}
\subtitle{KICC Workshop}
\date{23rd July 2025}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}
    \frametitle{Overview}
    
    \begin{columns}
        \column{0.5\textwidth}
        \begin{itemize}
            \item Follow-up to successful Part II/III training.
            \item Focus on ``doing'' not ``explaining''.
            \item Three layers of AI tools framework.
            \item Interactive demonstrations.
            \item 90 minutes then tea.
        \end{itemize}
        
        \vspace{0.3cm}
        
        \textbf{Common questions:}
        \begin{itemize}
            \item ``Which model should I use?''
            \item ``Don't these things hallucinate?''
            \item ``How do I get started?''
            \item ``Is this worth the investment?''
        \end{itemize}
        
        \column{0.5\textwidth}
        \begin{block}{Previous session covered:}
            \begin{itemize}
                \item Academic practice and Cambridge policies.
                \item How LLMs work and their limitations.
                \item Bias, hallucinations, and mitigation.
                \item Privacy and intellectual property.
                \item Model landscape and responsible usage.
            \end{itemize}
        \end{block}
    \end{columns}
    
    \vfill
    \centering
    \footnotesize{Previous slides: \tthref{docs.google.com/presentation/d/1rnbk7WZUjMIAGBzkN6i8v0tEe1imiHprvL1ZOjs2B20}}
\end{frame}

\begin{frame}
    \frametitle{First things first}
    If you've been busy, the past six months have seen a couple of inflection points
    \vspace{0.5cm}
    \begin{itemize}
        \item \textbf{February 2025}: Models acquired many of the capabilities we prize in our PhD students
            \begin{itemize}
                \item Models: o3 series, Gemini 2.5, Claude 3.5 Sonnet.
                \item Capabilities: Code development, mathematical reasoning, literature review, paper \& grant drafting.
            \end{itemize}
        \item \textbf{May 2025}: Agentic systems launched commercially
            \begin{itemize}
                \item Tools: Claude Code, Cursor agent mode, Deep Research.
                \item Capabilities: Writing test suites, assembling pip-installable codes, synthesising handwritten notes and TeX files, write/run/debug code in languages you don't know.
            \end{itemize}
        \item \textbf{The performance gap}: Free models lag significantly behind paid versions
    \end{itemize}
    
    \vspace{0.5cm}
    If your lived experience is typing a research question into free ChatGPT at the start of the year and finding it was rubbish, things have moved on
\end{frame}

\begin{frame}
    \frametitle{On AI Hype}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{itemize}
            \item These things are not ``intelligent'' in the way Silicon Valley wants you to believe (AGI is not just around the corner).
            \item \emph{Caveat emptor}: It is very hard for a human to read language, and not reflexively construct a mind behind it.
            \item You should not however think of them as mere ``next token predictors'':
                \begin{itemize}
                    \item The modern reinforcement learnt reasoning systems are significantly more than that.
                    \item Agentic tools ground these in reality.
                \end{itemize}
        \end{itemize}
        
        \column{0.48\textwidth}
        \begin{block}{A senior examiner's experiment}
            \begin{itemize}
                \item I got o3 to take the Part II exams.
                \item Prompt on exam day: ``You are a first class Part II Cambridge astronomy student... here is the syllabus... here is the question, answer it''.
                \item Toby Lovick transcribed answers onto written exam scripts.
                \item Slipped them amongst real scripts and marked blind.
                \item \textbf{Result}: Best student we've had across IoA history, even without coursework.
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{The AI landscape}
    \centering
    \begin{tabular}{l|c|c|c|}
        \visible<3->{Under the hood} & 
        \visible<3->{\includegraphics[valign=m, width=0.2\textwidth]{logos/microsoft}} & 
        \visible<3->{\includegraphics[valign=m, width=0.2\textwidth]{logos/amazon}} & 
        \visible<3->{\includegraphics[valign=m, width=0.2\textwidth]{logos/google}} \\[0.5cm]
        
        \visible<2->{Companies} & 
        \visible<2->{\includegraphics[valign=m, width=0.2\textwidth]{logos/openai}} & 
        \visible<2->{\includegraphics[valign=m, width=0.2\textwidth]{logos/anthropic}} & 
        \visible<2->{\includegraphics[valign=m, width=0.2\textwidth]{logos/deepmind}} \\[0.5cm]
        
        \visible<1->{Models} & 
        \visible<1->{\includegraphics[valign=m, width=0.2\textwidth]{figures/chatgpt.png}} & 
        \visible<1->{\includegraphics[valign=m, width=0.2\textwidth]{figures/claude.pdf}} & 
        \visible<1->{\includegraphics[valign=m, width=0.2\textwidth]{figures/gemini.pdf}}
    \end{tabular}
    
    \vspace{1cm}
    \visible<4->{
        \textbf{Others exist:} Perplexity, Poe, Character.ai, You.com, Cohere, xAI, Llama...
    }
    
    \vspace{0.5cm}
    \visible<5->{
        \textbf{But these are the three main pillars (July 2025).}
    }
\end{frame}

\begin{frame}
    \frametitle{Three layers of AI tools}
    \framesubtitle{A framework for understanding the landscape}
    \begin{block}{Layer 3: Agentic Systems}
        \vspace{-0.3cm}
        \begin{columns}
            \column{0.48\textwidth}
            \begin{itemize}
                \item Claude Code, cursor agent mode.
                \item Custom workflows.
            \end{itemize}
            \column{0.48\textwidth}
            \begin{itemize}
                \item Autonomous task completion.
                \item Steeper learning curve but transformative.
            \end{itemize}
        \end{columns}
    \end{block}
    
    \vspace{0.5cm}
    
    \begin{block}{Layer 2: Chat-based AI}
        \vspace{-0.3cm}
        \begin{columns}
            \column{0.48\textwidth}
            \begin{itemize}
                \item ChatGPT, Claude, Gemini.
                \item Web interfaces.
            \end{itemize}
            \column{0.48\textwidth}
            \begin{itemize}
                \item Interactive problem solving.
                \item Good for exploration and learning.
            \end{itemize}
        \end{columns}
    \end{block}
    
    \vspace{0.5cm}
    
    \begin{block}{Layer 1: Autocomplete}
        \vspace{-0.3cm}
        \begin{columns}
            \column{0.48\textwidth}
            \begin{itemize}
                \item GitHub Copilot.
                \item VS Code extensions.
            \end{itemize}
            \column{0.48\textwidth}
            \begin{itemize}
                \item Completes code as you type.
                \item Minimal learning curve.
            \end{itemize}
        \end{columns}
    \end{block}
\end{frame}

\begin{frame}
    \frametitle{Layer 1: Autocomplete}
    \framesubtitle{Standard code-completion, but powered by AI}
    
    \begin{columns}
        \column{0.6\textwidth}
        \begin{block}{Core idea}
            \begin{itemize}
                \item AI-powered code-completion.
                \item Trained on all of GitHub.
                \item Context-aware suggestions.
            \end{itemize}
        \end{block}
        
        \begin{block}{Recommendation}
            \begin{itemize}
                \item GitHub Copilot (Pro subscription).
                \item Free for university email holders.
                \item \url{github.com/settings/education/benefits}.
                \item Worth \$10/month.
                \item Also available in Cursor.
            \end{itemize}
        \end{block}

        \column{0.4\textwidth}
        \includegraphics[width=\textwidth]{figures/copilot_screenshot.png}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Layer 2: Chat-based AI}
    \framesubtitle{The usual chat versions}
    
    \begin{columns}
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/chatgpt_screenshot.png}
        
        ChatGPT
        
        \url{chat.openai.com}
        
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/claude_screenshot.png}
        
        Claude
        
        \url{claude.ai}
        
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/gemini_screenshot.png}
        
        Gemini
        
        \url{gemini.google.com}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Layer 2: Developer versions}
    \framesubtitle{Use their developer level tools}
    
    \begin{columns}
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/chatgpt_dev_screenshot.png}
        
        OpenAI Playground
        
        \url{platform.openai.com/playground}
        
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/claude_dev_screenshot.png}
        
        Claude Workbench
        
        \url{console.anthropic.com/workbench}
        
        \column{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/gemini_dev_screenshot.png}
        
        Google AI Studio
        
        \url{aistudio.google.com}
    \end{columns}
    
    \vspace{0.3cm}
    
    \textbf{These should be your default as a scientist} -- if you're happy using a command line, you shouldn't be satisfied by an interface that looks like a child's toy.
\end{frame}

\begin{frame}
    \frametitle{Layer 2: Chat-based AI}
    \framesubtitle{Use AI to improve AI}
    \begin{columns}
        \column{0.48\textwidth}
        \begin{itemize}
            \item Likely the layer of AI most familiar to you.
            \item ``chatting''/conversation can be very powerful for naturally tuning the attention/context of the model.
            \item For one shot work, a little prompt engineering goes a long way:
                \begin{itemize}
                    \item You can use these tools to improve prompts.
                \end{itemize}
        \end{itemize}
        \column{0.48\textwidth}
        \includegraphics<1>[width=\textwidth]{figures/prompt_demo1.png}%
        \includegraphics<2>[width=\textwidth]{figures/prompt_demo2.png}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Layer 2: Cost}
    \framesubtitle{Pay as you use vs subscription models}
    
    \begin{columns}
        \column{0.6\textwidth}
        \begin{itemize}
            \item Gemini is free to use (but you sign away your data).
            \item ChatGPT and Claude offer \$20/month subscriptions.
            \item API access available: pay per token consumed:
                \begin{itemize}
                    \item \$0.0002-0.015 per 1000 tokens (roughly 750 words).
                    \item Orders of magnitude cheaper for research use.
                    \item Only pay for what you actually use.
                \end{itemize}
            \item Never buy yearly subscriptions - AI evolves too rapidly.
            \item My group's preference: pay for API access via tokens.
                \begin{itemize}
                    \item Pay for what you use.
                    \item Allows you to try out the latest models.
                    \item API keys allow you to do all kinds of tricks at the command line.
                \end{itemize}
        \end{itemize}
        
        \column{0.4\textwidth}
        \includegraphics[width=\textwidth]{figures/cost_demo1.png}
        \includegraphics[width=\textwidth]{figures/cost_demo2.png}
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Layer 3: Agentic Systems}
    \framesubtitle{Autonomous task completion - transformative but complex}
    
    \begin{columns}
        \column{0.6\textwidth}
        \begin{itemize}
            \item Emerged commercially in April 2025.
            \item If you get too skilled at chat-based AI, your job ends up being a copy-paste-run-debug monkey.
            \item Agentic systems are what happens if you let ChatGPT edit files, run commands, and inspect the behaviour.
            \item This closes the scientific loop.
            \item Agentic systems are (partially) grounded in reality. Correctly guided, they can reduce hallucinations by testing their hypotheses.
            \item Claude Code (CLI), Cursor Agent mode, Gemini CLI.
            \item Deep research also falls into this category.
            \item Goal-based vs task-based.
        \end{itemize}
        
        \column{0.4\textwidth}
        \includegraphics<1>[width=\textwidth]{figures/agentic_demo.png}%
        \includegraphics<2>[width=\textwidth]{figures/agentic_demo2.png}%
    \end{columns}
\end{frame}

\begin{frame}
    \frametitle{Layer 3: Agentic Systems Cost}
    \framesubtitle{More expensive but potentially more valuable}
    
    \begin{itemize}
        \item Significantly more expensive than chat-based AI.
        \item Can consume hundreds of API calls per task.
        \item For agentic systems, subscriptions \emph{are} cheaper than pay-per-token.
        \item I was spending \$40/day on Claude Code with API tokens.
        \item Subscription gives you about 2hr coding session per day for \$20/month/user.
        \item Gemini CLI is ``free'' but risks going over free tier limit, and is less stable.
        \item In July 2025, Claude Code best in class.
        \item There are many further unknown costs of shifting your research toward ``hands-free coding''.
    \end{itemize}
\end{frame}

\begin{frame}
    \frametitle{Conclusions and Getting Started}
    
    \begin{columns}
        \column{0.5\textwidth}
        \begin{itemize}
            \item Use developer-level tools, not consumer interfaces.
            \item API access typically cheaper for research use.
            \item Capturing and owning content is critical -- use Otter for transcription.
            \item Export conversations and notes in formats that can be fed to LLMs.
            \item Get into the habit of recording information in exportable forms.
            \item Agentic systems are transformative but carry additional risks requiring careful consideration.
        \end{itemize}
        
        \column{0.5\textwidth}
        \begin{block}{To-do}
            \begin{itemize}
                \item Set up GitHub Copilot Pro (takes a couple of days to approve).
                \item Try AI Studio rather than ChatGPT.
                \item Get your PI to set up some tokens for your group to use (you only need ~O(\$10) to try it out).
                \item Try Google Gemini's Deep Research \textbf{button}.
                \item Test Claude Code/Gemini CLI for a month.
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

\end{document}

```

`sed_2025.tex`:

```tex
\documentclass[aspectratio=169]{beamer}
\usepackage{will_handley_beamer}
\usepackage{title_page}

% Commands
% --------
% - \arxiv{arxiv number}
% - \arxiv{<number>}            arxiv.org/abs/<number>
% - \oldarxiv{<arxiv number>}   arxiv.org/<number>
% - \doi{<doi>}                 doi.org/<doi>
% - \xkcd{<number>}             xkcd.com/<number>
% - \email{<email>}             <<email>>
% - \tthref{<website>}          <website>
% - \av[dist]{<quantity>}       <quantity>_{dist}
% - \student{<name>}{<detail>}{<photo>}

% A simple command for highlighting key terms
\newcommand{\keyterm}[1]{\textbf{\textcolor{C0}{#1}}}

% Talk details
% ------------
\title{A Statistician's Guide to the Galaxy (Fitting Zoo)}
\subtitle{An Introduction to the Statistical Foundations of SED Fitting}
\date{8 July 2025}

\begin{document}

% --- SLIDE 0: Title Slide ---
\begin{frame}
    \titlepage
\end{frame}

% --- SLIDE 1: SED Fitting Setup (2 minutes) ---
\begin{frame}
    \frametitle{The Goal: From Photons to Physics}
    \framesubtitle{Why SED fitting is a statistical inference problem}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{The Data $D$}
            We observe a galaxy's light (photometry or spectra) across different wavelengths. This is our dataset, $D$.
        \end{block}
        \includegraphics[width=\textwidth]{figures/sed.png}
        \column{0.49\textwidth}
        \begin{block}{The Model $\theta|M$}
            We want to infer the underlying physical properties, our parameters, $\theta$: Stellar Mass ($M_*$), Star Formation History (SFH), Dust content ($A_V$), Metallicity ($Z$), ...
        \end{block}
        \begin{block}{The Challenge}
            The parameter space is often:
            \begin{itemize}
                \item \keyterm{High-dimensional}: Many parameters to fit.
                \item \keyterm{Degenerate}: Different combinations of parameters can produce similar SEDs.
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

% --- SLIDE 2: Statistical Framework (2 minutes) ---
\begin{frame}
    \frametitle{The Language of Inference: }
    \framesubtitle{How we quantify what we learn from data}
    \begin{columns}
        \column{0.33\textwidth}
            \begin{block}{\C[1]{Prior}\hfill $\C[1]{\pi(\theta)}$}
                What we believe about the parameters \textit{before} we see the data. Our physical assumptions.
            \end{block}
        \vspace{2em}
        \begin{block}{\C[3]{Evidence}\hfill $\C[3]{\mathcal{Z}(D)}$}
            How we update our belief in the model using the data.
        \end{block}
        \column{0.33\textwidth}
\[\underbrace{\C[0]{\mathcal{P}(\theta|D)}}_{\C[0]{\text{Posterior}}} = \frac{\overbrace{\C[2]{\mathcal{L}(D|\theta)}}^{\C[2]{\text{Likelihood}}} \times \overbrace{\C[1]{\pi(\theta)}}^{\C[1]{\text{Prior}}}}{\underbrace{\C[3]{\mathcal{Z}(D)}}_{\C[3]{\text{Evidence}}}}\]
        \column{0.33\textwidth}
            \begin{block}{\C[2]{Likelihood}\hfill $\C[2]{\mathcal{L}(D|\theta)}$}
                How we update our belief in the parameters using the data.
            \end{block}
        \vspace{2em}
        \begin{block}{\C[0]{Posterior}\hfill $\C[0]{\mathcal{P}(\theta|D)}$}
            What we know about the parameters \textit{after} seeing the data. It's our updated state of knowledge.
        \end{block}
    \end{columns}
\end{frame}

% --- SLIDE 3: Chi-squared Maximization (3 minutes) ---
\begin{frame}
    \frametitle{The Simplest Approach: Optimization (e.g., $\chi^2$ Minimization)}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \only<1-6>{
        \begin{block}{How it Works: Hill Climbing}
            Imagine the parameter space is a landscape where lower $\chi^2$ (or higher likelihood) is ``downhill''.
            \begin{itemize}
                \item Start somewhere.
                \item Follow the steepest gradient downhill.
                \item Stop when you reach the bottom of a valley.
            \end{itemize}
        \end{block}
        \begin{block}{Advantages}
            \begin{itemize}
                \item \keyterm{Fast} and computationally cheap.
                \item Good for a quick first look.
            \end{itemize}
        \end{block}
        }
        \only<7-10>{
        \begin{block}{Limitations}
            \begin{itemize}
                \item Only gives a single \keyterm{point estimate} (the ``best fit'').
                \item \textbf{No uncertainty quantification!} Where are the error bars?
                \item Can easily get stuck in a \keyterm{local minimum}, missing the true global best fit.
            \end{itemize}
        \end{block}
        \begin{alertblock}{Key Message}
            Optimization is fast but gives an incomplete and potentially misleading picture. Science needs error bars.
        \end{alertblock}
        }
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<9>[width=\textwidth,page=9]{figures/himmelblau_gradient_ascent.pdf}%
            \includegraphics<10>[width=\textwidth,page=10]{figures/himmelblau_gradient_ascent.pdf}%
        \end{center}
    \end{columns}
    
\end{frame}

% --- SLIDE 4: Why do sampling? (3 minutes) ---
\begin{frame}
    \frametitle{Why do sampling?}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \begin{itemize}
            \item The cornerstone of numerical Bayesian inference is working with \textbf{samples}.
            \item Generate a set of representative parameters drawn in proportion to the posterior $\theta\sim\mathcal{P}$.
            \item The magic of marginalisation $\Rightarrow$ perform usual analysis on each sample in turn.
            \item The golden rule is \textbf{stay in samples} until the last moment before computing summary statistics/triangle plots because \[\boxed{f(\:\av{X}\:)\ne \av{\:f(X)\:}}\]
            \item Generally need $\sim\mathcal{O}(12)$ independent samples to compute a value and error bar.
        \end{itemize}
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics[width=\textwidth]{figures/himmelblau_samples.pdf}
        \end{center}
    \end{columns}
\end{frame}

% --- SLIDE 5: MCMC Sampling (4 minutes) ---
\begin{frame}
    \frametitle{The Classic Workhorse: Markov Chain Monte Carlo (MCMC)}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \only<1-5>{
        \begin{block}{How it Works (Metropolis-Hastings)}
            Imagine a ``walker'' exploring the parameter landscape.
            \begin{enumerate}
                \item Take a random step to a new position.
                \item If the new spot is ``higher'' (better likelihood), move there.
                \item If it's ``lower'', maybe move there anyway (with probability proportional to how much lower it is).
                \item Repeat millions of times. The path the walker takes traces the posterior distribution.
            \end{enumerate}
        \end{block}
        }
        \only<6-9>{
            \vspace{-1em}
        \begin{block}{Advantages \& Limitations}
            \begin{itemize}
                \item Explores the full posterior and gives uncertainties.
                \item \textcolor{red}{Limitation:} The walker can be inefficient. It can get ``stuck'' in a local high-likelihood region and fail to find other, separate modes.
                \item \textcolor{red}{Limitation:} Can be slow to explore highly correlated (``banana-shaped'') posteriors.
            \end{itemize}
        \end{block}
        \begin{alertblock}{Key Message}
            MCMC is a foundational sampling method, but its simple ``random walk'' can be inefficient in the complex parameter spaces of SED fitting.
        \end{alertblock}
        }
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[page=1]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<2>[page=2]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<3>[page=3]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<4>[page=4]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<5>[page=5]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<6>[page=6]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<7>[page=7]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<8>[page=8]{figures/himmelblau_mcmc.pdf}%
            \includegraphics<9>[page=9]{figures/himmelblau_mcmc.pdf}%
        \end{center}
    \end{columns}
\end{frame}

% --- SLIDE 6: Ensemble Sampling (emcee) (4 minutes) ---
\begin{frame}
    \frametitle{A Better Way: Ensemble Sampling (e.g., \texttt{emcee})}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \only<1-5>{
        \begin{block}{How it Works}
            Instead of one walker, we use an \keyterm{ensemble} of hundreds of walkers.
            \begin{itemize}
                \item The walkers don't move completely randomly.
                \item They propose new steps based on the positions of \textit{other} walkers in the ensemble.
                \item This allows the whole group to learn about the shape of the posterior (e.g., its correlations) and explore it more efficiently.
            \end{itemize}
        \end{block}
        }
        \only<6-9>{
        \vspace{-1em}
        \begin{block}{Advantages}
            \begin{itemize}
                \item Much better at exploring correlated, ``banana-shaped'' parameter spaces.
                \item More efficient ``mixing'' than a single chain.
                \item Easy to parallelize (one walker per CPU).
            \end{itemize}
        \end{block}
        \begin{block}{Limitation}
            \begin{itemize}
                \item Ensemble can still get trapped in one mode if other modes are very far away.
            \end{itemize}
        \end{block}
        \begin{alertblock}{Key Message}
            Ensemble samplers like \texttt{emcee} are a major improvement for many problems, especially those with parameter degeneracies.
        \end{alertblock}
        }
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_emcee.pdf}%
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_emcee.pdf}%
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_emcee.pdf}%
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_emcee.pdf}%
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_emcee.pdf}%
            \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_emcee.pdf}%
            \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_emcee.pdf}%
            \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_emcee.pdf}%
            \includegraphics<9>[width=\textwidth,page=9]{figures/himmelblau_emcee.pdf}%
        \end{center}
    \end{columns}
\end{frame}

% --- SLIDE 7: Why Evidence Calculation Matters (3 minutes) ---
\begin{frame}
    \frametitle{The Missing Piece: Why Evidence Calculation Matters}
    \framesubtitle{And why it's so hard to compute}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{Why Evidence is Important}
            \begin{itemize}
                \item \textbf{Model Comparison}: Bayes model theorem:
            \[\mathcal{P}(M|D) \propto \mathcal{Z}(D|M) \mathcal{P}(M)\]
            For SED fitting: Which stellar population model best explains the galaxy photometry?
        \item  \textbf{Occam's Razor}: Automatic complexity penalty
            \[\log \mathcal{Z} = \langle \log \mathcal{L} \rangle_{\mathcal{P}} - \mathcal{D}_{\text{KL}}(\mathcal{P}||\pi)\]
        \item \textbf{Bayesian Model Averaging}: Weighted model combinations
            \end{itemize}
        \end{block}
        \column{0.49\textwidth}
        \begin{block}{Why Evidence is Hard}
            \begin{itemize}
                \item The high-dimensional evidence integral:
            \[\mathcal{Z} = \int \mathcal{L}(\theta) \pi(\theta) d\theta\]
            \[\left(\text{from Bayes theorem}:\mathcal{P}(\theta|D) = \frac{\mathcal{L}(\theta) \pi(\theta)}{\mathcal{Z}}\right)\]
            \pause
        \item The difficulty is \textbf{not} that most of parameter space has $\mathcal{L} \approx 0$...
            \pause
        \item The difficulty is that we can't estimate \textbf{volume} $d\theta$ in high dimensions!
            \end{itemize}
        \end{block}
    \end{columns}
\end{frame}

% --- SLIDE 8: Nested Sampling (5 minutes) ---
\begin{frame}
    \frametitle{The State of the Art: Nested Sampling (e.g., \texttt{dynesty})}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \only<1-4>{
        \begin{block}{A Radically Different Approach}
            Instead of random walking, nested sampling attacks the problem from the outside-in.
            \begin{enumerate}
                \item Start with a set of ``live points'' scattered across the entire \keyterm{prior}.
                \item At each step: find the point with the \textit{worst} likelihood and discard it.
                \item Replace it with a new point drawn from the prior, but with a likelihood \textit{better} than the point you just discarded.
                \item This forces the set of live points to continuously ``shrink'' into regions of higher and higher likelihood.
            \end{enumerate}
        \end{block}
        }
        \only<5-8>{
        \begin{block}{Key Advantages}
            \begin{itemize}
                \item Naturally handles \keyterm{multimodality}. The shrinking cloud of points will find and explore all modes simultaneously.
                \item It calculates the \keyterm{Bayesian Evidence} ($\C[3]{\mathcal{Z}}$) as a primary output. This is essential for model comparison!
            \end{itemize}
        \end{block}
        \begin{alertblock}{Key Message}
            Nested sampling excels at exploring complex, multimodal posteriors and is the go-to method for Bayesian model comparison.
        \end{alertblock}
        }
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_ns.pdf}%
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_ns.pdf}%
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_ns.pdf}%
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_ns.pdf}%
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_ns.pdf}%
            \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_ns.pdf}%
            \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_ns.pdf}%
            \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_ns.pdf}%
        \end{center}
    \end{columns}
\end{frame}

% --- SLIDE 9: How Nested Sampling Estimates Volumes (2 minutes) ---
\begin{frame}
    \frametitle{How Nested Sampling Estimates Volumes: The Counting Trick}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \begin{block}{Volume Contraction}
            At each step, the volume contracts predictably:
            \[V_{i+1} = V_i \times \frac{n_{\text{inside}}}{n_{\text{total}}}\]
            indep.\ of dimensionality, geometry or topology
        \end{block}
        
        \begin{columns}[T]
            \column{0.45\textwidth}
            \begin{block}{Evidence}
                The evidence is computed as:
                \[\mathcal{Z} = \sum \mathcal{L}_i \Delta V_i\]
            \end{block}
            
            \column{0.45\textwidth}
            \begin{block}{Posterior}
                Each sample gets importance weight:
                \[w_i = \mathcal{L}_i \times \Delta V_i\]
            \end{block}
        \end{columns}
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_ns_counting_trick.pdf}
            \includegraphics<9>[width=\textwidth,page=9]{figures/himmelblau_ns_counting_trick.pdf}
        \end{center}
        \begin{center}
            \only<1>{\textbf{200 live points}}
            \only<2>{\textbf{Mark for deletion}}
            \only<3>{\textbf{Delete points}}
            \only<4>{\textbf{Repopulate}}
            \only<5>{\textbf{Next iteration}}
            \only<6>{\textbf{Delete again}}
            \only<7>{\textbf{Repopulate again}}
            \only<8>{\textbf{Complete sampling}}
            \only<9>{\textbf{Posterior reweighting}}
        \end{center}
    \end{columns}
\end{frame}

% --- SLIDE 10: Practical Guidance for Nested Sampling (2 minutes) ---
\begin{frame}
    \frametitle{Practical Guidance: How to Use Nested Sampling}
    \framesubtitle{Understanding resolution and reliability parameters}
    \vspace{-1em}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{Rejection Samplers}
            \begin{itemize}
                \item e.g. \texttt{MultiNest}, \texttt{UltraNest}, \texttt{nessai}
                \item Construct bounding regions, reject invalid points
                \item Efficient in low dimensions ($d \lesssim 10$)
                \item Exponentially inefficient in high dimensions
            \end{itemize}
        \end{block}
        \column{0.49\textwidth}
        \begin{block}{Chain-based Samplers}
            \begin{itemize}
                \item e.g. \texttt{PolyChord}, \texttt{dynesty}, \texttt{blackjax}
                \item Run Markov chains from live points
                \item Linear $\sim\mathcal{O}(d)$ scaling penalty
                \item Better for high-dimensional problems
            \end{itemize}
        \end{block}
    \end{columns}
    \vspace{5pt}
    \begin{alertblock}{Key Parameters}
        \begin{itemize}
            \item \textbf{Resolution parameter} $n_{\text{live}}$: Improves results as $\sim\mathcal{O}(n_{\text{live}}^{-1/2})$
            \item \textbf{Reliability parameters}: Don't improve results if set arbitrarily high, but introduce systematic errors if set too low
                \begin{itemize}
                    \item \texttt{MultiNest} efficiency \texttt{eff}, \texttt{PolyChord} chain length \texttt{n\_repeats}, \texttt{dynesty} \texttt{slices}
                \end{itemize}
        \end{itemize}
    \end{alertblock}
\end{frame}

% --- SLIDE 11: Method Comparison (2 minutes) ---
\begin{frame}
    \frametitle{Choosing Your Tool: A Summary}
    \framesubtitle{No single best method, only the right tool for the job}
    \begin{center}
        \begin{tabular}{|l|c|c|c|c|}
            \hline
            \textbf{Method} & \textbf{Speed} & \textbf{Uncertainties?} & \textbf{Handles Multimodality?} & \textbf{Evidence?} \\
            \hline
            \textbf{Optimization} ($\chi^2$) & \textcolor{green!50!black}{Very Fast} & \textcolor{red}{No} & \textcolor{red}{No} & \textcolor{red}{No} \\
            \hline
            \textbf{MCMC} (\texttt{pymc} etc) & \textcolor{orange}{Medium} & \textcolor{green!50!black}{Yes} & \textcolor{orange}{Poorly} & \textcolor{red}{No} \\
            \hline
            \textbf{Ensemble} (\texttt{emcee}) & \textcolor{orange}{Medium} & \textcolor{green!50!black}{Yes} & \textcolor{orange}{Okay} & \textcolor{red}{No} \\
            \hline
            \textbf{Nested} (\texttt{dynesty}) & \textcolor{red}{Slower} & \textcolor{green!50!black}{Yes} & \textcolor{green!50!black}{Excellently} & \textcolor{green!50!black}{Yes!} \\
            \hline
        \end{tabular}
    \end{center}
    \begin{block}{Practical Guidance}
        \begin{itemize}
            \item \textbf{Quick exploration / Sanity check?} $\rightarrow$ Use Optimization.
            \item \textbf{Simple, well-behaved posterior?} $\rightarrow$ \texttt{emcee} is a great choice.
            \item \textbf{Complex, possibly multimodal posterior?} $\rightarrow$ Use \texttt{dynesty}.
            \item \textbf{Need to compare different physical models?} $\rightarrow$ You \textit{must} use Nested Sampling.
        \end{itemize}
    \end{block}
\end{frame}

% --- SLIDE 12: AI in Scientific Code Development (2 minutes) ---
\begin{frame}
    \frametitle{The Future: AI in Scientific Code Development}
    \framesubtitle{How these tools themselves are evolving}
    \vspace{-1em}
    \begin{columns}[T]
        \column{0.49\textwidth}
        \begin{block}{The Real AI Revolution: LLMs}
            The biggest impact of AI will not be in analyzing data, but in helping us write the code to do it.
            \begin{itemize}
                \item \keyterm{Automated code translation}: LLMs can help port legacy Fortran/C++ models to modern, GPU-friendly \& differentiable frameworks like JAX or PyTorch.
            \end{itemize}
        \end{block}
        \column{0.49\textwidth}
        \begin{block}{The 80/20 Rule of Scientific Work}
            \begin{itemize}
                \item \textbf{80\% ``boring'' tasks}: Writing code, debugging, drafting \& reviewing papers, munging data, organising meetings...
                \item \textbf{20\% ``hard thinking''}: The actual scientific insight.
            \end{itemize}
            AI's biggest immediate impact is automating and accelerating the 80\%, freeing up human time for the 20\%.
        \end{block}
    \end{columns}
    \begin{alertblock}{Key Message}
        AI is not just a tool for analysis; it's about to fundamentally change how we develop, optimize, and deploy our science
    \end{alertblock}
\end{frame}

% --- SLIDE 13: Conclusions (1 minute) ---
\begin{frame}
    \frametitle{Conclusions \& What's Next}
    \framesubtitle{\tthref{github.com/handley-lab/group}}
    \tikz[overlay,remember picture]
        \node[anchor=north east] (A) at ($(current page.north east)+(0,0)$) {
        \includegraphics[width=0.06\textheight]{people/adam_ormondroyd.jpg}%
        \includegraphics[width=0.06\textheight]{people/charlotte_priestley.jpg}%
        \includegraphics[width=0.06\textheight]{people/claude.jpg}%
        \includegraphics[width=0.06\textheight]{people/david_yallup.jpg}%
        \includegraphics[width=0.06\textheight]{people/dily_ong.jpg}%
        \includegraphics[width=0.06\textheight]{people/gemini.jpg}%
        \includegraphics[width=0.06\textheight]{people/harry_bevins.jpg}%
        \includegraphics[width=0.06\textheight]{people/metha_prathaban.jpg}%
        \includegraphics[width=0.06\textheight]{people/ming_yang.jpg}%
        \includegraphics[width=0.06\textheight]{people/namu_kroupa.jpg}%
        \includegraphics[width=0.06\textheight]{people/openai.jpg}%
        \includegraphics[width=0.06\textheight]{people/sam_leeney.jpg}%
        \includegraphics[width=0.06\textheight]{people/sinah_legner.jpg}%
        \includegraphics[width=0.06\textheight]{people/toby_lovick.jpg}%
        \includegraphics[width=0.06\textheight]{people/wei-ning_deng.jpg}%
        \includegraphics[width=0.06\textheight]{people/will_handley.jpg}%
        \includegraphics[width=0.06\textheight]{people/will_templeton.jpg}%
    };
    \begin{block}{Key Takeaways}
        \begin{itemize}
            \item SED fitting is a problem of \keyterm{statistical inference}, not just optimization.
            \item The goal is the full \keyterm{posterior distribution}, which gives us parameters \textit{and} their uncertainties.
            \item \keyterm{Sampling} methods are the tools we use to map out the posterior.
            \item The choice of sampler—from MCMC to Ensemble to Nested—depends on the complexity of your problem and whether you need to do \keyterm{model comparison}.
        \end{itemize}
    \end{block}
    \begin{alertblock}{Next Up: David Yallup on ``GPU Accelerated Nested Sampling''}
        Now that we know \textit{why} nested sampling is so powerful, we'll hear about how to make it \textit{fast}!
    \end{alertblock}
\end{frame}

% --- APPENDIX: HMC Sampling ---
\appendix

\begin{frame}
    \frametitle{Appendix: Hamiltonian Monte Carlo (HMC)}
    \begin{columns}[T]
        \column{0.5\textwidth}
        \only<1-5>{
        \begin{block}{How it Works}
            Uses gradients to guide exploration more efficiently than random walks.
            \begin{enumerate}
                \item Treat parameters as ``particles'' with position and momentum.
                \item Use gradient of log-likelihood as ``force'' to guide movement.
                \item Propose coherent moves along gradient directions.
                \item Accept/reject using Metropolis criterion.
            \end{enumerate}
        \end{block}
        }
        \only<6-9>{
        \begin{block}{Advantages \& Requirements}
            \begin{itemize}
                \item Much more efficient than random walk for smooth posteriors.
                \item Requires gradients of the likelihood function.
                \item Can traverse parameter space much faster.
                \item Less likely to get stuck in local regions.
            \end{itemize}
        \end{block}
        \begin{alertblock}{Key Message}
            HMC leverages gradient information for efficient sampling, but requires differentiable models.
        \end{alertblock}
        }
        \column{0.5\textwidth}
        \vspace{-1.8em}
        \begin{center}
            \includegraphics<1>[width=\textwidth,page=1]{figures/himmelblau_hmc.pdf}%
            \includegraphics<2>[width=\textwidth,page=2]{figures/himmelblau_hmc.pdf}%
            \includegraphics<3>[width=\textwidth,page=3]{figures/himmelblau_hmc.pdf}%
            \includegraphics<4>[width=\textwidth,page=4]{figures/himmelblau_hmc.pdf}%
            \includegraphics<5>[width=\textwidth,page=5]{figures/himmelblau_hmc.pdf}%
            \includegraphics<6>[width=\textwidth,page=6]{figures/himmelblau_hmc.pdf}%
            \includegraphics<7>[width=\textwidth,page=7]{figures/himmelblau_hmc.pdf}%
            \includegraphics<8>[width=\textwidth,page=8]{figures/himmelblau_hmc.pdf}%
            \includegraphics<9>[width=\textwidth,page=9]{figures/himmelblau_hmc.pdf}%
        \end{center}
    \end{columns}
\end{frame}

\end{document}

```

`suhail_notes.md`:

```md
- 30-40 minute presentation
- Masters, PhD, Postdocs and faculty in attendance
    - it may be worth going through some pdeagagical material on what sampling is and why it is important for astronomy

```