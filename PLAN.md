# LMU 2025 Talk Plan

## Title: Next-Generation Model Comparison for Primordial Cosmology

**Duration**: 15 minutes  
**Slides**: 7 maximum  
**Audience**: LMU-Cambridge partnership, focus on collaboration opportunities

## Core Take-Home Messages

1. **GPU ≠ ML** - GPUs are massively parallel compute, not just for neural networks
2. **Modern languages have dual power** - Gradients AND parallelization are independent benefits (JAX, PyTorch, Julia)
3. **Classical algorithms on GPU beat ML** - Gold rush to ML misguided; classical algorithms on GPU are competitive when you factor in training costs and retain statistical guarantees
4. **AI as development accelerator** - The only practical way to handle the massive porting effort to GPU-native code

## Slide Structure

### Slide 1: Cosmological Context
**Source**: IOA 2024 talk - "Beginning the golden age of astronomy data" slide
- Next-generation data rates: SO/CMB-S4, Euclid, LSST/Rubin, DESI
- Telescope mosaic showing the scale
- Key point: Statistical weight demands rigorous systematics attention
- Sets stakes for computational challenge

### Slide 2: The GPU Revolution  
**Key message**: GPU ≠ ML
- HPC transitioning to GPU-first due to ML's influence on hardware development
- Show hardware trajectory: exascale machines, AI accelerators becoming standard HPC
- Traditional CPU-bound codes becoming obsolete
- Computational landscape fundamentally changing
- But: GPU capabilities extend far beyond ML (primary takeaway)

### Slide 3: Modern Languages - Dual Benefits
**Key message**: JAX/PyTorch/Julia good for TWO independent reasons
- Gradients (automatic differentiation)
- GPU parallelization  
- These are separate capabilities that happen to coexist
- Traditional physics benefits from both

### Slide 4: Classical Algorithms on GPU Beat ML
**Key message**: Gold rush to ML algorithms, but classical algorithms on GPU are competitive
- Show Toby's nested sampling results: 300x (CMB), >1000x (cosmic shear) speedups
- **Key point**: When you factor in ML training costs, classical GPU algorithms often win
- Classical methods retain statistical guarantees (proper uncertainties, evidence computation)
- ML: flashy but expensive training + black box
- Classical on GPU: transparent, statistically rigorous, often faster end-to-end
- **Perfect demonstration that GPU ≠ ML and classical > ML rush**

### Slide 5: AI-Accelerated Development
**Key message**: Only way to handle the porting effort
- Moving to GPU is massive amount of work
- AI code generation as enabler
- Our group's wholehearted embrace of this approach
- Show development workflow/paradigm

### Slide 6: Cambridge-LMU Synergies
- Cambridge CMB expertise + LMU galaxy survey strengths
- Cross-correlation opportunities enabled by our GPU framework
- Specific collaboration areas our pipeline enables
- Bridge to existing partnership themes

### Slide 7: Collaboration Opportunities
- Next-generation CMB experiments (CMB-S4, etc.)
- Joint project proposals
- What becomes computationally feasible with GPU-native approach
- Partnership potential and next steps

## Supporting Context Materials

### Available in `context/` directory:
- **desi3.pdf/txt** - Adam's latest DESI3 draft 
- **prathaban_handley_2509.04336/** - Latest Prathaban & Handley arXiv paper on GPU GW inference
- **metha_camgw_inference_techniques.tex** - Metha's GW inference work
- **toby.tex** - Toby's CMB/cosmic shear work (key for slide 4)

### Key Papers/Results to Reference:
- Prathaban et al. GPU GW inference: 20-40x speedups
- Toby's work: 300x (CMB), >1000x (cosmic shear) speedups  
- Your JAX-based complete inference pipeline
- BlackJAX framework integration

## LMU-Cambridge Partnership Context
- **Established**: 2017, 5-year funding cycles
- **Focus areas**: Cross-correlation CMB + galaxy surveys, cosmological parameter estimation, simulation-based inference
- **Recent workshop**: September 2024, next planned September 2025
- **Funding**: Regular calls for joint projects
- **Strategic significance**: World-leading collaboration in theoretical cosmology

## Timing
- ~2 minutes per slide
- Emphasis on collaboration and partnership opportunities
- Leave time for questions and discussion of joint projects