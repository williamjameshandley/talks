Please improve this prompt:

"""
I'm writing a latex beamer talk.

Here is an old talk that this one will be loosely based on

```tex
[INSERT old_talk.tex]
```

This should give you an indication of the style:
- minimal overlay
- columns
- bullet points in one column
- figures in the other column
- arxiv citations like \arxiv{1902.04029}

The talk is on the results from DiRAC allocations. I have finished one allocation 'dirac 13', and have now started 'dirac 17'. Here are the tex for the cases for support

DiRAC 13:
```tex
[INSERT dirac_13.tex]
```

DiRAC 17:
```tex
[INSERT dirac_17.tex]
```

Here is a section of my ERC grant that talks about the importance of quantifying tensions, in particular the bias toward LCDM through fiducial assumptions
```tex
[INSERT B2.tex]
```

In addition, here are some papers that survey the background material:

[Quantifying tensions in cosmological parameters: Interpreting the DES evidence ratio](https://arxiv.org/abs/1902.04029)
```tex
[INSERT 1902.04029/R.tex]
```
```bbl
[INSERT 1902.04029/R.bbl]
```

[Quantifying dimensionality: Bayesian cosmological model complexities](https://arxiv.org/abs/1903.06682)
```tex
[INSERT 1903.06682/D.tex]
```
```bbl
[INSERT 1903.06682/D.bbl]
```

[Quantifying Suspiciousness Within Correlated Data Sets](https://arxiv.org/abs/1910.07820)
```tex
[INSERT 1910.07820/correlated.tex]
```
```bbl
[INSERT 1910.07820/correlated.bbl]
```

[Curvature tension: evidence for a closed universe](https://arxiv.org/abs/1908.09139)
```tex
[INSERT 1908.09139/curvature_tension.tex]
```
```bbl
[INSERT 1908.09139/curvature_tension.bbl]
```


Here is an example script on how to use unimpeded
```python
[INSERT example.py]
```

Here is the unimpeded software

[INSERT unimpeded.md]

Here are some of slides I would like to produce, though not necessarily in that order
```tex
[INSERT slides.tex]
```

The talk is half an hour, so I would like to have fewer than 20 slides. Most of them are on the topic of unimpeded, but some of them are adverts for other work relevant to the rest of the audience. Feel free to add more content


Please populate these with content appropriately.
Relevant papers can be gathered from the bbl files. You should cite them in the text as \arxiv{<arxiv_id>}. Search online for the arxiv ids either from the titles, dois or ADS bibcodes.
You should use the papers to populate relevant figures for the background material. Assume I will provide these
Use your own knowledge, grounding and google search to add material
check all material carefully
Please put maximum effort into this. I would like this to showcase LLMs capabilities, which forms a cornerstone of my final few slides on how we are going to get to the next stage of cosmological robustness.
The date of the talk is 2025-05-20
Please produce just slides in beamer format, not a full latex document.
"""

I will paste in material into the prompt where the INSERT statements are, so please leave blocks where I can paste those in full
