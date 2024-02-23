import matplotlib.pyplot as plt

#scale=1, length=100, randomness=2
plt.xkcd()


width=5.74686
height=3.48863
pad_inches=0.2

plt.rcParams['figure.figsize'] = (width/2, width/2)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.size'] = 11.4
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['savefig.transparent'] = True
