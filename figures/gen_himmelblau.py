import numpy as np 
from anesthetic import MCMCSamples, NestedSamples
import tqdm

def loglikelihood(x):
    x, y = x
    return -(x**2+y-11)**2 - (x+y**2-7)**2

ndims = 2
columns = ['theta%i' % i for i in range(ndims)]
tex = {p: r'$\theta_%i$' % i  for i, p in enumerate(columns)}
roots = []

def ns_sim(ndims=2, nlive=125):
    """Brute force Nested Sampling run"""
    low=(-6,-6)
    high=(6,6)
    live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = np.array([loglikelihood(x) for x in live_points])
    live_birth_likes = np.ones(nlive) * -np.inf

    dead_points = []
    dead_likes = []
    birth_likes = []
    for _ in tqdm.tqdm(range(nlive*11)):
        i = np.argmin(live_likes)
        Lmin = live_likes[i]
        dead_points.append(live_points[i].copy())
        dead_likes.append(live_likes[i])
        birth_likes.append(live_birth_likes[i])
        live_birth_likes[i] = Lmin
        while live_likes[i] <= Lmin:
            live_points[i, :] = np.random.uniform(low=low, high=high, size=ndims) 
            live_likes[i] = loglikelihood(live_points[i])
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes

np.random.seed(0)
data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim()

ns = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_ns = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)

# Dead file for polychord
root = './chains/himmelblau'

ns[columns + ['logL', 'logL_birth']].to_csv(root + '_dead-birth.txt', sep=' ', index=False, header=False)
live_ns[columns + ['logL', 'logL_birth']].to_csv(root + '_phys_live-birth.txt', sep=' ', index=False, header=False)

# paramnames file
with open(root + '.paramnames', 'w') as f:
    for p in columns:
        f.write('%s\t%s\n' % (p, tex[p].replace('$','')))

# ranges file
with open(root + '.ranges', 'w') as f:
    f.write('%s\t-6\t6\n' % columns[0])
    f.write('%s\t-6\t6\n' % columns[1])
