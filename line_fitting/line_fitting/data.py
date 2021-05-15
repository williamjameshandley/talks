import numpy 

# Polynomial function
def f(x, theta):
    return sum(t*x**i for i, t in enumerate(theta))


# The true answer
def f_true(x):
    return f(x,[1,0,0,1])

numpy.random.seed(0)
x_min = 0
x_max = +1
n = 100
sigma_min = 0.1
sigma_max = 0.5
sigma = numpy.random.uniform(low=sigma_min, high=sigma_max, size=n)
x = numpy.random.uniform(low=x_min, high=x_max, size=n)
y = numpy.random.normal(loc=f_true(x), scale=sigma)

def poly(root):
    letters = 'abcd'
    terms = []
    for i, c in enumerate(root):
        if c=='1':
            if i==0:
                terms.append('%s' % (letters[0]))
            elif i==1:
                terms.append('%sx' % (letters[0]))
            else:
                terms.append('%sx^%i' % (letters[0],i))
            letters = letters[1:]
    return '$' + '+'.join(terms) + '$'

def label_axes(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')


def plot_function(ax, f=f_true):
    x_ = numpy.linspace(x_min,x_max,100)
    ax.plot(x_, f(x_))

def plot_points(ax, errors=True, n=None):
    x_ = numpy.linspace(x_min,x_max,100)
    if errors:
        ax.errorbar(x[:n], y[:n], yerr=sigma[:n], fmt='.', color='k',capthick=0.1,markersize=0,linewidth=0.1)
    else:
        ax.plot(x[:n], y[:n], 'k.')

def plot_diff(ax, f=f_true, n=None):
    for i, (xi, yi) in enumerate(zip(x,y)):
        if n is not None and i>=n:
            break
        ax.plot([xi,xi],[yi,f(xi)],'k-')
    plot_points(ax, errors=False, n=n)
    plot_function(ax, f)
