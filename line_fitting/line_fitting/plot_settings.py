import matplotlib

def  set_rcparams (fontsize = 10,  beamer=False):

    params  =  {
        # 'backend': 'ps',
        'axes.labelsize' :  fontsize ,
        'font.size' :  fontsize ,
        'legend.fontsize' :  fontsize ,
        'axes.titlesize' :  fontsize ,
        'xtick.labelsize' :  fontsize ,
        'ytick.labelsize' :  fontsize ,
        'text.usetex' :  True ,
        'font.family' :  'serif' ,
        'font.serif' :  'Computer Modern Roman' ,
        'font.sans-serif' :  'Computer Modern Roman' ,
        'ps.usedistiller' :  'xpdf' }

    if  beamer :
        params['font.family']  =  'sans-serif'
        preamble = r'\usepackage [cm]{sfmath}'
        matplotlib.rc('text.latex', preamble=preamble)

    matplotlib.rcParams.update( params )

set_rcparams(beamer=True)

params = {
        'backend': 'pdf',
          'errorbar.capsize':1,
          }
matplotlib.rcParams.update(params)

#print(matplotlib.rcParams.keys())
