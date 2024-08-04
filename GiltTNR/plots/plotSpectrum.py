import argparse, glob, os, re
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Parsing the arguments
parser = argparse.ArgumentParser(
    description='Script plotting tensor spectra as a function of iterations'
)
parser.add_argument('-d', '--dirs',
                    help='directories with log files (default: ./)',
                    type=lambda x: tuple(map(str, x.split())))
parser.add_argument('-o', '--out',
                    help='output file (default: No output, but show the plot)',
                    default='show')
parser.add_argument('--sings',
                    help='maximum number of sing. vals. to show (default: 30)',
                    type=np.int_, default=30)
parser.add_argument('--paper1',
                    help='uses parameters needed to reproduce Fig. 6 top (default: False)',
                    action="store_true")
parser.add_argument('--paper2',
                    help='uses parameters needed to reproduce Fig. 6 bottom (default: False)',
                    action="store_true")
args = parser.parse_args()

# Overwrites parameters when reproducing Fig. 6
if args.paper1:
    args.dirs  = ["logs/spectra_trg"]
    args.sings = 60
    args.out   = "plots/tensor_spectra_1.pdf"

elif args.paper2:
    args.dirs  = ["logs/spectra_gilt"]
    args.sings = 60
    args.out   = "plots/tensor_spectra_2.pdf"

# Find value of the variable in a given output file
def findExpression(expr, filename, parseas=float):
    file = open(filename, 'r')
    values = re.findall(expr, file.read())
    file.close()
    return [parseas(x) for x in values]

def findValue(var, filename, parseas=float):
    expr = var+'[\s=]*([0-9.\-e]+)'
    result = findExpression(expr, filename, parseas=parseas)
    return result

# Find the log files in the directories specified
filenames = []
for d in args.dirs:
    filenames += list(glob.glob(os.path.join(d, '*.log')))

# Creates space for subplots
fig = plt.figure(num=1, figsize=(8.16,7.06*6./20.))
n_subs = len(filenames)
ax=[]
ax.append(fig.add_subplot(1,5,1))
for i in range(2, n_subs+1):
    ax.append(fig.add_subplot(1,n_subs,i, sharey = ax[0]))

# Font sizes
smallsize = 9
footnotesize = 8

# Read the files.
outputs = []
betas = []
for name in filenames:
    print(name)
    try:
        beta = findValue('beta', name)[0]
        with open(name, 'r') as f:
            ls = f.readlines()
        outputs.append(ls)
        betas.append(beta)
    except IndexError:
        print("Trouble parsing output. Possible crash?")

# Parse the outputs
for sub, (beta, ls) in enumerate(sorted(zip(betas, outputs), reverse=True)):

    # Remove the timestamps and other header stuff and join to one string
    # without line breaks.
    dateformat = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    timeformat = "[0-9]{2}:[0-9]{2}:[0-9]{2}"

    for l in ls:
        match = re.search("^" + dateformat + " " + timeformat + "[^:]*: ", l)
        if match is not None:
            headlength = len(match.group(0))
            break

    s = ''.join(l[headlength:] for l in ls)

    # Find numpy arrays and plot them.
    matches = re.findall("pectrum[^\n]*\n\s*(\[[0-9,. e\-+\n]*\])", s)
    max_iter = len(matches)
    result_matrix = np.zeros((args.sings, max_iter))
    for i, m in enumerate(matches):
        m = m[2:-1].replace('\n', ' ').replace('  ', ' ')
        arr = np.fromstring(m, sep=' ')
        arr /= np.max(arr)
        arr = arr[:args.sings]
        result_matrix[:len(arr),i] = arr

    # Defines colors
    red = sns.xkcd_rgb["pale red"]
    gray = sns.xkcd_rgb["charcoal"]
    color = (red if args.paper2 else gray)

    # Produces the plot
    for line in result_matrix:
        ax[sub].plot(line, lw=1.5, mew=0, ms=4, marker="o", color=color, alpha=0.4)

    # Adds the subplot title with temperatures
    if args.paper1:
        T_c = 2./np.log(1.+2.**0.5) 
        T = 1./beta/T_c # temperature in the units of critical temperature
        if np.isclose(T, 1., rtol=1e-10):
            ax[sub].set_title(r"$T = T_c$", fontsize=smallsize)
        else:
            digits = int(np.ceil(-np.log10(np.around(np.abs(T-1.), decimals=14)))) # show only first non-trivial digit
            ax[sub].set_title(r"$T = %.*f\, T_c$" % (digits,T) , fontsize=smallsize)

    # y-axis ticks
    yticks = np.linspace(0.0, 1.0, 6)
    ax[sub].set_yticks(yticks)
    ax[sub].set_yticklabels([r"$%.1f$"%i for i in yticks], fontsize=smallsize)

    # x-axis ticks
    xticks = list(range(0, max_iter-1, 3))
    ax[sub].set_xticks(xticks)
    ax[sub].set_xticklabels([r"$2^{%i}$"%i for i in xticks], fontsize=smallsize)#, y=-0.01)
    ax[sub].set_xlim([0,20])

    if args.paper1:
        plt.setp(ax[sub].get_xticklabels(), visible=False)
    else:
        ax[sub].set_xlabel(r"$\mathrm{Linear\; system\; size}$", fontsize=smallsize)
    
    # Adds x-axis label only to the first subfigure
    if sub==0:
        if args.paper1:
            ax[sub].set_ylabel("$\mathrm{TRG}$\n $\mathrm{Singular\; value\; magnitude}$", fontsize=smallsize)
        elif args.paper2:
            ax[sub].set_ylabel("$\mathrm{Gilt}$-$\mathrm{TNR}$\n $\mathrm{Singular\; value\; magnitude}$", fontsize=smallsize)
        else:
            ax[sub].set_ylabel(r"$\mathrm{Singular\; value\; magnitude}$", fontsize=smallsize)
        ax[sub].set_ylim([-0.07,1.07])
    else:
        plt.setp(ax[sub].get_yticklabels(), visible=False)

    # Adds annotations in the final subfigure
    if args.paper2 and sub==n_subs-1:
        ax[sub].annotate(r"$\mathrm{first\; s.v.}$", xycoords='axes fraction', xy=(0.35, 0.9), textcoords='axes fraction', xytext=(0.5, 0.84),
            arrowprops=dict(arrowstyle='->', facecolor='black'), fontsize=footnotesize)
        ax[sub].annotate(r"$\mathrm{second\; s.v.}$", xycoords='axes fraction', xy=(0.35, 0.76), textcoords='axes fraction', xytext=(0.5, 0.76),
            arrowprops=dict(arrowstyle='->', facecolor='black'), fontsize=footnotesize)

# Shows or saves figure to file
if args.out == "show":
    plt.show()
else:
    fig.subplots_adjust(wspace=0.04)
    plt.savefig(args.out, bbox_inches='tight')
