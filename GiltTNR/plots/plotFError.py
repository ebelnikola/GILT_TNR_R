import argparse, glob, os, re
import numpy as np
import seaborn as sns
import datetime
from matplotlib import pyplot as plt

# Parsing the arguments
parser = argparse.ArgumentParser(
    description='Script plotting error in free energy at criticality'
)
parser.add_argument('-d', '--dir',
                    help='directory to scan (default: ./)',
                    default=os.getcwd())
parser.add_argument('-o', '--out',
                    help='output file (default: No output, but show the plot)',
                    default='show')
parser.add_argument('--maxchi',
                    help='maximum cg_chi to show (default: infinite)',
                    type=np.float_, default=np.inf)
parser.add_argument('-x', '--xaxis',
                    help='the quantity on the x axis (default: cg_chi)',
                    default="cg_chi")
parser.add_argument('--epses',
                    help='gilt_epses to show (default: (), meaning all)',
                    default=(), type=lambda x: tuple(map(float, x.split())))
parser.add_argument('--paper',
                    help='uses parameters needed to reproduce Fig. 5 (default: False)',
                    action="store_true")
args = parser.parse_args()

# Overwrites parameters when reproducing Fig. 5
if args.paper:
    args.dir = "logs/free_energy"
    args.maxchi = 65
    args.xaxis  = "cg_chi"
    args.epses = [-1, 8e-7]
    args.out    = "plots/ferror_vs_chi.pdf"

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

dateformat = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
timeformat = "[0-9]{2}:[0-9]{2}:[0-9]{2}"
timestampformat = dateformat + " " + timeformat

# Parse the interesting parts of the output files into one giant
# dictionary.
result_dict = {}
for filename in glob.glob(os.path.join(args.dir, '*.log')):
    print(filename)
    try:
        beta = findValue('beta', filename)[0]
        cg_chi = findValue('cg_chis = range\(0,', filename, parseas=int)[0] - 1
        gilt_eps = findValue('gilt_eps', filename)[0]
        timestamps = findExpression("\n(" + timestampformat + ")", filename,
                                    parseas=str)
        startstamp = timestamps[0]
        endstamp = timestamps[-1]
        starttime = datetime.datetime.strptime(startstamp, "%Y-%m-%d %H:%M:%S")
        endtime = datetime.datetime.strptime(endstamp, "%Y-%m-%d %H:%M:%S")
        walltime = (endtime - starttime).total_seconds()
        walltime = float(walltime)/60
        f_errors = findValue('off by', filename)
        f_error = f_errors[-1]
        result_dict[beta, cg_chi, gilt_eps] = (f_error, walltime)
    except IndexError:
        print("Trouble parsing output. Possible crash?")

# Pick from the dictionary the things we want to plot, in more plottable
# data structures.
dict_by_eps = {}
for key, value in result_dict.items():
    beta, cg_chi, gilt_eps = key
    f_error, walltime = value
    chi_list, error_list, walltime_list = dict_by_eps.get(gilt_eps, ([], [], []))
    chi_list.append(cg_chi)
    error_list.append(f_error)
    walltime_list.append(walltime)
    dict_by_eps[gilt_eps] = chi_list, error_list, walltime_list

if args.epses==():
    dict_by_eps = {k: v for k, v in dict_by_eps.items()}
else:
    dict_by_eps = {k: v for k, v in dict_by_eps.items() if k in args.epses}

# For the single columns size of PR
fig = plt.figure(num=1, figsize=(3.40457,3.40457*3./4.))
ax = fig.add_subplot(111) 

# Font sizes
smallsize = 9
footnotesize = 7

# Reproduces Fig. 5 from the paper
if args.paper:

    # Uses red and blue colors for 2D and 3D data respectively
    red = sns.xkcd_rgb["pale red"]
    gray = sns.xkcd_rgb["charcoal"]
    colors = [gray, red]
    labels = [r"$\mathrm{TRG}$", r"$\mathrm{Gilt}$-$\mathrm{TNR}$"]
    markers = ['s', 'o']

    # Produces the plot
    for gilt_eps, col, lab, mark in zip(sorted(dict_by_eps.keys(), reverse=False), colors, labels, markers):
        chis, errors, walltimes = map(np.array, dict_by_eps[gilt_eps])
        order = np.argsort(chis)
        chis, errors, walltimes = chis[order], errors[order], walltimes[order]
        chis, errors, walltimes = zip(*((c, e, w)
                                        for c, e, w in zip(chis, errors, walltimes)
                                        if c <= args.maxchi))
        errors = np.abs(errors)
        ax.semilogy(chis, errors, marker=mark, lw=1.5, ms=(5 if gilt_eps==-1 else 6), mew=0, color=col, label=lab)
    
        # Adds annotations
        for c, w, e in zip(chis[1:], walltimes[1:], errors[1:]):
            if gilt_eps==-1:
                ax.annotate(r"$%i$"%w, xy=(c,e*1.8), fontsize=footnotesize)
            else:
                if c==65:
                    ax.annotate(r"$%i \mathrm{\,min.}$" % w, xy=(c-3.0,e/3.4), fontsize=footnotesize)
                else:
                    ax.annotate(r"$%i$" % w, xy=(c-1.5,e/3.4), fontsize=footnotesize)

    ax.set_xlim([10, args.maxchi+5])
    ax.set_ylim([1e-11, 1e-4])

else:

    # Sets color cycle
    cm = plt.get_cmap('gist_rainbow')
    n_colors = len(dict_by_eps)
    ax.set_color_cycle([cm(1.*i/n_colors) for i in range(n_colors)])


    # Produces the plot
    for gilt_eps in sorted(dict_by_eps.keys(), reverse=True):
        chis, errors, walltimes = map(np.array, dict_by_eps[gilt_eps])
        order = np.argsort(chis)
        chis, errors, walltimes = chis[order], errors[order], walltimes[order]
        chis, errors, walltimes = zip(*((c, e, w)
                                        for c, e, w in zip(chis, errors, walltimes)
                                        if c <= args.maxchi))
        errors = np.abs(errors)
        if args.xaxis == "cg_chi":
            ax.semilogy(chis, errors, label=gilt_eps, marker='o', mew=0)
            for c, w, e in zip(chis, walltimes, errors):
                ax.annotate(r"$%i$" % w, xy=(c+1,e*1.1), fontsize=footnotesize)
        elif args.xaxis == "walltime":
            ax.loglog(walltimes, errors, label=gilt_eps, marker='o', mew=0)
            for c, w, e in zip(chis, walltimes, errors):
                ax.annotate(r"$%i$" % c, xy=(w,e), fontsize=footnotesize)

# Adds axes labels
if args.xaxis == "cg_chi":
    ax.set_xlabel(r"$\mathrm{Bond\; dimension\;}\chi$", fontsize=smallsize)
elif args.xaxis == "walltime":
    ax.set_xlabel("Wall time, in minutes")
ax.set_ylabel(r"$\mathrm{Relative\; error\; in\; free\; energy\; at\;} T=T_c$", fontsize=smallsize)

# Converts ticks' labels to LaTeX style
fig.canvas.draw()
new_xticklabels = [(r"$%s$" % t.get_text()).replace("$$", "$").replace("\\mathdefault", "") for t in ax.get_xticklabels()]
ax.set_xticklabels(new_xticklabels, fontsize=smallsize)
new_yticklabels = [(r"$%s$" % t.get_text()).replace("$$", "$").replace("\\mathdefault", "") for t in ax.get_yticklabels()]
ax.set_yticklabels(new_yticklabels, fontsize=smallsize)

# Adds the legend
ax.legend(fontsize=smallsize, fancybox=True, loc=0)

# Shows or saves figure to file
if args.out == "show":
    plt.show()
else:
    plt.savefig(args.out, bbox_inches='tight')