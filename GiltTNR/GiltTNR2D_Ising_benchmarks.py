import numpy as np
import sys
import os
import logging
import logging.config
import configparser
import datetime
import multilineformatter
import scipy.integrate as integrate
from scipy.optimize import fsolve
from GiltTNR2D import gilttnr_step
from tensors import Tensor, TensorZ2
from ncon import ncon
from yaml_config_parser import parse_argv
from math import exp,sqrt, cosh, sinh

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
"""filename = os.path.basename(__file__).replace(".py", "")"""
np.set_printoptions(precision=10)

default_pars = {
    # Parameters for this test file.
    "beta": np.log(1 + np.sqrt(2)) / 2,
    # TK addition 
    "Jratio": 1.,
    "total_iters": 6,
    "symmetry_tensors": True,
    "print_scaldims": 50,
    "print_spectra": True,
    "print_freeenergy": True,
    "log_dir": "logs",

    # Parameters for Gilt-TNR.
    "gilt_eps": 1e-6,
    "cg_chis": [1,2,3,4,5,6],
    "cg_eps": 1e-10,
    "verbosity": 100
}


def parse():
    pars = parse_argv(sys.argv)
    return pars


def apply_default_pars(pars, default_pars):
    for k, v in default_pars.items():
        if k not in pars:
            pars[k] = v
    return


def set_filehandler(logger, logfilename, pars):
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    filehandler.setLevel(logging.INFO)
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(tools_path + '/logging_default.conf')
    fmt = parser.get('formatter_default', 'format')
    datefmt = parser.get('formatter_default', 'datefmt')
    formatter = multilineformatter.MultilineFormatter(fmt=fmt, datefmt=datefmt)
    filehandler.setFormatter(formatter)
    rootlogger.addHandler(filehandler)
    return


def get_free_energy(A, log_fact, pars, iter_count):
    Z = ncon(A, [1,2,1,2]).value()
    log_Z = np.log(Z) + log_fact
    F = -log_Z/pars["beta"]
    f = F/(2*4**(iter_count))
    return f


def get_exact_free_energy(pars):
    beta = pars["beta"]
    sinh = np.sinh(2*beta)
    cosh = np.cosh(2*beta)
    def integrand(theta):
        res = np.log(cosh**2 +
                     sinh**2 * np.sqrt(1+sinh**(-4) -
                                       2*sinh**(-2)*np.cos(theta)))
        return res
    integral, err = integrate.quad(integrand, 0, np.pi)
    f = -(np.log(2)/2 + integral/(2*np.pi)) / beta
    return f


def get_A_spectrum(A):
    es = A.svd([0,1], [2,3])[1]
    es = es.to_ndarray()
    es /= np.max(es)
    es = -np.sort(-es)
    return es

def get_A_spectrum_flipped(A):
    # added by tk. Grouping of indices if flipped relative to get_A_spectrum
    es = A.svd([1,2], [3,0])[1]
    es = es.to_ndarray()
    es /= np.max(es)
    es = -np.sort(-es)
    return es

def get_A_spectrum_no_norm(A):
    es = A.svd([0,1], [2,3])[1]
    es = es.to_ndarray()
    es = -np.sort(-es)
    return es


def get_scaldims(A, pars):
    logging.info("Diagonalizing the transfer matrix.")
    # The cost of this scales as O(chi^6).
    transmat = ncon((A, A), [[3,-101,4,-1], [4,-102,3,-2]])
    es = transmat.eig([0,1], [2,3], hermitian=False)[0]
    # Extract the scaling dimensions from the eigenvalues of the
    # transfer matrix.
    es = es.to_ndarray()
    es = np.abs(es)
    es = -np.sort(-es)
    es[es==0] += 1e-16  # Ugly workaround for taking the log of zero.
    log_es = np.log(es)
    log_es -= np.max(log_es)
    log_es /= -np.pi
    return log_es






def get_initial_tensor(pars):
    hamiltonian = np.array([[-1, 1], [ 1,-1]])
    boltz = np.exp(-pars["beta"]*hamiltonian)
    A_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)

    u = np.array([[1, 1], [1,-1]]) / np.sqrt(2)
    u_dg = u.T.conjugate()
    A_0 = ncon((A_0, u, u, u_dg, u_dg),
               ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    if pars["symmetry_tensors"]:
        dim, qim = [1,1], [0,1]
        A_0 = TensorZ2.from_ndarray(A_0, shape=[dim]*4, qhape=[qim]*4,
                                    dirs=[1,1,-1,-1])
    else:
        A_0 = Tensor.from_ndarray(A_0)
    return A_0

# TK addition
def get_initial_tensor_aniso(pars):
    """ define NN tensor corresponding to Jv/Jh=Jratio and T = relT * T_c
        For all values of Jratio, relT=1 is the critical point.
        So T_c depends on Jratio. A is not normalized in any way
    """
    Jratio = pars["Jratio"]
    factor = 1./pars["relT"]

    # Jv, Jh are couplings in the vertical/horizontal directions. So coefs
    # in Hamiltonian are beta*Jv and beta*Jh. We want to choose beta_c and
    # Jv, Jh so that the model is critical and the ratio is Jratio. 
    Jv = Jratio
    Jh=1
    
    # The critical point beta_c for couplings Jv,Jh is given by 
    # sinh (2 beta_c Jv) sinh(2 beta_c Jh) =1
    def f(x):
        return sinh(2*x*Jv)*sinh(2*x*Jh) - 1
    betac = fsolve(f,1.)[0]
    
    # the Ising lattice is at 45 degs wrt tensor network. So meaning of
    # vertical and horizontal for Jv,Jh is ambiguous
    hamiltonian = np.array([[-1, 1], [ 1,-1]])
    boltz_v = np.exp(-factor*betac*Jv*hamiltonian)
    boltz_h = np.exp(-factor*betac*Jh*hamiltonian)
    A_0 = np.einsum('ab,bc,cd,da->abcd', boltz_v, boltz_h, boltz_v, boltz_h)

    # do gauge transformation so that legs have definite spinflip quantum number
    u = np.array([[1, 1], [1,-1]]) / np.sqrt(2)
    u_dg = u.T.conjugate()
    A_0 = ncon((A_0, u, u, u_dg, u_dg),
               ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))

    A_0[abs(A_0)<1e-10] =0

    # delete this printing eventually
    if False:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        print("A_0[",i,",",j,",",k,",",l,"]=",A_0[i,j,k,l])

    if pars["symmetry_tensors"]:
        dim, qim = [1,1], [0,1]
        A_0 = TensorZ2.from_ndarray(A_0, shape=[dim]*4, qhape=[qim]*4,
                                    dirs=[1,1,-1,-1])
    else:
        A_0 = Tensor.from_ndarray(A_0)
    return A_0


"""
if __name__ == "__main__":
    pars = parse()
    apply_default_pars(pars, default_pars)

    datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                             '%Y-%m-%d_%H-%M-%S')
    title_str = ('{}_{}_beta{}_cgchis{}_deeps{}'
                 .format(filename, datetime_str, pars["beta"],
                         pars["cg_chis"][-1], pars["gilt_eps"]))

    logfilename = "{}/{}.log".format(pars["log_dir"], title_str)
    rootlogger = logging.getLogger()
    set_filehandler(rootlogger, logfilename, pars)

    # - Infoprint -
    infostr = "\n{}\n".format("="*70)
    infostr += "Running {} with the following parameters:".format(filename)
    for k,v in sorted(pars.items()):
        infostr += "\n%s = %s"%(k, v)
    logging.info(infostr)

    A = get_initial_tensor(pars)
    log_fact = 0

    for iter_count in range(1, pars["total_iters"]+1):
        logging.info("\nIteration {}".format(iter_count))
        A, log_fact = gilttnr_step(A, log_fact, pars)

        if pars["print_spectra"]:
            es = get_A_spectrum(A)
            msg = "Spectrum of A:\n{}".format(es[:30])
            logging.info(msg)

        if pars["print_freeenergy"]:
            f = get_free_energy(A, log_fact, pars, iter_count)
            exact_f = get_exact_free_energy(pars)
            f_error = np.abs(f - exact_f)/exact_f
            msg = ("Free energy per site: {} ({}, off by {:.4e})"
                   .format(f, exact_f, f_error))
            logging.info(msg)

        if pars["print_scaldims"] > 0:
            scaldims = get_scaldims(A, pars)
            scaldims = scaldims[:pars["print_scaldims"]]
            msg = "Scaldims:\n{}".format(scaldims)
            logging.info(msg)
"""
