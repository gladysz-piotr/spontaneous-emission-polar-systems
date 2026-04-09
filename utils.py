import pickle
import numpy as np
from scipy.special import jv as J
from scipy.special import laguerre as L
from sympy import symbols, nsolve, besselj, Eq, I
from os.path import exists
from matplotlib import pyplot as plt
from math import factorial as fac
from numpy.linalg import norm
from tqdm import tqdm

def savePickle(file_path, data):
    """
    Function used to save results as a picle.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    f.close
    return "Saved."


def openPickle(file_path):
    """
    Function used to load results saved as a picle.
    """
    with open(file_path, "rb") as f:
        var = pickle.load(f)
        f.close
    return var


def energy_levels_show(w_0, w_l, N):
    """
    Plots energy ladder for the case os longitudinal coupling.
    """
    g_energies = [-0.5 * w_0 + i * w_l for i in range(N)]
    e_energies = [0.5 * w_0 + i * w_l for i in range(N)]
    g_labels = [fr'$|g\rangle|{i}\rangle_-$' for i in range(N)]
    e_labels = [fr'$|e\rangle|{i}\rangle_+$' for i in range(N)]

    fig, ax = plt.subplots(figsize=(4, 6))
    for i, (E, label) in enumerate(zip(g_energies, g_labels)):
        ax.hlines(y=E, xmin=-0.5, xmax=-0.05, color='green', linewidth=2)
        ax.text(-0.85, E, label, va='center', fontsize=12)
    for i, (E, label) in enumerate(zip(e_energies, e_labels)):
        ax.hlines(y=E, xmin=0.05, xmax=0.5, color='blue', linewidth=2)
        ax.text(0.55, E, label, va='center', fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(min(g_energies)-w_l, max(e_energies) + w_l)
    ax.set_xticks([])
    ax.set_ylabel(r'$\omega_0$')
    ax.set_title(fr'Energy levels for $\omega_0={w_0}$, $\omega_l={w_l}$')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.tight_layout()
    plt.show()


def matrix_element(bra, ket, g):
    '''
    bra, ket are the tuples of the form (n, +/-) and indicates the + or - part of the Hamilotnian,
    while n is the oscillator number
    '''
    if bra == ket:
        return 1
    elif bra[1] == ket[1]:
        return 0
    else:
        l = bra[0]
        n = ket[0]
        if bra[1] == "+":
            sign = (-1)**n
        else:
            sign = (-1)**l
        return np.sqrt(float(fac(n)*fac(l))) * np.exp(-2*g**2) * sign\
            * sum([
                (-1)**k * (2*g)**(n + l - 2*k) /
                (fac(int(k)) * fac(int(l - k)) * fac(int(n - k)))
                for k in range(0, np.min([n, l]) + 1)
            ])


def matrix_elements_show(ns, w_l, g_z, ranges):
    """
    Bar plots of the <n_+|n_-> matrix elements for given n_+s and given range of n_-s.
    """
    rows = int(np.floor(np.sqrt(len(ns))+0.5))
    cols = int(np.ceil(np.sqrt(len(ns))))

    fig, axs = plt.subplots(rows, cols, figsize=(16,9))

    psis = [(n, "+") for n in ns]
    vals = [[matrix_element((k, "-"), psi, g_z/w_l)**2 for k in ranges[i]] for i, psi in enumerate(psis)]

    [axs[i//3, i%3].bar(x, y) for i, x, y in zip(range(6), ranges, vals)]
    for ax, psi, val in zip(axs.flat, psis, vals):
        ax.set_title(fr"$|{{}}_-\langle k|{psi[0]}\rangle_{psi[1]}|^2$")
        ax.set_xlabel("k")
        print(f"{psi}: {np.round(sum(val),3)}")

    fig.suptitle(fr"$|{{}}_-\langle k|n\rangle_+|^2$ for $\Omega_z/\omega_l={np.round(g_z/w_l,5)}$")
    plt.tight_layout()
    plt.show()


def alpha_matrix_element(bra, alpha, g):
    """
    <\alpha|n_+-> matrix element for given bra = (n, +-).
    """
    n = bra[0]
    if bra[1] == "+":
        sign = 1
    else:
        sign = -1
    return np.sqrt(float(fac(n))) * np.exp(-0.5*(norm(alpha)**2 + g**2 + 2*sign*alpha*g))\
        * sum([
            sign**(n-k) * g**(n-k) * alpha**k / (fac(int(k))*fac(int(n-k)))
            for k in range(n+1)
        ])


def alpha_matrix_elements_show(alphas, ns, w_l, g_z, ranges, pm):
    """
    Bar plots of the <\alpha|n_+-> matrix elemetns for given coherents stets in alphas, given n range,
    and given sign pm.
    """
    rows = int(np.floor(np.sqrt(len(ns))+0.5))
    cols = int(np.ceil(np.sqrt(len(ns))))

    fig, axs = plt.subplots(rows, cols, figsize=(16,9))

    vals = [
        [
            norm(alpha_matrix_element((k, pm), alpha, g_z/w_l))**2 for k in ranges[i]
        ] for i, alpha in enumerate(alphas)
    ]

    [axs[i//3, i%3].bar(x, y) for i, x, y in zip(range(6), ranges, vals)]
    for ax, alpha, val in zip(axs.flat, alphas, vals):
        ax.set_title(fr"$|{{}}_{pm}\langle k|{np.round(alpha,5)}\rangle|^2$")
        ax.set_xlabel("k")
        print(f"{alpha}: {np.round(sum(val),3)}")

    fig.suptitle(fr"$|{{}}_{pm}\langle n|\alpha\rangle|^2$ for $\Omega_z/\omega_l={np.round(g_z/w_l,5)}$")
    plt.tight_layout()
    plt.show()


def g_n(n, g_z, w_l):
    """
    Returns effective coupling to the reservoir for the nth transition
    """
    return (1/np.sqrt(fac(int(n))) * np.exp(-0.5 * (g_z/w_l)**2) * (g_z/w_l)**n)


def energy_shift(w_0, w_l, N):
    fig, ax = plt.subplots(figsize=(6, 4))

    g_z = np.linspace(0, 2., 1000)

    y = sum([g_n(n, g_z, w_l)**2 for n in range(0, N, 1)])
    bessel = J(0, g_z/w_l)**2

    ax.plot(g_z/w_l, y, label="exp")
    ax.plot(g_z/w_l, bessel, label="bessel")

    ax.set_xlabel(r"$g_z$/$\omega_l$")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_ylim((0, 1.1))
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()


def laguerre_to_bessel_comparison():
    fig, ax = plt.subplots(figsize = (10, 8))

    g_z = np.linspace(0, 2., 1000)
    l_w = 2

    for n in [100, 1000, 10000]:
        x = g_z**2 / (4 * n)
        y_bessel = np.exp(0.5 * x) * J(0, 2*np.sqrt(n*x))
        y_laguerre = L(n)(x)
        label_bessel = f"J_0; n={n}"
        label_laguerre = f"L_{n}"
        ax.plot(g_z, y_bessel, linewidth=l_w, label=label_bessel)
        ax.plot(g_z, y_laguerre, linewidth=l_w, linestyle="--", label=label_laguerre)
    ax.grid()
    ax.legend()
    plt.show()


def displaced_Fock_matrix_element(N, n, n_prime, alpha):
    # ~ n_state = tensor(basis(N, n), qeye(2))
    # ~ n_prime_state = tensor(basis(N, n_prime), qeye(2))
    n_state = basis(N, n)
    n_prime_state = basis(N, n_prime)
    return (n_state.dag() * displace(N, alpha) * n_prime_state). norm()


def gamma_rate(w, d_eg, N, n, n_prime, alpha):
    D_nn_prime = displaced_Fock_matrix_element(N, n, n_prime, alpha)
    return D_nn_prime**2 * w**3 * d_eg**2

if __name__ == "__main__":
    from params import *

    n = 80
    num_range = np.array((range(82, 59, -1)))

    g_zs = np.array([0.01, 0.1, 0.2, 0.5, 1.0])

    fig, ax = plt.subplots(len(g_zs), figsize=(15, 8))

    color = "blue"
    color_CR = "green"
    for j, g_z in tqdm(enumerate(g_zs)):
        gammas = np.zeros(len(num_range))
        gammas_CR = np.zeros(len(num_range))

        for i, n_prime in enumerate(num_range):
            res_w = w_atom + (n - n_prime) * w_cavity
            res_w_CR = -w_atom + (n - n_prime) * w_cavity

            if res_w >= 0:
                gammas[i] = gamma_rate(res_w, 1., N, n, n_prime, g_z/w_cavity)
            if res_w_CR >= 0:
                gammas_CR[i] = gamma_rate(res_w_CR, 1., N, n, n_prime, -g_z/w_cavity)


        ax[j].bar(num_range+0.16, gammas, width = 0.3, color = color, label=r"emission $\Gamma$")
        ax[j].bar(num_range-0.16, gammas_CR, width = 0.3, color = color_CR, label=r"absorption $\Gamma_{CR}$")
        ax[j].grid()
        ax[j].set_title(fr"$g_z/\omega_L={np.round(g_z/w_cavity, 3)}$")
        ax[j].set_ylabel(r"$\Gamma$")
        ax[j].legend()


    ax[-1].set_xlabel("Fock state n'")
    plt.tight_layout()
    plt.show()
