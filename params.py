import numpy as np
from scipy.special import gammaln
from scipy.special import jv as J
from qutip import *
from math import factorial as fac


# physical parameters
hbar        = 1.0               # for simplicity set to 1
w_0         = 1.0               # two-level atom frequency
Gamma_0     = 0.01              # atom dissipation rate
kappa_0     = 0.0               # cavity mode dissipation rate
n_th_a      = 0.0               # avg number of thermal bath excitation

# numerical parameters
N           = 150               # number of cavity fock states

# operators
#atom
s_z         = tensor(sigmaz(), qeye(N))                     # sigma_z atom operator
s_m         = tensor(sigmam(), qeye(N))                     # anihilation atom operator
ee          = tensor(0.5 * (sigmaz() + qeye(2)), qeye(N))   # excited state atom operator
gg          = tensor(-0.5 * (sigmaz() - qeye(2)), qeye(N))  # ground state atom operator
#field
b           = tensor(qeye(2), destroy(N))                   # anihilation field operator
def D_L(alpha):                                             # displacement operator
    a = destroy(N)
    return (alpha * a.dag() - alpha.conjugate() * a).expm()
# atom-field
def H_0(w_0, w_L, Om_a, Om_x):                              # quantum Hamiltonian Schrodinger picture
    return (
          0.5 * hbar * w_0 * s_z
        + hbar * w_L * b.dag() * b
        + 0.25 * hbar * (Om_a * b.dag() + Om_a.conjugate() * b) * s_z
        + 0.5 * hbar * (s_m.dag() + s_m) * (Om_x * b.dag() + Om_x.conjugate() * b)
    )

# losses
def Gamma(init, final, w_atom, w_laser, coupling, option):
    """
    Input:
    init, final: (str, int); tuples with two values.
        str: "e" or "g" indicating excited or ground state of the atomic system
        int: integer indicating number of laser photons
    w_atom: float; value of atomic transition frequency
    w_laser: float; value of laser frequency
    coupling: complex; longitudinal coupling strength between atom and laser
    option: "quantum" or "classic", where for "quantum" exact formula is used
        while for "classic" the approximated formula for large "n_init" is used

    Output:
    float; gamma rate / gamma_0 without PDMs for given transition init -> final.
    If transition is forbidden value 0.0 is returned.
    """

    if init[0] == "e":
        sign = +1
    elif init[0] == "g":
        sign = -1
    else:
        raise ValueError

    if init[0] != final[0]:
        n_init = int(init[1])
        n_final = int(final[1])

    if n_final <= n_init + sign * w_atom/w_laser and n_init>=0 and n_final>=0:
        z = np.abs(coupling)/(2*w_laser)

        if option == "quantum_small":
            log_coeff = (
                + 0.5*(gammaln(n_init+1) + gammaln(n_final+1))
                - 0.5*z**2
            )
            coeff = (sign)**n_init * (-1 * sign)**n_final * np.exp(log_coeff)
            summ = sum([
                ((-1)**k / (fac(k) * fac(n_init - k) * fac(n_final - k))) * (np.abs(coupling) / (2 *   w_laser))**(n_init + n_final - 2*k) for k in range(0, min(n_init, n_final) + 1)
            ])
            prefactor = ((sign * w_atom + (n_init - n_final) * w_laser) / w_atom)**3
            rate = prefactor * (coeff * summ)**2

        elif option == "quantum_large":
            ln_C = -0.5 * np.abs(z)**2 + 0.5 * (gammaln(n_init+1) + gammaln(n_final+1))

            Kmax = min(n_init, n_final)
            K = np.arange(0, Kmax+1)

            ln_Sk_vec = (
                  (n_init + n_final - 2*K) * np.log(z)
                - gammaln(K + 1)
                - gammaln(n_init - K + 1)
                - gammaln(n_final - K + 1)
            )

            ln_Sk_max = np.max(ln_Sk_vec)
            shifted_Sk = np.exp(ln_Sk_vec - ln_Sk_max)
            pos_sum = np.sum(shifted_Sk[::2])
            neg_sum = np.sum(shifted_Sk[1::2])
            ln_Sk_sum = ln_Sk_max + np.log(np.abs(pos_sum - neg_sum))

            prefactor = ((sign * w_atom + (n_init - n_final) * w_laser) / w_atom)**3
            ln_G = 2 * (ln_C + ln_Sk_sum)
            rate = prefactor * np.exp(ln_G)

        elif option == "classic":
            p = n_init - n_final
            arg = np.abs(coupling) * np.sqrt(n_init) / w_laser
            prefactor = ((sign * w_atom + p * w_laser) / w_atom)**3
            rate = prefactor * J(p, arg)**2

        elif option == "classic_optimal":
            p_max = 1000
            G_g = sum([
                J(p, coupling * np.sqrt(n_init) / w_laser)**2 * (p * w_laser / w_atom - 1)**3
                for p in range(int(w_atom/w_laser), p_max+1, 1)
            ])
            G_e = 1 + 1.5 * (coupling / w_atom)**2 * n_init + G_g
            return (G_e, G_g)

        return (rate/prefactor, rate)

    return (0.0, 0.0)

c_ops = []
rate = kappa_0 * (1 + n_th_a)  # cavity relaxation
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)
rate = kappa_0 * n_th_a       # cavity excitation if temperature > 0
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())
rate = Gamma_0                # qubit relaxation
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * s_m)
########################################################################

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    fig, ax = plt.subplots(1, figsize=(8, 5))

    G_large = []
    G_small = []
    G_class = []
    ns = np.arange(0, 100+1)
    w_L = 1.0
    Om_a = .5
    for n in tqdm(ns):
        n = int(n)
        init = ["e", n]
        final = ["g", n-3]
        try:
            G_small.append(Gamma(init, final, w_0, w_L, Om_a, "quantum_small"))
        except:
            print(f"Final n={n}")
        G_large.append(Gamma(init, final, w_0, w_L, Om_a, "quantum_large"))
        G_class.append(Gamma(init, final, w_0, w_L, Om_a, "classic"))

    ax.plot(np.sqrt(ns), G_small, "o", label="small")
    ax.plot(np.sqrt(ns), G_large, "-", label="large")
    ax.plot(np.sqrt(ns), G_large, "--", label="classic")

    ax.set_xlabel(r"$\sqrt{n}$")
    ax.set_ylabel(r"$\Gamma/\Gamma_0$")
    ax.legend()
    plt.show()
