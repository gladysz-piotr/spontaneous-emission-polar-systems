import sys
from tqdm import tqdm
from params import *
from utils import energy_levels_show
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib import rcParams


if __name__ == '__main__':
    num = sys.argv[1]

    #####---------- Fig. 1 ----------#####
    if num == "1":
        fontsize = 18
        rcParams.update({'font.size': fontsize})
        linewidth = 3
        fig, axs = plt.subplots(1, 3, figsize=(10, 5.5), sharey=False)
        w_Ls = [0.7, 1.0, 1.3]
        minE = -0.8
        maxE = 3.8
        for i, w_l in enumerate(w_Ls):
            Ng = int((maxE + 0.5 * w_0) / w_l) + 1
            Ne = int((maxE - 0.5 * w_0) / w_l) + 1
            g_energies = [-0.5 * w_0 + _ * w_l for _ in range(Ng)]
            e_energies = [0.5 * w_0 + _ * w_l for _ in range(Ne)]
            g_labels = [fr'$|{_}\rangle_-$' for _ in range(Ng)]
            e_labels = [fr'$|{_}\rangle_+$' for _ in range(Ne)]

            for j, (E, label) in enumerate(zip(g_energies, g_labels)):
                axs[i].hlines(y=E, xmin=-0.5, xmax=-0.05, color='green', linewidth=linewidth)
                axs[i].text(-0.8, E-0.05, label, va='center', fontsize=fontsize)
            for j, (E, label) in enumerate(zip(e_energies, e_labels)):
                axs[i].hlines(y=E, xmin=0.05, xmax=0.5, color='blue', linewidth=linewidth)
                axs[i].text(0.55, E-0.05, label, va='center', fontsize=fontsize)

            axs[i].set_xticks([])
            axs[i].set_title(fr'$\omega_L/\omega_0={w_l}$')
            axs[i].spines[['top', 'right', 'bottom']].set_visible(False)
            axs[i].set_xlim(-0.85, 0.85)
            axs[i].set_ylim(minE, maxE)
        axs[0].set_ylabel(fr'$\omega_0$ [a.u.]', fontsize=fontsize)
        plt.tight_layout()
        plt.show()

    #####---------- Fig. 2 ----------#####
    if num == "2":
        fontsize = 18
        rcParams.update({'font.size': fontsize})
        fig = plt.figure(figsize=(16, 7.5))
        gs = GridSpec(2, 2,
            height_ratios=[1, 1],
            width_ratios=[3,1],
            left=0.1,
            right=1.03,
            top=0.95,
            bottom=0.1,
            hspace=0.38,
            wspace=-0.0
        )

        # Parameters
        top_plots_num   = 2
        Om_top_limits   = [2.0, 2.0]
        w_top_limits    = [2.0, 2.0]
        top_labels      = ["a", "b"]
        top_label_pos   = [0.03, 0.9]

        bot_plots_num   = 4
        Om_bot_limits   = [2.0, 2.0, 2.0, 2.0]
        w_bot_limits    = [1.0, 0.5, 0.33, 1.0]
        bot_labels      = ["c", "d", "e", "f"]
        bot_label_pos   = [0.07, 0.9]

        Om_num          = 401
        w_num           = 401
        opt             = "quantum_small"
        min_w           = 0.01
        vmin            = 0.0
        cmap            = "cividis"

        # ---------------- TOP ROW ----------------
        # Plots setup
        gs_top      = gs[0,0].subgridspec(1, top_plots_num, wspace=0.1)
        axs_top     = [fig.add_subplot(gs_top[0]),]
        for i in range(1,top_plots_num):
            axs_top.append(fig.add_subplot(gs_top[i], sharey=axs_top[0]))
            axs_top[-1].tick_params(labelleft=False)

        # Data grid
        Om_a_vals   = [np.linspace(0.0, Om_top_limits[i], Om_num) for i in range(len(Om_top_limits))]
        w_L_vals    = [np.linspace(min_w, w_top_limits[i], w_num) for i in range(len(w_top_limits))]
        Gammas      = [np.zeros((len(Om_a), len(w_L))) for Om_a, w_L in zip(Om_a_vals, w_L_vals)]
        cs_top      = []

        # Compute top row
        for k in range(top_plots_num):
            for i, om_a in enumerate(tqdm(Om_a_vals[k])):
                for j, w in enumerate(w_L_vals[k]):
                    if k == 0:  # total gamma rate e,0
                        Gammas[k][i, j] = sum(
                            Gamma(("e", 0), ("g", n), w_0, w, om_a, opt)[1]
                            for n in range(int(w_0 / w), -1, -1)
                        )
                    else:
                        Gammas[k][i, j] = Gamma(("e", 0), ("g", 0), w_0, w, om_a, opt)[1]

        # Plot top row
        for k in range(top_plots_num):
            cs_top.append(axs_top[k].pcolormesh(w_L_vals[k], Om_a_vals[k], Gammas[k], shading='auto', cmap=cmap))
            if k == 0:
                axs_top[k].set_title(r"Total rate $\Gamma_{e,0}$")
                axs_top[k].set_ylabel(r"$\Omega_a/\omega_0$")
            else:
                axs_top[k].set_title(r"$\Gamma_{e00}$")
            axs_top[k].set_xlabel(r"$\omega_L/\omega_0$")
            axs_top[k].text(
                top_label_pos[0]*w_L_vals[k][-1],
                top_label_pos[1]*Om_a_vals[k][-1],
                top_labels[k],
                color="white",
                fontsize=fontsize
            )
        #---------------------------------------------

        # ---------------- BOTTOM ROW ----------------
        # Plots setup
        gs_bot = gs[1,0].subgridspec(1, bot_plots_num, wspace=0.25)
        axs_bot = [fig.add_subplot(gs_bot[0]),]
        for i in range(1, bot_plots_num):
            axs_bot.append(fig.add_subplot(gs_bot[i], sharey=axs_bot[0]))
            axs_bot[-1].tick_params(labelleft=False)

        # Data grid
        Om_a_vals   = [np.linspace(0.0, Om_bot_limits[i], Om_num) for i in range(len(Om_bot_limits))]
        w_L_vals    = [np.linspace(min_w, w_bot_limits[i], w_num//2+1) for i in range(len(w_bot_limits))]
        Gammas      = [np.zeros((len(Om_a), len(w_L))) for Om_a, w_L in zip(Om_a_vals, w_L_vals)]
        cs_bot      = []

        # Compute top row
        for k in range(bot_plots_num):
            for i, om_a in enumerate(tqdm(Om_a_vals[k])):
                for j, w in enumerate(w_L_vals[k]):
                    if k == bot_plots_num-1:  # total gamma rate for n'>0
                        Gammas[k][i, j] = sum(
                            Gamma(("e", 0), ("g", n), w_0, w, om_a, opt)[1]
                            for n in range(int(w_0 / w), 0, -1)
                        )
                    else:
                        Gammas[k][i, j] = Gamma(("e", 0), ("g", k+1), w_0, w, om_a, opt)[1]

        # Plot bot row
        vmax = np.max(Gammas[-1])
        for k in range(bot_plots_num):
            cs_bot.append(axs_bot[k].pcolormesh(w_L_vals[k], Om_a_vals[k], Gammas[k], shading='auto', cmap=cmap, vmax=vmax))
            if k == 0:
                axs_bot[k].set_ylabel(r"$\Omega_a/\omega_0$")
            if k == bot_plots_num-1:
                axs_bot[k].set_title(r"$\Sigma_{n^\prime >0}\Gamma_{e0n^\prime}$")
            else:
                axs_bot[k].set_title(fr"$\Gamma_{{e0{k+1}}}$")
            axs_bot[k].set_xlabel(r"$\omega_L/\omega_0$")
            axs_bot[k].text(
                bot_label_pos[0]*w_L_vals[k][-1],
                bot_label_pos[1]*Om_a_vals[k][-1],
                bot_labels[k],
                color="white",
                fontsize=fontsize
            )

        # ---------------- SHARED COLORBAR ----------------
        fig.colorbar(cs_top[0], ax=axs_top, label=r"$\Gamma/\Gamma_0$")
        fig.colorbar(cs_bot[0], ax=axs_bot, label=r"$\Gamma/\Gamma_0$")

        #---------------------------------------------

        # ----------------LADDER---------------------
        linewidth = 3
        gs_left = gs[:, 1]
        ax_left = fig.add_subplot(gs_left)
        w_l     = 0.26
        minE    = -0.6
        maxE    = 0.9
        g_xmin  = -0.5
        g_xmax  = -0.05
        e_xmin  = 0.05
        e_xmax  = 0.5
        Ng = int((maxE + 0.5 * w_0) / w_l) + 1
        Ne = int((maxE - 0.5 * w_0) / w_l) + 1

        g_energies = [-0.5 * w_0 + _ * w_l for _ in range(Ng)]
        e_energies = [0.5 * w_0 + _ * w_l for _ in range(Ne)]
        g_labels = [fr'$|{_}\rangle_-$' for _ in range(Ng)]
        e_labels = [fr'$|{_}\rangle_+$' for _ in range(Ne)]

        for j, (E, label) in enumerate(zip(g_energies, g_labels)):
            ax_left.hlines(y=E, xmin=g_xmin, xmax=g_xmax, color='green', linewidth=linewidth)
            ax_left.text(-0.65, E-0.05, label, va='center', fontsize=fontsize)
        for j, (E, label) in enumerate(zip(e_energies, e_labels)):
            ax_left.hlines(y=E, xmin=e_xmin, xmax=e_xmax, color='blue', linewidth=linewidth)
            ax_left.text(0.5, E-0.05, label, va='center', fontsize=fontsize)

        ax_left.set_xticks([])
        ax_left.set_yticks([])
        # ~ ax_left.set_title(fr'$\omega_L/\omega_0={w_l}$')
        ax_left.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax_left.set_xlim(-0.7, 0.95)
        ax_left.set_ylim(minE, maxE)

        # setup arrows
        Anum        = 4
        text_xs_off = [0.09, 0.07, 0.1, 0.2]
        text_ys     = [0.1, 0.04, 0.04, 0.03]
        margin      = 0.03
        arrow_g_x   = g_xmax - margin
        arrows_g_y  = g_energies[0:Anum]
        arrow_e_x   = e_xmax - margin
        arrow_e_y   = e_energies[0]
        offset_x    = (g_xmax - g_xmin - 2 * margin)/(Anum - 1)
        offset_y    = 0.01
        for i in range(Anum):
            ax_left.annotate(
                "",
                xytext=(arrow_e_x-offset_x*i , arrow_e_y-offset_y),
                xy=(arrow_g_x-offset_x*i, arrows_g_y[i]+offset_y),
                arrowprops=dict(arrowstyle="->", lw=2-0.5*i)
            )
            ax_left.text(
                arrow_g_x-offset_x*i + text_xs_off[i],
                arrows_g_y[i] + text_ys[i],
                fr"$\Gamma_{{e0{i}}}$",
                color="black",
                fontsize=fontsize-2
            )

        ax_left.text(-0.7, 0.9, "g", color="black", fontsize=fontsize)

        # ~ plt.tight_layout()
        plt.show()

    ############################################

    #####---------- Fig. 3 ----------#####
    if num == "3":
        # Setup figure
        fontsize = 18
        rcParams.update({'font.size': fontsize})
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), width_ratios=[1, 2.5])
        fig.subplots_adjust(
            left=0.11,   # space from left edge
            right=0.97, # space from right edge
            bottom=0.16,
            top=0.9,
            wspace=0.35, # space between columns
            #hspace=0.4  # space between rows
        )

        # Parameters
        opt         = "quantum_small"
        min_w       = 1.0
        vmin        = 0.0
        vmax        = 1.0
        cmap        = "cividis"
        Om_a        = np.linspace(0.0, 10.0, 201)
        w_L_vals    = np.linspace(min_w, 3.0, 201)
        G_10_matrix = np.zeros((len(Om_a), len(w_L_vals)))
        minE        = -0.8
        maxE        = 2.2
        w_l         = 1.5
        linewidth   = 3

        # Energy ladders
        Ng = int((maxE + 0.5 * w_0) / w_l) + 1
        Ne = int((maxE - 0.5 * w_0) / w_l) + 1
        g_energies = [-0.5 * w_0 + _ * w_l for _ in range(Ng)]
        e_energies = [0.5 * w_0 + _ * w_l for _ in range(Ne)]
        g_labels = [fr'$|{_}\rangle_-$' for _ in range(Ng)]
        e_labels = [fr'$|{_}\rangle_+$' for _ in range(Ne)]
        arrow_points= [(-0.275, g_energies[1]-0.02), (0.275, e_energies[0]+0.02)]

        for j, (E, label) in enumerate(zip(g_energies, g_labels)):
            axs[0].hlines(y=E, xmin=-0.5, xmax=-0.05, color='green', linewidth=linewidth)
            axs[0].text(-0.8, E-0.05, label, va='center', fontsize=fontsize)
        for j, (E, label) in enumerate(zip(e_energies, e_labels)):
            axs[0].hlines(y=E, xmin=0.05, xmax=0.5, color='blue', linewidth=linewidth)
            axs[0].text(0.55, E-0.05, label, va='center', fontsize=fontsize)

        axs[0].annotate(
            '',                    # no text
            xy=arrow_points[1],      # arrow head (data coords)
            xytext=arrow_points[0],  # arrow tail
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black')
        )

        axs[0].set_xticks([])
        axs[0].set_title(fr'$\omega_L/\omega_0={w_l}$')
        axs[0].spines[['top', 'right', 'bottom']].set_visible(False)
        axs[0].set_xlim(-0.85, 0.85)
        axs[0].set_ylim(minE, maxE)
        axs[0].set_ylabel(fr'$\omega_0$ [a.u.]', fontsize=fontsize)
        axs[0].text(-.75, 1.92, "a", color="black", fontsize=fontsize)

        # Heatmap
        for i, om_a in enumerate(tqdm(Om_a)):
            for j, w in enumerate(w_L_vals):
                G_10_matrix[i, j] = Gamma(("g", 1), ("e", 0), w_0, w, om_a, opt)[1]

        c = axs[1].pcolormesh(w_L_vals, Om_a, G_10_matrix, shading='auto', cmap=cmap)
        cc = axs[1].contour(w_L_vals, Om_a, G_10_matrix, levels=[1.0, 3.0], colors='white', linewidths=1.0)
        axs[1].plot(w_L_vals, 2 * w_L_vals, linestyle="--", color="white", linewidth=1.0)
        axs[1].set_title(r"Total rate $\Gamma_{g,1}$")
        axs[1].set_xlabel(r"$\omega_L/\omega_0$")
        axs[1].set_ylabel(r"$\Omega_a/\omega_0$")
        axs[1].text(1.05, 9.1, "b", color="white", fontsize=fontsize)

        fig.colorbar(c, ax=axs[1], label=r"$\Gamma/\Gamma_0$")
        plt.show()
    ############################################


    #####---------- Fig. 4 ----------#####
    if num == "4":
        # Setup figure
        fontsize = 18
        rcParams.update({'font.size': fontsize})
        fig, ax    = plt.subplots(1, 1, figsize=(7, 4.5), sharey=True)
        fig.subplots_adjust(
            left=0.15,   # space from left edge
            right=0.93, # space from right edge
            bottom=0.15,
            top=0.95,
            wspace=0.15, # space between columns
            hspace=0.35  # space between rows
        )

        # Parameters
        Om_a        = 0.001
        w_L_vals    = (0.9,)
        n_min       = 10
        n_max       = 5000
        p_max       = 5
        p_range     = range(0, p_max)
        ns_classic  = np.linspace(n_min, n_max, 201)**2
        ns_quantum  = np.linspace(n_min, n_max, 21)**2
        colors = ['blue', 'orange', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

        # Plots
        for i, w in enumerate(w_L_vals):
            args_classic = Om_a * np.sqrt(ns_classic) / w
            args_quantum = Om_a * np.sqrt(ns_quantum) / w
            ordering = 5*len(p_range)
            for p in tqdm(p_range):
                col = colors[p]
                G_c = [Gamma(("e", n), ("g", n-p), w_0, w, Om_a, "classic")[0] for n in ns_classic]
                ax.plot(np.sqrt(ns_classic), G_c, color=col, zorder=ordering, label=f"p={p}")
                G_q = [Gamma(("e", n), ("g", n-p), w_0, w, Om_a, "quantum_large")[0] for n in tqdm(ns_quantum)]
                ax.plot(np.sqrt(ns_quantum), G_q, "o", zorder=ordering, color=col)
                ordering -= 5

            ax.set_xlim([np.sqrt(ns_classic)[0], np.sqrt(ns_classic)[-1]])
            # ~ ax.set_title(fr"$\Omega_a/\omega_0 = {Om_a}$, $\omega_L/\omega_0={w}$")
            ax.set_xlabel(r"$\sqrt{n}$")
            if i == 0:
                ax.set_ylabel(r"$|{}_\pm\langle n | n-p \rangle_\mp|^2$")
            ax.grid()
            ax.legend(loc=1)
        plt.show()
    ############################################

    #####---------- Fig. 5 ----------#####
    if num == "5":
        # Setup figure
        fontsize = 18
        rcParams.update({'font.size': fontsize})
        fig, axs    = plt.subplots(1, 3, figsize=(10, 4))
        fig.subplots_adjust(
            left=0.10,   # space from left edge
            right=0.95, # space from right edge
            bottom=0.2,
            top=0.85,
            wspace=0.35, # space between columns
            hspace=0.4  # space between rows
        )

        # Parameters
        Om_a        = 0.001
        w_laser     = [0.1, 0.5, 2.0]
        n_min       = 1
        n_max       = 8000
        x_max       = 10.0
        points_num  = 1001
        labels      = ["a", "b", "c"]

        # Plots
        for i, w in enumerate(tqdm(w_laser)):
            ns = np.linspace(n_min, x_max * w / Om_a, points_num)**2
            args = Om_a * np.sqrt(ns) / w
            # ~ gamma_total = []
            # ~ for p in tqdm(range(-int(w_0 / w), 101, 1)):
                # ~ gamma_p = [Gamma(("e", n), ("g", n-p), w_0, w, Om_a, "classic")[1] for n in ns]
                # ~ gamma_total.append(gamma_p)
                # ~ axs[i].plot(args, gamma_p, color="gray")
            # ~ gamma_total = np.array(gamma_total)
            gamma = np.array([Gamma(("e", n), ("g", n), w_0, w, Om_a, "classic_optimal") for n in ns])
            # ~ axs[i].plot(args, sum(gamma_total), label="total")
            # ~ axs[i].plot(args, 1 + 3 * (w/w_0)**2 * args**2 / 2, "--", label="approx")
            axs[i].plot(args, gamma[:,0], "-", label=r"$\Gamma_e$")
            axs[i].plot(args, gamma[:,1], "-", label=r"$\Gamma_g$")

            axs[i].grid()
            axs[i].set_title(
                #fr"$\Omega_a/\omega_0 = {Om_a}$"
                #+ "\n"
                #+
                fr"$\omega_L/\omega_0 = {w}$"
            )
            axs[i].locator_params(axis='y', nbins=6)
            axs[i].locator_params(axis='x', nbins=5)
            axs[i].set_xlim((args[0],args[-1]))
            # ~ axs[i].set_xlabel(r"${\Omega_a}\sqrt{[\bar{n}]}/\omega_L$")
            axs[i].set_xlabel(r"$\frac{\Omega_a}{\omega_L}\sqrt{\mathcal{[\bar{n}]}}$")
            axs[i].text(
                0.3,
                0.95 * gamma[:,0][-1],
                labels[i],
                color="black",
                fontsize=fontsize
            )

        axs[0].set_ylabel(r"$\Gamma/\Gamma_0$")
        axs[2].legend(loc="center left")

        plt.show()


    #####---------- Fig. 6 ----------#####
    if num == "6":

        fontsize = 18
        rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(1, figsize=(5, 4))
        fig.subplots_adjust(
            left=0.18,   # space from left edge
            right=0.95, # space from right edge
            bottom=0.2,
            top=0.85,
            wspace=0.35, # space between columns
            hspace=0.4  # space between rows
        )
        ax.locator_params(axis='y', nbins=6)
        colors = ["blue", "orange", "green"]

        ns = np.arange(0, 50+1)**2
        ps = [0, 1, 2]
        G_large = np.zeros((len(ps), len(ns)))
        G_small = np.zeros((len(ps), len(ns)))*np.nan
        G_class = np.zeros((len(ps), len(ns)))
        w_L = 1.0
        Om_a = .1
        for i, p in enumerate(ps):
            for j, n in enumerate(tqdm(ns)):
                n = int(n)
                init = ["e", n]
                final = ["g", n-p]
                factor = (w_0 / (w_0 + p * w_L))**3
                if n<=400:
                    G_small[i,j] = factor * Gamma(init, final, w_0, w_L, Om_a, "quantum_small")[1]
                G_large[i,j] = factor * Gamma(init, final, w_0, w_L, Om_a, "quantum_large")[1]
                G_class[i,j] = factor * Gamma(init, final, w_0, w_L, Om_a, "classic")[1]

            ax.plot(np.sqrt(ns), G_small[i], "o", color=colors[p])
            ax.plot(np.sqrt(ns), G_large[i], "-", color=colors[p], label=fr"$\ell-n=${p}")
            #ax.plot(np.sqrt(ns), G_large, "--", color=colors[p], label="classic")

        # ~ ax.set_title(fr"$\Omega_a/\omega_0 = {Om_a}$, $\omega_L/\omega_0 = {w_L}$")
        ax.set_xlabel(r"$\sqrt{\ell}$")
        ax.set_ylabel(r"$\Gamma/\Gamma_0$")
        ax.set_xlim((np.sqrt(ns[0]), np.sqrt(ns[-1])))
        ax.grid()
        ax.legend()
        plt.show()


    #####---------- Fig. 7 ----------#####
    if num == "7":
        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.subplots_adjust(
            left=0.17,   # space from left edge
            right=0.95, # space from right edge
            bottom=0.15,
            top=0.85,
            wspace=0.35, # space between columns
            hspace=0.4  # space between rows
        )

        # Parameters
        opt     = "classic"
        cmap    = "cividis"
        Om_a    = 0.001
        w_Ls    = np.linspace(0.01, 3.0, 101)
        ns      = np.linspace(1e1, 1e4, 101)**2
        Gammas  = []

        # Plots
        Gammas.append(np.zeros((len(w_Ls), len(ns))))
        for i, w in enumerate(tqdm(w_Ls)):
            for j, n in enumerate(ns):
                ps = range(-int(w_0 / w), 1001, 1)
                Gammas[-1][i, j] = sum([
                    Gamma(("e", int(n)), ("g", int(n-p)), w_0, w, Om_a, opt)[1] for p in ps
                ])

        c = ax.pcolormesh(
            np.sqrt(ns),
            w_Ls,
            Gammas[0],
            norm=LogNorm(),
            shading='auto',
            cmap=cmap,
        )
        ax.set_title(r"$\Omega_a/\omega_0 = 0.001$")
        ax.set_ylabel(r"$\omega_L/\omega_0$")
        ax.set_xlabel(r"$\sqrt{n}$")


        # ---------------- SHARED COLORBAR ----------------
        fig.colorbar(c, ax=ax, label=r"$\Gamma/\Gamma_0$")

        plt.show()
