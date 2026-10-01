"""
PMR3220 - Parte II: simulacao cinematica e dinamica de uma ortese de membro inferior.

Modelo plano de 2 graus de liberdade com base fixa no quadril D (tronco imovel):
  elo 1 = coxa (D -> O), elo 2 = perna + pe (O -> P -> R, com PR perpendicular a OP).
Notacao do metodo matricial (PMR3220): bases 0 (fixa, em D), 1 (coxa) e 2 (perna+pe),
  [0R1] = Rot(theta1, z), 0rO1 = 0;  [1R2] = Rot(theta2, z), 1rO2 = (l1, 0);  0r = [0R1](1rO2 + [1R2] 2r).
Coordenadas generalizadas q = [theta1, theta2]:
  theta1 = angulo de X0 para X1 (quadril),
  theta2 = angulo RELATIVO de X1 para X2 (joelho);
  theta1 + theta2 = orientacao absoluta da perna.
Trajetorias: polinomio de 5o grau com velocidade e aceleracao nulas nos extremos.
Torques: metodo de Newton-Euler (equilibrio dinamico de cada elo), com as aceleracoes
obtidas pelas derivadas das matrizes de rotacao (mudanca de base de ponto).
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIG = os.path.join(RAIZ, "figuras")
DADOS = os.path.join(RAIZ, "dados")

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})

G = 9.81
L1, L2, L3 = 0.40, 0.45, 0.25     # DO, OP, PR (m)

# Configuracoes (graus) lidas da Figura 2 do enunciado
CONF = {
    "inicial": dict(th1=-90.0, phi=0.0),
    "final I": dict(th1=280.13 - 360.0, phi=-120.0),       # angulo interno no joelho = 60 graus
    "final II": dict(th1=10.07, phi=-(180.0 - 70.1)),       # angulo interno no joelho = 70,1 graus
}

# ---------------------------------------------------------------------------
# Antropometria (Winter, 2009, tab. 4.1): fracao de massa, CM a partir do
# extremo proximal e raio de giracao em torno do CM (fracoes do comprimento)
# ---------------------------------------------------------------------------
WINTER = dict(coxa=(0.100, 0.433, 0.323), perna=(0.0465, 0.433, 0.302), pe=(0.0145, 0.50, 0.475))

# ---------------------------------------------------------------------------
# Ortese: materiais e componentes (massas estimadas a partir da geometria)
# ---------------------------------------------------------------------------
MAT = {
    "Al 6061-T6": dict(rho=2700.0, E=68.9e9, Sy=276e6, Su=310e6),
    "PP copolimero": dict(rho=905.0, E=1.3e9, Sy=28e6),
    "Aco AISI 304": dict(rho=8000.0, E=193e9, Sy=215e6, Su=505e6),
}
HASTE_B, HASTE_H = 0.020, 0.005   # secao das hastes laterais (m): largura no plano sagital x espessura
M_ATUADOR_JOELHO = 1.2            # kg (motor BLDC + reducao harmonica, estator fixo na coxa)


def componentes_ortese():
    """Lista (elo, massa, posicao do CM no referencial do elo, I proprio) dos componentes.
    Referencial do elo 1: origem em D, x ao longo de D->O.
    Referencial do elo 2: origem em O, x ao longo de O->P, y no sentido P->R."""
    rho_al = MAT["Al 6061-T6"]["rho"]
    rho_pp = MAT["PP copolimero"]["rho"]
    A = HASTE_B * HASTE_H
    comp = []
    # hastes medial e lateral (barras esbeltas)
    m = 2 * rho_al * A * L1
    comp.append(("coxa", "hastes Al (2x)", m, (L1 / 2, 0.0), m * L1**2 / 12))
    m = 2 * rho_al * A * L2
    comp.append(("perna", "hastes Al (2x)", m, (L2 / 2, 0.0), m * L2**2 / 12))
    # bracadeiras em PP 3 mm (area desenvolvida ~ 0,030 m^2 cada) + tiras
    m_cuff = rho_pp * 0.030 * 0.003 + 0.05
    comp.append(("coxa", "bracadeira proximal PP", m_cuff, (0.10, 0.0), 0.0))
    comp.append(("coxa", "bracadeira distal PP", m_cuff, (0.30, 0.0), 0.0))
    comp.append(("perna", "bracadeira PP", m_cuff, (0.15, 0.0), 0.0))
    # palmilha/estribo em PP 4 mm (0,25 x 0,09 m) + estribo Al
    m_pe = rho_pp * L3 * 0.09 * 0.004 + 0.10
    comp.append(("perna", "palmilha PP + estribo", m_pe, (L2, L3 / 2), m_pe * L3**2 / 12))
    # articulacao do joelho (eixo e mancais em aco 304) e atuador do joelho
    comp.append(("coxa", "articulacao joelho (aco 304)", 0.20, (L1, 0.0), 0.0))
    comp.append(("coxa", "atuador do joelho", M_ATUADOR_JOELHO, (L1, 0.0), 0.5 * M_ATUADOR_JOELHO * 0.045**2))
    return comp


def parametros(M, com_ortese=True):
    """Massa, posicao do CM (no referencial do elo) e inercia baricentrica dos elos."""
    corpos = {"coxa": [], "perna": []}
    mc, xc, rc = WINTER["coxa"]
    mc *= M
    corpos["coxa"].append((mc, np.array([xc * L1, 0.0]), mc * (rc * L1) ** 2))
    mp, xp, rp = WINTER["perna"]
    mp *= M
    corpos["perna"].append((mp, np.array([xp * L2, 0.0]), mp * (rp * L2) ** 2))
    mf, xf, rf = WINTER["pe"]
    mf *= M
    corpos["perna"].append((mf, np.array([L2, xf * L3]), mf * (rf * L3) ** 2))
    if com_ortese:
        for elo, _, m, pos, I in componentes_ortese():
            corpos[elo].append((m, np.array(pos), I))
    out = {}
    for elo, lst in corpos.items():
        m = sum(c[0] for c in lst)
        r = sum(c[0] * c[1] for c in lst) / m
        I = sum(c[2] + c[0] * np.sum((c[1] - r) ** 2) for c in lst)   # Steiner
        out[elo] = dict(m=m, r=r, I=I)
    return out


# ---------------------------------------------------------------------------
# Trajetoria polinomial de 5o grau
# ---------------------------------------------------------------------------
def quintica(q0, qf, T, t):
    tau = np.clip(t / T, 0, 1)
    d = qf - q0
    q = q0 + d * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    qd = d / T * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
    qdd = d / T**2 * (60 * tau - 180 * tau**2 + 120 * tau**3)
    return q, qd, qdd


# ---------------------------------------------------------------------------
# Dinamica: Newton-Euler
# ---------------------------------------------------------------------------
def cruz(a, b):
    """Componente z do produto vetorial no plano, [a~ b]_z = a_x b_y - a_y b_x."""
    return a[0] * b[1] - a[1] * b[0]


def newton_euler(par, q, qd, qdd):
    """Torques do quadril (tau_Q) e do joelho (tau_J) por Newton-Euler.
    Elo 2, momentos em O:  tau_J = I2 al2 + [(rG2 - rO) x m2 (aG2 - g)]_z
    Elos 1+2, momentos em D (fixo): tau_Q = I1 al1 + I2 al2 + sum [rGi x mi (aGi - g)]_z"""
    th1, th2 = q
    d1, d2 = qd
    dd1, dd2 = qdd
    m1, I1, a1 = par["coxa"]["m"], par["coxa"]["I"], par["coxa"]["r"][0]
    m2, I2 = par["perna"]["m"], par["perna"]["I"]
    bx, by = par["perna"]["r"]
    c1, s1 = np.cos(th1), np.sin(th1)
    c12, s12 = np.cos(th1 + th2), np.sin(th1 + th2)
    w2, al2 = d1 + d2, dd1 + dd2
    # posicoes por mudanca de base: 0rG1 = [0R1] 1rG1, 0rO = [0R1] 1rO, 0rG2 = 0rO + [0R2] 2rG2
    rG1 = np.array([a1 * c1, a1 * s1])
    rO = np.array([L1 * c1, L1 * s1])
    rOG2 = np.array([bx * c12 - by * s12, bx * s12 + by * c12])
    rG2 = rO + rOG2
    # aceleracoes: a = alfa k x r - w^2 r (derivadas segundas das transformacoes)
    aG1 = a1 * np.array([-s1 * dd1 - c1 * d1**2, c1 * dd1 - s1 * d1**2])
    aO = L1 * np.array([-s1 * dd1 - c1 * d1**2, c1 * dd1 - s1 * d1**2])
    aG2 = aO + np.array([-rOG2[1] * al2 - rOG2[0] * w2**2, rOG2[0] * al2 - rOG2[1] * w2**2])
    g = np.array([0.0 * np.ones_like(th1), -G * np.ones_like(th1)])
    FG1 = m1 * (aG1 - g)
    FG2 = m2 * (aG2 - g)
    tauJ = I2 * al2 + cruz(rOG2, FG2)
    tauQ = I1 * dd1 + I2 * al2 + cruz(rG1, FG1) + cruz(rG2, FG2)
    return tauQ, tauJ


def torques(par, q, qd, qdd):
    """Torques totais e suas parcelas: gravitacional (q, 0, 0), inercial (so aceleracoes)
    e centripeta/Coriolis (so velocidades)."""
    z = np.zeros_like(np.asarray(q[0], dtype=float))
    tQ, tJ = newton_euler(par, q, qd, qdd)
    gQ, gJ = newton_euler(par, q, (z, z), (z, z))
    iQ, iJ = newton_euler(par, q, (z, z), qdd)
    cQ, cJ = newton_euler(par, q, qd, (z, z))
    return tQ, tJ, dict(grav1=gQ, grav2=gJ, inercial1=iQ - gQ, inercial2=iJ - gJ, centr1=cQ - gQ, centr2=cJ - gJ)


def ciclo(par, final, T, n=801):
    """Ciclo completo: ida (0..T) e retorno (T..2T) com quinticas."""
    q0 = np.radians([CONF["inicial"]["th1"], CONF["inicial"]["phi"]])
    qf = np.radians([CONF[final]["th1"], CONF[final]["phi"]])
    t = np.linspace(0, 2 * T, n)
    Q = np.zeros((n, 2)); Qd = np.zeros((n, 2)); Qdd = np.zeros((n, 2))
    for j in range(2):
        ida = t <= T
        Q[ida, j], Qd[ida, j], Qdd[ida, j] = quintica(q0[j], qf[j], T, t[ida])
        Q[~ida, j], Qd[~ida, j], Qdd[~ida, j] = quintica(qf[j], q0[j], T, t[~ida] - T)
    tau1, tau2, parc = torques(par, Q.T, Qd.T, Qdd.T)
    return t, Q, Qd, Qdd, tau1, tau2, parc


def main():
    os.makedirs(FIG, exist_ok=True)
    res = {}
    par80 = parametros(80.0)
    par60 = parametros(60.0)
    res["parametros_80kg"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par80.items()}
    res["parametros_60kg"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par60.items()}
    res["componentes_ortese"] = [dict(elo=c[0], item=c[1], m=c[2], pos=list(c[3]), I=c[4]) for c in componentes_ortese()]
    res["massa_ortese"] = sum(c[2] for c in componentes_ortese())
    par_sem = parametros(80.0, com_ortese=False)
    res["parametros_80kg_sem_ortese"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par_sem.items()}

    # conferencia: em repouso na configuracao inicial, tau_Q = tau_J = m2 g bx? (so o pe a frente)
    tq0, tj0 = newton_euler(par80, np.radians([-90.0, 0.0]), (0.0, 0.0), (0.0, 0.0))
    res["repouso_inicial"] = dict(tau_Q=float(tq0), tau_J=float(tj0),
                                  m2_g_by=float(par80["perna"]["m"] * G * par80["perna"]["r"][1]))

    T = 2.0
    res["T"] = T
    for final in ("final I", "final II"):
        for M, par in ((80.0, par80), (60.0, par60)):
            t, Q, Qd, Qdd, t1, t2, parc = ciclo(par, final, T)
            chave = f"{final}|{M:.0f}kg"
            res[chave] = dict(
                tau_quadril_max=float(np.max(np.abs(t1))), tau_joelho_max=float(np.max(np.abs(t2))),
                w_coxa_max=float(np.degrees(np.max(np.abs(Qd[:, 0])))),
                w_perna_max=float(np.degrees(np.max(np.abs(Qd[:, 0] + Qd[:, 1])))),
                w_joelho_max=float(np.degrees(np.max(np.abs(Qd[:, 1])))),
                a_coxa_max=float(np.degrees(np.max(np.abs(Qdd[:, 0])))),
                a_perna_max=float(np.degrees(np.max(np.abs(Qdd[:, 0] + Qdd[:, 1])))),
                a_joelho_max=float(np.degrees(np.max(np.abs(Qdd[:, 1])))),
                grav_quadril_max=float(np.max(np.abs(parc["grav1"]))),
                grav_joelho_max=float(np.max(np.abs(parc["grav2"]))),
            )

    # ---------------- figuras ----------------
    for final, tag in (("final I", "I"), ("final II", "II")):
        t, Q, Qd, Qdd, t1, t2, parc = ciclo(par80, final, T)
        fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.2), sharex=True)
        th2 = Q[:, 0] + Q[:, 1]
        ax[0].plot(t, np.degrees(Q[:, 0]), label=r"$\theta_1$ (coxa)")
        ax[0].plot(t, np.degrees(th2), label=r"$\theta_1+\theta_2$ (perna)")
        ax[0].plot(t, np.degrees(Q[:, 1]), "--", label=r"$\theta_2$ (joelho, relativo)")
        ax[0].set_ylabel("orientação (°)")
        ax[1].plot(t, np.degrees(Qd[:, 0]))
        ax[1].plot(t, np.degrees(Qd[:, 0] + Qd[:, 1]))
        ax[1].plot(t, np.degrees(Qd[:, 1]), "--")
        ax[1].set_ylabel("vel. angular (°/s)")
        ax[2].plot(t, np.degrees(Qdd[:, 0]))
        ax[2].plot(t, np.degrees(Qdd[:, 0] + Qdd[:, 1]))
        ax[2].plot(t, np.degrees(Qdd[:, 1]), "--")
        ax[2].set_ylabel("acel. angular (°/s²)")
        ax[2].set_xlabel("t (s)")
        ax[0].legend(fontsize=7, ncol=3, loc="best")
        for a in ax:
            a.grid(alpha=0.3)
            a.axvline(T, color="k", lw=0.6, ls=":")
        ax[0].set_title(f"Cinemática – configuração {final} (T = {T:.0f} s por trecho)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"p2_cinematica_{tag}.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
        ax[0].plot(t, t1, "k", lw=1.4, label="total")
        ax[0].plot(t, parc["grav1"], label="gravitacional")
        ax[0].plot(t, parc["inercial1"], label="inercial")
        ax[0].plot(t, parc["centr1"], label="centríp./Coriolis")
        ax[0].set_ylabel(r"$\tau_{quadril}$ (N·m)")
        ax[1].plot(t, t2, "k", lw=1.4)
        ax[1].plot(t, parc["grav2"])
        ax[1].plot(t, parc["inercial2"])
        ax[1].plot(t, parc["centr2"])
        _, _, _, _, t1b, t2b, _ = ciclo(par60, final, T)
        ax[0].plot(t, t1b, "k:", lw=1, label="total (60 kg)")
        ax[1].plot(t, t2b, "k:", lw=1)
        for a in ax:
            lo, hi = a.get_ylim()
            a.set_ylim(lo, hi + 0.45 * (hi - lo))
        ax[0].legend(fontsize=6.5, ncol=5, loc="upper center")
        ax[1].set_ylabel(r"$\tau_{joelho}$ (N·m)")
        ax[1].set_xlabel("t (s)")
        for a in ax:
            a.grid(alpha=0.3)
            a.axvline(T, color="k", lw=0.6, ls=":")
        ax[0].set_title(f"Torques dos atuadores – configuração {final} (80 kg; pontilhado: 60 kg)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"p2_torques_{tag}.pdf"))
        plt.close(fig)

        # instantaneos do movimento (estroboscopia)
        fig, ax = plt.subplots(figsize=(3.4, 3.6))
        for k, tk in enumerate(np.linspace(0, T, 9)):
            i = np.argmin(np.abs(t - tk))
            a1, a2 = Q[i, 0], Q[i, 0] + Q[i, 1]
            O = L1 * np.array([np.cos(a1), np.sin(a1)])
            P = O + L2 * np.array([np.cos(a2), np.sin(a2)])
            R = P + L3 * np.array([np.cos(a2 + np.pi / 2), np.sin(a2 + np.pi / 2)])
            cor = plt.cm.viridis(k / 8)
            ax.plot([0, O[0], P[0], R[0]], [0, O[1], P[1], R[1]], "-o", color=cor, ms=2, lw=1)
        ax.plot(0, 0, "ks")
        ax.annotate("D", (0, 0), xytext=(4, 2), textcoords="offset points")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"inicial → {final}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"p2_estrobo_{tag}.pdf"))
        plt.close(fig)

    with open(os.path.join(DADOS, "resultado_parte2.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
