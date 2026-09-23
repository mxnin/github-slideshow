"""
PMR3220 - Parte II: simulacao cinematica e dinamica de uma ortese de membro inferior.

Modelo plano de 2 graus de liberdade com base fixa no quadril D (tronco imovel):
  elo 1 = coxa (D -> O), elo 2 = perna + pe (O -> P -> R, com PR perpendicular a OP).
Coordenadas generalizadas:
  theta1 = angulo absoluto da coxa (a partir de +x, anti-horario),
  phi    = angulo relativo do joelho (perna em relacao a coxa),
  theta2 = theta1 + phi = angulo absoluto da perna.
Trajetorias: polinomio de 5o grau com velocidade e aceleracao nulas nos extremos.
Torques: equacoes de Lagrange (verificadas simbolicamente com sympy).
"""
import json
import os

import numpy as np
import sympy as sp

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
    "Al 6061-T6": dict(rho=2700.0, E=68.9e9, Sy=276e6, Su=310e6, Se=96.5e6),
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
# Dinamica: Lagrange simbolico (verificacao) + forma fechada numerica
# ---------------------------------------------------------------------------
def deduz_lagrange():
    t = sp.symbols("t")
    m1, m2, I1, I2, a1, c2, gam, l1, g = sp.symbols("m1 m2 I1 I2 a1 c2 gamma L1 g", positive=True)
    th1 = sp.Function("theta1")(t)
    ph = sp.Function("phi")(t)
    x1 = a1 * sp.Matrix([sp.cos(th1), sp.sin(th1)])
    x2 = l1 * sp.Matrix([sp.cos(th1), sp.sin(th1)]) + c2 * sp.Matrix([sp.cos(th1 + ph + gam), sp.sin(th1 + ph + gam)])
    v1, v2 = x1.diff(t), x2.diff(t)
    T = sp.Rational(1, 2) * (m1 * v1.dot(v1) + m2 * v2.dot(v2) + I1 * th1.diff(t) ** 2 + I2 * (th1.diff(t) + ph.diff(t)) ** 2)
    V = g * (m1 * x1[1] + m2 * x2[1])
    Lg = T - V
    tau = [sp.simplify(sp.diff(Lg.diff(q.diff(t)), t) - Lg.diff(q)) for q in (th1, ph)]
    return tau, (th1, ph, t)


def torques(par, q, qd, qdd):
    """q = (theta1, phi). Retorna (tau_quadril, tau_joelho) em N.m."""
    th1, ph = q
    w1, wp = qd
    a1d, apd = qdd
    m1, I1, a1 = par["coxa"]["m"], par["coxa"]["I"], par["coxa"]["r"][0]
    m2, I2 = par["perna"]["m"], par["perna"]["I"]
    r2 = par["perna"]["r"]
    c2 = np.hypot(*r2)
    gam = np.arctan2(r2[1], r2[0])
    cb = np.cos(ph + gam)
    sb = np.sin(ph + gam)
    M11 = I1 + m1 * a1**2 + I2 + m2 * (L1**2 + c2**2 + 2 * L1 * c2 * cb)
    M12 = I2 + m2 * (c2**2 + L1 * c2 * cb)
    M22 = I2 + m2 * c2**2
    h = m2 * L1 * c2 * sb
    g1 = (m1 * a1 + m2 * L1) * G * np.cos(th1) + m2 * c2 * G * np.cos(th1 + ph + gam)
    g2 = m2 * c2 * G * np.cos(th1 + ph + gam)
    tau1 = M11 * a1d + M12 * apd - h * (2 * w1 * wp + wp**2) + g1
    tau2 = M12 * a1d + M22 * apd + h * w1**2 + g2
    return tau1, tau2, dict(inercial1=M11 * a1d + M12 * apd, centr1=-h * (2 * w1 * wp + wp**2), grav1=g1,
                            inercial2=M12 * a1d + M22 * apd, centr2=h * w1**2, grav2=g2)


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


def verifica_newton_euler(par, q, qd, qdd):
    """Verificacao independente por Newton-Euler (equilibrio dinamico de cada elo)."""
    th1, ph = q
    w1, wp = qd
    al1, alp = qdd
    th2, w2, al2 = th1 + ph, w1 + wp, al1 + alp
    m1, I1, a1 = par["coxa"]["m"], par["coxa"]["I"], par["coxa"]["r"][0]
    m2, I2, r2 = par["perna"]["m"], par["perna"]["I"], par["perna"]["r"]
    e = lambda a: np.array([np.cos(a), np.sin(a)])
    n = lambda a: np.array([-np.sin(a), np.cos(a)])
    O = L1 * e(th1)
    aO = L1 * (al1 * n(th1) - w1**2 * e(th1))
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    rc2 = R2 @ r2
    a_c2 = aO + al2 * np.array([-rc2[1], rc2[0]]) - w2**2 * rc2
    a_c1 = a1 * (al1 * n(th1) - w1**2 * e(th1))
    gv = np.array([0, -G])
    F_O = m2 * (a_c2 - gv)                         # forca do elo 1 no elo 2
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    tau2 = I2 * al2 + cross(rc2, F_O)              # momentos em torno do CM2: tau2 + (-rc2) x F_O = I2 al2
    # elo 1: forcas em D (F_D) e -F_O em O
    F_D = m1 * (a_c1 - gv) + F_O
    rc1 = a1 * e(th1)
    tau1 = I1 * al1 + tau2 - cross(-rc1, F_D) - cross(O - rc1, -F_O)
    return tau1, tau2


def main():
    os.makedirs(FIG, exist_ok=True)
    res = {}
    # verificacao simbolica
    tau_sym, _ = deduz_lagrange()
    res["lagrange_simbolico"] = [str(sp.simplify(x)) for x in tau_sym]

    par80 = parametros(80.0)
    par60 = parametros(60.0)
    res["parametros_80kg"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par80.items()}
    res["parametros_60kg"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par60.items()}
    res["componentes_ortese"] = [dict(elo=c[0], item=c[1], m=c[2], pos=list(c[3]), I=c[4]) for c in componentes_ortese()]
    res["massa_ortese"] = sum(c[2] for c in componentes_ortese())
    par_sem = parametros(80.0, com_ortese=False)
    res["parametros_80kg_sem_ortese"] = {k: dict(m=v["m"], r=v["r"].tolist(), I=v["I"]) for k, v in par_sem.items()}

    # verificacao Lagrange x Newton-Euler em um estado aleatorio
    rng = np.random.default_rng(1)
    q, qd, qdd = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    tl = torques(par80, q, qd, qdd)[:2]
    tn = verifica_newton_euler(par80, q, qd, qdd)
    res["verificacao_NE"] = dict(lagrange=list(map(float, tl)), newton_euler=list(map(float, tn)))

    T = 2.0
    res["T"] = T
    for final in ("final I", "final II"):
        for M, par in ((80.0, par80), (60.0, par60)):
            t, Q, Qd, Qdd, t1, t2, parc = ciclo(par, final, T)
            P1, P2 = t1 * Qd[:, 0], t2 * Qd[:, 1]
            chave = f"{final}|{M:.0f}kg"
            res[chave] = dict(
                tau_quadril_max=float(np.max(np.abs(t1))), tau_joelho_max=float(np.max(np.abs(t2))),
                tau_quadril_rms=float(np.sqrt(np.mean(t1**2))), tau_joelho_rms=float(np.sqrt(np.mean(t2**2))),
                pot_quadril_max=float(np.max(np.abs(P1))), pot_joelho_max=float(np.max(np.abs(P2))),
                w_coxa_max=float(np.degrees(np.max(np.abs(Qd[:, 0])))),
                w_perna_max=float(np.degrees(np.max(np.abs(Qd[:, 0] + Qd[:, 1])))),
                w_joelho_max=float(np.degrees(np.max(np.abs(Qd[:, 1])))),
                a_coxa_max=float(np.degrees(np.max(np.abs(Qdd[:, 0])))),
                a_perna_max=float(np.degrees(np.max(np.abs(Qdd[:, 0] + Qdd[:, 1])))),
                a_joelho_max=float(np.degrees(np.max(np.abs(Qdd[:, 1])))),
                grav_quadril_max=float(np.max(np.abs(parc["grav1"]))),
                grav_joelho_max=float(np.max(np.abs(parc["grav2"]))),
            )
        # sensibilidade ao tempo T (80 kg)
        sens = []
        for TT in (1.0, 1.5, 2.0, 3.0, 4.0):
            _, _, _, _, t1, t2, _ = ciclo(par80, final, TT)
            sens.append((TT, float(np.max(np.abs(t1))), float(np.max(np.abs(t2)))))
        res[f"sens_T|{final}"] = sens

    # ---------------- figuras ----------------
    for final, tag in (("final I", "I"), ("final II", "II")):
        t, Q, Qd, Qdd, t1, t2, parc = ciclo(par80, final, T)
        fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.2), sharex=True)
        th2 = Q[:, 0] + Q[:, 1]
        ax[0].plot(t, np.degrees(Q[:, 0]), label=r"$\theta_1$ (coxa)")
        ax[0].plot(t, np.degrees(th2), label=r"$\theta_2=\theta_1+\varphi$ (perna)")
        ax[0].plot(t, np.degrees(Q[:, 1]), "--", label=r"$\varphi$ (joelho, relativo)")
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

    # ---------------- verificacao estrutural (hastes Al) ----------------
    tau_j = max(res["final I|80kg"]["tau_joelho_max"], res["final II|80kg"]["tau_joelho_max"])
    tau_q = max(res["final I|80kg"]["tau_quadril_max"], res["final II|80kg"]["tau_quadril_max"])
    W = HASTE_H * HASTE_B**2 / 6       # modulo resistente (flexao no plano sagital)
    Sy = MAT["Al 6061-T6"]["Sy"]
    Se = MAT["Al 6061-T6"]["Se"]
    Su = MAT["Al 6061-T6"]["Su"]
    est = {}
    for nome, Mf in (("haste da perna (torque do joelho)", tau_j), ("haste da coxa (torque do quadril)", tau_q)):
        sig = Mf / 2 / W                # duas hastes dividem o momento
        # ciclo alternado 0 -> max -> 0: sigma_a = sigma_m = sig/2 (Goodman modificado)
        sa = sm = sig / 2
        n_goodman = 1 / (sa / Se + sm / Su)
        est[nome] = dict(M=Mf, sigma_MPa=sig / 1e6, n_escoamento=Sy / sig, n_goodman=n_goodman)
    res["estrutural"] = est
    res["W_mm3"] = W * 1e9

    with open(os.path.join(DADOS, "resultado_parte2.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "lagrange_simbolico"}, indent=2, default=float))
    for s in res["lagrange_simbolico"]:
        print(s[:400])


if __name__ == "__main__":
    main()
