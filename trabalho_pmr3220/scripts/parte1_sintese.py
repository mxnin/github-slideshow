"""
PMR3220 - Parte I: sintese dimensional de um quadrilatero articulado (joelho policentrico)
pela equacao da diade padrao (tres posicoes de precisao), a partir das MEDICOES do
Grupo C (rastreamento no software Tracker, arquivo dados/tracker_rascunho.csv).

Somente ferramentas da disciplina:
  - pose de um corpo rigido no plano a partir de dois pontos (analise cinematica 2D);
  - sintese dimensional analitica: diade padrao [Rot(beta)-I] W + [Rot(alpha)-I] Z = delta,
    resolvida por blocos de matrizes 2x2 (regra de Cramer por blocos), com as escolhas livres (beta2, beta3) por diade
    determinadas por varredura;
  - analise de posicao (cadeia fechada) pelo metodo de Newton-Raphson.
Notacao (metodo matricial): base 0 = coxa (fixa, origem no ponto medio de 3-4, eixos da
imagem: movimento angular do quadril desconsiderado); base 4 = perna, coincidente com a
base 0 no quadro de referencia 307 (membro estendido). Unidades: mm e graus.
"""
import csv
import itertools
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

FRAME_REF = 307
FRAMES = [307, 390, 423, 456]
MARC_COXA = [1, 2, 3, 4]
MARC_PERNA = [5, 6, 7, 8, 9]
PONTOS_PE = [10, 11]
PONTO_TORNOZELO = 8

# requisitos de projeto (regiao do joelho, mm)
CAIXA_FIXOS = dict(xmin=-60, xmax=60, ymin=-40, ymax=60)
CAIXA_MOVEIS = dict(xmin=-60, xmax=60, ymin=-120, ymax=0)   # pivos moveis na parte proximal da perna (base 4)
ELO_MIN, ELO_MAX = 20.0, 120.0
MU_MIN = 30.0
ERRO_ACEITAVEL = 1.0   # mm: abaixo disso (<< incerteza das medicoes), prefere-se o maior angulo de transmissao

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def muda_base(a, p, r):
    """Mudanca de base de ponto: 0r = [0R4] 4r + 0rO4, com [0R4] = Rot(a, z) e 0rO4 = p."""
    return rot(a) @ np.asarray(r) + p


# ---------------------------------------------------------------------------
# 1) dados medidos -> poses ([0R4], 0rO4) da perna (corpo rigido definido por dois pontos)
# ---------------------------------------------------------------------------
def le_tracker(caminho):
    D = {}
    with open(caminho) as f:
        for r in csv.DictReader(f):
            D.setdefault(int(r["frame"]), {})[int(r["ponto"])] = 1000.0 * np.array([float(r["x_m"]), float(r["y_m"])])
    return D


def na_coxa(P, k):
    """0r_k = imr_k - (imr_3 + imr_4)/2  ([imR0] = I: rotacao do quadril desconsiderada)."""
    return P[k] - (P[3] + P[4]) / 2


def poses_medidas(D):
    """Pose da perna em cada quadro a partir de J (ponto medio de 5-6) e P (ponto 8):
    alpha_j = beta_j - beta_ref  e  0rO4 = 0rJ - Rot(alpha_j) 4rJ."""
    J = {fr: (na_coxa(D[fr], 5) + na_coxa(D[fr], 6)) / 2 for fr in D}
    P = {fr: na_coxa(D[fr], PONTO_TORNOZELO) for fr in D}
    beta = {fr: np.arctan2(*(P[fr] - J[fr])[::-1]) for fr in D}
    J4, P4 = J[FRAME_REF], P[FRAME_REF]                     # base 4 = base 0 no quadro de referencia
    poses = {}
    for fr in sorted(D):
        a = (beta[fr] - beta[FRAME_REF] + np.pi) % (2 * np.pi) - np.pi
        if a > 0.5:                                          # flexao alem de 180 graus
            a -= 2 * np.pi
        p = J[fr] - rot(a) @ J4
        comp = np.linalg.norm(P[fr] - J[fr])
        poses[fr] = dict(alpha=a, p=p, flexao=-np.degrees(a) + 0.0, JP=comp,
                         dP=float(np.linalg.norm(muda_base(a, p, P4) - P[fr])))
    return poses, J4, P4


# ---------------------------------------------------------------------------
# 2) sintese pela diade padrao
# ---------------------------------------------------------------------------
def diade(alphas, deltas, b2, b3):
    """Resolve [Rot(b_j) - I] W + [Rot(a_j) - I] Z = delta_j (j = 2, 3) por blocos 2x2:
    Delta = A2 B3 - B2 A3,  W = Delta^-1 (B3 d2 - B2 d3),  Z = Delta^-1 (A2 d3 - A3 d2)
    (A_j e B_j comutam, pois tem a forma [[a, -b], [b, a]])."""
    I = np.eye(2)
    A2, A3 = rot(b2) - I, rot(b3) - I
    B2, B3 = rot(alphas[0]) - I, rot(alphas[1]) - I
    Delta = A2 @ B3 - B2 @ A3
    if abs(np.linalg.det(Delta)) < 1e-9:
        return None
    Dinv = np.linalg.inv(Delta)
    W = Dinv @ (B3 @ deltas[0] - B2 @ deltas[1])
    Z = Dinv @ (A2 @ deltas[1] - A3 @ deltas[0])
    return W, Z


def candidatos_diade(poses, prec, verif, passo, centro=None, janela=None):
    """Varre as escolhas livres (beta2, beta3) de uma diade. Para cada par, calcula
    W e Z, os pivos fixo (0rC = 0rO4 - Z - W) e movel (4rQ), o comprimento |W| e o erro
    da diade na posicao de verificacao, e = | |0rQ(verif) - 0rC| - |W| |."""
    p1 = poses[prec[0]]
    alphas = [poses[f]["alpha"] - p1["alpha"] for f in prec[1:]]
    deltas = [poses[f]["p"] - p1["p"] for f in prec[1:]]
    if centro is None:
        g = np.radians(np.arange(-180, 180, passo))
        grade = itertools.product(g, g)
    else:
        g2 = centro[0] + np.radians(np.arange(-janela, janela + 1e-9, passo))
        g3 = centro[1] + np.radians(np.arange(-janela, janela + 1e-9, passo))
        grade = itertools.product(g2, g3)
    out = []
    for b2, b3 in grade:
        sol = diade(alphas, deltas, b2, b3)
        if sol is None:
            continue
        W, Z = sol
        Q1 = p1["p"] - Z                                  # 0rQ(1) = 0rO4(1) - Z
        C = Q1 - W                                        # 0rC = 0rQ(1) - W
        Q4 = rot(p1["alpha"]).T @ (Q1 - p1["p"])          # 4rQ = [0R4]^T (0rQ - 0rO4)
        l = np.linalg.norm(W)
        if not (CAIXA_FIXOS["xmin"] <= C[0] <= CAIXA_FIXOS["xmax"] and CAIXA_FIXOS["ymin"] <= C[1] <= CAIXA_FIXOS["ymax"]):
            continue
        if not (ELO_MIN <= l <= ELO_MAX):
            continue
        if not (CAIXA_MOVEIS["xmin"] <= Q4[0] <= CAIXA_MOVEIS["xmax"] and CAIXA_MOVEIS["ymin"] <= Q4[1] <= CAIXA_MOVEIS["ymax"]):
            continue
        pv = poses[verif]
        e = abs(np.linalg.norm(muda_base(pv["alpha"], pv["p"], Q4) - C) - l)
        out.append(dict(b2=b2, b3=b3, W=W, Z=Z, C=C, Q4=Q4, l=l, e=e))
    out.sort(key=lambda d: d["e"])
    return out


def montagem(C, L, Q4, N4, alphas, pose_ini):
    """Analise de posicao: para cada orientacao alpha da perna, resolve a equacao de
    fechamento f(theta1, theta3) = 0 por Newton-Raphson, partindo do passo anterior."""
    l1 = np.linalg.norm(muda_base(pose_ini["alpha"], pose_ini["p"], Q4) - C)
    l3 = np.linalg.norm(muda_base(pose_ini["alpha"], pose_ini["p"], N4) - L)
    Q0 = muda_base(pose_ini["alpha"], pose_ini["p"], Q4)
    N0 = muda_base(pose_ini["alpha"], pose_ini["p"], N4)
    t1 = np.arctan2(*(Q0 - C)[::-1])
    t3 = np.arctan2(*(N0 - L)[::-1])
    QN4 = N4 - Q4
    l2 = np.linalg.norm(QN4)
    psi0 = np.arctan2(QN4[1], QN4[0])
    out = []
    for a in alphas:
        psi = psi0 + a
        for _ in range(30):
            f = np.array([C[0] + l1 * np.cos(t1) + l2 * np.cos(psi) - L[0] - l3 * np.cos(t3),
                          C[1] + l1 * np.sin(t1) + l2 * np.sin(psi) - L[1] - l3 * np.sin(t3)])
            J = np.array([[-l1 * np.sin(t1), l3 * np.sin(t3)], [l1 * np.cos(t1), -l3 * np.cos(t3)]])
            if abs(np.linalg.det(J)) < 1e-9:
                return None
            d = np.linalg.solve(J, -f)
            t1, t3 = t1 + d[0], t3 + d[1]
            if np.max(np.abs(d)) < 1e-12:
                break
        if np.max(np.abs(f)) > 1e-6:
            return None
        if out and (abs(t1 - out[-1]["t1"]) > 0.35 or abs(t3 - out[-1]["t3"]) > 0.35):
            return None                                   # salto de ramo
        Q = C + l1 * np.array([np.cos(t1), np.sin(t1)])
        p = Q - rot(a) @ Q4
        dd = np.degrees((psi - t3) % np.pi)              # angulo entre o acoplador (X2) e o balancim LN (X3)
        mu = min(dd, 180 - dd)
        out.append(dict(alpha=a, p=p, t1=t1, t3=t3, psi=psi, mu=mu))
    return out


def polo(C, L, t1, t3):
    d1, d2 = np.array([np.cos(t1), np.sin(t1)]), np.array([np.cos(t3), np.sin(t3)])
    s = np.linalg.solve(np.array([d1, -d2]).T, L - C)
    return C + s[0] * d1


def avalia_par(d1, d2, poses, prec, verif, P4, grade):
    C, L, Q4, N4 = d1["C"], d2["C"], d1["Q4"], d2["Q4"]
    l0, l2 = np.linalg.norm(C - L), np.linalg.norm(Q4 - N4)
    if not (ELO_MIN <= l0 <= ELO_MAX and ELO_MIN <= l2 <= ELO_MAX):
        return None
    traj = montagem(C, L, Q4, N4, [-np.radians(g) for g in grade], poses[FRAME_REF])
    if traj is None:
        return None
    mus = [t["mu"] for t in traj]
    if min(mus) < MU_MIN:
        return None
    porphi = dict(zip(grade, traj))
    erros = {}
    for f in FRAMES:
        t = porphi[round(poses[f]["flexao"], 6)]
        erros[f] = float(np.linalg.norm(muda_base(t["alpha"], t["p"], P4)
                                        - muda_base(poses[f]["alpha"], poses[f]["p"], P4)))
    return dict(C=C, L=L, Q4=Q4, N4=N4, traj=traj, grade=grade, erros=erros, mus=mus,
                custo=max(erros[verif], ERRO_ACEITAVEL) - 0.01 * min(mus), betas=(d1["b2"], d1["b3"]), gammas=(d2["b2"], d2["b3"]),
                W=d1["W"], Z=d1["Z"], U=d2["W"], S=d2["Z"])


def sintetiza(poses, prec, verif, P4):
    phi_max = max(poses[f]["flexao"] for f in FRAMES)
    grade = sorted({round(g, 6) for g in np.arange(0, phi_max + 1.0, 1.0)} | {round(poses[f]["flexao"], 6) for f in FRAMES})
    cand = candidatos_diade(poses, prec, verif, passo=4.0)[:60]
    melhor = None
    for d1, d2 in itertools.combinations(cand, 2):
        for a, b in ((d1, d2), (d2, d1)):
            r = avalia_par(a, b, poses, prec, verif, P4, grade)
            if r is not None and (melhor is None or r["custo"] < melhor["custo"]):
                melhor = r
    if melhor is None:
        return None
    # refinamento: varredura mais fina em torno das escolhas livres encontradas
    c1 = candidatos_diade(poses, prec, verif, 0.5, melhor["betas"], 4.0)[:25]
    c2 = candidatos_diade(poses, prec, verif, 0.5, melhor["gammas"], 4.0)[:25]
    for a in c1:
        for b in c2:
            r = avalia_par(a, b, poses, prec, verif, P4, grade)
            if r is not None and r["custo"] < melhor["custo"]:
                melhor = r
    return melhor


# ---------------------------------------------------------------------------
# 3) programa principal
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG, exist_ok=True)
    D = le_tracker(os.path.join(DADOS, "tracker_rascunho.csv"))
    poses, J4, P4 = poses_medidas(D)

    with open(os.path.join(DADOS, "poses_medidas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "flexao_graus", "x_O4_mm", "y_O4_mm", "x_P_mm", "y_P_mm", "JP_mm", "dP_mm"])
        for fr, ps in sorted(poses.items()):
            tp = muda_base(ps["alpha"], ps["p"], P4)
            w.writerow([fr, f"{ps['flexao']:.2f}", f"{ps['p'][0]:.2f}", f"{ps['p'][1]:.2f}",
                        f"{tp[0]:.1f}", f"{tp[1]:.1f}", f"{ps['JP']:.0f}", f"{ps['dP']:.1f}"])

    combos = []
    for trio in itertools.combinations(FRAMES[1:], 2):
        prec = [FRAME_REF, *trio]
        verif = [f for f in FRAMES if f not in prec][0]
        r = sintetiza(poses, prec, verif, P4)
        if r is not None:
            combos.append((r["custo"], prec, [verif], r))
            print("precisao", prec, "verif", verif, "erro %.2f mm" % r["erros"][verif], "mu_min %.1f" % min(r["mus"]),
                  "betas", np.round(np.degrees(r["betas"]), 1), "gammas", np.round(np.degrees(r["gammas"]), 1))
    combos.sort(key=lambda z: z[0])
    _, prec, cheque, r = combos[0]
    C, L, Q4, N4, traj, grade = r["C"], r["L"], r["Q4"], r["N4"], r["traj"], r["grade"]
    comp = dict(l0=float(np.linalg.norm(C - L)), l1=float(np.linalg.norm(r["W"])),
                l2=float(np.linalg.norm(Q4 - N4)), l3=float(np.linalg.norm(r["U"])))
    ls = sorted(comp.values())
    polos = [polo(C, L, t["t1"], t["t3"]) for t in traj]
    p1 = poses[prec[0]]
    resultado = dict(
        frames=FRAMES, frame_ref=FRAME_REF, precisao=prec, cheque=cheque,
        poses={str(f): dict(flexao=float(poses[f]["flexao"]), x=float(poses[f]["p"][0]), y=float(poses[f]["p"][1]),
                            JP=float(poses[f]["JP"]), dP=float(poses[f]["dP"])) for f in poses},
        J4=J4.tolist(), P4=P4.tolist(), Q4=Q4.tolist(), N4=N4.tolist(), C=C.tolist(), L=L.tolist(), comprimentos=comp,
        grashof=bool(ls[0] + ls[3] <= ls[1] + ls[2]), grashof_sl=ls[0] + ls[3], grashof_pq=ls[1] + ls[2],
        grashof_ord=ls, erros={str(k): v for k, v in r["erros"].items()},
        mu_min=float(min(r["mus"])), mu_max=float(max(r["mus"])),
        polo_0=polos[0].tolist(), polo_fim=polos[-1].tolist(), phi_max=float(grade[-1]),
        diade=dict(alpha2=float(np.degrees(poses[prec[1]]["alpha"] - p1["alpha"])),
                   alpha3=float(np.degrees(poses[prec[2]]["alpha"] - p1["alpha"])),
                   delta2=(poses[prec[1]]["p"] - p1["p"]).tolist(), delta3=(poses[prec[2]]["p"] - p1["p"]).tolist(),
                   beta2=float(np.degrees(r["betas"][0])), beta3=float(np.degrees(r["betas"][1])),
                   gamma2=float(np.degrees(r["gammas"][0])), gamma3=float(np.degrees(r["gammas"][1])),
                   W=r["W"].tolist(), Z=r["Z"].tolist(), U=r["U"].tolist(), S=r["S"].tolist()),
        combinacoes=[dict(precisao=c[1], cheque=c[2], erro_cheque=float(c[3]["erros"][c[2][0]]),
                          mu_min=float(min(c[3]["mus"]))) for c in combos],
    )
    with open(os.path.join(DADOS, "resultado_parte1.json"), "w") as f:
        json.dump(resultado, f, indent=2)
    print(json.dumps({k: v for k, v in resultado.items() if k != "poses"}, indent=2))

    # ------------------------------ figuras ------------------------------
    cores = {307: "#1b9e77", 390: "#d95f02", 423: "#7570b3", 456: "#e7298a"}
    # (a) marcadores na base da coxa
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for fr in FRAMES:
        pts = {k: na_coxa(D[fr], k) for k in range(1, 12)}
        for grupo, estilo in ((MARC_COXA, "s"), (MARC_PERNA, "o"), (PONTOS_PE, "^")):
            xy = np.array([pts[k] for k in grupo])
            ax.plot(xy[:, 0], xy[:, 1], estilo, color=cores[fr], ms=3.5)
        Jf = (pts[5] + pts[6]) / 2
        seq = np.array([Jf, pts[8]])
        ax.plot(seq[:, 0], seq[:, 1], "-", color=cores[fr], lw=1.2,
                label=f"frame {fr} ($\\varphi$ = {poses[fr]['flexao']:.1f}°)")
    ax.plot(0, 0, "k+", ms=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("$^0x$ (mm)")
    ax.set_ylabel("$^0y$ (mm)")
    ax.set_title("Marcadores na base 0 (coxa)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_medicoes.pdf"))
    plt.close(fig)

    # (b) mecanismo + trajetoria do tornozelo
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.8))
    idx = {round(g, 6): i for i, g in enumerate(grade)}
    for fr in FRAMES:
        t = traj[idx[round(poses[fr]["flexao"], 6)]]
        Q, N = muda_base(t["alpha"], t["p"], Q4), muda_base(t["alpha"], t["p"], N4)
        cor = cores[fr]
        ax[0].plot([C[0], Q[0]], [C[1], Q[1]], "-o", color=cor, ms=3)
        ax[0].plot([L[0], N[0]], [L[1], N[1]], "-o", color=cor, ms=3)
        ax[0].plot([Q[0], N[0]], [Q[1], N[1]], "--", color=cor, lw=1)
    pol = np.array(polos)
    ax[0].plot(pol[:, 0], pol[:, 1], "r.-", ms=2, lw=0.7, label="polo $I_{13}$ (centroide)")
    ax[0].plot(*C, "ks", ms=5)
    ax[0].plot(*L, "ks", ms=5)
    ax[0].annotate("C", C, xytext=(-10, 4), textcoords="offset points")
    ax[0].annotate("L", L, xytext=(4, 4), textcoords="offset points")
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6, loc="lower left")
    ax[0].set_xlabel("$^0x$ (mm)")
    ax[0].set_ylabel("$^0y$ (mm)")
    ax[0].set_title("Quadrilátero sintetizado (base 0)")
    tor = np.array([muda_base(t["alpha"], t["p"], P4) for t in traj])
    ax[1].plot(tor[:, 0], tor[:, 1], "r-", lw=1.2, label="mecanismo")
    for fr in FRAMES:
        tm = muda_base(poses[fr]["alpha"], poses[fr]["p"], P4)
        mk = "*" if fr in prec else "o"
        ax[1].plot(*tm, mk, color=cores[fr], ms=9 if mk == "*" else 6,
                   label=f"{fr} ({'precisão' if fr in prec else 'verificação'})")
    ax[1].plot(0, 0, "k+")
    ax[1].set_aspect("equal")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6)
    ax[1].set_xlabel("$^0x$ (mm)")
    ax[1].set_ylabel("$^0y$ (mm)")
    ax[1].set_title("Trajetória do tornozelo (ponto 8)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_mecanismo.pdf"))
    plt.close(fig)

    # (c) angulo de transmissao
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.plot(grade, r["mus"])
    ax.axhline(30, color="r", ls="--", lw=0.8)
    for fr in FRAMES:
        ax.axvline(poses[fr]["flexao"], color=cores[fr], ls=":", lw=0.8)
    ax.set_xlabel("flexão do joelho $\\varphi$ (°)")
    ax.set_ylabel("$\\mu$ (°)")
    ax.set_title("Ângulo de transmissão")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_erro_mu.pdf"))
    plt.close(fig)

    # (d) gabarito 1:1 da maquete
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(105, 285, "PMR3220 – Grupo C – Gabarito 1:1 da maquete (imprimir em escala 100%)", ha="center", fontsize=10)
    ax.plot([20, 120], [270, 270], "k-")
    ax.text(70, 272, "escala: 100 mm", ha="center", fontsize=8)

    def elo(x0, y0, c, nome):
        w = 12.0
        ax.add_patch(plt.Rectangle((x0 - w / 2, y0 - w / 2), c + w, w, fill=False, lw=0.8))
        for xx in (x0, x0 + c):
            ax.add_patch(plt.Circle((xx, y0), 2.1, fill=False, lw=0.8))
        ax.text(x0 + c / 2, y0 + 8, f"{nome}: {c:.1f} mm entre centros (furo Ø4,2)", ha="center", fontsize=7)

    elo(30, 240, comp["l1"], "Elo CQ")
    elo(30, 215, comp["l3"], "Elo LN")

    def placa(orig, pts, nomes, titulo):
        pts = np.array(pts)
        mn, mx = pts.min(0) - 15, pts.max(0) + 15
        ax.add_patch(plt.Rectangle(orig + mn, *(mx - mn), fill=False, lw=0.8))
        for p_, n in zip(pts, nomes):
            ax.add_patch(plt.Circle(orig + p_, 2.1, fill=False, lw=0.8))
            ax.text(*(orig + p_ + np.array([3, 3])), n, fontsize=7)
        ax.plot(*(orig + np.array([0.0, 0.0])), "k+", ms=8)
        ax.text(*(orig + np.array([mn[0], mx[1] + 3])), titulo, fontsize=7)

    placa(np.array([60.0, 130.0]), [C, L, np.zeros(2)], ["C", "L", ""], "Placa distal da coxa (+ = origem da base 0)")
    placa(np.array([160.0, 150.0]), [Q4, N4, np.zeros(2)], ["Q", "N", ""], "Placa proximal da perna (+ = origem da base 4)")
    ax.text(105, 45, "Coxa e perna nas dimensões medidas (coxa ≈ 375 mm; joelho–tornozelo ≈ 420 mm).\n"
                     "Elos CQ e LN em planos diferentes (um de cada lado das placas).",
            ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "p1_gabarito_maquete.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
