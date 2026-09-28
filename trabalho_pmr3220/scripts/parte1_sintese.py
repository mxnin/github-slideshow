"""
PMR3220 - Parte I: sintese dimensional de um quadrilatero articulado (joelho policentrico)
por posicoes de precisao do elo acoplador (perna), a partir das MEDICOES do Grupo C
(rastreamento no software Tracker, arquivo dados/tracker_rascunho.csv).

Notacao do metodo matricial (PMR3220):
  base 0 = coxa (fixa): origem no ponto medio dos marcadores 3 e 4 (extremidade distal da
           coxa), eixos paralelos aos da imagem (movimento angular do quadril desconsiderado);
  base 4 = perna: coincide com a base 0 no frame de referencia (307, membro estendido);
  bases 1, 2, 3 = elos CQ, QN (acoplador) e LN do quadrilatero.
Unidades: mm e graus. Flexao do joelho phi > 0  <=>  rotacao da perna alpha = -phi.
"""
import csv
import itertools
import json
import os

import numpy as np
from scipy.optimize import fsolve, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIG = os.path.join(RAIZ, "figuras")
DADOS = os.path.join(RAIZ, "dados")

FRAME_REF = 307                       # membro estendido (configuracao inicial)
FRAMES = [307, 390, 423, 456]         # frames selecionados pelo grupo
MARC_COXA = [1, 2, 3, 4]
MARC_PERNA = [5, 6, 7, 8, 9]          # marcadores usados na pose da perna
PONTO_TORNOZELO = 8                   # conexao perna-pe
PONTOS_PE = [10, 11]

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})


# ---------------------------------------------------------------------------
# utilitarios do metodo matricial
# ---------------------------------------------------------------------------
def rot(a):
    """Bloco 2x2 de Rot(a, z)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def T_hom(a, p):
    """Transformacao homogenea plana [Rot(a, z) | p; 0 0 1]."""
    T = np.eye(3)
    T[:2, :2] = rot(a)
    T[:2, 2] = p
    return T


def aplica(T, r):
    return (T @ np.array([r[0], r[1], 1.0]))[:2]


# ---------------------------------------------------------------------------
# 1) dados medidos -> poses 0T4 da perna
# ---------------------------------------------------------------------------
def le_tracker(caminho):
    D = {}
    with open(caminho) as f:
        for r in csv.DictReader(f):
            D.setdefault(int(r["frame"]), {})[int(r["ponto"])] = 1000.0 * np.array([float(r["x_m"]), float(r["y_m"])])
    return D


def base_coxa(P):
    """imT0 da base da coxa em um frame. O movimento angular do quadril (rotacao da coxa
    em relacao ao tronco) e desconsiderado: a coxa mantem a orientacao da imagem
    (X0 horizontal para a frente, Y0 vertical para cima) e acompanha apenas a translacao
    da regiao do joelho, com origem no ponto medio dos marcadores 3 e 4. Os marcadores
    1 e 2 (quadril) nao sao usados na Parte I; o comprimento 1-2 -> 3-4 e so informativo."""
    o = (P[3] + P[4]) / 2
    comp = np.linalg.norm((P[1] + P[2]) / 2 - o)
    return T_hom(0.0, o), comp


def na_coxa(P, k):
    T, _ = base_coxa(P)
    return aplica(np.linalg.inv(T), P[k])            # 0r = imR0^T (imr - imr_O)


def procrustes(U, X):
    """Pose (alpha, p) que minimiza sum |Rot(alpha) u_k + p - x_k|^2."""
    ub, xb = U.mean(0), X.mean(0)
    u, x = U - ub, X - xb
    a = np.arctan2(np.sum(u[:, 0] * x[:, 1] - u[:, 1] * x[:, 0]), np.sum(u[:, 0] * x[:, 0] + u[:, 1] * x[:, 1]))
    p = xb - rot(a) @ ub
    res = np.sqrt(np.mean(np.sum((X - (U @ rot(a).T + p)) ** 2, axis=1)))
    return a, p, res


def poses_medidas(D):
    U = np.array([na_coxa(D[FRAME_REF], k) for k in MARC_PERNA])       # 4r_k (base 4 = base 0 no frame 307)
    poses = {}
    for fr in sorted(D):
        X = np.array([na_coxa(D[fr], k) for k in MARC_PERNA])
        a, p, res = procrustes(U, X)
        _, comp = base_coxa(D[fr])
        poses[fr] = dict(alpha=a, p=p, flexao=-np.degrees(a) + 0.0, resid=res, coxa=comp)
    locais = {k: na_coxa(D[FRAME_REF], k) for k in MARC_PERNA + PONTOS_PE}
    return poses, locais


# ---------------------------------------------------------------------------
# 2) sintese (ponto-centro) e analise de posicao (cadeia fechada)
# ---------------------------------------------------------------------------
def circuncentro(p1, p2, p3):
    A = 2.0 * np.array([p2 - p1, p3 - p1])
    b = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    return np.linalg.solve(A, b)


def sintese(poses3, Q4, N4):
    Qs = [aplica(T_hom(ps["alpha"], ps["p"]), Q4) for ps in poses3]
    Ns = [aplica(T_hom(ps["alpha"], ps["p"]), N4) for ps in poses3]
    return circuncentro(*Qs), circuncentro(*Ns)


def montagem(C, L, Q4, N4, pose_ini, alphas):
    """Para cada rotacao alpha da perna, resolve a posicao 0rO4 que satisfaz
    |Q - C| = l1 e |N - L| = l3 (equacao de fechamento), com continuidade de ramo."""
    l1 = np.linalg.norm(aplica(T_hom(pose_ini["alpha"], pose_ini["p"]), Q4) - C)
    l3 = np.linalg.norm(aplica(T_hom(pose_ini["alpha"], pose_ini["p"]), N4) - L)
    t = np.array(pose_ini["p"], dtype=float)
    out = []
    for a in alphas:
        def f(tt):
            q = tt + rot(a) @ Q4
            n = tt + rot(a) @ N4
            return [np.linalg.norm(q - C) - l1, np.linalg.norm(n - L) - l3]
        sol, _, ok, _ = fsolve(f, t, full_output=True, xtol=1e-12)
        if ok != 1 or np.max(np.abs(f(sol))) > 1e-6 or np.linalg.norm(sol - t) > 20:
            return None
        t = sol
        out.append(dict(alpha=a, p=sol.copy()))
    return out


def angulo_transmissao(L, Q, N):
    u, v = Q - N, L - N
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return min(ang, 180 - ang)


def polo(C, L, Q, N):
    d1, d2 = Q - C, N - L
    s = np.linalg.solve(np.array([d1, -d2]).T, L - C)
    return C + s[0] * d1


def avalia(x, poses, prec, cheque, P4, retorna=False):
    Q4, N4 = x[:2], x[2:]
    if np.linalg.norm(Q4 - N4) < 15:
        return 1e9
    try:
        C, L = sintese([poses[f] for f in prec], Q4, N4)
    except np.linalg.LinAlgError:
        return 1e9
    phi_max = max(poses[f]["flexao"] for f in FRAMES)
    grade = sorted(set(np.round(np.arange(0, phi_max + 0.5, 1.0), 6)) | {round(poses[f]["flexao"], 6) for f in FRAMES})
    grade = [g for g in grade if g >= 0]
    traj = montagem(C, L, Q4, N4, poses[FRAME_REF], [-np.radians(g) for g in grade])
    if traj is None:
        return 1e9
    porphi = dict(zip(grade, traj))
    erros = {}
    for f in FRAMES:
        ps = porphi[round(poses[f]["flexao"], 6)]
        erros[f] = float(np.linalg.norm(aplica(T_hom(ps["alpha"], ps["p"]), P4) - aplica(T_hom(poses[f]["alpha"], poses[f]["p"]), P4)))
    mus = [angulo_transmissao(L, aplica(T_hom(t["alpha"], t["p"]), Q4), aplica(T_hom(t["alpha"], t["p"]), N4)) for t in traj]
    pen = 50 * max(0, 30 - min(mus))
    for piv in (C, L):                                  # pivos fixos na regiao distal da coxa
        pen += 5 * (max(0, abs(piv[0]) - 60) + max(0, piv[1] - 60) + max(0, -40 - piv[1]))
    l1 = np.linalg.norm(aplica(T_hom(0, [0, 0]), Q4) - C)
    comps = [np.linalg.norm(C - L), np.linalg.norm(Q4 - C), np.linalg.norm(Q4 - N4), np.linalg.norm(N4 - L)]
    for c in comps:
        pen += 5 * (max(0, 20 - c) + max(0, c - 120))
    custo = sum(erros[f] for f in cheque) + pen
    if retorna:
        return dict(C=C, L=L, Q4=Q4, N4=N4, grade=grade, traj=traj, erros=erros, mus=mus, custo=custo, comps=comps)
    return custo


def otimiza(poses, prec, cheque, P4, semente=3220):
    rng = np.random.default_rng(semente)
    melhor = (1e9, None)
    for _ in range(3000):
        x = rng.uniform([-60, -90, -60, -90], [60, 0, 60, 0])
        c = avalia(x, poses, prec, cheque, P4)
        if c < melhor[0]:
            melhor = (c, x)
    if melhor[1] is None:
        return None
    r = minimize(lambda x: avalia(x, poses, prec, cheque, P4), melhor[1], method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-5, maxiter=4000))
    x = np.round(r.x, 1)
    return avalia(x, poses, prec, cheque, P4, retorna=True)


# ---------------------------------------------------------------------------
# 3) programa principal
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG, exist_ok=True)
    D = le_tracker(os.path.join(DADOS, "tracker_rascunho.csv"))
    poses, locais = poses_medidas(D)
    P4 = locais[PONTO_TORNOZELO]

    with open(os.path.join(DADOS, "poses_medidas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "flexao_graus", "x_O4_mm", "y_O4_mm", "x_P_mm", "y_P_mm", "residuo_mm", "coxa_mm"])
        for fr, ps in sorted(poses.items()):
            tp = aplica(T_hom(ps["alpha"], ps["p"]), P4)
            w.writerow([fr, f"{ps['flexao']:.2f}", f"{ps['p'][0]:.2f}", f"{ps['p'][1]:.2f}",
                        f"{tp[0]:.1f}", f"{tp[1]:.1f}", f"{ps['resid']:.1f}", f"{ps['coxa']:.0f}"])

    # escolha das tres posicoes de precisao: todas as combinacoes que incluem a extensao
    combos = []
    for trio in itertools.combinations(FRAMES[1:], 2):
        prec = [FRAME_REF, *trio]
        cheque = [f for f in FRAMES if f not in prec]
        r = otimiza(poses, prec, cheque, P4)
        if r is not None:
            combos.append((r["custo"], prec, cheque, r))
            print("precisao", prec, "cheque", cheque, "erro cheque %.2f mm" % sum(r["erros"][c] for c in cheque),
                  "mu_min %.1f" % min(r["mus"]))
    combos.sort(key=lambda z: z[0])
    _, prec, cheque, r = combos[0]
    C, L, Q4, N4, traj, grade = r["C"], r["L"], r["Q4"], r["N4"], r["traj"], r["grade"]
    Q0 = aplica(T_hom(0, [0, 0]), Q4)
    N0 = aplica(T_hom(0, [0, 0]), N4)
    comp = dict(l0=float(np.linalg.norm(C - L)), l1=float(np.linalg.norm(Q0 - C)),
                l2=float(np.linalg.norm(Q4 - N4)), l3=float(np.linalg.norm(N0 - L)))
    ls = sorted(comp.values())
    polos = [polo(C, L, aplica(T_hom(t["alpha"], t["p"]), Q4), aplica(T_hom(t["alpha"], t["p"]), N4)) for t in traj]
    resultado = dict(
        frames=FRAMES, frame_ref=FRAME_REF, precisao=prec, cheque=cheque,
        poses={str(f): dict(flexao=float(poses[f]["flexao"]), x=float(poses[f]["p"][0]), y=float(poses[f]["p"][1]),
                            resid=float(poses[f]["resid"]), coxa=float(poses[f]["coxa"])) for f in poses},
        P4=P4.tolist(), Q4=Q4.tolist(), N4=N4.tolist(), C=C.tolist(), L=L.tolist(), comprimentos=comp,
        grashof=bool(ls[0] + ls[3] <= ls[1] + ls[2]), grashof_sl=ls[0] + ls[3], grashof_pq=ls[1] + ls[2],
        grashof_ord=ls, erros={str(k): v for k, v in r["erros"].items()},
        mu_min=float(min(r["mus"])), mu_max=float(max(r["mus"])),
        polo_0=polos[0].tolist(), polo_fim=polos[-1].tolist(), phi_max=float(grade[-1]),
        combinacoes=[dict(precisao=c[1], cheque=c[2], erro_cheque=float(sum(c[3]["erros"][k] for k in c[2])),
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
        seq = np.array([pts[k] for k in [5, 6, 9, 8, 7, 5]])
        ax.plot(seq[:, 0], seq[:, 1], "-", color=cores[fr], lw=0.8,
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
        T = T_hom(t["alpha"], t["p"])
        Q, N = aplica(T, Q4), aplica(T, N4)
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
    tor = np.array([aplica(T_hom(t["alpha"], t["p"]), P4) for t in traj])
    ax[1].plot(tor[:, 0], tor[:, 1], "r-", lw=1.2, label="mecanismo")
    for fr in FRAMES:
        tm = aplica(T_hom(poses[fr]["alpha"], poses[fr]["p"]), P4)
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
