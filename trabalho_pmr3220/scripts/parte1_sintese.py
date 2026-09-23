"""
PMR3220 - Parte I: sintese dimensional de um quadrilatero articulado (joelho policentrico)
por tres posicoes de precisao do elo acoplador (perna).

Convencoes (plano sagital, perna direita vista pelo lado direito):
  - referencial da COXA (fixo): origem no centro do joelho em extensao,
    x apontando para frente (anterior), y apontando para cima (proximal); unidades em mm.
  - referencial da PERNA (acoplador): mesma origem/orientacao quando o joelho esta estendido.
  - flexao do joelho phi > 0 corresponde a uma rotacao da perna no sentido HORARIO
    (angulo do acoplador = -phi).

Dados de entrada: arquivo CSV com as poses medidas da perna em relacao a coxa
(dados/poses_medidas.csv). Se o arquivo nao existir, ele e gerado a partir de um
modelo anatomico de quatro barras cruzadas (ligamentos cruzados, Menschik 1974 /
O'Connor et al. 1989) - valores de REFERENCIA que devem ser substituidos pelas
medicoes experimentais do grupo.
"""
import csv
import json
import os

import numpy as np
from scipy.optimize import brentq, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIG = os.path.join(RAIZ, "figuras")
DADOS = os.path.join(RAIZ, "dados")

L_PERNA = 450.0   # mm, joelho -> tornozelo (OP)
L_PE = 250.0      # mm, tornozelo -> ponta do pe (PR)
PHI_PREC = [0.0, 45.0, 90.0]           # posicoes de precisao (graus de flexao)
PHI_MAX = 120.0

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def pose_ponto(pose, p_local):
    """pose = (x, y, ang) do referencial da perna; p_local em coordenadas da perna."""
    x, y, a = pose
    return np.array([x, y]) + rot(a) @ np.asarray(p_local)


# ---------------------------------------------------------------------------
# 1) Modelo anatomico de referencia (quatro barras cruzadas dos ligamentos)
# ---------------------------------------------------------------------------
# Insercoes (mm) na configuracao estendida, referencial da coxa.
ACL_F = np.array([-12.0, 22.0])   # LCA - insercao femoral (posterior)
PCL_F = np.array([8.0, 20.0])     # LCP - insercao femoral (anterior)
ACL_T = np.array([12.0, -5.0])    # LCA - insercao tibial (anterior)
PCL_T = np.array([-16.0, -10.0])  # LCP - insercao tibial (posterior)
L_ACL = np.linalg.norm(ACL_T - ACL_F)
L_PCL = np.linalg.norm(PCL_T - PCL_F)


def pose_anatomica(phi_deg, chute=(0.0, 0.0)):
    """Pose da perna (x, y, ang) para uma flexao phi, impondo comprimentos
    constantes das fibras isometricas do LCA e do LCP."""
    a = -np.radians(phi_deg)

    def res(t):
        t = np.asarray(t)
        pa = t + rot(a) @ ACL_T
        pp = t + rot(a) @ PCL_T
        return [np.linalg.norm(pa - ACL_F) - L_ACL, np.linalg.norm(pp - PCL_F) - L_PCL]

    from scipy.optimize import fsolve
    t = fsolve(res, chute, xtol=1e-12)
    return np.array([t[0], t[1], a])


def gera_poses_referencia(caminho):
    phis = np.arange(0.0, PHI_MAX + 1e-9, 15.0)
    chute = (0.0, 0.0)
    linhas = []
    for ph in phis:
        p = pose_anatomica(ph, chute)
        chute = p[:2]
        tornozelo = pose_ponto(p, [0.0, -L_PERNA])
        linhas.append((ph, p[0], p[1], tornozelo[0], tornozelo[1]))
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flexao_graus", "joelho_x_mm", "joelho_y_mm", "tornozelo_x_mm", "tornozelo_y_mm"])
        for l in linhas:
            w.writerow([f"{v:.3f}" for v in l])


def le_poses(caminho):
    """Le poses: posicao do ponto de referencia do joelho (origem da perna) e do
    tornozelo; o angulo da perna e obtido do segmento joelho->tornozelo."""
    poses = {}
    with open(caminho) as f:
        for r in csv.DictReader(f):
            ph = float(r["flexao_graus"])
            k = np.array([float(r["joelho_x_mm"]), float(r["joelho_y_mm"])])
            t = np.array([float(r["tornozelo_x_mm"]), float(r["tornozelo_y_mm"])])
            d = t - k
            ang = np.arctan2(d[1], d[0]) + np.pi / 2  # perna estendida aponta para -y
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            poses[ph] = np.array([k[0], k[1], ang])
    return poses


# ---------------------------------------------------------------------------
# 2) Sintese por tres posicoes de precisao (pivos moveis escolhidos)
# ---------------------------------------------------------------------------
def circuncentro(p1, p2, p3):
    """Centro do circulo que passa por tres pontos (solucao do sistema linear das
    mediatrizes): 2(p2-p1).c = |p2|^2-|p1|^2 ; 2(p3-p1).c = |p3|^2-|p1|^2."""
    A = 2.0 * np.array([p2 - p1, p3 - p1])
    b = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    return np.linalg.solve(A, b)


def sintese(poses3, Q_loc, N_loc):
    Qs = [pose_ponto(p, Q_loc) for p in poses3]
    Ns = [pose_ponto(p, N_loc) for p in poses3]
    C = circuncentro(*Qs)
    L = circuncentro(*Ns)
    return C, L, Qs, Ns


def montagem(C, L, Q_loc, N_loc, pose_ini, angulos):
    """Simula o quadrilatero: para cada angulo do acoplador (perna) resolve a
    posicao da perna que satisfaz |Q-C| = r_C e |N-L| = r_L (continuidade de ramo)."""
    rC = np.linalg.norm(pose_ponto(pose_ini, Q_loc) - C)
    rL = np.linalg.norm(pose_ponto(pose_ini, N_loc) - L)
    from scipy.optimize import fsolve
    t = pose_ini[:2].copy()
    saida = []
    for a in angulos:
        def res(tt):
            q = tt + rot(a) @ Q_loc
            n = tt + rot(a) @ N_loc
            return [np.linalg.norm(q - C) - rC, np.linalg.norm(n - L) - rL]
        sol, info, ok, _ = fsolve(res, t, full_output=True, xtol=1e-12)
        if ok != 1 or np.max(np.abs(res(sol))) > 1e-6 or np.linalg.norm(sol - t) > 15:
            return None
        t = sol
        saida.append(np.array([t[0], t[1], a]))
    return np.array(saida)


def angulo_transmissao(C, L, Q, N):
    """angulo entre o acoplador (Q-N) e o balancim de saida (L-N)."""
    u = Q - N
    v = L - N
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return min(ang, 180 - ang)


def centro_instantaneo(C, L, Q, N):
    """Polo relativo perna/coxa: intersecao das retas C-Q e L-N."""
    d1 = Q - C
    d2 = N - L
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, L - C)
    return C + s[0] * d1


def avalia(Q_loc, N_loc, poses_ref, phis_ref, retorna=False):
    p3 = [poses_ref[p] for p in PHI_PREC]
    try:
        C, L, _, _ = sintese(p3, Q_loc, N_loc)
    except np.linalg.LinAlgError:
        return 1e9
    angs = -np.radians(np.arange(0, PHI_MAX + 0.01, 1.0))
    traj = montagem(C, L, Q_loc, N_loc, p3[0], angs)
    if traj is None:
        return 1e9
    # erro estrutural no tornozelo para os angulos de referencia
    phis_sim = np.arange(0, PHI_MAX + 0.01, 1.0)
    erros = []
    for ph in phis_ref:
        pr = poses_ref[ph]
        ps = traj[int(round(ph))]
        erros.append(np.linalg.norm(pose_ponto(pr, [0, -L_PERNA]) - pose_ponto(ps, [0, -L_PERNA])))
    # restricoes de projeto (penalidades)
    pen = 0.0
    mu_min = 1e9
    for ps in traj:
        Q = pose_ponto(ps, Q_loc)
        N = pose_ponto(ps, N_loc)
        mu_min = min(mu_min, angulo_transmissao(C, L, Q, N))
    pen += 50 * max(0, 30 - mu_min)
    for piv in (C, L):
        # pivos fixos devem ficar no volume da extremidade distal da coxa
        pen += 5 * max(0, abs(piv[0]) - 45) + 5 * max(0, piv[1] - 90) + 5 * max(0, -piv[1])
    for lng in (np.linalg.norm(pose_ponto(p3[0], Q_loc) - C), np.linalg.norm(pose_ponto(p3[0], N_loc) - L),
                np.linalg.norm(Q_loc - N_loc), np.linalg.norm(C - L)):
        pen += 5 * max(0, 20 - lng) + 5 * max(0, lng - 90)
    custo = max(erros) + pen
    if retorna:
        return dict(C=C, L=L, traj=traj, erros=erros, mu_min=mu_min, phis_sim=phis_sim, custo=custo)
    return custo


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DADOS, exist_ok=True)
    arq = os.path.join(DADOS, "poses_medidas.csv")
    if not os.path.exists(arq):
        gera_poses_referencia(arq)
    poses_ref = le_poses(arq)
    phis_ref = sorted(poses_ref)

    # busca em grade + refinamento (Nelder-Mead) dos pivos moveis na perna
    melhor = (1e9, None)
    rng = np.random.default_rng(3220)
    for _ in range(1500):
        x = rng.uniform([-40, -45, -40, -45], [40, -5, 40, -5])
        if np.linalg.norm(x[:2] - x[2:]) < 20:
            continue
        c = avalia(x[:2], x[2:], poses_ref, phis_ref)
        if c < melhor[0]:
            melhor = (c, x)
    res = minimize(lambda x: avalia(x[:2], x[2:], poses_ref, phis_ref), melhor[1],
                   method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=4000))
    x = np.round(res.x, 1)  # cotas de fabricacao em decimos de mm
    Q_loc, N_loc = x[:2], x[2:]
    r = avalia(Q_loc, N_loc, poses_ref, phis_ref, retorna=True)
    C, L, traj = r["C"], r["L"], r["traj"]
    p0 = poses_ref[0.0]
    Q0, N0 = pose_ponto(p0, Q_loc), pose_ponto(p0, N_loc)

    comp = dict(
        CL_quadro=float(np.linalg.norm(C - L)),
        CQ=float(np.linalg.norm(Q0 - C)),
        LN=float(np.linalg.norm(N0 - L)),
        QN_acoplador=float(np.linalg.norm(Q_loc - N_loc)),
    )
    ls = sorted(comp.values())
    grashof = ls[0] + ls[3] <= ls[1] + ls[2]

    # verificacao nas posicoes de precisao
    prec = []
    for ph in PHI_PREC:
        pr = poses_ref[ph]
        ps = traj[int(round(ph))]
        prec.append(float(np.linalg.norm(pose_ponto(pr, [0, -L_PERNA]) - pose_ponto(ps, [0, -L_PERNA]))))

    # centros instantaneos: mecanismo e modelo anatomico
    ci_mec, ci_anat = [], []
    for ph in range(0, int(PHI_MAX) + 1, 5):
        ps = traj[ph]
        ci_mec.append(centro_instantaneo(C, L, pose_ponto(ps, Q_loc), pose_ponto(ps, N_loc)))
        pa = pose_anatomica(ph, pose_anatomica(max(ph - 5, 0))[:2])
        ci_anat.append(centro_instantaneo(ACL_F, PCL_F, pose_ponto(pa, ACL_T), pose_ponto(pa, PCL_T)))
    ci_mec, ci_anat = np.array(ci_mec), np.array(ci_anat)

    # angulo de transmissao ao longo do movimento
    mus = [angulo_transmissao(C, L, pose_ponto(ps, Q_loc), pose_ponto(ps, N_loc)) for ps in traj]

    resultado = dict(
        dados_referencia=dict(L_ACL=float(L_ACL), L_PCL=float(L_PCL),
                              femoral=float(np.linalg.norm(ACL_F - PCL_F)),
                              tibial=float(np.linalg.norm(ACL_T - PCL_T))),
        posicoes_precisao={f"{ph:.0f}": [float(v) for v in (poses_ref[ph][0], poses_ref[ph][1], np.degrees(poses_ref[ph][2]))]
                           for ph in PHI_PREC},
        Q_perna=Q_loc.tolist(), N_perna=N_loc.tolist(), C_coxa=C.tolist(), L_coxa=L.tolist(),
        comprimentos=comp, grashof=bool(grashof),
        erro_tornozelo_max=float(max(r["erros"])),
        erros_tornozelo={f"{ph:.0f}": float(e) for ph, e in zip(phis_ref, r["erros"])},
        erro_precisao=prec, mu_min=float(min(mus)), mu_max=float(max(mus)),
        ci_mec_0=ci_mec[0].tolist(), ci_mec_120=ci_mec[-1].tolist(),
    )
    with open(os.path.join(DADOS, "resultado_parte1.json"), "w") as f:
        json.dump(resultado, f, indent=2)
    print(json.dumps(resultado, indent=2))

    # ------------------------------ figuras ------------------------------
    # (a) mecanismo em varias posicoes
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
    cores = plt.cm.viridis(np.linspace(0, 0.9, 4))
    for cor, ph in zip(cores, [0, 45, 90, 120]):
        ps = traj[ph]
        Q, N = pose_ponto(ps, Q_loc), pose_ponto(ps, N_loc)
        ax[0].plot([C[0], Q[0]], [C[1], Q[1]], "-o", color=cor, ms=3)
        ax[0].plot([L[0], N[0]], [L[1], N[1]], "-o", color=cor, ms=3)
        ax[0].plot([Q[0], N[0]], [Q[1], N[1]], "--", color=cor, lw=1)
        ax[0].annotate(f"{ph}°", Q, fontsize=7, color=cor, xytext=(3, -8), textcoords="offset points")
    ax[0].plot(*C, "ks", ms=5)
    ax[0].plot(*L, "ks", ms=5)
    ax[0].annotate("C", C, xytext=(-10, 4), textcoords="offset points")
    ax[0].annotate("L", L, xytext=(4, 4), textcoords="offset points")
    ax[0].plot(ci_mec[:, 0], ci_mec[:, 1], "r.-", ms=3, lw=0.8, label="polo do mecanismo")
    ax[0].plot(ci_anat[:, 0], ci_anat[:, 1], "b.-", ms=3, lw=0.8, label="polo anatômico (ref.)")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (mm) – anterior")
    ax[0].set_ylabel("y (mm) – proximal")
    ax[0].set_title("Quadrilátero sintetizado (referencial da coxa)")
    ax[0].legend(fontsize=6, loc="lower left")
    ax[0].grid(alpha=0.3)
    # (b) trajetoria do tornozelo
    tor_s = np.array([pose_ponto(ps, [0, -L_PERNA]) for ps in traj])
    tor_r = np.array([pose_ponto(poses_ref[ph], [0, -L_PERNA]) for ph in phis_ref])
    ax[1].plot(tor_s[:, 0], tor_s[:, 1], "r-", lw=1.2, label="mecanismo")
    ax[1].plot(tor_r[:, 0], tor_r[:, 1], "bo", ms=3, label="referência (medido)")
    for ph in PHI_PREC:
        tp = pose_ponto(poses_ref[ph], [0, -L_PERNA])
        ax[1].plot(*tp, "k*", ms=9)
    ax[1].plot([], [], "k*", label="posições de precisão")
    ax[1].plot(0, 0, "k+")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x (mm)")
    ax[1].set_ylabel("y (mm)")
    ax[1].set_title("Trajetória do tornozelo P")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_mecanismo.pdf"))
    plt.close(fig)

    # (c) erro e angulo de transmissao
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].plot(phis_ref, r["erros"], "o-", ms=3)
    for ph in PHI_PREC:
        ax[0].axvline(ph, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("flexão do joelho (°)")
    ax[0].set_ylabel("erro no tornozelo (mm)")
    ax[0].set_title("Erro estrutural")
    ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(0, PHI_MAX + 0.01, 1.0), mus)
    ax[1].axhline(30, color="r", ls="--", lw=0.8)
    ax[1].set_xlabel("flexão do joelho (°)")
    ax[1].set_ylabel(r"$\mu$ (°)")
    ax[1].set_title("Ângulo de transmissão")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_erro_mu.pdf"))
    plt.close(fig)

    # (d) gabarito 1:1 da maquete (elos com furos)
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(105, 285, "PMR3220 – Gabarito 1:1 da maquete (imprimir em escala 100%)", ha="center", fontsize=10)
    ax.plot([20, 120], [270, 270], "k-")
    ax.text(70, 272, "escala: 100 mm", ha="center", fontsize=8)

    def elo(x0, y0, comp_mm, nome):
        w = 12.0
        ax.add_patch(plt.Rectangle((x0 - w / 2, y0 - w / 2), comp_mm + w, w, fill=False, lw=0.8))
        for xx in (x0, x0 + comp_mm):
            ax.add_patch(plt.Circle((xx + w / 2 - w / 2, y0), 2.1, fill=False, lw=0.8))
        ax.text(x0 + comp_mm / 2, y0 + 8, f"{nome}: {comp_mm:.1f} mm entre centros (furo Ø4,2)", ha="center", fontsize=7)

    elo(30, 240, comp["CQ"], "Elo CQ")
    elo(30, 215, comp["LN"], "Elo LN")
    # placa da coxa com C e L e placa da perna com Q e N (coordenadas reais)
    def placa(orig, pontos, nomes, titulo):
        pts = np.array(pontos)
        mn, mx = pts.min(0) - 15, pts.max(0) + 15
        ax.add_patch(plt.Rectangle(orig + mn, *(mx - mn), fill=False, lw=0.8))
        for p, n in zip(pts, nomes):
            ax.add_patch(plt.Circle(orig + p, 2.1, fill=False, lw=0.8))
            ax.text(*(orig + p + np.array([3, 3])), n, fontsize=7)
        ax.plot(*(orig + np.array([0, 0])), "k+", ms=8)
        ax.text(*(orig + np.array([mn[0], mx[1] + 3])), titulo, fontsize=7)
    placa(np.array([60.0, 130.0]), [C, L], ["C", "L"], "Placa distal da coxa (+ = centro do joelho em extensão)")
    placa(np.array([150.0, 130.0]), [Q_loc, N_loc], ["Q", "N"], "Placa proximal da perna")
    ax.text(105, 60, "Coxa: prolongar a placa da coxa até 400 mm (quadril D).\n"
                     "Perna: prolongar a placa da perna até 450 mm (tornozelo P) e fixar o pé (250 mm) a 90°.",
            ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "p1_gabarito_maquete.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
