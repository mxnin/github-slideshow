"""
PMR3220 - Relatorio parcial (Parte I, ate o processamento das medicoes).

Le os dados do Tracker (dados/tracker_rascunho.csv), escreve os marcadores na base da
coxa (origem no ponto medio de 3-4, eixos paralelos aos da imagem: o movimento angular
do quadril e desconsiderado) e calcula a flexao do joelho pelo angulo do segmento da
perna, do ponto medio de 5-6 (abaixo do joelho) ao ponto 8 (tornozelo).
Gera dados/poses_parcial.csv, dados/valores_parcial.tex e figuras/parcial_medicoes.pdf.
"""
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
FIG = os.path.join(RAIZ, "figuras")
FRAME_REF = 307
FRAMES = [307, 390, 423, 456]

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})


def f(x, n=1):
    return f"{x:.{n}f}".replace(".", "{,}").replace("-", r"\ensuremath{-}")


def main():
    D = {}
    with open(os.path.join(DADOS, "tracker_rascunho.csv")) as fh:
        for r in csv.DictReader(fh):
            D.setdefault(int(r["frame"]), {})[int(r["ponto"])] = 1000.0 * np.array([float(r["x_m"]), float(r["y_m"])])

    def na_coxa(fr, k):                      # 0r_k = imr_k - (imr_3 + imr_4)/2   ([imR0] = I)
        P = D[fr]
        return P[k] - (P[3] + P[4]) / 2

    lin = {}
    for fr in sorted(D):
        J = (na_coxa(fr, 5) + na_coxa(fr, 6)) / 2
        T = na_coxa(fr, 8)
        v = T - J
        lin[fr] = dict(J=J, T=T, ang=np.degrees(np.arctan2(v[1], v[0])), comp=np.linalg.norm(v))
    a0 = lin[FRAME_REF]["ang"]
    for fr in lin:
        lin[fr]["flexao"] = (a0 - lin[fr]["ang"]) % 360.0

    with open(os.path.join(DADOS, "poses_parcial.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "flexao_graus", "x_J_mm", "y_J_mm", "x_P_mm", "y_P_mm", "perna_mm"])
        for fr, d in sorted(lin.items()):
            w.writerow([fr, f"{d['flexao']:.1f}", f"{d['J'][0]:.1f}", f"{d['J'][1]:.1f}",
                        f"{d['T'][0]:.1f}", f"{d['T'][1]:.1f}", f"{d['comp']:.0f}"])

    L = []
    m = lambda nome, val: L.append(f"\\newcommand{{\\{nome}}}{{{val}}}")
    for fr, nome in zip(FRAMES, "ABCD"):
        m(f"phi{nome}", f(lin[fr]["flexao"]))
    comps = [lin[fr]["comp"] for fr in FRAMES]
    m("pernamin", f(min(comps), 0)); m("pernamax", f(max(comps), 0))
    desl = np.linalg.norm(lin[FRAMES[-1]]["J"] - lin[FRAME_REF]["J"])
    m("deslJ", f(desl, 0))
    rows = []
    for fr, d in sorted(lin.items()):
        tag = "" if fr in FRAMES else r" $\dagger$"
        rows.append(f"{fr}{tag} & {f(d['flexao'])} & {f(d['J'][0])} & {f(d['J'][1])} & "
                    f"{f(d['T'][0])} & {f(d['T'][1])} & {f(d['comp'], 0)} \\\\")
    L.append("\\newcommand{\\tabelaposes}{" + "\n".join(rows) + "}")
    with open(os.path.join(DADOS, "valores_parcial.tex"), "w") as fh:
        fh.write("% gerado por scripts/parte1_parcial.py -- nao editar\n" + "\n".join(L) + "\n")

    cores = {307: "#1b9e77", 390: "#d95f02", 423: "#7570b3", 456: "#e7298a"}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for fr in FRAMES:
        pts = {k: na_coxa(fr, k) for k in range(1, 12)}
        for grupo, estilo in (([1, 2, 3, 4], "s"), ([5, 6, 7, 8, 9], "o"), ([10, 11], "^")):
            xy = np.array([pts[k] for k in grupo])
            ax.plot(xy[:, 0], xy[:, 1], estilo, color=cores[fr], ms=3.5)
        J, T = lin[fr]["J"], lin[fr]["T"]
        ax.plot([J[0], T[0]], [J[1], T[1]], "-", color=cores[fr], lw=1.2,
                label=f"frame {fr} ($\\varphi$ = {lin[fr]['flexao']:.1f}°)")
    ax.plot(0, 0, "k+", ms=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("$^0x$ (mm)")
    ax.set_ylabel("$^0y$ (mm)")
    ax.set_title("Marcadores na base 0 (coxa)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parcial_medicoes.pdf"))
    plt.close(fig)
    for fr in sorted(lin):
        print(fr, round(lin[fr]["flexao"], 1), np.round(lin[fr]["J"], 1), round(lin[fr]["comp"]))


if __name__ == "__main__":
    main()
