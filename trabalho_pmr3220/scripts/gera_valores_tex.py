"""Gera dados/valores.tex (macros e tabelas) a partir dos resultados numericos,
evitando transcricao manual de numeros para o relatorio."""
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, "dados")


def f(x, n=1):
    return f"{x:.{n}f}".replace(".", "{,}").replace("-", r"\ensuremath{-}")


def main():
    r1 = json.load(open(os.path.join(D, "resultado_parte1.json")))
    r2 = json.load(open(os.path.join(D, "resultado_parte2.json")))
    L = []
    m = lambda nome, val: L.append(f"\\newcommand{{\\{nome}}}{{{val}}}")

    # ---------------- Parte I ----------------
    ref = r1["dados_referencia"]
    m("LACL", f(ref["L_ACL"])); m("LPCL", f(ref["L_PCL"]))
    m("Lfem", f(ref["femoral"])); m("Ltib", f(ref["tibial"]))
    Q, N, C, Lc = r1["Q_perna"], r1["N_perna"], r1["C_coxa"], r1["L_coxa"]
    m("Qx", f(Q[0])); m("Qy", f(Q[1])); m("Nx", f(N[0])); m("Ny", f(N[1]))
    m("Cx", f(C[0])); m("Cy", f(C[1])); m("Lx", f(Lc[0])); m("Ly", f(Lc[1]))
    c = r1["comprimentos"]
    m("rCL", f(c["CL_quadro"])); m("rCQ", f(c["CQ"])); m("rLN", f(c["LN"])); m("rQN", f(c["QN_acoplador"]))
    m("errmax", f(r1["erro_tornozelo_max"], 2)); m("mumin", f(r1["mu_min"])); m("mumax", f(r1["mu_max"]))
    m("grashof", "satisfaz" if r1["grashof"] else "não satisfaz")
    ls = sorted(c.values())
    m("grashofs", f(ls[0])); m("grashofl", f(ls[3])); m("grashofp", f(ls[1])); m("grashofq", f(ls[2]))
    m("grashofsl", f(ls[0] + ls[3])); m("grashofpq", f(ls[1] + ls[2]))
    m("ciOx", f(r1["ci_mec_0"][0])); m("ciOy", f(r1["ci_mec_0"][1]))
    m("ciFx", f(r1["ci_mec_120"][0])); m("ciFy", f(r1["ci_mec_120"][1]))
    pp = r1["posicoes_precisao"]
    for k, nome in (("0", "A"), ("45", "B"), ("90", "C")):
        m(f"pp{nome}x", f(pp[k][0], 2)); m(f"pp{nome}y", f(pp[k][1], 2)); m(f"pp{nome}a", f(pp[k][2]))

    # tabela de poses de referencia
    import csv
    linhas = []
    with open(os.path.join(D, "poses_medidas.csv")) as fh:
        for row in csv.DictReader(fh):
            ph = float(row["flexao_graus"])
            e = r1["erros_tornozelo"].get(f"{ph:.0f}", 0.0)
            est = r"$\star$" if ph in (0, 45, 90) else ""
            linhas.append(f"{f(ph,0)}{est} & {f(float(row['joelho_x_mm']),2)} & {f(float(row['joelho_y_mm']),2)} & "
                          f"{f(float(row['tornozelo_x_mm']),1)} & {f(float(row['tornozelo_y_mm']),1)} & {f(e,2)} \\\\")
    L.append("\\newcommand{\\tabelaposes}{" + "\n".join(linhas) + "}")

    # ---------------- Parte II ----------------
    for M in ("80", "60"):
        p = r2[f"parametros_{M}kg"]
        tag = "A" if M == "80" else "B"
        m(f"mUm{tag}", f(p["coxa"]["m"], 2)); m(f"aUm{tag}", f(p["coxa"]["r"][0], 3)); m(f"IUm{tag}", f(p["coxa"]["I"], 3))
        m(f"mDois{tag}", f(p["perna"]["m"], 2)); m(f"bxDois{tag}", f(p["perna"]["r"][0], 3))
        m(f"byDois{tag}", f(p["perna"]["r"][1], 3)); m(f"IDois{tag}", f(p["perna"]["I"], 3))
    ps = r2["parametros_80kg_sem_ortese"]
    m("mUmS", f(ps["coxa"]["m"], 2)); m("mDoisS", f(ps["perna"]["m"], 2))
    m("massaOrtese", f(r2["massa_ortese"], 2))
    comp = []
    for c_ in r2["componentes_ortese"]:
        nomes = {"hastes Al (2x)": "hastes laterais Al 6061-T6 (2x)", "bracadeira proximal PP": "braçadeira proximal PP",
                 "bracadeira distal PP": "braçadeira distal PP", "bracadeira PP": "braçadeira PP",
                 "palmilha PP + estribo": "palmilha PP + estribo", "articulacao joelho (aco 304)": "articulação do joelho (aço AISI 304)",
                 "atuador do joelho": "atuador do joelho (motor + redutor)"}
        comp.append(f"{c_['elo']} & {nomes.get(c_['item'], c_['item'])} & "
                    f"{f(c_['m'],3)} & ({f(c_['pos'][0],3)}; {f(c_['pos'][1],3)}) \\\\")
    L.append("\\newcommand{\\tabelaortese}{" + "\n".join(comp) + "}")

    res_lin = []
    for fin in ("final I", "final II"):
        for M in ("80", "60"):
            x = r2[f"{fin}|{M}kg"]
            res_lin.append(f"{fin} & {M} & {f(x['tau_quadril_max'])} & {f(x['tau_quadril_rms'])} & {f(x['pot_quadril_max'])} & "
                           f"{f(x['tau_joelho_max'])} & {f(x['tau_joelho_rms'])} & {f(x['pot_joelho_max'])} \\\\")
    L.append("\\newcommand{\\tabelatorques}{" + "\n".join(res_lin) + "}")

    cin = []
    for fin in ("final I", "final II"):
        x = r2[f"{fin}|80kg"]
        cin.append(f"{fin} & {f(x['w_coxa_max'])} & {f(x['w_perna_max'])} & {f(x['w_joelho_max'])} & "
                   f"{f(x['a_coxa_max'])} & {f(x['a_perna_max'])} & {f(x['a_joelho_max'])} \\\\")
    L.append("\\newcommand{\\tabelacinematica}{" + "\n".join(cin) + "}")

    sens = []
    s1, s2 = r2["sens_T|final I"], r2["sens_T|final II"]
    for a, b in zip(s1, s2):
        sens.append(f"{f(a[0],1)} & {f(a[1])} & {f(a[2])} & {f(b[1])} & {f(b[2])} \\\\")
    L.append("\\newcommand{\\tabelasens}{" + "\n".join(sens) + "}")

    x1, x2 = r2["final I|80kg"], r2["final II|80kg"]
    m("tqImax", f(x1["tau_quadril_max"])); m("tjImax", f(x1["tau_joelho_max"]))
    m("tqIImax", f(x2["tau_quadril_max"])); m("tjIImax", f(x2["tau_joelho_max"]))
    m("pqIImax", f(x2["pot_quadril_max"])); m("pjImax", f(x1["pot_joelho_max"]))
    m("wmaxII", f(x2["w_coxa_max"])); m("wmaxI", f(x1["w_joelho_max"]))
    est = r2["estrutural"]
    k = list(est)
    m("sigPerna", f(est[k[0]]["sigma_MPa"])); m("nyPerna", f(est[k[0]]["n_escoamento"])); m("ngPerna", f(est[k[0]]["n_goodman"]))
    m("sigCoxa", f(est[k[1]]["sigma_MPa"])); m("nyCoxa", f(est[k[1]]["n_escoamento"])); m("ngCoxa", f(est[k[1]]["n_goodman"]))
    m("Wmod", f(r2["W_mm3"], 0))
    ne = r2["verificacao_NE"]
    m("neLq", f(ne["lagrange"][0], 6)); m("neNq", f(ne["newton_euler"][0], 6))
    m("neLj", f(ne["lagrange"][1], 6)); m("neNj", f(ne["newton_euler"][1], 6))

    with open(os.path.join(D, "valores.tex"), "w") as fh:
        fh.write("% Arquivo gerado automaticamente por scripts/gera_valores_tex.py -- nao editar\n")
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
