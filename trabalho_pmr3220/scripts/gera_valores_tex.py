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

    # ---------------- Parte I (medicoes do Grupo C) ----------------
    Q, N, C, Lc, P4 = r1["Q4"], r1["N4"], r1["C"], r1["L"], r1["P4"]
    m("Qx", f(Q[0])); m("Qy", f(Q[1])); m("Nx", f(N[0])); m("Ny", f(N[1]))
    m("Cx", f(C[0])); m("Cy", f(C[1])); m("Lx", f(Lc[0])); m("Ly", f(Lc[1]))
    m("Px", f(P4[0])); m("Py", f(P4[1]))
    c = r1["comprimentos"]
    m("rCL", f(c["l0"])); m("rCQ", f(c["l1"])); m("rQN", f(c["l2"])); m("rLN", f(c["l3"]))
    m("mumin", f(r1["mu_min"])); m("mumax", f(r1["mu_max"]))
    m("muQmin", f(r1["muQ_min"])); m("muQmax", f(r1["muQ_max"]))
    m("mumindois", f(min(r1["mu_min"], r1["muQ_min"])))
    m("grashof", "satisfaz" if r1["grashof"] else "não satisfaz")
    ls = r1["grashof_ord"]
    m("grashofs", f(ls[0])); m("grashofp", f(ls[1])); m("grashofq", f(ls[2])); m("grashofl", f(ls[3]))
    m("grashofsl", f(r1["grashof_sl"])); m("grashofpq", f(r1["grashof_pq"]))
    m("grashofsinal", r"\le" if r1["grashof"] else ">")
    m("ciOx", f(r1["polo_0"][0])); m("ciOy", f(r1["polo_0"][1]))
    m("ciFx", f(r1["polo_fim"][0])); m("ciFy", f(r1["polo_fim"][1]))
    m("phimax", f(r1["phi_max"]))
    po = r1["poses"]
    for k, nome in zip(r1["frames"], ["A", "B", "C", "D", "E"]):
        m(f"fr{nome}", str(k)); m(f"phi{nome}", f(po[str(k)]["flexao"]))
    m("frRef", str(r1["frame_ref"]))
    pf = r1["precisao_phi"]
    m("precphiA", f(pf[0])); m("precphiB", f(pf[1])); m("precphiC", f(pf[2]))
    m("ecurva", f(r1["e_curva"])); m("emedmax", f(max(r1["erros"].values())))
    m("emedmin", f(min(r1["erros"].values())))
    aj = r1["ajuste"]
    m("resmax", f(max(aj["residuos"].values())))
    for i, nome in enumerate(("Zero", "Um", "Dois")):
        m(f"cx{nome}", f(aj["cx"][i], [2, 4, 6][i])); m(f"cy{nome}", f(aj["cy"][i], [2, 4, 6][i]))
    JPs = [po[str(k)]["JP"] for k in r1["frames"]]
    m("JPmin", f(min(JPs), 0)); m("JPmax", f(max(JPs), 0))
    m("JPvar", f(100 * (max(JPs) - min(JPs)) / min(JPs), 0))
    m("Jx", f(r1["J0"][0])); m("Jy", f(r1["J0"][1]))
    import numpy as _np
    J0 = _np.array(r1["J0"])
    desl = max(_np.hypot(po[str(k)]["x"] - J0[0], po[str(k)]["y"] - J0[1]) for k in r1["frames"])
    m("deslO", f(desl, 0))
    m("rmaxelo", f(max(c.values()))); m("rminelo", f(min(c.values())))
    dd = r1["diade"]
    m("dalphaA", f(dd["alpha2"])); m("dalphaB", f(dd["alpha3"]))
    m("ddeltaAx", f(dd["delta2"][0])); m("ddeltaAy", f(dd["delta2"][1]))
    m("ddeltaBx", f(dd["delta3"][0])); m("ddeltaBy", f(dd["delta3"][1]))
    m("dOx", f(dd["rO4_1"][0])); m("dOy", f(dd["rO4_1"][1]))
    m("dbetaA", f(dd["beta2"])); m("dbetaB", f(dd["beta3"]))
    m("dgamA", f(dd["gamma2"])); m("dgamB", f(dd["gamma3"]))
    for nm in ("W", "Z", "U", "S"):
        m(f"d{nm}x", f(dd[nm][0])); m(f"d{nm}y", f(dd[nm][1]))
    import csv
    linhas = []
    with open(os.path.join(D, "poses_medidas.csv")) as fh:
        for row in csv.DictReader(fh):
            fr = int(row["frame"])
            tag = "" if fr in r1["frames"] else r" $\dagger$"
            linhas.append(f"{fr}{tag} & {f(float(row['flexao_graus']))} & {f(float(row['x_O4_mm']))} & {f(float(row['y_O4_mm']))} & "
                          f"{f(float(row['x_P_mm']))} & {f(float(row['y_P_mm']))} & {f(float(row['JP_mm']), 0)} & "
                          + (f"{f(aj['residuos'][str(fr)])} & {f(r1['erros'][str(fr)])}" if fr in r1["frames"] else "-- & --") + r" \\")
    L.append("\\newcommand{\\tabelaposes}{" + "\n".join(linhas) + "}")
    comb = []
    for cb in r1["combinacoes"]:
        comb.append(f"{cb['escolha']} & {'; '.join(f(g) for g in cb['phis'])} & {f(cb['e_curva'], 2)} & {f(cb['e_med_max'])} & {f(cb['mu_min'])} \\\\")
    L.append("\\newcommand{\\tabelacombos}{" + "\n".join(comb) + "}")

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
            res_lin.append(f"{fin} & {M} & {f(x['tau_quadril_max'])} & {f(x['tau_joelho_max'])} \\\\")
    L.append("\\newcommand{\\tabelatorques}{" + "\n".join(res_lin) + "}")

    cin = []
    for fin in ("final I", "final II"):
        x = r2[f"{fin}|80kg"]
        cin.append(f"{fin} & {f(x['w_coxa_max'])} & {f(x['w_perna_max'])} & {f(x['w_joelho_max'])} & "
                   f"{f(x['a_coxa_max'])} & {f(x['a_perna_max'])} & {f(x['a_joelho_max'])} \\\\")
    L.append("\\newcommand{\\tabelacinematica}{" + "\n".join(cin) + "}")

    x1, x2 = r2["final I|80kg"], r2["final II|80kg"]
    m("tqImax", f(x1["tau_quadril_max"])); m("tjImax", f(x1["tau_joelho_max"]))
    m("tqIImax", f(x2["tau_quadril_max"])); m("tjIImax", f(x2["tau_joelho_max"]))
    m("wmaxII", f(x2["w_coxa_max"])); m("wmaxI", f(x1["w_joelho_max"]))
    rp = r2["repouso_inicial"]
    m("repQ", f(rp["tau_Q"], 3)); m("repJ", f(rp["tau_J"], 3)); m("repmgb", f(rp["m2_g_by"], 3))

    with open(os.path.join(D, "valores.tex"), "w") as fh:
        fh.write("% Arquivo gerado automaticamente por scripts/gera_valores_tex.py -- nao editar\n")
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
