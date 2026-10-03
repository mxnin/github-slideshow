"""Gera o desenho AutoCAD (DXF) do joelho de quatro barras.

Reproduz a síntese de três posições do rascunho HTML (joelho-4-barras-rascunho.html)
com os parâmetros de projeto padrão e desenha, em mm e escala 1:1 no Model:
  - fêmur (barra fixa 1) com o côndilo, pivôs O2 e O4;
  - LCA (barra 2), tíbia/acoplador (barra 3) e LCP (barra 4) na pose P1;
  - poses de precisão P1–P3 tracejadas e a centroide fixa (lugar do CI);
  - cotas a, b, c, d e sequência de flexão 0°/40°/80°/120°.
O layout "A3" traz as vistas em escala, tabelas e carimbo.

Uso:  pip install ezdxf matplotlib && python gerar_dxf.py
"""
import math
import ezdxf
from ezdxf.enums import TextEntityAlignment

# ---------------- parâmetros de projeto (iguais ao HTML) ----------------
R, K, S0 = 22.0, 0.5, 3.0          # raio do côndilo, razão de rolamento, contato inicial (mm)
PHI_A, PHI_B = 0.0, 120.0          # intervalo de síntese (°)
A_T = (4.0, -12.0)                 # pivô do LCA na tíbia (local)
B_T = (-18.0, -4.0)                # pivô do LCP na tíbia (local)
TIB = [(-30, 0), (24, 0), (23, -7), (15, -17), (13, -62), (-13, -62), (-15, -17), (-27, -9)]
D2R = math.pi / 180


# ---------------- vetores ----------------
def add(p, q): return (p[0] + q[0], p[1] + q[1])
def sub(p, q): return (p[0] - q[0], p[1] - q[1])
def mul(p, s): return (p[0] * s, p[1] * s)
def norm(p): return math.hypot(*p)
def ang(p): return math.atan2(p[1], p[0])
def cross(p, q): return p[0] * q[1] - p[1] * q[0]
def rot(p, t):
    c, s = math.cos(t), math.sin(t)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


# ---------------- síntese e cinemática ----------------
def desired_pose(phi):
    s = S0 - K * R * phi
    C = (-R * math.sin(phi), -R * math.cos(phi))
    return {"T": sub(C, rot((s, 0), -phi)), "th": -phi}


def to_femur(pt, pose): return add(pose["T"], rot(pt, pose["th"]))


def circumcenter(p1, p2, p3):
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    s1, s2, s3 = (p[0] ** 2 + p[1] ** 2 for p in (p1, p2, p3))
    return ((s1 * (p2[1] - p3[1]) + s2 * (p3[1] - p1[1]) + s3 * (p1[1] - p2[1])) / d,
            (s1 * (p3[0] - p2[0]) + s2 * (p1[0] - p3[0]) + s3 * (p2[0] - p1[0])) / d)


def synthesize():
    phis = [((PHI_A + PHI_B) / 2 - (PHI_B - PHI_A) / 2 * math.cos((2 * j - 1) * math.pi / 6)) * D2R
            for j in (1, 2, 3)]
    poses = [desired_pose(f) for f in phis]
    Aj = [to_femur(A_T, p) for p in poses]
    Bj = [to_femur(B_T, p) for p in poses]
    O2, O4 = circumcenter(*Aj), circumcenter(*Bj)
    a, c = norm(sub(Aj[0], O2)), norm(sub(Bj[0], O4))
    b, d = norm(sub(B_T, A_T)), norm(sub(O4, O2))
    AB0 = rot(sub(B_T, A_T), poses[0]["th"])
    br = math.copysign(1, cross(sub(Aj[0], O2), sub(sub(O4, AB0), O2)))
    return dict(phis=phis, poses=poses, Aj=Aj, Bj=Bj, O2=O2, O4=O4, a=a, b=b, c=c, d=d, br=br)


def position(phi, S):
    AB = rot(sub(B_T, A_T), -phi)
    Q = sub(S["O4"], AB)
    e = sub(Q, S["O2"]); dd = norm(e)
    if not (abs(S["a"] - S["c"]) <= dd <= S["a"] + S["c"]):
        return None
    x = (S["a"] ** 2 - S["c"] ** 2 + dd ** 2) / (2 * dd)
    h = math.sqrt(max(0.0, S["a"] ** 2 - x * x))
    u = mul(e, 1 / dd); n = (-u[1], u[0])
    A = add(add(S["O2"], mul(u, x)), mul(n, h))
    if math.copysign(1, cross(sub(A, S["O2"]), e)) != S["br"]:
        A = add(add(S["O2"], mul(u, x)), mul(n, -h))
    B = add(A, AB)
    return dict(A=A, B=B, T=sub(A, rot(A_T, -phi)), th3=ang(AB))


def icr(p, S):
    d2, d4 = sub(p["A"], S["O2"]), sub(p["B"], S["O4"])
    den = cross(d2, d4)
    if abs(den) < 1e-9:
        return None
    return add(S["O2"], mul(d2, cross(sub(S["O4"], S["O2"]), d4) / den))


def transmission(p, S):
    t4 = ang(sub(p["B"], S["O4"]))
    mu = abs(p["th3"] - t4) % (2 * math.pi)
    if mu > math.pi: mu = 2 * math.pi - mu
    if mu > math.pi / 2: mu = math.pi - mu
    return mu / D2R


S = synthesize()
sweep = [PHI_A + (PHI_B - PHI_A) * i / 120 for i in range(121)]
centrode, mu_min, err_max = [], 999.0, 0.0
for f in sweep:
    p = position(f * D2R, S)
    assert p, f"mecanismo não monta em φ={f}"
    ci = icr(p, S)
    if ci: centrode.append(ci)
    mu_min = min(mu_min, transmission(p, S))
    err_max = max(err_max, norm(sub(p["T"], desired_pose(f * D2R)["T"])))

L = sorted([("a", S["a"]), ("b", S["b"]), ("c", S["c"]), ("d", S["d"])], key=lambda t: t[1])
grashof = L[0][1] + L[3][1] <= L[1][1] + L[2][1]
if grashof:
    cls = {"d": "Grashof - dupla manivela", "b": "Grashof - duplo balancim"}.get(L[0][0], "Grashof - manivela-balancim")
else:
    cls = "Nao Grashof (triplo balancim)"


def fmt(v, n=2): return f"{v:.{n}f}".replace(".", ",")
def fpt(p, n=2): return f"({fmt(p[0], n)}; {fmt(p[1], n)})"


# ---------------- documento DXF ----------------
doc = ezdxf.new("R2010", setup=True)
doc.units = ezdxf.units.MM
doc.header["$INSUNITS"] = 4
doc.header["$MEASUREMENT"] = 1
doc.styles.add("CAD", font="arial.ttf")

LAYERS = {  # nome: (cor ACI, tipo de linha, espessura 1/100 mm)
    "FEMUR": (8, "CONTINUOUS", 50),
    "CONDILO_REF": (8, "CENTER", 18),
    "LCA_BARRA2": (1, "CONTINUOUS", 70),
    "LCP_BARRA4": (4, "CONTINUOUS", 70),
    "TIBIA_BARRA3": (30, "CONTINUOUS", 50),
    "ACOPLADOR_AB": (30, "CONTINUOUS", 50),
    "PIVOS": (7, "CONTINUOUS", 35),
    "POSES_PRECISAO": (9, "DASHED", 18),
    "CENTROIDE": (5, "CONTINUOUS", 35),
    "CI": (5, "CONTINUOUS", 25),
    "EIXOS": (8, "CENTER", 13),
    "COTAS": (3, "CONTINUOUS", 18),
    "TEXTO": (7, "CONTINUOUS", 25),
    "SEQUENCIA": (7, "CONTINUOUS", 25),
    "FORMATO": (7, "CONTINUOUS", 50),
    "VIEWPORT": (8, "CONTINUOUS", 13),
}
for name, (color, lt, lw) in LAYERS.items():
    doc.layers.add(name, color=color, linetype=lt, lineweight=lw)
doc.layers.get("VIEWPORT").off()  # contorno das janelas não aparece na impressão
doc.layers.get("VIEWPORT").dxf.plot = 0
doc.header["$LTSCALE"] = 0.25

dim = doc.dimstyles.new("COTA_MECANISMO")
dim.dxf.dimtxt = 1.8; dim.dxf.dimasz = 1.4; dim.dxf.dimexe = 0.8; dim.dxf.dimexo = 0.6
dim.dxf.dimgap = 0.5; dim.dxf.dimdec = 1; dim.dxf.dimdsep = ord(","); dim.dxf.dimtad = 1
dim.dxf.dimtxsty = "CAD"; dim.dxf.dimclrd = 3; dim.dxf.dimclre = 3; dim.dxf.dimclrt = 3

msp = doc.modelspace()


def text(layout, s, p, h, layer="TEXTO", align=TextEntityAlignment.LEFT, color=None):
    t = layout.add_text(s, height=h, dxfattribs={"layer": layer, "style": "CAD"})
    if color is not None: t.dxf.color = color
    t.set_placement(p, align=align)
    return t


def poly(layout, pts, layer, closed=True, **kw):
    return layout.add_lwpolyline(pts, close=closed, dxfattribs={"layer": layer, **kw})


def femur(origin, layout=msp, hatch=True):
    """Contorno do fêmur: haste 30x50 unida ao côndilo de raio R."""
    ox, oy = origin
    yi = math.sqrt(R * R - 15 * 15)
    a0 = math.degrees(math.atan2(yi, 15))      # ponto (15, yi)
    a1 = 180 - a0                              # ponto (-15, yi)
    # polilinha com arco (bulge) do lado inferior: de (-15,yi) a (15,yi) passando por baixo
    sweep_ang = 360 - (a1 - a0)
    bulge = math.tan(math.radians(sweep_ang) / 4)
    pts = [(ox + 15, oy + yi, 0), (ox + 15, oy + 50, 0), (ox - 15, oy + 50, 0), (ox - 15, oy + yi, bulge)]
    pl = layout.add_lwpolyline(pts, format="xyb", close=True, dxfattribs={"layer": "FEMUR"})
    if hatch:
        h = layout.add_hatch(color=254, dxfattribs={"layer": "FEMUR"})
        h.paths.add_polyline_path(pts, is_closed=True)
    return pl


def tibia(pose, origin=(0, 0), layer="TIBIA_BARRA3", hatch=True, layout=msp):
    pts = [add(origin, to_femur(q, pose)) for q in TIB]
    pl = poly(layout, pts, layer)
    if hatch:
        h = layout.add_hatch(color=254, dxfattribs={"layer": layer})
        h.paths.add_polyline_path(pts, is_closed=True)
    return pl


def pin(p, ground=False, r=1.6, layout=msp):
    if ground:  # símbolo de apoio fixo (triângulo + hachura)
        tri = [add(p, (0, 0)), add(p, (-3.2, 5.5)), add(p, (3.2, 5.5))]
        poly(layout, tri, "PIVOS")
        layout.add_line(add(p, (-4.5, 5.5)), add(p, (4.5, 5.5)), dxfattribs={"layer": "PIVOS"})
        for i in range(4):
            x = -4 + i * 2.6
            layout.add_line(add(p, (x, 5.5)), add(p, (x + 1.8, 7.3)), dxfattribs={"layer": "PIVOS"})
    layout.add_circle(p, r, dxfattribs={"layer": "PIVOS"})
    layout.add_line(add(p, (-r * 0.6, 0)), add(p, (r * 0.6, 0)), dxfattribs={"layer": "EIXOS"})
    layout.add_line(add(p, (0, -r * 0.6)), add(p, (0, r * 0.6)), dxfattribs={"layer": "EIXOS"})


def mechanism(phi, origin=(0, 0), hatch=True, lw_scale=1.0, layout=msp):
    p = position(phi, S)
    O2, O4 = add(origin, S["O2"]), add(origin, S["O4"])
    A, B = add(origin, p["A"]), add(origin, p["B"])
    femur(origin, layout, hatch)
    layout.add_circle(origin, R, dxfattribs={"layer": "CONDILO_REF"})
    tibia({"T": p["T"], "th": -phi}, origin, hatch=hatch, layout=layout)
    layout.add_line(A, B, dxfattribs={"layer": "ACOPLADOR_AB"})
    layout.add_line(O2, A, dxfattribs={"layer": "LCA_BARRA2"})
    layout.add_line(O4, B, dxfattribs={"layer": "LCP_BARRA4"})
    for q, g in ((O2, True), (O4, True), (A, False), (B, False)):
        pin(q, g, layout=layout)
    return p, O2, O4, A, B


# ---------------- vista principal (origem = centro do côndilo) ----------------
phi1 = S["phis"][0]
p1, O2, O4, A, B = mechanism(phi1)

# poses de precisão P1–P3 (tracejadas) com as barras nessas poses
for j, (pose, Aj, Bj) in enumerate(zip(S["poses"], S["Aj"], S["Bj"])):
    tibia(pose, layer="POSES_PRECISAO", hatch=False)
    if j:
        msp.add_line(S["O2"], Aj, dxfattribs={"layer": "POSES_PRECISAO"})
        msp.add_line(S["O4"], Bj, dxfattribs={"layer": "POSES_PRECISAO"})
        msp.add_line(Aj, Bj, dxfattribs={"layer": "POSES_PRECISAO"})
    text(msp, f"P{j + 1}", to_femur((0, -67), pose), 2.5, layer="POSES_PRECISAO",
         align=TextEntityAlignment.MIDDLE_CENTER)

# centroide fixa e CI atual
msp.add_spline(centrode, dxfattribs={"layer": "CENTROIDE"})
ci = icr(p1, S)
msp.add_circle(ci, 1.0, dxfattribs={"layer": "CI"})
h = msp.add_hatch(color=5, dxfattribs={"layer": "CI"}); h.paths.add_edge_path().add_arc(ci, 1.0, 0, 360)
for P0, Q0 in ((O2, A), (O4, B)):  # prolongamentos das barras até o CI
    msp.add_line(Q0, ci, dxfattribs={"layer": "CI", "linetype": "DASHED"})
text(msp, "CI", add(ci, (2, 1.5)), 2.5, layer="CI")
text(msp, "centroide fixa", add(centrode[0], (-3, -1)), 1.8, layer="CENTROIDE", align=TextEntityAlignment.RIGHT)

# eixos do referencial do fêmur
msp.add_line((-28, 0), (28, 0), dxfattribs={"layer": "EIXOS"})
msp.add_line((0, -28), (0, 55), dxfattribs={"layer": "EIXOS"})

# rótulos
text(msp, "O2", add(O2, (-7, 1.5)), 2.5)
text(msp, "O4", add(O4, (3, 1.5)), 2.5)
text(msp, "A", add(A, (2.5, -3.5)), 2.5)
text(msp, "B", add(B, (-5, -4)), 2.5)
text(msp, "FEMUR (1)", (17, 42), 2.5)
text(msp, "TIBIA (3)", to_femur((16, -46), S["poses"][0]), 2.5)
text(msp, "x", (29, -1), 2.2); text(msp, "y", (1, 52), 2.2)

# cotas a, b, c, d
def adim(p, q, dist, label):
    d = msp.add_aligned_dim(p1=p, p2=q, distance=dist, dimstyle="COTA_MECANISMO",
                            override={"dimpost": f"{label} = <>"}, dxfattribs={"layer": "COTAS"})
    d.render()

adim(O2, A, 6, "LCA a")
adim(A, B, -6, "tibia b")
adim(O4, B, -6, "LCP c")
adim(O2, O4, 9, "femur d")
# raio do côndilo
rd = msp.add_radius_dim(center=(0, 0), radius=R, angle=135, dimstyle="COTA_MECANISMO",
                        override={"dimpost": "R<>"}, dxfattribs={"layer": "COTAS"})
rd.render()

# ---------------- sequência de flexão (abaixo da vista principal) ----------------
SEQ_Y, SEQ_DX = -230.0, 160.0
SEQ = [0, 40, 80, 120]
for i, f in enumerate(SEQ):
    org = (i * SEQ_DX, SEQ_Y)
    mechanism(f * D2R, org)
    text(msp, f"phi = {f} graus", add(org, (0, -88)), 7, align=TextEntityAlignment.MIDDLE_CENTER)
    msp.add_spline([add(org, q) for q in centrode], dxfattribs={"layer": "CENTROIDE"})

# ---------------- layout A3 ----------------
psp = doc.layouts.new("A3")
psp.page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
for vp in psp.query("VIEWPORT"):
    if vp.dxf.id != 1: psp.delete_entity(vp)

poly(psp, [(10, 10), (410, 10), (410, 287), (10, 287)], "FORMATO", const_width=0.5)

# janela 1: vista principal, escala 5:4
MAIN_SCALE = 1.25
v1 = psp.add_viewport(center=(120, 175), size=(214, 196), view_center_point=(-19, -20), view_height=196 / MAIN_SCALE,
                      dxfattribs={"layer": "VIEWPORT"})
# janela 2: sequência, escala 1:3
SEQ_SCALE = 1 / 3
v2 = psp.add_viewport(center=(120, 43), size=(210, 62), view_center_point=(1.5 * SEQ_DX - 18, SEQ_Y - 22),
                      view_height=62 / SEQ_SCALE, dxfattribs={"layer": "VIEWPORT"})
text(psp, "VISTA SAGITAL - POSE P1 (phi = %s graus)   ESC. 5:4" % fmt(phi1 / D2R), (15, 279), 3.5)
text(psp, "SEQUENCIA DE FLEXAO   ESC. 1:3", (15, 76), 3.0)
psp.add_line((10, 73), (230, 73), dxfattribs={"layer": "FORMATO"})
psp.add_line((230, 10), (230, 287), dxfattribs={"layer": "FORMATO"})


def table(x, y, title, rows, widths, h=2.5, rh=6.0):
    text(psp, title, (x, y + 2), 3.2)
    W = sum(widths)
    for i, row in enumerate(rows):
        yy = y - i * rh
        psp.add_line((x, yy), (x + W, yy), dxfattribs={"layer": "FORMATO"})
        cx = x
        for w, cell in zip(widths, row):
            text(psp, cell, (cx + 1.5, yy - rh + 1.8), h)
            cx += w
    yb = y - len(rows) * rh
    psp.add_line((x, yb), (x + W, yb), dxfattribs={"layer": "FORMATO"})
    cx = x
    for w in [0] + widths:
        cx += w
        psp.add_line((cx, y), (cx, yb), dxfattribs={"layer": "FORMATO"})
    return yb


y = 278
y = table(236, y, "DADOS DA SINTESE", [
    ["Raio do condilo R", f"{fmt(R, 1)} mm"],
    ["Razao de rolamento k", fmt(K, 2)],
    ["Contato inicial s0", f"{fmt(S0, 1)} mm"],
    ["Intervalo phi", f"{fmt(PHI_A, 0)} a {fmt(PHI_B, 0)} graus"],
    ["A' (LCA na tibia)", fpt(A_T, 1) + " mm"],
    ["B' (LCP na tibia)", fpt(B_T, 1) + " mm"],
], [95, 73]) - 12
y = table(236, y, "POSES DE PRECISAO (Chebyshev)", [
    ["Pose", "phi (graus)", "A j (mm)", "B j (mm)"],
] + [[f"P{j + 1}", fmt(S["phis"][j] / D2R), fpt(S["Aj"][j], 1), fpt(S["Bj"][j], 1)] for j in range(3)],
    [18, 30, 60, 60]) - 12
y = table(236, y, "RESULTADO", [
    ["Pivo fixo O2 (LCA)", fpt(S["O2"]) + " mm"],
    ["Pivo fixo O4 (LCP)", fpt(S["O4"]) + " mm"],
    ["a  LCA (barra 2)", f"{fmt(S['a'])} mm"],
    ["b  tibia AB (barra 3)", f"{fmt(S['b'])} mm"],
    ["c  LCP (barra 4)", f"{fmt(S['c'])} mm"],
    ["d  femur O2O4 (barra 1)", f"{fmt(S['d'])} mm"],
    ["Classificacao", cls],
    ["mu minimo (0-120 graus)", f"{fmt(mu_min, 1)} graus"],
    ["Erro estrutural max.", f"{fmt(err_max, 3)} mm"],
], [95, 73])

# legenda de camadas
ly = y - 10
text(psp, "LEGENDA", (236, ly), 3.2)
for i, (lay, lab) in enumerate([("LCA_BARRA2", "LCA - barra 2"), ("LCP_BARRA4", "LCP - barra 4"),
                                ("ACOPLADOR_AB", "Tibia - acoplador 3"), ("CENTROIDE", "Centroide fixa / CI"),
                                ("POSES_PRECISAO", "Poses de precisao P1-P3")]):
    yy = ly - 6 - i * 5
    psp.add_line((238, yy + 1), (252, yy + 1), dxfattribs={"layer": lay, "lineweight": 50})
    text(psp, lab, (256, yy), 2.5)

# carimbo
cb = [(236, 16), (404, 16), (404, 52), (236, 52)]
poly(psp, cb, "FORMATO")
for yy in (40, 28):
    psp.add_line((236, yy), (404, yy), dxfattribs={"layer": "FORMATO"})
psp.add_line((330, 16), (330, 40), dxfattribs={"layer": "FORMATO"})
text(psp, "JOELHO DE QUATRO BARRAS - QUADRILATERO CRUZADO", (239, 44), 3.8)
text(psp, "Projeto de mecanismos - sintese de 3 posicoes", (239, 32), 2.8)
text(psp, "Unidades: mm   |   Model 1:1", (239, 20), 2.8)
text(psp, "Folha A3", (333, 32), 2.8)
text(psp, "Desenho 01/01", (333, 20), 2.8)

out = "joelho-4-barras.dxf"
doc.saveas(out)
print(f"O2={S['O2']}  O4={S['O4']}  a={S['a']:.3f} b={S['b']:.3f} c={S['c']:.3f} d={S['d']:.3f}")
print(f"{cls}; mu_min={mu_min:.1f}; err_max={err_max:.4f}; salvo em {out}")
