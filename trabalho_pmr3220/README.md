# PMR3220 – Prótese e órtese de membro inferior humano

Relatório em LaTeX (`main.tex` → `main.pdf`) e códigos de cálculo em Python.

| Pasta/arquivo | Conteúdo |
|---|---|
| `main.tex`, `referencias.bib` | relatório (ABNT via biblatex-abnt) |
| `scripts/parte1_sintese.py` | Parte I – síntese do quadrilátero por 3 posições de precisão, gabarito da maquete |
| `scripts/parte2_ortese.py` | Parte II – interpolação de 5º grau, dinâmica (Lagrange + verificação Newton–Euler), torques |
| `scripts/gera_valores_tex.py` | exporta os resultados para `dados/valores.tex` (usado pelo relatório) |
| `dados/poses_medidas.csv` | poses da perna em relação à coxa (**substituir pelas medições do grupo**) |
| `figuras/` | figuras geradas + figuras do enunciado |

## Como gerar
```bash
pip install numpy scipy sympy matplotlib
make            # roda os scripts, gera valores.tex e compila o PDF (latexmk + biber)
```
Para usar as medições experimentais: edite `dados/poses_medidas.csv`, apague
`dados/resultado_parte1.json` e rode `make` de novo.
Pendências antes de entregar: nomes/NUSP na capa, fotos da medição e da maquete.
