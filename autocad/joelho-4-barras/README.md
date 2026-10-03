# Joelho de quatro barras — desenho AutoCAD

- `joelho-4-barras.dxf` — abre direto no AutoCAD (DXF R2010). Para DWG: `SAVEAS` → *AutoCAD Drawing (*.dwg)*.
  - **Model**: geometria em mm, escala 1:1, origem no centro do côndilo (referencial do fêmur).
  - **Layout A3**: vista sagital na pose P1 (5:4), sequência de flexão 0°/40°/80°/120° (1:3), tabelas e carimbo.
  - Camadas: `FEMUR`, `LCA_BARRA2`, `LCP_BARRA4`, `TIBIA_BARRA3`, `ACOPLADOR_AB`, `PIVOS`, `POSES_PRECISAO`, `CENTROIDE`, `CI`, `COTAS`, `TEXTO`…
- `gerar_dxf.py` — refaz a síntese de três posições do rascunho HTML e gera o DXF. Para mudar o projeto, edite os parâmetros no topo e rode `pip install ezdxf && python gerar_dxf.py`.
- `preview-A3.png` — prévia da folha.
