PY ?= python3

all: main.pdf

dados/resultado_parte1.json: scripts/parte1_sintese.py
	$(PY) scripts/parte1_sintese.py > /dev/null

dados/resultado_parte2.json: scripts/parte2_ortese.py
	$(PY) scripts/parte2_ortese.py > /dev/null

dados/valores.tex: dados/resultado_parte1.json dados/resultado_parte2.json scripts/gera_valores_tex.py
	$(PY) scripts/gera_valores_tex.py

main.pdf: main.tex referencias.bib dados/valores.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
	rm -f main.bbl main.run.xml

.PHONY: all clean
