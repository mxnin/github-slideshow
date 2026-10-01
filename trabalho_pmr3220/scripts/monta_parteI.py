"""Gera main_parteI.tex (relatorio apenas da Parte I) a partir de main.tex."""
import os
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(RAIZ, "main.tex")).read()


def cut(a, b, new=""):
    global s
    i = s.index(a)
    j = s.index(b, i)
    s = s[:i] + new + s[j:]


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


# Parte II inteira
cut(r"\section{Parte II -- Modelagem cinemática}", r"\section{Conclusões}")
# introducao: so a Parte I
cut(r"\begin{enumerate}[label=\textbf{Parte \Roman*.},leftmargin=*]", r"\section{Objetivos}", r"""Este documento trata da primeira parte do trabalho: a síntese dimensional de um
quadrilátero articulado plano cujo acoplador (conjunto perna--pé artificial) reproduza o
movimento relativo entre a perna e a coxa, medido experimentalmente em um integrante do
grupo, e a construção de uma maquete para avaliar qualitativamente o funcionamento
(\cref{fig:enunciado}).

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.42\textwidth]{figuras/enunciado_fig1.png}
  \caption{Diagrama cinemático do enunciado: pivôs $C$ e $L$ na coxa e $Q$ e $N$ na perna.}
  \label{fig:enunciado}
\end{figure}

""")
rep(r"No joelho protético (\cref{fig:enunciado}a)", r"No joelho protético (\cref{fig:enunciado})")
# objetivos: so os da Parte I
cut(r"\section{Objetivos}", r"\section{Fundamentação teórica}", r"""\section{Objetivos}

\begin{itemize}
  \item Desenvolver a síntese dimensional de um mecanismo quadrilátero articulado,
  capaz de reproduzir de forma aproximada o movimento relativo entre a coxa e o
  conjunto perna--pé de uma prótese de membro inferior.
  \item Determinar as dimensões e as posições das articulações do mecanismo, utilizando
  como referência as posições do membro inferior obtidas experimentalmente.
  \item Construir uma maquete que permita avaliar qualitativamente o funcionamento do
  mecanismo sintetizado.
\end{itemize}

Em todas as deduções, os passos intermediários são mostrados, de modo que os
resultados possam ser reproduzidos à mão. Os cálculos foram implementados em Python
(NumPy e Matplotlib), e os valores numéricos citados no texto são lidos
automaticamente dos resultados (\cref{sec:anexos}).

""")
# conclusoes: sem o item da Parte II
i = s.index(r"  \item \textbf{Parte II}: a interpolação")
j = s.index(r"  \item \textbf{Próximos passos}", i)
s = s[:i] + s[j:]
rep(r"""  \item \textbf{Próximos passos}: construir e ensaiar a maquete e refinar a síntese com mais
  quadros do vídeo.""", r"""  \item \textbf{Próximos passos}: construir e ensaiar a maquete, refinar a síntese com mais
  quadros do vídeo e realizar a Parte~II (órtese).""")
# anexos: sem o programa e o trecho da Parte II
i = s.index(r"  \item[\texttt{parte2\_ortese.py}]")
j = s.index(r"  \item[\texttt{gera\_valores\_tex.py}]", i)
s = s[:i] + s[j:]
i = s.index(r"Trecho central da Parte II")
j = s.index(r"\end{lstlisting}", i) + len(r"\end{lstlisting}")
s = s[:i] + s[j:]
s = s.replace(r"""e os programas em Python usados nas análises.""", r"""e os programas em Python usados na análise.""")
open(os.path.join(RAIZ, "main_parteI.tex"), "w").write(s)
print("ok")
