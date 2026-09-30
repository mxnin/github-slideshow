"""Gera main_parcial.tex (relatorio parcial: Parte I ate o processamento dos dados) a partir de main.tex."""
import os
src=open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'main.tex')).read()
def between(a,b):
    i=src.index(a); j=src.index(b,i); return src[i:j]
pre=src[:src.index(r'\begin{document}')].replace(r'\input{dados/valores.tex}', r'\input{dados/valores_parcial.tex}')
capa=between(r'\begin{document}', r'\section{Introdução}')
intro=r'''\section{Introdução}

O movimento do membro inferior humano resulta da interação entre diferentes segmentos
e articulações, sendo o joelho particularmente importante para a mobilidade entre a
coxa e a perna. Entretanto, seu movimento relativo não pode ser rigorosamente
representado por uma simples junta de revolução: no plano sagital, os côndilos
femorais rolam e deslizam sobre o platô tibial, de modo que o centro instantâneo de
rotação (CIR) descreve uma trajetória -- a \emph{centroide} -- em vez de permanecer fixo
\cite{menschik1974,oconnor1989,dathe2016}. Desde \textcite{menschik1974}, essa cinemática
é modelada por um quadrilátero articulado cruzado, formado pelas fibras isométricas
dos ligamentos cruzados anterior (LCA) e posterior (LCP) e pelos segmentos ósseos que
unem suas inserções \cite{oconnor1989,zavatsky1992}.

Nesse contexto, o projeto de próteses e órteses constitui uma aplicação relevante da
teoria de mecanismos, permitindo reproduzir ou auxiliar movimentos do membro inferior
em situações de amputação ou de comprometimento da capacidade motora. Este relatório
parcial trata da primeira etapa do trabalho: o projeto de uma prótese de joelho e
perna capaz de reproduzir o movimento relativo entre a coxa e o conjunto perna--pé por
meio de um quadrilátero articulado plano (\cref{fig:enunciado}). São apresentados a
fundamentação do modelo, a definição da geometria, a aquisição experimental do
movimento em um integrante do grupo e o processamento dos dados que servirá de base
para a síntese dimensional do mecanismo.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.42\textwidth]{figuras/enunciado_fig1.png}
  \caption{Diagrama cinemático do enunciado: pivôs $C$ e $L$ na coxa e $Q$ e $N$ na perna.}
  \label{fig:enunciado}
\end{figure}

\section{Objetivos}

\begin{itemize}
  \item Desenvolver a síntese dimensional de um mecanismo quadrilátero articulado,
  capaz de reproduzir de forma aproximada o movimento relativo entre a coxa e o
  conjunto perna--pé de uma prótese de membro inferior.
  \item Determinar as dimensões e as posições das articulações do mecanismo, utilizando
  como referência as posições do membro inferior obtidas experimentalmente.
  \item Realizar a análise cinemática de uma órtese para membro inferior, determinando
  as posições, velocidades e acelerações angulares dos segmentos ao longo de um ciclo
  de movimento.
  \item Realizar a análise dinâmica do sistema, determinando os torques necessários nos
  atuadores do joelho e da articulação entre a coxa e o tronco para a execução do
  movimento considerado.
\end{itemize}

Neste relatório parcial são desenvolvidas as etapas preparatórias do primeiro e do
segundo objetivos: a formulação do modelo do quadrilátero e a obtenção, a partir das
medições, do movimento relativo entre a coxa e a perna.

'''
fund=between(r'\section{Fundamentação teórica}', r'\subsection{Síntese dimensional analítica')
fund=fund.replace(r'No joelho protético (\cref{fig:enunciado}a)', r'No joelho protético (\cref{fig:enunciado})')
metod=between(r'\section{Parte I -- Metodologia}', r'\subsection{Processamento dos dados')
metod=metod.replace(r'\section{Parte I -- Metodologia}', r'\section{Metodologia}')
proc=r'''\subsection{Processamento dos dados}\label{sec:processamento}

As coordenadas do \emph{Tracker} estão na base da imagem. O que interessa ao quadrilátero
do joelho é apenas o movimento \emph{relativo} entre a coxa e a perna; por isso, o
movimento angular do quadril (a rotação da coxa em relação ao tronco, que também ocorre
levemente durante o experimento) foi desconsiderado, e a coxa foi tratada como o elo
fixo do mecanismo. Os dados foram processados em dois passos.

\paragraph{1º passo -- base da coxa em cada quadro.} A base $0$ tem origem no ponto médio
dos marcadores 3 e 4 (extremidade distal da coxa, junto ao joelho) e eixos paralelos aos
da imagem: $X_0$ horizontal, para a frente, e $Y_0$ vertical, para cima. Como a rotação do
quadril é desconsiderada, $\Rm{im}{0} = \Rot{0}{z_0} = I$, e a base da coxa apenas
acompanha a translação da região do joelho entre os quadros. Com
$\rv{im}{O_0} = \tfrac12\big(\rv{im}{3} + \rv{im}{4}\big)$, a mudança de base inversa de
\eqref{eq:mudancabase} fornece as coordenadas de cada marcador $k$ na base da coxa:
\begin{equation}
  \rv{0}{k} = \Rm{im}{0}^{\mathsf T}\big(\rv{im}{k} - \rv{im}{O_0}\big) = \rv{im}{k} - \tfrac12\big(\rv{im}{3} + \rv{im}{4}\big).
\end{equation}
Os marcadores 1 e 2 (quadril) não entram no processamento.

\paragraph{2º passo -- orientação da perna e flexão do joelho.} A perna é representada pelo
segmento que vai do ponto $J$, médio dos marcadores 5 e 6 (logo abaixo do joelho), ao
ponto $P$, marcador 8 (conexão perna--pé). Na base da coxa, sua orientação é
\begin{equation}
  \beta_j = \operatorname{atan2}\big({}^0y_P - {}^0y_J,\ {}^0x_P - {}^0x_J\big),
\end{equation}
e a flexão do joelho no quadro $j$ é medida em relação ao quadro 307 (membro estendido):
\begin{equation}
  \varphi_j = \beta_{307} - \beta_j \pmod{\SI{360}{\degree}} .
\end{equation}

\begin{table}[htbp]
  \centering
  \caption{Posição da perna na base $0$ (coxa) obtida das medições (mm): ponto $J$ (abaixo
  do joelho), ponto $P$ (tornozelo) e comprimento $\overline{JP}$. $\dagger$: quadro não
  utilizado na análise.}
  \label{tab:poses}
  \begin{tabular}{lrrrrrr}
    \toprule
    Quadro & $\varphi$ (\si{\degree}) & ${}^0x_{J}$ & ${}^0y_{J}$ & ${}^0x_{P}$ & ${}^0y_{P}$ & $\overline{JP}$ \\
    \midrule
    \tabelaposes
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{figuras/parcial_medicoes.pdf}
  \caption{Marcadores dos quadros selecionados escritos na base da coxa, com origem entre
  os pontos 3 e 4 ($\square$: coxa, $\circ$: perna, $\triangle$: pé; linha: segmento $JP$).}
  \label{fig:medicoes}
\end{figure}

Os quadros selecionados correspondem a flexões de \phiA, \phiB, \phiC{} e
\phiD\si{\degree} (\cref{tab:poses,fig:medicoes}), próximas das configurações inicial
(\SI{0}{\degree}), intermediárias (cerca de \SI{45}{\degree} e \SI{90}{\degree}) e de
máxima flexão. O comprimento do segmento $JP$ varia apenas entre \pernamin{} e
\pernamax~mm, o que confirma a coerência da marcação ao longo do movimento (as
diferenças decorrem dos artefatos de pele e da marcação manual). Observa-se ainda que
o ponto $J$, logo abaixo do joelho, desloca-se cerca de \deslJ~mm em relação à coxa entre
a extensão e a flexão máxima: a perna não gira em torno de um ponto fixo, o que
confirma que uma junta de revolução simples não reproduz o movimento e motiva o uso do
quadrilátero articulado.

'''
concl=r'''\section{Conclusões}

Nesta etapa, foram definidos a geometria de representação do membro inferior e o
modelo do joelho protético como quadrilátero articulado, formulado pelo método
matricial. O movimento do membro de um integrante do grupo foi filmado e rastreado no
\emph{Tracker}, e os dados foram escritos na base da coxa, desconsiderando o movimento
angular do quadril. Os quatro quadros selecionados cobrem flexões do joelho de
\phiA{} a \phiD\si{\degree}, e o deslocamento de cerca de \deslJ~mm da região proximal
da perna em relação à coxa confirma que o movimento relativo não é uma rotação pura.

As próximas etapas são: a síntese dimensional do quadrilátero a partir das posições
medidas, a construção da maquete e a análise cinemática e dinâmica da órtese.

'''
anex=between(r'\section{Materiais e anexos}', r'\subsection{Programas}')
anex=anex.replace(r'''no software \emph{Tracker}, a planilha com as coordenadas obtidas durante o rastreamento
e os programas em Python usados nas análises.''', r'''no software \emph{Tracker}, a planilha com as coordenadas obtidas durante o rastreamento
e o programa em Python usado no processamento (\texttt{scripts/parte1\_parcial.py}).''')
fim=r'''
\clearpage
\printbibliography[heading=bibintoc,title={Referências}]

\end{document}
'''
import os; open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'main_parcial.tex'),'w').write(pre+capa+intro+fund+metod+proc+concl+anex+fim)
