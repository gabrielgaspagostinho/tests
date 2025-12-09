\documentclass[11pt, a4paper]{article}

% --- UNIVERSAL PREAMBLE BLOCK ---
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}
\usepackage{fontspec}

% Configuração de idioma para Português
\usepackage[portuguese, bidi=basic, provide=*]{babel}

\babelprovide[import, onchar=ids fonts]{portuguese}
\babelprovide[import, onchar=ids fonts]{english}

% Set default/Latin font to Sans Serif in the main (rm) slot
\babelfont{rm}{Noto Sans}
% Set monospace font for code
\babelfont{tt}{Noto Sans Mono}

% Add because main language is not English
\usepackage{enumitem}
\setlist[itemize]{label=-}

% --- EXTRA PACKAGES FOR DOCUMENTATION ---
\usepackage{booktabs}   % Para tabelas bonitas
\usepackage{xcolor}     % Para cores em caixas de código (opcional, mas bom)
\usepackage{titlesec}   % Para formatar títulos

% Configuração simples para blocos de código
\usepackage{fancyvrb}
\DefineVerbatimEnvironment{Code}{Verbatim}{frame=single, fontsize=\small}

% Hyperref deve ser o último
\usepackage[hidelinks]{hyperref}

% Metadados do PDF
\title{\textbf{Manual Completo de Engenharia de Pacotes ROS 2: \\ Dependências e Ecossistema}}
\author{Equipe de Desenvolvimento Harpia}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Este documento serve como a referência definitiva para o gerenciamento de pacotes e dependências no ROS 2 (Jazzy/Humble). Ele cobre desde a teoria fundamental do manifesto \texttt{package.xml} até técnicas avançadas de resolução de conflitos e ferramentas de diagnóstico.
\end{abstract}

\tableofcontents
\newpage

\section{Fundamentos Teóricos: A Arquitetura de Distribuição}

No desenvolvimento de robótica profissional, o código nunca existe no vácuo. Um robô é um ``Sistema de Sistemas''.

\subsection{O Problema da Reproducibilidade}
Quando você desenvolve um nó de visão computacional, ele depende de:
\begin{enumerate}
    \item \textbf{Middleware:} ROS 2 (\texttt{rclpy}, \texttt{std\_msgs}).
    \item \textbf{Bibliotecas de Sistema:} Drivers, codecs (\texttt{libusb}, \texttt{ffmpeg}).
    \item \textbf{Bibliotecas de Linguagem:} Pacotes Python/C++ (\texttt{numpy}, \texttt{opencv}, \texttt{boost}).
\end{enumerate}

Sem um sistema de gerenciamento estrito, enviar seu código para outro robô ou para a nuvem (CI/CD) resultaria no clássico ``funciona na minha máquina''.

\subsection{A Solução: Abstração via Chaves (Rosdep Keys)}
O ROS não depende diretamente do gerenciador de pacotes do OS (\texttt{apt} ou \texttt{dnf}). Ele usa uma camada de abstração.

\begin{itemize}
    \item \textbf{No \texttt{package.xml}:} Você pede uma funcionalidade abstrata, ex: \texttt{python-numpy}.
    \item \textbf{O \texttt{rosdep}:} Consulta o banco de dados \texttt{rosdistro} e traduz isso para o pacote real do sistema operacional.
    \begin{itemize}
        \item Ubuntu: \texttt{python3-numpy}
        \item Fedora: \texttt{numpy}
        \item Arch: \texttt{python-numpy}
    \end{itemize}
\end{itemize}

Isso garante que o mesmo código fonte compile em diferentes sistemas operacionais.

\section{Anatomia Profunda do \texttt{package.xml} (REP-149)}

O arquivo \texttt{package.xml} é o contrato do seu pacote. Ele segue a especificação \textbf{REP-149} e deve usar \texttt{format="3"}.

\subsection{Tags de Dependência Detalhadas}

A escolha correta da tag afeta a ordem de compilação do \texttt{colcon} e o tamanho da imagem Docker final.

\begin{center}
\small
\begin{tabular}{@{}p{0.22\textwidth} p{0.35\textwidth} p{0.35\textwidth}@{}}
\toprule
\textbf{Tag} & \textbf{Função} & \textbf{Contexto Principal} \\
\midrule
\texttt{<buildtool\_depend>} & Ferramentas necessárias para iniciar a compilação (orquestração). & \texttt{ament\_cmake}, \texttt{ament\_python} \\
\midrule
\texttt{<build\_depend>} & Arquivos necessários \textit{apenas} durante a compilação (headers, geradores). & Geradores de msg, libs estáticas C++. \\
\midrule
\texttt{<exec\_depend>} & Bibliotecas necessárias para rodar o binário ou script. Crucial para Python. & \texttt{rclpy}, \texttt{numpy}, \texttt{opencv}. \\
\midrule
\texttt{<build\_export\_depend>} & Dependência transitiva. Se alguém compilar \textit{contra} seu pacote, precisará disso. & Libs C++ complexas (\texttt{pcl}, \texttt{eigen}). \\
\midrule
\texttt{<depend>} & ``Meta-tag''. Instala para Build, Execução e Exportação. & Libs padrão C++ (\texttt{rclcpp}, \texttt{std\_msgs}). \\
\midrule
\texttt{<test\_depend>} & Apenas para rodar a suíte de testes. & \texttt{ament\_lint\_auto}, \texttt{pytest}. \\
\midrule
\texttt{<group\_depend>} & Dependência de um grupo de pacotes. & Stacks como \texttt{nav2\_bringup}. \\
\bottomrule
\end{tabular}
\end{center}

\subsection{Versionamento de Dependências (Advanced)}

Você pode (e deve) especificar versões mínimas para evitar quebras de API.

\begin{Code}
<!-- Exige especificamente a versão 2.0 ou superior do numpy -->
<exec_depend version_gte="2.0.0">python3-numpy</exec_depend>

<!-- Exige uma versão específica do pacote de navegação -->
<depend version_eq="1.1.0">nav2_core</depend>
\end{Code}

Atributos disponíveis: \texttt{version\_lt} (less than), \texttt{version\_lte} (less or equal), \texttt{version\_gt} (greater than), \texttt{version\_gte} (greater or equal).

\subsection{A Tag \texttt{<export>} e o Sistema de Build}

Esta seção diz ao \texttt{colcon} qual sistema de build usar. Se isso estiver errado, o pacote não será detectado.

\textbf{Para C++:}
\begin{Code}
<export>
  <build_type>ament_cmake</build_type>
</export>
\end{Code}

\textbf{Para Python:}
\begin{Code}
<export>
  <build_type>ament_python</build_type>
</export>
\end{Code}

\section{O Rosdep: Ferramenta e Prática}

\subsection{Ciclo de Vida do Rosdep}

\begin{enumerate}
    \item \texttt{sudo rosdep init}: Cria o arquivo \texttt{/etc/ros/rosdep/sources.list.d/20-default.list}. Faz isso apenas uma vez na instalação do ROS.
    \item \texttt{rosdep update}: Baixa o banco de dados YAML do GitHub (\texttt{ros/rosdistro}) para o cache do seu usuário (\texttt{\textasciitilde/.ros/rosdep}). Deve ser rodado periodicamente.
    \item \texttt{rosdep install}: Lê seus arquivos locais e chama o \texttt{apt} ou \texttt{pip}.
\end{enumerate}

\subsection{O Comando ``Bala de Prata'' Explicado}

\begin{Code}
rosdep install --from-paths src --ignore-src -y -r
\end{Code}

\begin{itemize}
    \item \texttt{--from-paths src}: Escaneia recursivamente a pasta \texttt{src}.
    \item \texttt{--ignore-src}: Se uma dependência (ex: \texttt{cv\_bridge}) já estiver clonada na pasta \texttt{src} como código-fonte, \textbf{ignora} a versão do sistema (\texttt{apt}) e usa a sua. Isso é vital para desenvolvimento (Overlaying).
    \item \texttt{-y}: Assume ``yes'' para instalações do sistema.
    \item \texttt{-r}: (Continue on error) Tenta instalar o restante mesmo se um pacote falhar. Útil em ambientes instáveis.
\end{itemize}

\subsection{Lidando com Chaves Personalizadas (Custom Keys)}

Quando você precisa de uma biblioteca que não existe no ecossistema ROS oficial.

1. Crie um arquivo \texttt{custom\_rules.yaml} na raiz do workspace:
\begin{Code}
# Nome da chave que você usará no package.xml
minha_lib_ia_v2:
  ubuntu: # OS
    pip:  # Instalador
      packages: [super-ai-lib-v2] # Nome real no PyPI
\end{Code}

2. Aponte o rosdep para este arquivo:
\begin{Code}
echo "yaml file://$(pwd)/custom_rules.yaml" | \
sudo tee /etc/ros/rosdep/sources.list.d/10-local.list
\end{Code}

3. Atualize o cache:
\begin{Code}
rosdep update
\end{Code}

\section{Ferramentas de Diagnóstico e Visualização}

Antes de compilar, é útil entender a árvore de dependências.

\subsection{colcon graph}
O sistema de build \texttt{colcon} sabe exatamente a ordem de compilação baseada no \texttt{package.xml}.

\begin{Code}
# Lista a ordem topológica de processamento
colcon graph

# Gera uma visualização gráfica (requer graphviz)
colcon graph --dot | dot -Tpng -o deps.png
\end{Code}

\subsection{ros2 pkg}
Inspeciona pacotes já instalados ou no ambiente.

\begin{Code}
# Lista dependências diretas de um pacote
ros2 pkg xml yolo_detector

# Verifica onde um pacote está instalado (Sistema vs. Overlay)
ros2 pkg prefix cv_bridge
\end{Code}

\section{Melhores Práticas e Anti-Patterns}

\subsection{O que Fazer (Boas Práticas)}
\begin{itemize}
    \item \textbf{Minimalismo:} Use \texttt{<exec\_depend>} para Python. Instalar compiladores C++ para um nó Python é desperdício de tempo de CI/CD.
    \item \textbf{Separação:} Mantenha arquivos \texttt{.msg} em um pacote separado (ex: \texttt{my\_interfaces}). Isso evita dependências cíclicas.
    \item \textbf{Linting:} Adicione \texttt{<test\_depend>ament\_lint\_auto</test\_depend>} para garantir qualidade de código.
\end{itemize}

\subsection{O que Não Fazer (Anti-Patterns)}
\begin{itemize}
    \item \textbf{Hardcoding:} Nunca assuma que uma biblioteca está instalada só porque funcionou no seu PC. Se não está no \texttt{package.xml}, ela não existe.
    \item \textbf{Dependência Cíclica:} Pacote A depende de B, e B depende de A. O \texttt{colcon} falhará.
    \item \textbf{Misturar Gerenciadores:} Evite usar \texttt{sudo pip install} globalmente. Deixe o \texttt{rosdep} gerenciar isso ou use ambientes virtuais/Docker.
\end{itemize}

\section{Resumo de Comandos Úteis}

\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Ação} & \textbf{Comando} \\
\midrule
Instalar dependências & \texttt{rosdep install --from-paths src --ignore-src -y} \\
Checar chaves faltantes & \texttt{rosdep check --from-paths src --ignore-src} \\
Encontrar nome do pacote & \texttt{rosdep resolve python-numpy} \\
Ver ordem de compilação & \texttt{colcon graph} \\
\bottomrule
\end{tabular}
\end{center}

\end{document}
