# Projeto 3: Alteração de Taxa e Detecção de Tons

Este repositório contém os códigos-fonte, dados e resultados do Projeto 3 da disciplina de Processamento Digital de Sinais (EEL7522) da Universidade Federal de Santa Catarina (UFSC).

## Aluno
- Caio Missfeld Carlos (Matrícula: 22202674)

## Estrutura do Projeto

- **teste.wav**: Áudio de voz gravado (7.35 segundos, 48 kHz, mono) utilizado como entrada.
- **Codigos_Scilab/**: Scripts `.sce` correspondentes a cada item da tarefa:
  - `script_4_1.sce`: Redução de taxa (dizimação) por fator 4 com filtro passa-baixas Hamming ($\Omega_c = \pi/4$).
  - `script_4_2.sce`: Aumento de taxa (interpolação) por fator 4 com filtro anti-imagem Hamming ($\Omega_c = \pi/4$).
  - `script_4_3.sce`: Redução e aumento de taxa por fator 2 via estrutura polifásica ($\Omega_c = \pi/2$).
  - `script_4_4.sce`: Detecção de tom ($\cos(n\pi/7)$) em quadros de 2500 amostras via algoritmo de Goertzel.
- **process_voice.py**: Script Python de validação e geração dos gráficos e métricas do relatório.
- **Audios_Processados/**: Arquivos `.wav` resultantes dos processamentos.
- **Graficos/**: Figuras dos espectros e energias geradas para o relatório.
- **relatorio_pds.pdf**: Relatório final completo em formato PDF.
