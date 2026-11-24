# -*- coding: utf-8 -*-
"""
PROJETO 3 DE PDS - ALTERACAO DE TAXA E DETECCAO DE TONS (GOERTZEL)
Script de simulacao em Python para validar os scripts Scilab, gerar os
arquivos de audio processados e salvar os graficos do relatorio.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

# Configuracao visual padrao dos graficos
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

DIR_PROJETO = r"D:\Programação\PDS\Projeto3"
PASTA_AUDIOS = os.path.join(DIR_PROJETO, "Audios_Processados")
PASTA_GRAFICOS = os.path.join(DIR_PROJETO, "Graficos")
ARQUIVO_ENTRADA = os.path.join(DIR_PROJETO, "teste.wav")

os.makedirs(PASTA_AUDIOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# 1. Carregamento do audio (apenas um canal, sem DC e normalizado)
fs, dados = wavfile.read(ARQUIVO_ENTRADA)
if dados.ndim > 1:
    sinal_x = dados[:, 0].astype(np.float64)
else:
    sinal_x = dados.astype(np.float64)

sinal_x = sinal_x - np.mean(sinal_x)
sinal_x = sinal_x / np.max(np.abs(sinal_x))
N_total = len(sinal_x)
print(f"Audio carregado: Fs = {fs} Hz, N = {N_total} amostras ({N_total/fs:.2f} s)")


def salvar_wav(nome_arq, vetor, taxa):
    """Normaliza o sinal para [-1, 1] e grava em 16-bit PCM."""
    pico = max(abs(np.min(vetor)), abs(np.max(vetor)))
    norm = vetor / pico if pico > 0 else vetor
    caminho = os.path.join(PASTA_AUDIOS, nome_arq)
    wavfile.write(caminho, int(taxa), np.int16(norm * 32767))
    return norm


def calcular_espectro(sig, taxa, ini_seg=2.0, N_fft=4096):
    """Retorna eixo de frequencias e magnitude normalizada de um trecho ativo."""
    ini = int(ini_seg * taxa)
    if ini + N_fft > len(sig):
        ini = max(0, len(sig) - N_fft)
    trecho = sig[ini : ini + N_fft] * np.hanning(N_fft)
    mag = np.abs(np.fft.fft(trecho))[: N_fft // 2]
    f = np.fft.fftfreq(N_fft, d=1.0 / taxa)[: N_fft // 2]
    return f, mag / (np.max(mag) + 1e-12), mag


# =========================================================================
# ITEM 4.1: REDUCAO DE TAXA (DIZIMACAO) POR FATOR 4
# =========================================================================
print("Executando Item 4.1 (Dizimacao por fator 4)...")
M = 4
fs_4 = fs // M  # 12000 Hz

# Filtro FIR passa-baixas Hamming com corte em OmegaC = pi/4 (0.25 de Nyquist)
ordem_4 = 65
h_lp4 = firwin(ordem_4, cutoff=1.0 / M, window='hamming')

# Filtragem anti-aliasing com compensacao de atraso de grupo
x_filt4 = np.convolve(sinal_x, h_lp4, mode='same')

# Dizimacao por fator 4
z = x_filt4[::M]
salvar_wav("output_4_1_dizimado_4x.wav", z, fs_4)

# Graficos de espectro: entrada x[n] vs dizimado z[n]
f_x, X_norm, _ = calcular_espectro(sinal_x, fs, ini_seg=2.0, N_fft=4096)
f_z, Z_norm, _ = calcular_espectro(z, fs_4, ini_seg=2.0, N_fft=2048)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(f_x, X_norm, color='#2B6CB0', linewidth=1.1, label="Sinal de Entrada x[n]")
plt.axvline(fs_4 / 2, color='#E53E3E', linestyle='--', linewidth=1.2, label=f"Corte do Filtro Anti-Aliasing ({fs_4//2} Hz)")
plt.title(f"Espectro do Sinal de Entrada x[n] (Fs = {fs} Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, 12000)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f_z, Z_norm, color='#C53030', linewidth=1.1, label="Sinal Dizimado z[n]")
plt.title(f"Espectro do Sinal Dizimado z[n] (Fator 4 -> Nova Taxa Fs/4 = {fs_4} Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, fs_4 / 2)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PASTA_GRAFICOS, "espectro_4_1_dizimacao.png"), dpi=150)
plt.close()


# =========================================================================
# ITEM 4.2: AUMENTO DE TAXA (INTERPOLACAO) POR FATOR 4
# =========================================================================
print("Executando Item 4.2 (Interpolacao por fator 4)...")
L = 4

# Expansao: insere 3 zeros entre cada amostra de z[n]
z_up = np.zeros(len(z) * L)
z_up[::L] = z

# Filtro passa-baixas anti-imagem (mesmo corte pi/4, janela Hamming, ganho L=4)
h_interp4 = L * h_lp4
y_interp4 = np.convolve(z_up, h_interp4, mode='same')
salvar_wav("output_4_2_interpolado_4x.wav", y_interp4, fs)

f_y4, Y4_norm, _ = calcular_espectro(y_interp4, fs, ini_seg=2.0, N_fft=4096)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(f_x, X_norm, color='#2B6CB0', linewidth=1.1, label="Entrada Original x[n]")
plt.axvline(fs_4 / 2, color='#718096', linestyle=':', linewidth=1.1, label="Limite de Banda (6000 Hz)")
plt.title(f"Espectro do Sinal de Entrada x[n] (Fs = {fs} Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, 12000)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f_y4, Y4_norm, color='#2F855A', linewidth=1.1, label="Saída Interpolada y[n] (4x)")
plt.axvline(fs_4 / 2, color='#718096', linestyle=':', linewidth=1.1, label="Corte Anti-Imagem (6000 Hz)")
plt.title(f"Espectro do Sinal Reconstruído y[n] após Aumento de Taxa 4x + Filtro Hamming (Fs = {fs} Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, 12000)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PASTA_GRAFICOS, "espectro_4_2_interpolacao.png"), dpi=150)
plt.close()


# =========================================================================
# ITEM 4.3: ESTRUTURA POLIFASICA (DIZIMACAO 2x + INTERPOLACAO 2x)
# =========================================================================
print("Executando Item 4.3 (Estrutura Polifasica fator 2)...")
N_pol = 2
L_pol = 64  # Comprimento par -> 2 ramos polifasicos de 32 coeficientes cada
h_pol = firwin(L_pol, cutoff=1.0 / N_pol, window='hamming')

# Decomposicao polifasica em 2 ramos: indices pares e impares
E0 = h_pol[0::2]
E1 = h_pol[1::2]

# 1) Dizimador Polifasico por fator 2
# x0[m] = x[2m], x1[m] = x[2m - 1]
x0 = sinal_x[0::2]
x_atrasado = np.concatenate(([0.0], sinal_x[:-1]))
x1 = x_atrasado[0::2]

z_poli = lfilter(E0, [1.0], x0) + lfilter(E1, [1.0], x1)

# 2) Interpolador Polifasico por fator 2 (ganho 2 nos ramos)
G0 = N_pol * E0
G1 = N_pol * E1

y_ramo0 = lfilter(G0, [1.0], z_poli)
y_ramo1 = lfilter(G1, [1.0], z_poli)

# Comutador na saida: intercala as amostras pares e impares
y_poli = np.zeros(2 * len(z_poli))
y_poli[0::2] = y_ramo0
y_poli[1::2] = y_ramo1

salvar_wav("output_4_3_polifasico_2x.wav", y_poli, fs)

f_yp, Yp_norm, _ = calcular_espectro(y_poli, fs, ini_seg=2.0, N_fft=4096)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(f_x, X_norm, color='#2B6CB0', linewidth=1.2, alpha=0.85, label="Entrada Original x[n]")
plt.plot(f_yp, Yp_norm, color='#DD6B20', linestyle='--', linewidth=1.1, alpha=0.9, label="Saída Polifásica y[n]")
plt.axvline(fs / 4, color='#718096', linestyle=':', linewidth=1.1, label=f"Corte pi/2 ({fs//4} Hz)")
plt.title("Comparação Espectral: Entrada x[n] vs Saída Polifásica y[n] (0 a 15 kHz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, 15000)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f_x, X_norm, color='#2B6CB0', linewidth=1.2, alpha=0.85, label="Entrada Original x[n]")
plt.plot(f_yp, Yp_norm, color='#DD6B20', linestyle='--', linewidth=1.1, alpha=0.9, label="Saída Polifásica y[n]")
plt.title("Detalhe na Faixa Principal da Voz (Zoom: 0 a 4000 Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude Normalizada")
plt.xlim(0, 4000)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PASTA_GRAFICOS, "espectro_4_3_polifasico.png"), dpi=150)
plt.close()


# =========================================================================
# ITEM 4.4: DETECCAO DE TONS (ALGORITMO DE GOERTZEL)
# =========================================================================
print("Executando Item 4.4 (Deteccao de Tons com Goertzel)...")
omega0 = np.pi / 7.0
f0_hz = (omega0 / (2.0 * np.pi)) * fs  # ~3428.57 Hz
L_tom = 10000
N_quadro = 2500
num_quadros = N_total // N_quadro

# Insercao exata nos quadros 41, 42, 43 e 44 (0-based: indices 100000 a 109999)
ini_tom = 100000
fim_tom = ini_tom + L_tom
quadros_inseridos = [41, 42, 43, 44]  # 1-based

n_tom = np.arange(L_tom)
A1 = 0.9 * np.max(np.abs(sinal_x))  # Amplitude proxima do maximo (0.9)
A2 = A1 / 4.0                        # 1/4 da amplitude anterior (0.225)

tom1 = A1 * np.cos(n_tom * omega0)
tom2 = A2 * np.cos(n_tom * omega0)

x_tom1 = sinal_x.copy()
x_tom1[ini_tom:fim_tom] += tom1

x_tom2 = sinal_x.copy()
x_tom2[ini_tom:fim_tom] += tom2

salvar_wav("output_4_4_com_tom_A.wav", x_tom1, fs)
salvar_wav("output_4_4_com_tom_A4.wav", x_tom2, fs)


def goertzel_por_quadro(sig, w0, Nq, n_q):
    """Calcula a energia |y[N]|^2 do filtro de Goertzel em cada quadro de Nq amostras."""
    coef = 2.0 * np.cos(w0)
    energias = np.zeros(n_q)
    for q in range(n_q):
        quadro = sig[q * Nq : (q + 1) * Nq]
        v1 = 0.0
        v2 = 0.0
        for amostra in quadro:
            v0 = amostra + coef * v1 - v2
            v2 = v1
            v1 = v0
        energias[q] = v1 * v1 + v2 * v2 - coef * v1 * v2
    return energias


energias_A1 = goertzel_por_quadro(x_tom1, omega0, N_quadro, num_quadros)
energias_A2 = goertzel_por_quadro(x_tom2, omega0, N_quadro, num_quadros)

# Grafico 1 do Item 4.4: Espectros com e sem o tom (no trecho onde o tom esta presente)
N_fft_tom = 4096
w_hn = np.hanning(N_fft_tom)
f_t = np.fft.fftfreq(N_fft_tom, d=1.0 / fs)[: N_fft_tom // 2]

mag_sem = np.abs(np.fft.fft(sinal_x[ini_tom : ini_tom + N_fft_tom] * w_hn))[: N_fft_tom // 2]
mag_com1 = np.abs(np.fft.fft(x_tom1[ini_tom : ini_tom + N_fft_tom] * w_hn))[: N_fft_tom // 2]
mag_com2 = np.abs(np.fft.fft(x_tom2[ini_tom : ini_tom + N_fft_tom] * w_hn))[: N_fft_tom // 2]

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(f_t, mag_sem, color='#2B6CB0', linewidth=1.1)
plt.title("Espectro do Sinal de Entrada SEM o Tom (Trecho: Quadros 41–44)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)

plt.subplot(3, 1, 2)
plt.plot(f_t, mag_com1, color='#C53030', linewidth=1.1)
plt.axvline(f0_hz, color='#1A202C', linestyle='--', linewidth=1.0, label=f"Tom em pi/7 ({f0_hz:.1f} Hz)")
plt.title(f"Espectro COM o Tom Adicionado — Amplitude A = {A1:.2f}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(f_t, mag_com2, color='#805AD5', linewidth=1.1)
plt.axvline(f0_hz, color='#1A202C', linestyle='--', linewidth=1.0, label=f"Tom em pi/7 ({f0_hz:.1f} Hz)")
plt.title(f"Espectro COM o Tom Adicionado — Amplitude A/4 = {A2:.3f}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PASTA_GRAFICOS, "espectro_4_4_com_sem_tom.png"), dpi=150)
plt.close()

# Grafico 2 do Item 4.4: Energia do Filtro de Goertzel por Quadro (A e A/4)
quadros_idx = np.arange(1, num_quadros + 1)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.bar(quadros_idx, energias_A1, color='#2B6CB0', width=0.8)
plt.axvspan(40.5, 44.5, color='#FEB2B2', alpha=0.45, label="Quadros com o Tom (41 a 44)")
plt.title(f"Energia na Saída do Filtro de Goertzel por Quadro (Amplitude A = {A1:.2f})")
plt.xlabel("Número do Quadro (2500 amostras por quadro)")
plt.ylabel("Energia |y[N]|²")
plt.xlim(0, num_quadros + 1)
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(quadros_idx, energias_A2, color='#805AD5', width=0.8)
plt.axvspan(40.5, 44.5, color='#FEB2B2', alpha=0.45, label="Quadros com o Tom (41 a 44)")
plt.title(f"Energia na Saída do Filtro de Goertzel por Quadro (Amplitude A/4 = {A2:.3f})")
plt.xlabel("Número do Quadro (2500 amostras por quadro)")
plt.ylabel("Energia |y[N]|²")
plt.xlim(0, num_quadros + 1)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PASTA_GRAFICOS, "energia_4_4_goertzel.png"), dpi=150)
plt.close()

mascara_fora = np.ones(num_quadros, dtype=bool)
mascara_fora[40:44] = False

print(f"Goertzel A={A1:.2f} -> Media nos quadros 41-44: {np.mean(energias_A1[40:44]):.1f} | Max fora: {np.max(energias_A1[mascara_fora]):.1f}")
print(f"Goertzel A/4={A2:.3f} -> Media nos quadros 41-44: {np.mean(energias_A2[40:44]):.1f} | Max fora: {np.max(energias_A2[mascara_fora]):.1f}")
print("Simulacoes do Projeto 3 concluidas!")
