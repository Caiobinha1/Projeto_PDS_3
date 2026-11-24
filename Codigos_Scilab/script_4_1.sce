// =========================================================================
// PROJETO 3 DE PDS - ITEM 4.1: REDUCAO DE TAXA (DIZIMACAO) POR FATOR 4
// =========================================================================

clc;
clear;

// Carrega o audio gravado (usando apenas um canal)
arquivo_entrada = "../teste.wav";
try
    [x, Fs] = wavread(arquivo_entrada);
catch
    [x, Fs] = audioread(arquivo_entrada);
end

[linhas, colunas] = size(x);
if linhas > 1 then
    sinal_x = x(1, :);
else
    sinal_x = x';
end

// Remove nivel DC e normaliza a entrada
sinal_x = sinal_x - mean(sinal_x);
sinal_x = sinal_x / max(abs(sinal_x));

disp("Taxa de amostragem original Fs = " + string(Fs) + " Hz");
M = 4;
Fs_novo = Fs / M;
disp("Nova taxa apos dizimacao Fs/4 = " + string(Fs_novo) + " Hz");

// Projeto do filtro FIR passa-baixas anti-aliasing com janela de Hamming
// Corte em OmegaC = pi/4 rad/amostra -> no Scilab wfir usa f_c/Fs = (pi/4)/(2*pi) = 0.125
ordem = 65;
fc_norm = 1 / (2 * M); // 0.125
[h_lp, hm, fr] = wfir('lp', ordem, [fc_norm, 0], 'hm', [0, 0]);

// Aplica a filtragem passa-baixas antes de reduzir a taxa
disp("Filtrando o sinal antes da dizimacao...");
x_filtrado = convol(h_lp, sinal_x);
// Ajusta o tamanho para compensar o atraso do filtro
atraso = floor(ordem / 2);
x_filtrado = x_filtrado(atraso + 1 : atraso + length(sinal_x));

// Reducao de taxa por fator 4 (pega 1 amostra a cada 4)
z = x_filtrado(1:M:$);

// Normaliza e salva o audio dizimado na nova taxa Fs/4
normalix = max(abs(min(z)), abs(max(z)));
z_norm = z / normalix;
nome_saida = "../Audios_Processados/output_4_1_dizimado_4x.wav";
try
    wavwrite(z_norm, Fs_novo, nome_saida);
catch
    audiowrite(nome_saida, z_norm', Fs_novo);
end
disp("Audio dizimado salvo em: " + nome_saida);

// Analise espectral via FFT (trecho ativo da voz)
N_fft = 4096;
ini_x = round(2.0 * Fs);
ini_z = round(2.0 * Fs_novo);

trecho_x = sinal_x(ini_x : ini_x + N_fft - 1) .* window('hn', N_fft);
trecho_z = z(ini_z : ini_z + N_fft - 1) .* window('hn', N_fft);

X_mag = abs(fft(trecho_x));
Z_mag = abs(fft(trecho_z));

f_x = (0 : N_fft/2 - 1) * (Fs / N_fft);
f_z = (0 : N_fft/2 - 1) * (Fs_novo / N_fft);

// Graficos dos espectros de entrada x[n] e saida z[n]
scf(1); clf();
subplot(2, 1, 1);
plot(f_x, X_mag(1:N_fft/2) / max(X_mag(1:N_fft/2)), 'b');
xtitle("Espectro Entrada x[n] (Fs=" + string(Fs) + " Hz)", "Freq (Hz)", "Mag Norm");

subplot(2, 1, 2);
plot(f_z, Z_mag(1:N_fft/2) / max(Z_mag(1:N_fft/2)), 'r');
xtitle("Espectro Dizimado z[n] (Fs/4=" + string(Fs_novo) + " Hz)", "Freq (Hz)", "Mag Norm");

disp("Item 4.1 finalizado!");
