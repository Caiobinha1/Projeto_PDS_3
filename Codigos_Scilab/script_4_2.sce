// =========================================================================
// PROJETO 3 DE PDS - ITEM 4.2: AUMENTO DE TAXA (INTERPOLACAO) POR FATOR 4
// =========================================================================

clc;
clear;

// Carrega o sinal de entrada original x[n] e o sinal dizimado z[n] do item 4.1
try
    [x, Fs] = wavread("../teste.wav");
    [z, Fs_z] = wavread("../Audios_Processados/output_4_1_dizimado_4x.wav");
catch
    [x, Fs] = audioread("../teste.wav");
    [z, Fs_z] = audioread("../Audios_Processados/output_4_1_dizimado_4x.wav");
end

if size(x, 1) > 1 then sinal_x = x(1, :); else sinal_x = x'; end
if size(z, 1) > 1 then sinal_z = z(1, :); else sinal_z = z'; end

sinal_x = sinal_x - mean(sinal_x);
sinal_x = sinal_x / max(abs(sinal_x));

L = 4;
N_z = length(sinal_z);

// 1. Expansao (insercao de L-1 = 3 zeros entre cada amostra de z[n])
disp("Expandindo z[n] por fator 4 (inserindo zeros)...");
z_up = zeros(1, L * N_z);
z_up(1:L:$) = sinal_z;

// 2. Filtro passa-baixas anti-imagem com corte OmegaC = pi/4 e janela Hamming
// Multiplicamos os coeficientes por L = 4 para recuperar o ganho de amplitude
ordem = 65;
fc_norm = 1 / (2 * L); // 0.125
[h_lp, hm, fr] = wfir('lp', ordem, [fc_norm, 0], 'hm', [0, 0]);
h_interp = L * h_lp;

// 3. Filtragem apos o aumento de taxa -> gera o sinal y[n]
disp("Aplicando filtro passa-baixas de interpolacao...");
y_full = convol(h_interp, z_up);
atraso = floor(ordem / 2);
y = y_full(atraso + 1 : atraso + length(z_up));

// Normaliza e salva o sinal reconstruido y[n] na taxa Fs original
normalix = max(abs(min(y)), abs(max(y)));
y_norm = y / normalix;
nome_saida = "../Audios_Processados/output_4_2_interpolado_4x.wav";
try
    wavwrite(y_norm, Fs, nome_saida);
catch
    audiowrite(nome_saida, y_norm', Fs);
end
disp("Audio interpolado salvo em: " + nome_saida);

// Comparacao espectral entre entrada x[n] e saida interpolada y[n]
N_fft = 4096;
ini = round(2.0 * Fs);

trecho_x = sinal_x(ini : ini + N_fft - 1) .* window('hn', N_fft);
trecho_y = y(ini : ini + N_fft - 1) .* window('hn', N_fft);

X_mag = abs(fft(trecho_x));
Y_mag = abs(fft(trecho_y));
f = (0 : N_fft/2 - 1) * (Fs / N_fft);

scf(2); clf();
subplot(2, 1, 1);
plot(f, X_mag(1:N_fft/2) / max(X_mag(1:N_fft/2)), 'b');
xtitle("Espectro do Sinal Original x[n]", "Frequencia (Hz)", "Magnitude Normalizada");

subplot(2, 1, 2);
plot(f, Y_mag(1:N_fft/2) / max(Y_mag(1:N_fft/2)), 'g');
xtitle("Espectro do Sinal Reconstruido y[n] (Aumento 4x + Filtro Hamming)", "Frequencia (Hz)", "Magnitude Normalizada");

disp("Item 4.2 finalizado!");
