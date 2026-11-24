// =========================================================================
// PROJETO 3 DE PDS - ITEM 4.3: ESTRUTURA POLIFASICA (FATOR 2)
// =========================================================================

clc;
clear;

// Carrega o audio de voz original (1 canal)
try
    [x, Fs] = wavread("../teste.wav");
catch
    [x, Fs] = audioread("../teste.wav");
end

if size(x, 1) > 1 then sinal_x = x(1, :); else sinal_x = x'; end
sinal_x = sinal_x - mean(sinal_x);
sinal_x = sinal_x / max(abs(sinal_x));

N_fator = 2;

// Projeto do filtro passa-baixas prototipo com corte em pi/N = pi/2
// Comprimento par (64 amostras) para dividir em 2 ramos polifasicos iguais de 32
L_filtro = 64;
fc_norm = 1 / (2 * N_fator); // 0.25 (equivalente a pi/2 rad/amostra)
[h, hm, fr] = wfir('lp', L_filtro, [fc_norm, 0], 'hm', [0, 0]);

// Decomposicao polifasica de h[n] em 2 ramos: E0 (pares) e E1 (impares)
// Lembrando que no Scilab os indices comecam em 1: h(1)=h[0], h(2)=h[1]...
E0 = h(1:2:$);
E1 = h(2:2:$);

// -------------------------------------------------------------------------
// PARTE 1: DIZIMACAO POR FATOR 2 USANDO ESTRUTURA POLIFASICA
// -------------------------------------------------------------------------
disp("Executando dizimacao polifasica (fator 2)...");
// Divide a entrada em dois sub-sinais na taxa reduzida Fs/2
// Ramo 0: x[2m] -> indices impares do Scilab (1, 3, 5...)
x0 = sinal_x(1:2:$);
// Ramo 1: x[2m - 1] -> sinal atrasado de 1 amostra e dizimado por 2
x_atrasado = [0, sinal_x(1:$-1)];
x1 = x_atrasado(1:2:$);

// Filtra cada ramo na taxa baixa e soma os resultados
y_dec0 = convol(E0, x0);
y_dec1 = convol(E1, x1);
tam_dec = length(x0);
z_poli = y_dec0(1:tam_dec) + y_dec1(1:tam_dec);

// -------------------------------------------------------------------------
// PARTE 2: INTERPOLACAO POR FATOR 2 USANDO ESTRUTURA POLIFASICA
// -------------------------------------------------------------------------
disp("Executando interpolacao polifasica (fator 2)...");
// Aplica ganho N=2 nos ramos do filtro de interpolacao
G0 = N_fator * E0;
G1 = N_fator * E1;

// Filtra z_poli em paralelo na taxa baixa (sem multiplicar por zeros)
y_ramo0 = convol(G0, z_poli);
y_ramo1 = convol(G1, z_poli);

y_ramo0 = y_ramo0(1:tam_dec);
y_ramo1 = y_ramo1(1:tam_dec);

// Comutador: intercala as saidas dos dois ramos para voltar a taxa Fs
y_poli = zeros(1, 2 * tam_dec);
y_poli(1:2:$) = y_ramo0;
y_poli(2:2:$) = y_ramo1;

// Normaliza e salva o audio processado pela estrutura polifasica
normalix = max(abs(min(y_poli)), abs(max(y_poli)));
y_norm = y_poli / normalix;
nome_saida = "../Audios_Processados/output_4_3_polifasico_2x.wav";
try
    wavwrite(y_norm, Fs, nome_saida);
catch
    audiowrite(nome_saida, y_norm', Fs);
end
disp("Audio polifasico salvo em: " + nome_saida);

// Comparacao espectral via FFT entre entrada x[n] e saida y[n]
N_fft = 4096;
ini = round(2.0 * Fs);
trecho_x = sinal_x(ini : ini + N_fft - 1) .* window('hn', N_fft);
trecho_y = y_poli(ini : ini + N_fft - 1) .* window('hn', N_fft);

X_mag = abs(fft(trecho_x));
Y_mag = abs(fft(trecho_y));
f = (0 : N_fft/2 - 1) * (Fs / N_fft);

scf(3); clf();
plot(f, X_mag(1:N_fft/2) / max(X_mag(1:N_fft/2)), 'b');
plot(f, Y_mag(1:N_fft/2) / max(Y_mag(1:N_fft/2)), 'r--');
legend(["Entrada x[n]", "Saida Polifasica y[n]"]);
xtitle("Comparacao Espectral - Estrutura Polifasica (Fator 2)", "Frequencia (Hz)", "Magnitude Normalizada");

disp("Item 4.3 finalizado!");
