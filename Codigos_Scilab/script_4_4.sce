// =========================================================================
// PROJETO 3 DE PDS - ITEM 4.4: DETECCAO DE TONS (ALGORITMO DE GOERTZEL)
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

N_total = length(sinal_x);
max_x = max(abs(sinal_x));

// Parametros do tom: tom[n] = A * cos(n * pi / 7), 10000 amostras
omega0 = %pi / 7;
L_tom = 10000;
n_tom = 0 : L_tom - 1;

// Quadros de 2500 amostras -> 10000 amostras ocupam exatamente 4 quadros
N_quadro = 2500;
num_quadros = floor(N_total / N_quadro);

// Inserimos o tom a partir da amostra 100001 ate 110000 (Quadros 41, 42, 43 e 44)
ini_tom = 100001;
fim_tom = ini_tom + L_tom - 1;
disp("Tom inserido nas amostras " + string(ini_tom) + " a " + string(fim_tom) + " (Quadros 41 a 44)");

// Funcao auxiliar para rodar o algoritmo de Goertzel em cada quadro de 2500 amostras
function energias = rodar_goertzel(sinal_in, w0, Nq, n_quadros)
    coef = 2 * cos(w0);
    energias = zeros(1, n_quadros);
    for q = 1:n_quadros
        idx_i = (q - 1) * Nq + 1;
        idx_f = q * Nq;
        quadro = sinal_in(idx_i:idx_f);
        
        // Equacao de diferencas recursiva de Goertzel
        v1 = 0; // v[n-1]
        v2 = 0; // v[n-2]
        for n = 1:Nq
            v0 = quadro(n) + coef * v1 - v2;
            v2 = v1;
            v1 = v0;
        end
        
        // Energia na saida do filtro de Goertzel: |y[N]|^2
        energias(q) = v1^2 + v2^2 - coef * v1 * v2;
    end
endfunction

// --- TESTE 1: Amplitude A proxima do maximo do sinal de entrada ---
A1 = 0.9 * max_x;
tom1 = A1 * cos(n_tom * omega0);
x_com_tom1 = sinal_x;
x_com_tom1(ini_tom:fim_tom) = x_com_tom1(ini_tom:fim_tom) + tom1;

energias_A1 = rodar_goertzel(x_com_tom1, omega0, N_quadro, num_quadros);

// --- TESTE 2: Amplitude reduzida para A/4 ---
A2 = A1 / 4;
tom2 = A2 * cos(n_tom * omega0);
x_com_tom2 = sinal_x;
x_com_tom2(ini_tom:fim_tom) = x_com_tom2(ini_tom:fim_tom) + tom2;

energias_A2 = rodar_goertzel(x_com_tom2, omega0, N_quadro, num_quadros);

// Exibe os valores de energia em volta dos quadros onde o tom foi inserido
disp("=== Energias Goertzel (Quadros 38 a 47) ===");
for q = 38:47
    disp("Q" + string(q) + " | E(A)=" + string(energias_A1(q)) + " | E(A/4)=" + string(energias_A2(q)));
end

// Plota os graficos de energia por quadro
scf(4); clf();
subplot(2, 1, 1);
plot(1:num_quadros, energias_A1, 'b');
xtitle("Energia Goertzel - A=" + string(A1), "Quadro (2500 am.)", "Energia");

subplot(2, 1, 2);
plot(1:num_quadros, energias_A2, 'r');
xtitle("Energia Goertzel - A/4=" + string(A2), "Quadro (2500 am.)", "Energia");

// Plota os espectros com e sem o tom no trecho onde o tom foi inserido
N_fft = 4096;
trecho_sem = sinal_x(ini_tom : ini_tom + N_fft - 1) .* window('hn', N_fft);
trecho_com = x_com_tom1(ini_tom : ini_tom + N_fft - 1) .* window('hn', N_fft);
f = (0 : N_fft/2 - 1) * (Fs / N_fft);

scf(5); clf();
subplot(2, 1, 1);
plot(f, abs(fft(trecho_sem))(1:N_fft/2), 'b');
xtitle("Espectro de Entrada SEM o Tom", "Frequencia (Hz)", "Magnitude");

subplot(2, 1, 2);
plot(f, abs(fft(trecho_com))(1:N_fft/2), 'r');
xtitle("Espectro COM o Tom (pi/7 ~ 3428.6 Hz)", "Frequencia (Hz)", "Magnitude");

disp("Item 4.4 finalizado!");
