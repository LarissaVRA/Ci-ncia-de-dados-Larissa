clc; clear; close all;

% === 1. Ler o áudio ===
filename = 'ruido_corte.wav';   
[x, Fs] = audioread(filename);  

% Verificar se o arquivo foi encontrado
if isempty(x)
    error('Arquivo "%s" não encontrado! Verifique o nome e a pasta.', filename);
end

% Se for estéreo, converter para mono
if size(x, 2) == 2
    x = mean(x, 2);
    disp('Áudio convertido de estéreo para mono');
end

% Normalizar sinal
x = x / max(abs(x));

% === 2. Adicionar ruído branco gaussiano ===
SNR_dB = 10;  % Relação sinal-ruído em dB
Px = mean(x.^2);
SNR_linear = 10^(SNR_dB/10);
Pn = Px / SNR_linear;
noise = sqrt(Pn) * randn(size(x));
x_noisy = x + noise;

% === 3. Redução de ruído via FFT ===
N = length(x_noisy);
X = fft(x_noisy);
f = (0:N-1)*(Fs/N);
fc = 2000;  % frequência de corte em Hz

H = (f <= fc | f >= (Fs - fc))';
if size(H, 2) > 1
    H = H';
end

X_filtered = X .* H;
x_denoised_fft = real(ifft(X_filtered));

% === 4. REDUÇÃO DE RUÍDO VIA WAVELET (COM TOOLBOX) ===
disp('Aplicando redução de ruído com Wavelet Toolbox...');

% Método 1: Usando wdenoise (recomendado - mais simples)
wavelet_name = 'db4';
level = 5;

x_denoised_wavelet = wdenoise(x_noisy, level, ...
    'Wavelet', wavelet_name, ...
    'DenoisingMethod', 'SURE', ...
    'ThresholdRule', 'Soft');

% Ajustar tamanho se necessário
x_denoised_wavelet = x_denoised_wavelet(1:N);

% Método 2: Usando wavedec + wthresh + waverec (abordagem direta)
[C, L] = wavedec(x_noisy, level, wavelet_name);

% Calcular threshold usando o primeiro nível de detalhe
D1 = detcoef(C, L, 1);
sigma = median(abs(D1)) / 0.6745;

% Aplicar threshold em todos os níveis de detalhe
C_denoised = C; % Copiar a estrutura original

% Percorrer cada nível de detalhe
current_pos = L(1) + 1; % Pular coeficientes de aproximação

for i = 1:level
    % Tamanho dos coeficientes deste nível
    coeff_length = L(i+1);
    
    % Calcular threshold para este nível
    threshold = sigma * sqrt(2 * log(coeff_length));
    
    % Aplicar threshold suave nos coeficientes de detalhe
    start_idx = current_pos;
    end_idx = current_pos + coeff_length - 1;
    
    C_denoised(start_idx:end_idx) = wthresh(C(start_idx:end_idx), 's', threshold);
    
    % Avançar para o próximo nível
    current_pos = end_idx + 1;
end

% Reconstruir o sinal
x_denoised_wavelet2 = waverec(C_denoised, L, wavelet_name);
x_denoised_wavelet2 = x_denoised_wavelet2(1:N);

% Método 3: Usando wden (função mais antiga mas funcional)
% [x_denoised_wavelet3, ~, ~, ~, ~] = wden(x_noisy, 'heursure', 's', 'one', level, wavelet_name);

% === 5. Plotar comparações ===
t = (0:N-1)/Fs;

figure('Position', [100, 100, 1200, 800]);

subplot(4,1,1);
plot(t, x);
title('Sinal Original');
xlabel('Tempo [s]'); ylabel('Amplitude');
grid on;
xlim([0, t(end)]);

subplot(4,1,2);
plot(t, x_noisy);
title(sprintf('Sinal com Ruído Branco (SNR = %d dB)', SNR_dB));
xlabel('Tempo [s]'); ylabel('Amplitude');
grid on;
xlim([0, t(end)]);

subplot(4,1,3);
plot(t, x_denoised_fft);
title('Redução de Ruído - Método FFT (Passa-Baixa)');
xlabel('Tempo [s]'); ylabel('Amplitude');
grid on;
xlim([0, t(end)]);

subplot(4,1,4);
plot(t, x_denoised_wavelet);
title('Redução de Ruído - Wavelet (wdenoise)');
xlabel('Tempo [s]'); ylabel('Amplitude');
grid on;
xlim([0, t(end)]);

% === 6. Plot adicional comparando os métodos wavelet ===
figure('Position', [200, 200, 1000, 600]);

subplot(2,1,1);
plot(t, x_denoised_wavelet, 'b', 'LineWidth', 1.5); hold on;
plot(t, x_denoised_wavelet2, 'r--', 'LineWidth', 1);
title('Comparação dos Métodos Wavelet');
xlabel('Tempo [s]'); ylabel('Amplitude');
legend('wdenoise (Automático)', 'wavedec + wthresh (Manual)', 'Location', 'best');
grid on;
xlim([0.1, 0.15]); % Zoom

subplot(2,1,2);
plot(t, x, 'k', 'LineWidth', 2); hold on;
plot(t, x_denoised_wavelet, 'b', 'LineWidth', 1);
plot(t, x_denoised_fft, 'g', 'LineWidth', 1);
title('Comparação com Original (Zoom)');
xlabel('Tempo [s]'); ylabel('Amplitude');
legend('Original', 'Wavelet', 'FFT', 'Location', 'best');
grid on;
xlim([0.1, 0.15]);

% === 7. Plot dos coeficientes wavelet ===
figure('Position', [300, 100, 1200, 400]);

subplot(1,3,1);
plot(C, 'b');
title('Coeficientes Wavelet - Originais');
xlabel('Coeficientes'); ylabel('Valor');
grid on;

subplot(1,3,2);
plot(C_denoised, 'r');
title('Coeficientes Wavelet - Após Threshold');
xlabel('Coeficientes'); ylabel('Valor');
grid on;

subplot(1,3,3);
plot(abs(C) - abs(C_denoised), 'g');
title('Coeficientes Removidos (Ruído)');
xlabel('Coeficientes'); ylabel('Valor');
grid on;

% === 8. Calcular métricas de qualidade ===
fprintf('\n=== MÉTRICAS DE QUALIDADE ===\n');

% SNR Original vs Ruidoso
snr_noisy = 10*log10(mean(x.^2)/mean((x_noisy-x).^2));

% SNR após processamento
snr_fft = 10*log10(mean(x.^2)/mean((x_denoised_fft-x).^2));
snr_wavelet1 = 10*log10(mean(x.^2)/mean((x_denoised_wavelet-x).^2));
snr_wavelet2 = 10*log10(mean(x.^2)/mean((x_denoised_wavelet2-x).^2));

fprintf('SNR Ruidoso: %.2f dB\n', snr_noisy);
fprintf('SNR após FFT: %.2f dB\n', snr_fft);
fprintf('SNR após wdenoise: %.2f dB\n', snr_wavelet1);
fprintf('SNR após wavedec+wthresh: %.2f dB\n', snr_wavelet2);

% MSE
mse_fft = mean((x - x_denoised_fft).^2);
mse_wavelet1 = mean((x - x_denoised_wavelet).^2);
mse_wavelet2 = mean((x - x_denoised_wavelet2).^2);

fprintf('\nMSE FFT: %.6f\n', mse_fft);
fprintf('MSE wdenoise: %.6f\n', mse_wavelet1);
fprintf('MSE wavedec+wthresh: %.6f\n', mse_wavelet2);

% === 9. Plot comparativo dos espectros ===
figure('Position', [400, 200, 1000, 400]);

f_plot = f(1:floor(N/2));
X_orig = abs(fft(x));
X_noisy = abs(fft(x_noisy));
X_fft = abs(fft(x_denoised_fft));
X_wavelet = abs(fft(x_denoised_wavelet));

plot(f_plot, X_orig(1:length(f_plot)), 'k', 'LineWidth', 2); hold on;
plot(f_plot, X_noisy(1:length(f_plot)), 'r', 'LineWidth', 1);
plot(f_plot, X_fft(1:length(f_plot)), 'g', 'LineWidth', 1);
plot(f_plot, X_wavelet(1:length(f_plot)), 'b', 'LineWidth', 1);
title('Espectros de Frequência - Comparação');
xlabel('Frequência [Hz]'); ylabel('Magnitude');
legend('Original', 'Ruidoso', 'FFT Filter', 'Wavelet');
grid on;

% === 10. Reproduzir áudios ===
disp(' ');
disp('=== REPRODUÇÃO DOS ÁUDIOS ===');

% Normalizar para reprodução
x_play = x / max(abs(x));
x_noisy_play = x_noisy / max(abs(x_noisy));
x_fft_play = x_denoised_fft / max(abs(x_denoised_fft));
x_wavelet_play = x_denoised_wavelet / max(abs(x_denoised_wavelet));

disp('Reproduzindo: Original');
sound(x_play, Fs);
pause(N/Fs + 1);

disp('Reproduzindo: Com Ruído');
sound(x_noisy_play, Fs);
pause(N/Fs + 1);

disp('Reproduzindo: Reduzido por FFT');
sound(x_fft_play, Fs);
pause(N/Fs + 1);

disp('Reproduzindo: Reduzido por Wavelet');
sound(x_wavelet_play, Fs);

% === Informações adicionais ===
fprintf('\n=== CONFIGURAÇÃO WAVELET ===\n');
fprintf('Wavelet usada: %s\n', wavelet_name);
fprintf('Níveis de decomposição: %d\n', level);
fprintf('Threshold calculado: %.4f\n', sigma * sqrt(2 * log(length(D1))));
fprintf('Comprimento do sinal: %d amostras\n', N);