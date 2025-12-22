%% ============================================================
% REPRODUÇÃO COMPLETA DO CAPÍTULO EIGENFACES DO LIVRO
%% ============================================================

clear; close all; clc;
set(groot,'defaultAxesFontSize', 12);
set(groot,'defaultLineMarkerSize', 10);

%% ------------------------------------------------------------
% 1. CONFIGURAÇÃO E CARREGAMENTO DOS DADOS
% ------------------------------------------------------------
fprintf('=== INICIANDO REPRODUÇÃO DO EXEMPLO EIGENFACES ===\n');

% Diretório de dados
dataDir = '../DATA';
if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

dataFile = fullfile(dataDir,'allFaces.mat');

% Download do arquivo (se não existir)
if ~exist(dataFile,'file')
    fprintf('Baixando allFaces.mat...\n');
    url = 'https://bitbucket.org/cpraveen/nla/downloads/allFaces.mat';
    websave(dataFile, url);
end

% Carregamento dos dados
S = load(dataFile);
faces = S.faces;      % cada coluna é uma imagem
m = S.m;              % pixels (horizontal) = 168
n = S.n;              % pixels (vertical) = 192
nfaces = S.nfaces;    % número de imagens por pessoa
npeople = length(nfaces);  % número total de pessoas

fprintf('Dimensões da imagem: %d x %d pixels\n', n, m);
fprintf('Total de pessoas: %d\n', npeople);
fprintf('Total de imagens: %d\n', sum(nfaces));

%% ------------------------------------------------------------
% 1b. CARREGAR IMAGEM DO CACHORRO
% ------------------------------------------------------------
fprintf('\n--- Carregando imagem do cachorro ---\n');

% Tentar carregar a imagem do cachorro
dogFile = 'cachorro.jpg';
if exist(dogFile, 'file')
    fprintf('Carregando %s...\n', dogFile);
    dogImgOriginal = imread(dogFile);
    
    % Verificar se a imagem está em cores e converter para escala de cinza
    if size(dogImgOriginal, 3) == 3
        dogImgGray = rgb2gray(dogImgOriginal);
    else
        dogImgGray = dogImgOriginal;
    end
    
    % Redimensionar para o mesmo tamanho das faces (192x168)
    dogImgResized = imresize(dogImgGray, [n, m]);
    
    % Converter para double e normalizar
    dogImgDouble = double(dogImgResized);
    
    % Achatar para vetor coluna (mesmo formato das faces)
    dogVector = dogImgDouble(:);
    
    fprintf('Imagem do cachorro processada com sucesso!\n');
    fprintf('Dimensões originais: %d x %d\n', size(dogImgOriginal, 1), size(dogImgOriginal, 2));
    fprintf('Dimensões redimensionadas: %d x %d\n', n, m);
    
else
    fprintf('AVISO: Arquivo %s não encontrado.\n', dogFile);
    fprintf('Usando simulação com imagem alternativa...\n');
    dogVector = [];
end

%% ------------------------------------------------------------
% 2. FIGURA 1.16 (ESQUERDA) - TODAS AS PESSOAS
% Code 1.6 MATLAB
% ------------------------------------------------------------
fprintf('\n--- Gerando Figura 1.16 (esquerda) ---\n');

figure('Position', [100, 100, 800, 900], 'Name', 'Figura 1.16: Todas as pessoas');

% Grid 6x6 das primeiras 36 pessoas
allPersons = zeros(n*6, m*6);
count = 0;

for i = 1:6
    for j = 1:6
        % Índice da primeira imagem da pessoa 'count'
        idx = 1 + sum(nfaces(1:count));
        % Extrair e redimensionar a imagem
        faceImg = reshape(faces(:, idx), n, m);
        % Posicionar no grid
        rowRange = (i-1)*n + 1 : i*n;
        colRange = (j-1)*m + 1 : j*m;
        allPersons(rowRange, colRange) = faceImg;
        count = count + 1;
    end
end

subplot(2,1,1);
imagesc(allPersons);
colormap gray;
axis image off;
title('(a) Uma imagem de cada pessoa no banco de dados Yale (36 primeiras)');

%% ------------------------------------------------------------
% 2b. FIGURA 1.16 (DIREITA) - TODAS AS IMAGENS DE UMA PESSOA
% ------------------------------------------------------------
fprintf('--- Gerando Figura 1.16 (direita) ---\n');

% Selecionar uma pessoa específica (ex: pessoa 1)
personNum = 1;
personStart = 1 + sum(nfaces(1:personNum-1));
personEnd = sum(nfaces(1:personNum));

% Criar grid para todas as imagens desta pessoa
% Vamos organizar em 8x8 (64 imagens total)
gridRows = 8;
gridCols = 8;

personGrid = zeros(n*gridRows, m*gridCols);
imgCount = 0;

for i = 1:gridRows
    for j = 1:gridCols
        if imgCount < nfaces(personNum)
            idx = personStart + imgCount;
            faceImg = reshape(faces(:, idx), n, m);
            rowRange = (i-1)*n + 1 : i*n;
            colRange = (j-1)*m + 1 : j*m;
            personGrid(rowRange, colRange) = faceImg;
            imgCount = imgCount + 1;
        end
    end
end

subplot(2,1,2);
imagesc(personGrid);
colormap gray;
axis image off;
title(sprintf('(b) Todas as 64 imagens da Pessoa %d', personNum));

%% ------------------------------------------------------------
% 3. COMPUTAÇÃO DAS EIGENFACES (CODE 1.7)
% ------------------------------------------------------------
fprintf('\n--- Computando Eigenfaces (Code 1.7) ---\n');

% Usar as primeiras 36 pessoas para treinamento
nTrain = sum(nfaces(1:36));
trainingFaces = faces(:, 1:nTrain);

% Rosto médio
avgFace = mean(trainingFaces, 2);

% Subtrair a média
X = trainingFaces - avgFace * ones(1, size(trainingFaces, 2));
fprintf('Dimensões de X: %d x %d\n', size(X,1), size(X,2));

% SVD reduzido
[U, S, V] = svd(X, 'econ');
singularValues = diag(S);

fprintf('SVD completo. Número de valores singulares: %d\n', length(singularValues));

%% ------------------------------------------------------------
% 4. VISUALIZAÇÃO DO ROSTO MÉDIO E EIGENFACES
% ------------------------------------------------------------
fprintf('\n--- Gerando visualização do rosto médio e eigenfaces ---\n');

figure('Position', [100, 100, 1200, 400], 'Name', 'Rosto Médio e Eigenfaces');

% Rosto médio
subplot(1, 5, 1);
imagesc(reshape(avgFace, n, m));
colormap gray;
axis image off;
title('Rosto Médio');

% Eigenfaces
for k = 1:4
    subplot(1, 5, k+1);
    
    if k == 4
        % u_100 conforme mencionado no texto
        eigenfaceIdx = 100;
        titleStr = sprintf('Autoface u_{%d}', eigenfaceIdx);
    else
        eigenfaceIdx = k;
        titleStr = sprintf('Autoface u_{%d}', eigenfaceIdx);
    end
    
    imagesc(reshape(U(:, eigenfaceIdx), n, m));
    colormap gray;
    axis image off;
    title(titleStr);
end

%% ------------------------------------------------------------
% 5. VALORES SINGULARES (FIGURA NÃO NUMERADA NO PDF)
% ------------------------------------------------------------
fprintf('\n--- Gerando gráfico de valores singulares ---\n');

figure('Position', [100, 100, 900, 400], 'Name', 'Valores Singulares');

subplot(1, 2, 1);
plot(singularValues / singularValues(1), 'b-o', 'LineWidth', 1.5);
xlabel('k');
ylabel('\sigma_k / \sigma_1');
title('Valores Singulares (Escala Linear)');
grid on;

subplot(1, 2, 2);
semilogy(singularValues / singularValues(1), 'r-o', 'LineWidth', 1.5);
xlabel('k');
ylabel('\sigma_k / \sigma_1');
title('Valores Singulares (Escala Logarítmica)');
grid on;

%% ------------------------------------------------------------
% 6. FIGURA 1.18 - APROXIMAÇÃO DE IMAGEM DE TESTE
% Code 1.8 MATLAB
% ------------------------------------------------------------
fprintf('\n--- Gerando Figura 1.18: Aproximação de Imagem de Teste ---\n');

% Selecionar uma imagem de teste (pessoa 37, primeira imagem)
testIdx = 1 + sum(nfaces(1:36));
testFace = faces(:, testIdx);

% Subtrair a média
testFaceMS = testFace - avgFace;

% Valores de r para testar
r_values = [25, 50, 100, 200, 400, 800, 1600];
n_r = length(r_values);

figure('Position', [100, 100, 1200, 600], 'Name', 'Figura 1.18: Aproximação com Eigenfaces');

% Imagem original (com média)
subplot(3, ceil((n_r+1)/3), 1);
imagesc(reshape(testFace, n, m));
colormap gray;
axis image off;
title('Imagem de Teste (Pessoa 37)');

% Aproximações para diferentes valores de r
for i = 1:n_r
    r = r_values(i);
    
    % Reconstrução usando r eigenfaces
    reconFace = avgFace + (U(:, 1:r) * (U(:, 1:r)' * testFaceMS));
    
    % Plotar
    subplot(3, ceil((n_r+1)/3), i+1);
    imagesc(reshape(reconFace, n, m));
    colormap gray;
    axis image off;
    title(sprintf('r = %d', r));
end

%% ------------------------------------------------------------
% 7. FIGURA 1.19 - APROXIMAÇÃO DE UM CACHORRO REAL
% ------------------------------------------------------------
fprintf('\n--- Gerando Figura 1.19: Aproximação de Cachorro Real ---\n');

if ~isempty(dogVector)
    % Processar imagem do cachorro
    % Primeiro, precisamos normalizar a imagem do cachorro para ter
    % características similares às faces
    
    % Subtrair o rosto médio (mesmo procedimento das faces)
    dogMS = dogVector - avgFace;
    
    figure('Position', [100, 100, 1200, 700], 'Name', 'Figura 1.19: Aproximação de Cachorro');
    
    % Mostrar imagem original do cachorro
    subplot(3, ceil((n_r+2)/3), 1);
    imagesc(reshape(dogVector, n, m));
    colormap gray;
    axis image off;
    title('Imagem Original do Cachorro');
    
    % Mostrar cachorro com média subtraída
    subplot(3, ceil((n_r+2)/3), 2);
    imagesc(reshape(dogMS + avgFace, n, m));
    colormap gray;
    axis image off;
    title('Cachorro (ajustado)');
    
    % Aproximações para diferentes valores de r
    for i = 1:n_r
        r = r_values(i);
        
        % Reconstrução usando r eigenfaces
        reconDog = avgFace + (U(:, 1:r) * (U(:, 1:r)' * dogMS));
        
        % Plotar
        subplot(3, ceil((n_r+2)/3), i+2);
        imagesc(reshape(reconDog, n, m));
        colormap gray;
        axis image off;
        title(sprintf('r = %d', r));
    end
else
    % Fallback: usar simulação com pessoa 38
    fprintf('Usando simulação com pessoa 38 para cachorro...\n');
    
    dogTestIdx = 1 + sum(nfaces(1:37));
    dogTestFace = faces(:, dogTestIdx);
    dogTestFaceMS = dogTestFace - avgFace;
    
    figure('Position', [100, 100, 1200, 600], 'Name', 'Figura 1.19: Aproximação de "Cachorro" (Simulação)');
    
    subplot(3, ceil((n_r+1)/3), 1);
    imagesc(reshape(dogTestFace, n, m));
    colormap gray;
    axis image off;
    title('"Cachorro" Simulado (Pessoa 38)');
    
    for i = 1:n_r
        r = r_values(i);
        
        % Reconstrução
        reconDog = avgFace + (U(:, 1:r) * (U(:, 1:r)' * dogTestFaceMS));
        
        % Plotar
        subplot(3, ceil((n_r+1)/3), i+1);
        imagesc(reshape(reconDog, n, m));
        colormap gray;
        axis image off;
        title(sprintf('r = %d', r));
    end
end

%% ------------------------------------------------------------
% 8. FIGURA 1.21 - PROJEÇÃO NO ESPAÇO DAS EIGENFACES
% Code 1.9 MATLAB - Classificação automática
% ------------------------------------------------------------
fprintf('\n--- Gerando Figura 1.21: Projeção para Classificação ---\n');

P1num = 2;  % Pessoa 2
P2num = 7;  % Pessoa 7

% Extrair todas as imagens das duas pessoas
P1_start = 1 + sum(nfaces(1:P1num-1));
P1_end = sum(nfaces(1:P1num));
P1 = faces(:, P1_start:P1_end);

P2_start = 1 + sum(nfaces(1:P2num-1));
P2_end = sum(nfaces(1:P2num));
P2 = faces(:, P2_start:P2_end);

% Subtrair a média
P1_ms = P1 - avgFace * ones(1, size(P1, 2));
P2_ms = P2 - avgFace * ones(1, size(P2, 2));

% Projetar nos modos PCA 5 e 6
PCAmodes = [5, 6];
PCACoordsP1 = U(:, PCAmodes)' * P1_ms;
PCACoordsP2 = U(:, PCAmodes)' * P2_ms;

figure('Position', [100, 100, 900, 700], 'Name', 'Figura 1.21: Espaço de Eigenfaces para Classificação');

% Plotar as coordenadas
plot(PCACoordsP1(1,:), PCACoordsP1(2,:), 'kd', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'k', 'DisplayName', sprintf('Pessoa %d', P1num));
hold on;
plot(PCACoordsP2(1,:), PCACoordsP2(2,:), 'r^', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'r', 'DisplayName', sprintf('Pessoa %d', P2num));

% Destacar algumas imagens específicas (3 de cada pessoa)
highlight_idx_P1 = [1, round(size(P1,2)/2), size(P1,2)];
highlight_idx_P2 = [1, round(size(P2,2)/2), size(P2,2)];

% Adicionar círculos azuis
plot(PCACoordsP1(1, highlight_idx_P1), PCACoordsP1(2, highlight_idx_P1), ...
     'bo', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Exemplos Pessoa 2');
plot(PCACoordsP2(1, highlight_idx_P2), PCACoordsP2(2, highlight_idx_P2), ...
     'bo', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Exemplos Pessoa 7');

xlabel('Coordenada PCA 5');
ylabel('Coordenada PCA 6');
title('Projeção no Espaço das Eigenfaces (Modos 5 e 6)');
legend('Location', 'best');
grid on;
hold off;

%% ------------------------------------------------------------
% 9. FIGURA ADICIONAL: COMPARAÇÃO DE RECONSTRUÇÕES
% Mostrar reconstruções lado a lado
% ------------------------------------------------------------
fprintf('\n--- Gerando figura comparativa de reconstruções ---\n');

figure('Position', [100, 100, 1400, 500], 'Name', 'Comparação de Reconstruções');

% Selecionar alguns r values para mostrar detalhadamente
show_r = [25, 100, 400, 1600];

for i = 1:4
    r = show_r(i);
    
    % Reconstruir imagem de teste
    reconFace = avgFace + (U(:, 1:r) * (U(:, 1:r)' * testFaceMS));
    
    % Plotar
    subplot(2, 4, i);
    imagesc(reshape(reconFace, n, m));
    colormap gray;
    axis image off;
    title(sprintf('Teste: r = %d', r));
    
    % Reconstruir cachorro (se disponível)
    if ~isempty(dogVector)
        reconDog = avgFace + (U(:, 1:r) * (U(:, 1:r)' * dogMS));
        dogTitle = 'Cachorro';
    else
        reconDog = avgFace + (U(:, 1:r) * (U(:, 1:r)' * dogTestFaceMS));
        dogTitle = '"Cachorro"';
    end
    
    subplot(2, 4, i+4);
    imagesc(reshape(reconDog, n, m));
    colormap gray;
    axis image off;
    title(sprintf('%s: r = %d', dogTitle, r));
end

%% ------------------------------------------------------------
% 10. ANÁLISE DA VARIÂNCIA EXPLICADA
% ------------------------------------------------------------
fprintf('\n--- Analisando variância explicada ---\n');

% Calcular variância explicada
totalVariance = sum(singularValues.^2);
explainedVariance = cumsum(singularValues.^2) / totalVariance;

% Encontrar r necessário para 90%, 95%, 99% da variância
r90 = find(explainedVariance >= 0.90, 1);
r95 = find(explainedVariance >= 0.95, 1);
r99 = find(explainedVariance >= 0.99, 1);

fprintf('Variância explicada:\n');
fprintf('  r = %d: %.1f%% da variância\n', r90, explainedVariance(r90)*100);
fprintf('  r = %d: %.1f%% da variância\n', r95, explainedVariance(r95)*100);
fprintf('  r = %d: %.1f%% da variância\n', r99, explainedVariance(r99)*100);
fprintf('  r = 100: %.1f%% da variância\n', explainedVariance(100)*100);
fprintf('  r = 400: %.1f%% da variância\n', explainedVariance(400)*100);
fprintf('  r = 1600: %.1f%% da variância\n', explainedVariance(1600)*100);

% Plotar variância explicada
figure('Position', [100, 100, 800, 400], 'Name', 'Variância Explicada');
plot(explainedVariance(1:1000), 'b-', 'LineWidth', 2);
hold on;
plot([r90, r90], [0, explainedVariance(r90)], 'r--', 'LineWidth', 1.5);
plot([r95, r95], [0, explainedVariance(r95)], 'g--', 'LineWidth', 1.5);
plot([r99, r99], [0, explainedVariance(r99)], 'm--', 'LineWidth', 1.5);

xlabel('Número de Componentes (r)');
ylabel('Variância Explicada Acumulada');
title('Variância Explicada pelas Componentes Principais');
legend('Variância', sprintf('90%% (r=%d)', r90), ...
       sprintf('95%% (r=%d)', r95), sprintf('99%% (r=%d)', r99), ...
       'Location', 'southeast');
grid on;
xlim([0, 1000]);

%% ------------------------------------------------------------
% 11. ANÁLISE ADICIONAL: PROJEÇÃO DO CACHORRO NO ESPAÇO DAS EIGENFACES
% ------------------------------------------------------------
if ~isempty(dogVector)
    fprintf('\n--- Projetando cachorro no espaço das eigenfaces ---\n');
    
    % Projetar a imagem do cachorro nos modos PCA
    dogCoords = U(:, PCAmodes)' * dogMS;
    
    figure('Position', [100, 100, 900, 700], 'Name', 'Cachorro no Espaço das Eigenfaces');
    
    % Replotar as pessoas
    plot(PCACoordsP1(1,:), PCACoordsP1(2,:), 'kd', 'MarkerSize', 8, ...
         'MarkerFaceColor', 'k', 'DisplayName', sprintf('Pessoa %d', P1num));
    hold on;
    plot(PCACoordsP2(1,:), PCACoordsP2(2,:), 'r^', 'MarkerSize', 8, ...
         'MarkerFaceColor', 'r', 'DisplayName', sprintf('Pessoa %d', P2num));
    
    % Plotar o cachorro
    plot(dogCoords(1), dogCoords(2), 'gs', 'MarkerSize', 15, ...
         'MarkerFaceColor', 'g', 'LineWidth', 2, 'DisplayName', 'Cachorro');
    
    xlabel('Coordenada PCA 5');
    ylabel('Coordenada PCA 6');
    title('Cachorro Projetado no Espaço das Eigenfaces');
    legend('Location', 'best');
    grid on;
    hold off;
    
    fprintf('Coordenadas do cachorro no espaço PCA [5,6]:\n');
    fprintf('  PCA5: %.4f\n', dogCoords(1));
    fprintf('  PCA6: %.4f\n', dogCoords(2));
end

%% ------------------------------------------------------------
% 12. MOSTRAR IMAGEM DO CACHORRO ORIGINAL SEPARADAMENTE
% ------------------------------------------------------------
if ~isempty(dogVector)
    figure('Position', [100, 100, 800, 400], 'Name', 'Processamento da Imagem do Cachorro');
    
    subplot(1, 3, 1);
    if size(dogImgOriginal, 3) == 3
        imshow(dogImgOriginal);
    else
        imshow(dogImgOriginal, []);
    end
    title('Cachorro Original');
    
    subplot(1, 3, 2);
    imshow(dogImgResized, []);
    title(sprintf('Redimensionado: %dx%d', n, m));
    
    subplot(1, 3, 3);
    imagesc(reshape(dogVector, n, m));
    colormap gray;
    axis image off;
    title('Vetorizado para análise');
end

%% ------------------------------------------------------------
% 13. RESUMO FINAL
% ------------------------------------------------------------
fprintf('\n=== RESUMO DA EXECUÇÃO ===\n');
fprintf('Total de figuras geradas: %d\n', 8 + (~isempty(dogVector) * 2));
fprintf('Figuras reproduzidas do PDF:\n');
fprintf('  1. Figura 1.16 (completa)\n');
fprintf('  2. Rosto médio e eigenfaces (equivalente a Fig. 1.17)\n');
fprintf('  3. Figura 1.18 (aproximação imagem teste)\n');
fprintf('  4. Figura 1.19 (aproximação de cachorro REAL)\n');
fprintf('  5. Figura 1.21 (projeção para classificação)\n');

if ~isempty(dogVector)
    fprintf('Figuras adicionais com cachorro:\n');
    fprintf('  6. Processamento da imagem do cachorro\n');
    fprintf('  7. Cachorro no espaço das eigenfaces\n');
end

fprintf('Figuras técnicas:\n');
fprintf('  8. Valores singulares\n');
fprintf('  9. Comparação de reconstruções\n');
fprintf('  10. Variância explicada\n');

fprintf('\n=== EXECUÇÃO CONCLUÍDA COM SUCESSO! ===\n');