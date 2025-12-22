%% ============================================================
% EIGENBEEF COMPLETO - TREINO E TESTE
% ============================================================

clear; close all; clc;
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultLineMarkerSize', 8);

fprintf('===========================================\n');
fprintf('     EIGENBEEF - ANÁLISE COMPLETA\n');
fprintf('     VERSÃO OTIMIZADA PARA MEMÓRIA\n');
fprintf('===========================================\n\n');

%% ------------------------------------------------------------
% 1. CONFIGURAÇÃO DAS PASTAS
% ------------------------------------------------------------
trainDir = 'train';
testDir = 'test';

% Verificar existência das pastas
if ~exist(trainDir, 'dir')
    error('❌ Pasta de TREINO não encontrada: %s', trainDir);
end
if ~exist(testDir, 'dir')
    fprintf('⚠️  Pasta de TESTE não encontrada. Continuando apenas com treino.\n');
    testDir = [];
end

%% ------------------------------------------------------------
% 2. FUNÇÃO PARA CARREGAR IMAGENS (OTIMIZADA)
% ------------------------------------------------------------
function [images, labels, fileCount, imgDims] = loadImagesFromFolder(folderPath, labelValue, targetSize)
    % targetSize: [altura, largura] para redimensionamento
    
    fprintf('   Carregando de: %s\n', folderPath);
    
    % Extensões suportadas
    extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'};
    
    % Encontrar TODAS as imagens recursivamente
    allImages = [];
    for ext = extensions
        imageFiles = dir(fullfile(folderPath, '**', ext{1}));
        allImages = [allImages; imageFiles];
    end
    
    numImages = length(allImages);
    fprintf('     Encontradas: %d imagens\n', numImages);
    
    if numImages == 0
        images = [];
        labels = [];
        fileCount = 0;
        imgDims = [];
        return;
    end
    
    % Determinar número máximo de imagens para processar
    MAX_IMAGES = 500; % Limitar número de imagens para evitar memória excessiva
    if numImages > MAX_IMAGES
        fprintf('     Limite de %d imagens aplicado. Selecionando aleatoriamente...\n', MAX_IMAGES);
        selectedIdx = randperm(numImages, MAX_IMAGES);
        allImages = allImages(selectedIdx);
        numImages = MAX_IMAGES;
    end
    
    % Redimensionar para dimensões menores
    targetHeight = targetSize(1);
    targetWidth = targetSize(2);
    
    % Inicializar matriz com dimensões reduzidas
    images = zeros(targetHeight * targetWidth, numImages, 'single'); % Usar single para economizar memória
    labels = labelValue * ones(1, numImages);
    
    % Carregar cada imagem
    for i = 1:numImages
        try
            imgPath = fullfile(allImages(i).folder, allImages(i).name);
            img = imread(imgPath);
            
            % Progresso
            if mod(i, 50) == 0
                fprintf('     Processadas: %d/%d\n', i, numImages);
            end
            
            % Pré-processamento
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            
            % Redimensionar para dimensões fixas
            img = imresize(img, [targetHeight, targetWidth]);
            
            % Converter para single (metade da memória do double) e achatatar
            images(:, i) = single(img(:));
            
        catch ME
            fprintf('     ⚠️  Erro na imagem %s: %s\n', allImages(i).name, ME.message);
            % Preencher com zeros em caso de erro
            images(:, i) = zeros(targetHeight * targetWidth, 1, 'single');
        end
    end
    
    fileCount = numImages;
    imgDims = [targetHeight, targetWidth];
    fprintf('     ✅ Carregadas: %d imagens (%d x %d pixels)\n', numImages, targetHeight, targetWidth);
end

%% ------------------------------------------------------------
% 3. CARREGAR DADOS DE TREINO
% ------------------------------------------------------------
fprintf('\n📥 CARREGANDO DADOS DE TREINO...\n');

% Definir dimensões menores para as imagens
TARGET_SIZE = [100, 100]; % 100x100 pixels = 10,000 pixels por imagem
n = TARGET_SIZE(1); % altura
m = TARGET_SIZE(2); % largura

% Carregar imagens da pasta de treino (primeira subpasta)
subDirs = dir(trainDir);
subDirs = subDirs([subDirs.isdir]);
subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'}));

if length(subDirs) >= 2
    % Se há duas subpastas, assumir que são fresh e rotten
    trainFreshPath = fullfile(trainDir, subDirs(1).name);
    trainRottenPath = fullfile(trainDir, subDirs(2).name);
    
    [trainFresh, labelsFresh, numTrainFresh, dimsFresh] = loadImagesFromFolder(trainFreshPath, 1, TARGET_SIZE);
    [trainRotten, labelsRotten, numTrainRotten, dimsRotten] = loadImagesFromFolder(trainRottenPath, 2, TARGET_SIZE);
    
elseif length(subDirs) == 1
    % Se só uma pasta, dividir os dados
    [allTrain, allLabels, numAll, dims] = loadImagesFromFolder(fullfile(trainDir, subDirs(1).name), 1, TARGET_SIZE);
    
    % Dividir 50/50
    splitPoint = floor(numAll/2);
    trainFresh = allTrain(:, 1:splitPoint);
    labelsFresh = 1 * ones(1, splitPoint);
    trainRotten = allTrain(:, splitPoint+1:end);
    labelsRotten = 2 * ones(1, size(trainRotten, 2));
    
    numTrainFresh = splitPoint;
    numTrainRotten = size(trainRotten, 2);
else
    error('❌ Nenhuma subpasta encontrada em %s', trainDir);
end

% Combinar dados de treino
trainingFaces = [trainFresh, trainRotten];
trainingLabels = [labelsFresh, labelsRotten];
numTrainTotal = size(trainingFaces, 2);

fprintf('\n📊 ESTATÍSTICAS DE TREINO:\n');
fprintf('   - Total de imagens: %d\n', numTrainTotal);
fprintf('   - Fresh: %d (%.1f%%)\n', numTrainFresh, numTrainFresh/numTrainTotal*100);
fprintf('   - Rotten: %d (%.1f%%)\n', numTrainRotten, numTrainRotten/numTrainTotal*100);
fprintf('   - Dimensões: %d x %d pixels\n', m, n);

%% ------------------------------------------------------------
% 4. CARREGAR DADOS DE TESTE (SE DISPONÍVEL)
% ------------------------------------------------------------
if ~isempty(testDir)
    fprintf('\n📥 CARREGANDO DADOS DE TESTE...\n');
    
    testSubDirs = dir(testDir);
    testSubDirs = testSubDirs([testSubDirs.isdir]);
    testSubDirs = testSubDirs(~ismember({testSubDirs.name}, {'.', '..'}));
    
    if length(testSubDirs) >= 2
        % Duas subpastas
        testFreshPath = fullfile(testDir, testSubDirs(1).name);
        testRottenPath = fullfile(testDir, testSubDirs(2).name);
        
        [testFresh, testLabelsFresh, numTestFresh] = loadImagesFromFolder(testFreshPath, 1, TARGET_SIZE);
        [testRotten, testLabelsRotten, numTestRotten] = loadImagesFromFolder(testRottenPath, 2, TARGET_SIZE);
        
    elseif length(testSubDirs) == 1
        % Uma subpasta, dividir
        [allTest, allTestLabels, numAllTest] = loadImagesFromFolder(fullfile(testDir, testSubDirs(1).name), 1, TARGET_SIZE);
        
        splitPoint = floor(numAllTest/2);
        testFresh = allTest(:, 1:splitPoint);
        testLabelsFresh = 1 * ones(1, splitPoint);
        testRotten = allTest(:, splitPoint+1:end);
        testLabelsRotten = 2 * ones(1, size(testRotten, 2));
        
        numTestFresh = splitPoint;
        numTestRotten = size(testRotten, 2);
    else
        testFresh = [];
        testRotten = [];
        numTestFresh = 0;
        numTestRotten = 0;
    end
    
    % Combinar dados de teste
    if numTestFresh > 0 || numTestRotten > 0
        testingFaces = [testFresh, testRotten];
        testingLabels = [testLabelsFresh, testLabelsRotten];
        numTestTotal = size(testingFaces, 2);
        
        fprintf('\n📊 ESTATÍSTICAS DE TESTE:\n');
        fprintf('   - Total de imagens: %d\n', numTestTotal);
        fprintf('   - Fresh: %d\n', numTestFresh);
        fprintf('   - Rotten: %d\n', numTestRotten);
    else
        testingFaces = [];
        testingLabels = [];
        numTestTotal = 0;
        fprintf('   ⚠️  Nenhuma imagem de teste encontrada.\n');
    end
else
    testingFaces = [];
    testingLabels = [];
    numTestTotal = 0;
end

%% ------------------------------------------------------------
% 5. CÁLCULO DO EIGENBEEF (SVD ECONÔMICO)
% ------------------------------------------------------------
fprintf('\n🧮 Calculando Eigenbeef (SVD)...\n');

% Converter para double para SVD (necessário para precisão)
trainingFacesDouble = double(trainingFaces);

% Calcar carne média
avgFace = mean(trainingFacesDouble, 2);
fprintf('   ✅ Carne média calculada\n');

% Centralizar dados
X = trainingFacesDouble - avgFace;

% Aplicar SVD econômico (com menos componentes se necessário)
MAX_COMPONENTS = min(100, size(X, 2) - 1); % Limitar componentes principais
fprintf('   Aplicando SVD com no máximo %d componentes...\n', MAX_COMPONENTS);

% Usar svds para obter apenas os primeiros componentes
[U, S, V] = svds(X, MAX_COMPONENTS);
singularValues = diag(S);

fprintf('   ✅ SVD concluído!\n');
fprintf('   - Componentes principais: %d\n', size(U, 2));
fprintf('   - Valor singular máximo: %.4f\n', singularValues(1));

%% ------------------------------------------------------------
% 6. FIGURA 1: GRID DE IMAGENS DE TREINO
% ------------------------------------------------------------
fprintf('\n🎨 Gerando Figura 1: Grid de imagens de treino...\n');

% Selecionar 36 imagens para o grid (como no eigenface)
numGrid = min(36, numTrainTotal);
gridIndices = randperm(numTrainTotal, numGrid);

% Criar grid 6x6
gridRows = 6;
gridCols = 6;
gridImage = zeros(n * gridRows, m * gridCols);

for i = 1:gridRows
    for j = 1:gridCols
        idx = (i-1)*gridCols + j;
        if idx <= numGrid
            imgIdx = gridIndices(idx);
            img = reshape(trainingFacesDouble(:, imgIdx), n, m);
            
            rowRange = (i-1)*n + 1 : i*n;
            colRange = (j-1)*m + 1 : j*m;
            gridImage(rowRange, colRange) = img;
        end
    end
end

figure('Position', [100, 100, 800, 800], 'Name', 'Figura 1: Imagens de Treino');
imagesc(gridImage);
colormap gray;
axis image off;
title(sprintf('36 Amostras de Carne (Treino: %d fresh, %d rotten)', ...
      sum(trainingLabels(gridIndices) == 1), sum(trainingLabels(gridIndices) == 2)));

%% ------------------------------------------------------------
% 7. FIGURA 2: CARNE MÉDIA E EIGENBEEFS
% ------------------------------------------------------------
fprintf('\n🎨 Gerando Figura 2: Carne média e eigenbeefs...\n');

figure('Position', [100, 100, 1400, 300], 'Name', 'Figura 2: Carne Média e Eigenbeefs');

% 1. Carne média
subplot(1, 5, 1);
imagesc(reshape(avgFace, n, m));
colormap gray;
axis image off;
title('Carne Média');
colorbar;

% 2. Eigenbeef 1
subplot(1, 5, 2);
imagesc(reshape(U(:, 1), n, m));
colormap gray;
axis image off;
title('Eigenbeef u_1');
colorbar;

% 3. Eigenbeef 2
subplot(1, 5, 3);
imagesc(reshape(U(:, 2), n, m));
colormap gray;
axis image off;
title('Eigenbeef u_2');
colorbar;

% 4. Eigenbeef 3
subplot(1, 5, 4);
imagesc(reshape(U(:, 3), n, m));
colormap gray;
axis image off;
title('Eigenbeef u_3');
colorbar;

% 5. Eigenbeef 100 (ou o último se houver menos)
eigenIdx = min(100, size(U, 2));
subplot(1, 5, 5);
imagesc(reshape(U(:, eigenIdx), n, m));
colormap gray;
axis image off;
title(sprintf('Eigenbeef u_{%d}', eigenIdx));
colorbar;

%% ------------------------------------------------------------
% 8. FIGURA 3: VALORES SINGULARES
% ------------------------------------------------------------
fprintf('\n📈 Gerando Figura 3: Valores singulares...\n');

figure('Position', [100, 100, 1000, 400], 'Name', 'Figura 3: Valores Singulares');

subplot(1, 2, 1);
plot(singularValues / singularValues(1), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('k');
ylabel('\sigma_k / \sigma_1');
title('Valores Singulares (Escala Linear)');
grid on;

subplot(1, 2, 2);
semilogy(singularValues / singularValues(1), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('k');
ylabel('\sigma_k / \sigma_1');
title('Valores Singulares (Escala Logarítmica)');
grid on;

%% ------------------------------------------------------------
% 9. FIGURA 4: RECONSTRUÇÃO DE IMAGEM DE TESTE
% ------------------------------------------------------------
fprintf('\n🔧 Gerando Figura 4: Reconstrução de imagem de teste...\n');

if numTestTotal > 0
    % Converter imagem de teste para double
    testingFacesDouble = double(testingFaces);
    
    % Usar a primeira imagem de teste
    testFace = testingFacesDouble(:, 1);
    testLabel = testingLabels(1);
    testFaceMS = testFace - avgFace;
    
    % Valores de r para testar
    r_values = [5, 10, 25, 50, 75, 100]; % Ajustado para máximo de 100 componentes
    r_values = r_values(r_values <= size(U, 2));
    
    figure('Position', [100, 100, 1200, 400], 'Name', 'Figura 4: Reconstrução de Imagem de Teste');
    
    % Imagem original
    subplot(2, length(r_values)+1, 1);
    imagesc(reshape(testFace, n, m));
    colormap gray;
    axis image off;
    if testLabel == 1
        title('Teste (Fresh)');
    else
        title('Teste (Rotten)');
    end
    
    % Reconstruções
    for i = 1:length(r_values)
        r = r_values(i);
        reconFace = avgFace + (U(:, 1:r) * (U(:, 1:r)' * testFaceMS));
        
        subplot(2, length(r_values)+1, i+1);
        imagesc(reshape(reconFace, n, m));
        colormap gray;
        axis image off;
        title(sprintf('r = %d', r));
    end
else
    fprintf('   ⚠️  Sem dados de teste para reconstrução.\n');
end

%% ------------------------------------------------------------
% 10. FIGURA 5: PROJEÇÃO PARA CLASSIFICAÇÃO
% ------------------------------------------------------------
fprintf('\n🎯 Gerando Figura 5: Projeção para classificação...\n');

% Calcular centroides das classes no espaço original
freshIndices = find(trainingLabels == 1);
rottenIndices = find(trainingLabels == 2);

if length(freshIndices) >= 10 && length(rottenIndices) >= 10
    % Selecionar amostras aleatórias de cada classe
    numSamples = min(50, min(length(freshIndices), length(rottenIndices)));
    freshSamples = freshIndices(randperm(length(freshIndices), numSamples));
    rottenSamples = rottenIndices(randperm(length(rottenIndices), numSamples));
    
    % Extrair as imagens
    P1 = trainingFacesDouble(:, freshSamples);  % "Pessoa 1" = Fresh
    P2 = trainingFacesDouble(:, rottenSamples); % "Pessoa 2" = Rotten
    
    % Centralizar
    P1_ms = P1 - avgFace;
    P2_ms = P2 - avgFace;
    
    % Projetar nos eigenbeefs 1 e 2 (ou 5 e 6 se disponíveis)
    if size(U, 2) >= 6
        PCAmodes = [5, 6];
    elseif size(U, 2) >= 2
        PCAmodes = [1, 2];
    else
        PCAmodes = [1, min(2, size(U, 2))];
    end
    
    PCACoordsP1 = U(:, PCAmodes)' * P1_ms;
    PCACoordsP2 = U(:, PCAmodes)' * P2_ms;
    
    figure('Position', [100, 100, 900, 700], 'Name', 'Figura 5: Espaço de Classificação Eigenbeef');
    
    % Plotar projeções
    plot(PCACoordsP1(1, :), PCACoordsP1(2, :), 'kd', 'MarkerSize', 8, ...
         'MarkerFaceColor', 'k', 'DisplayName', 'Fresh');
    hold on;
    plot(PCACoordsP2(1, :), PCACoordsP2(2, :), 'r^', 'MarkerSize', 8, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'Rotten');
    
    % Destacar 3 exemplos de cada
    highlightP1 = [1, floor(numSamples/2), numSamples];
    highlightP2 = [1, floor(numSamples/2), numSamples];
    
    plot(PCACoordsP1(1, highlightP1), PCACoordsP1(2, highlightP1), ...
         'bo', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Exemplos Fresh');
    plot(PCACoordsP2(1, highlightP2), PCACoordsP2(2, highlightP2), ...
         'bo', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Exemplos Rotten');
    
    xlabel(sprintf('Eigenbeef %d', PCAmodes(1)));
    ylabel(sprintf('Eigenbeef %d', PCAmodes(2)));
    title('Projeção no Espaço Eigenbeef para Classificação');
    legend('Location', 'best');
    grid on;
    hold off;
else
    fprintf('   ⚠️  Classes muito pequenas para projeção.\n');
end

%% ------------------------------------------------------------
% 11. ANÁLISE DE VARIÂNCIA EXPLICADA
% ------------------------------------------------------------
fprintf('\n📊 Analisando variância explicada...\n');

% Calcular variância explicada
totalVariance = sum(singularValues.^2);
explainedVariance = cumsum(singularValues.^2) / totalVariance;

% Encontrar número de componentes para diferentes níveis
varianceLevels = [0.50, 0.75, 0.90, 0.95, 0.99];
numComponentsNeeded = zeros(size(varianceLevels));

for i = 1:length(varianceLevels)
    idx = find(explainedVariance >= varianceLevels(i), 1);
    if ~isempty(idx)
        numComponentsNeeded(i) = idx;
    end
end

figure('Position', [100, 100, 800, 500], 'Name', 'Variância Explicada');

plot(explainedVariance, 'b-', 'LineWidth', 3);
hold on;

% Linhas de referência
colors = {'r--', 'g--', 'm--', 'c--', 'k--'};
for i = 1:length(varianceLevels)
    if numComponentsNeeded(i) > 0
        plot([numComponentsNeeded(i), numComponentsNeeded(i)], [0, varianceLevels(i)], ...
             colors{i}, 'LineWidth', 1.5);
        plot([0, numComponentsNeeded(i)], [varianceLevels(i), varianceLevels(i)], ...
             colors{i}, 'LineWidth', 1.5);
    end
end

xlabel('Número de Componentes (r)');
ylabel('Variância Explicada Acumulada');
title('Variância Explicada pelos Eigenbeefs');
legendStr = ['Variância', arrayfun(@(v) sprintf('%.0f%%', v*100), varianceLevels, 'UniformOutput', false)];
legend(legendStr, 'Location', 'southeast');
grid on;
xlim([0, min(100, length(singularValues))]);
ylim([0, 1.05]);

%% ------------------------------------------------------------
% 12. CLASSIFICAÇÃO SIMPLES USANDO EIGENBEEFS
% ------------------------------------------------------------
fprintf('\n🤖 Testando classificação com eigenbeefs...\n');

if numTestTotal > 0
    % Usar primeiros r eigenbeefs para características
    r_classify = min(50, size(U, 2));
    
    % Projetar dados de treino
    trainFeatures = U(:, 1:r_classify)' * X;
    
    % Separar por classe
    trainFreshFeatures = trainFeatures(:, trainingLabels == 1);
    trainRottenFeatures = trainFeatures(:, trainingLabels == 2);
    
    % Calcular centroides
    centroidFresh = mean(trainFreshFeatures, 2);
    centroidRotten = mean(trainRottenFeatures, 2);
    
    % Classificar dados de teste
    testingFacesDouble = double(testingFaces); % Garantir que está em double
    testFeatures = U(:, 1:r_classify)' * (testingFacesDouble - avgFace);
    
    % Distância aos centroides
    distToFresh = sum((testFeatures - centroidFresh).^2, 1);
    distToRotten = sum((testFeatures - centroidRotten).^2, 1);
    
    % Prever labels
    predictedLabels = 1 + (distToRotten < distToFresh);  % 1=fresh, 2=rotten
    
    % Calcular acurácia
    accuracy = sum(predictedLabels == testingLabels) / numTestTotal * 100;
    
    % Calcular matriz de confusão MANUALMENTE (sem confusionmat)
    C = zeros(2, 2);
    for i = 1:numTestTotal
        trueLabel = testingLabels(i);
        predLabel = predictedLabels(i);
        C(trueLabel, predLabel) = C(trueLabel, predLabel) + 1;
    end
    
    fprintf('   📊 RESULTADOS DE CLASSIFICAÇÃO:\n');
    fprintf('      - Acurácia: %.2f%%\n', accuracy);
    fprintf('      - Matriz de confusão:\n');
    fprintf('        Verdadeiro\\Previsto   Fresh   Rotten\n');
    fprintf('        Fresh                 %6d   %6d\n', C(1,1), C(1,2));
    fprintf('        Rotten                %6d   %6d\n', C(2,1), C(2,2));
    
    % Calcular métricas adicionais
    TP = C(1,1); % True Positive: Fresh classificado como Fresh
    FP = C(2,1); % False Positive: Rotten classificado como Fresh
    TN = C(2,2); % True Negative: Rotten classificado como Rotten
    FN = C(1,2); % False Negative: Fresh classificado como Rotten
    
    precision = TP / (TP + FP) * 100;
    recall = TP / (TP + FN) * 100;
    specificity = TN / (TN + FP) * 100;
    
    fprintf('      - Métricas adicionais:\n');
    fprintf('        * Precisão (Fresh): %.2f%%\n', precision);
    fprintf('        * Recall (Fresh): %.2f%%\n', recall);
    fprintf('        * Especificidade: %.2f%%\n', specificity);
    
    % Plotar matriz de confusão
    figure('Position', [100, 100, 600, 500], 'Name', 'Matriz de Confusão');
    imagesc(C);
    colormap(flipud(gray));
    colorbar;
    
    % Adicionar valores
    textStrings = num2str(C(:), '%d');
    textStrings = strtrim(cellstr(textStrings));
    
    [x, y] = meshgrid(1:size(C, 2), 1:size(C, 1));
    text(x(:), y(:), textStrings, ...
         'HorizontalAlignment', 'center', ...
         'Color', 'white', 'FontWeight', 'bold', 'FontSize', 14);
    
    xticks(1:2);
    yticks(1:2);
    xticklabels({'Fresh', 'Rotten'});
    yticklabels({'Fresh', 'Rotten'});
    xlabel('Predito');
    ylabel('Verdadeiro');
    title(sprintf('Matriz de Confusão (Acurácia: %.1f%%)', accuracy));
else
    fprintf('   ⚠️  Sem dados de teste para classificação.\n');
end

%% ------------------------------------------------------------
% 13. SALVAR RESULTADOS
% ------------------------------------------------------------
fprintf('\n💾 Salvando resultados...\n');

% Criar estrutura com resultados
results = struct();
results.avgFace = avgFace;
results.eigenbeefs = U;
results.singularValues = singularValues;
results.trainingLabels = trainingLabels;
results.numTrain = numTrainTotal;
results.numTrainFresh = numTrainFresh;
results.numTrainRotten = numTrainRotten;

if numTestTotal > 0
    results.testingLabels = testingLabels;
    results.numTest = numTestTotal;
    results.numTestFresh = numTestFresh;
    results.numTestRotten = numTestRotten;
    
    % Adicionar resultados de classificação se disponíveis
    if exist('accuracy', 'var')
        results.accuracy = accuracy;
        results.confusionMatrix = C;
        results.predictedLabels = predictedLabels;
        
        if exist('precision', 'var')
            results.precision = precision;
            results.recall = recall;
            results.specificity = specificity;
        end
    end
end

results.imageDims = [n, m];
results.explainedVariance = explainedVariance;

% Salvar
save('eigenbeef_complete_results.mat', '-struct', 'results');
fprintf('   ✅ Resultados salvos em: eigenbeef_complete_results.mat\n');

%% ------------------------------------------------------------
% 14. RESUMO FINAL
% ------------------------------------------------------------
fprintf('\n===========================================\n');
fprintf('📋 RESUMO DA ANÁLISE EIGENBEEF COMPLETA\n');
fprintf('===========================================\n\n');

fprintf('DADOS PROCESSADOS:\n');
fprintf('   TREINO:\n');
fprintf('     - Total: %d imagens\n', numTrainTotal);
fprintf('     - Fresh: %d (%.1f%%)\n', numTrainFresh, numTrainFresh/numTrainTotal*100);
fprintf('     - Rotten: %d (%.1f%%)\n', numTrainRotten, numTrainRotten/numTrainTotal*100);

if numTestTotal > 0
    fprintf('   TESTE:\n');
    fprintf('     - Total: %d imagens\n', numTestTotal);
    fprintf('     - Fresh: %d\n', numTestFresh);
    fprintf('     - Rotten: %d\n', numTestRotten);
end

fprintf('\nANÁLISE SVD:\n');
fprintf('   - Dimensões: %d x %d pixels\n', m, n);
fprintf('   - Eigenbeefs calculados: %d\n', size(U, 2));
fprintf('   - Valor singular máximo: %.4f\n', singularValues(1));
fprintf('   - Variância total: %.4e\n', totalVariance);

fprintf('\nVARIÂNCIA EXPLICADA:\n');
for i = 1:length(varianceLevels)
    if numComponentsNeeded(i) > 0
        fprintf('   - %.0f%%: %d componentes\n', varianceLevels(i)*100, numComponentsNeeded(i));
    end
end

fprintf('\nFIGURAS GERADAS:\n');
fprintf('   1. Grid de imagens de treino (6x6)\n');
fprintf('   2. Carne média e eigenbeefs\n');
fprintf('   3. Valores singulares\n');
if numTestTotal > 0
    fprintf('   4. Reconstrução de imagem de teste\n');
end
fprintf('   5. Espaço de classificação\n');
fprintf('   6. Variância explicada\n');
if numTestTotal > 0
    fprintf('   7. Matriz de confusão\n');
end

if numTestTotal > 0 && exist('accuracy', 'var')
    fprintf('\nCLASSIFICAÇÃO:\n');
    fprintf('   - Acurácia no teste: %.2f%%\n', accuracy);
    fprintf('   - Precisão (Fresh): %.2f%%\n', precision);
    fprintf('   - Recall (Fresh): %.2f%%\n', recall);
    fprintf('   - Especificidade: %.2f%%\n', specificity);
end

fprintf('\n✨ ANÁLISE EIGENBEEF CONCLUÍDA COM SUCESSO!\n');
fprintf('   Verifique as figuras geradas.\n');
fprintf('   Resultados salvos para uso futuro.\n');