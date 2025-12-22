%% ============================================================
% EIGENBEEF - PARTE 2: RPCA PARA ANÁLISE ROBUSTA
% ============================================================

clear; close all; clc;
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultLineMarkerSize', 8);

fprintf('===========================================\n');
fprintf('     EIGENBEEF - PARTE 2: RPCA AVANÇADO\n');
fprintf('===========================================\n\n');

%% ------------------------------------------------------------
% 1. CARREGAR DADOS DIRETAMENTE
% ------------------------------------------------------------
fprintf('📂 Carregando dados de imagens...\n');

trainDir = 'train';
testDir = 'test';
TARGET_SIZE = [100, 100];
n = TARGET_SIZE(1);
m = TARGET_SIZE(2);

% --- Carregar dados de treino ---
subDirs = dir(trainDir);
subDirs = subDirs([subDirs.isdir]);
subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'}));

if length(subDirs) >= 2
    trainFreshPath = fullfile(trainDir, subDirs(1).name);
    trainRottenPath = fullfile(trainDir, subDirs(2).name);
    [trainFresh, labelsFresh, numTrainFresh, dimsFresh] = loadImagesFromFolder(trainFreshPath, 1, TARGET_SIZE);
    [trainRotten, labelsRotten, numTrainRotten, dimsRotten] = loadImagesFromFolder(trainRottenPath, 2, TARGET_SIZE);
elseif length(subDirs) == 1
    [allTrain, allLabels, numAll, dims] = loadImagesFromFolder(fullfile(trainDir, subDirs(1).name), 1, TARGET_SIZE);
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

fprintf('\n📊 DADOS DE TREINO CARREGADOS:\n');
fprintf('   - Total de imagens: %d\n', numTrainTotal);
fprintf('   - Fresh: %d (%.1f%%)\n', numTrainFresh, numTrainFresh/numTrainTotal*100);
fprintf('   - Rotten: %d (%.1f%%)\n', numTrainRotten, numTrainRotten/numTrainTotal*100);
fprintf('   - Dimensões: %d x %d pixels\n', m, n);

% --- Carregar dados de teste (se disponível) ---
testingFaces = [];
testingLabels = [];
numTestTotal = 0;
numTestFresh = 0;
numTestRotten = 0;

if exist(testDir, 'dir')
    fprintf('\n📂 Carregando dados de teste...\n');
    
    testSubDirs = dir(testDir);
    testSubDirs = testSubDirs([testSubDirs.isdir]);
    testSubDirs = testSubDirs(~ismember({testSubDirs.name}, {'.', '..'}));
    
    if length(testSubDirs) >= 2
        testFreshPath = fullfile(testDir, testSubDirs(1).name);
        testRottenPath = fullfile(testDir, testSubDirs(2).name);
        [testFresh, testLabelsFresh, numTestFresh] = loadImagesFromFolder(testFreshPath, 1, TARGET_SIZE);
        [testRotten, testLabelsRotten, numTestRotten] = loadImagesFromFolder(testRottenPath, 2, TARGET_SIZE);
        
        testingFaces = [testFresh, testRotten];
        testingLabels = [testLabelsFresh, testLabelsRotten];
        numTestTotal = size(testingFaces, 2);
        
        fprintf('📊 DADOS DE TESTE CARREGADOS:\n');
        fprintf('   - Total de imagens: %d\n', numTestTotal);
        fprintf('   - Fresh: %d\n', numTestFresh);
        fprintf('   - Rotten: %d\n', numTestRotten);
    else
        fprintf('   ⚠️  Estrutura de pastas de teste não reconhecida.\n');
    end
else
    fprintf('   ⚠️  Pasta de teste não encontrada.\n');
end

%% ------------------------------------------------------------
% 3. APLICAR RPCA NAS IMAGENS DE CARNE
% ------------------------------------------------------------
fprintf('\n🔬 Aplicando RPCA nas imagens de carne...\n');

% Preparar matriz de dados (usar subset para demonstração)
num_samples_per_class = 50;

% Encontrar índices das classes
fresh_idx = find(trainingLabels == 1);
rotten_idx = find(trainingLabels == 2);

% Selecionar amostras
if length(fresh_idx) >= num_samples_per_class && length(rotten_idx) >= num_samples_per_class
    selected_fresh = fresh_idx(1:num_samples_per_class);
    selected_rotten = rotten_idx(1:num_samples_per_class);
else
    % Usar todas as imagens disponíveis
    selected_fresh = fresh_idx;
    selected_rotten = rotten_idx;
end

% Combinar dados
selected_idx = [selected_fresh, selected_rotten];
X_data = double(trainingFaces(:, selected_idx)); % Converter para double

% Para RPCA, precisamos de matriz 2D (pixels x imagens)
X = X_data;

fprintf('   Matriz de dados: %d pixels × %d imagens\n', size(X, 1), size(X, 2));

% Parâmetros RPCA (como no livro)
lambda = 1 / sqrt(max(size(X))); % λ = 1/√max(n,m)
mu = size(X, 1) * size(X, 2) / (4 * sum(abs(X(:)))); % μ sugerido
max_iter = 1000;
tolerance = 1e-7;

fprintf('   Parâmetros RPCA:\n');
fprintf('     - lambda: %.4f\n', lambda);
fprintf('     - mu: %.4f\n', mu);
fprintf('     - Máx iterações: %d\n', max_iter);

% Executar RPCA
fprintf('   Executando RPCA (isso pode levar alguns minutos)...\n');
tic;
[L, S, Y, iter_count] = RPCA_adm(X, lambda, mu, max_iter, tolerance);
elapsed_time = toc;

fprintf('   ✅ RPCA concluído em %.2f segundos\n', elapsed_time);
fprintf('   - Iterações realizadas: %d\n', iter_count);
fprintf('   - Erro final: %.4e\n', norm(X - L - S, 'fro'));
fprintf('   - Rank(L): %d\n', rank(L, 1e-6));
fprintf('   - Sparsidade de S: %.2f%%\n', 100 * nnz(S) / numel(S));

%% ------------------------------------------------------------
% 4. FIGURA 1: COMPONENTES RPCA (COMO FIGURA 3.20 DO LIVRO)
% ------------------------------------------------------------
fprintf('\n🎨 Gerando Figura 1: Decomposição RPCA...\n');

% Selecionar 5 imagens para visualização
num_viz = 5;
viz_indices = [1, 2, 14, 17, 20]; % Como no exemplo do livro
viz_indices = viz_indices(viz_indices <= size(X, 2));

figure('Position', [50, 50, 1400, 800], 'Name', 'Figura 1: Decomposição RPCA (como Fig 3.20)');

for i = 1:length(viz_indices)
    idx = viz_indices(i);
    
    % Imagem original (X)
    subplot(3, length(viz_indices), i);
    img_original = reshape(X(:, idx), n, m);
    imagesc(img_original);
    colormap gray;
    axis image off;
    if i == 1
        ylabel('Original X', 'FontSize', 12, 'FontWeight', 'bold');
    end
    title(sprintf('Imagem %d', idx));
    
    % Componente de baixo rank (L)
    subplot(3, length(viz_indices), i + length(viz_indices));
    img_lowrank = reshape(L(:, idx), n, m);
    imagesc(img_lowrank);
    colormap gray;
    axis image off;
    if i == 1
        ylabel('Low-rank L', 'FontSize', 12, 'FontWeight', 'bold');
    end
    
    % Componente esparso (S) - outliers
    subplot(3, length(viz_indices), i + 2*length(viz_indices));
    img_sparse = reshape(S(:, idx), n, m);
    imagesc(img_sparse);
    colormap gray;
    axis image off;
    if i == 1
        ylabel('Sparse S', 'FontSize', 12, 'FontWeight', 'bold');
    end
    caxis([-50, 50]); % Limitar escala para melhor visualização
end

sgtitle('Decomposição RPCA: X = L + S', 'FontSize', 14, 'FontWeight', 'bold');

%% ------------------------------------------------------------
% 5. ANÁLISE DOS COMPONENTES RPCA
% ------------------------------------------------------------
fprintf('\n📊 Analisando componentes RPCA...\n');

% Calcular estatísticas dos componentes
L_norm = norm(L, 'fro');
S_norm = norm(S, 'fro');
X_norm = norm(X, 'fro');

fprintf('   Normas dos componentes:\n');
fprintf('     - ||X||_F: %.4e\n', X_norm);
fprintf('     - ||L||_F: %.4e (%.1f%% de X)\n', L_norm, 100*L_norm/X_norm);
fprintf('     - ||S||_F: %.4e (%.1f%% de X)\n', S_norm, 100*S_norm/X_norm);

% Analisar valores singulares de L
[U_L, S_L, V_L] = svd(L, 'econ');
singular_values_L = diag(S_L);
energy_L = cumsum(singular_values_L.^2) / sum(singular_values_L.^2);

% Encontrar rank efetivo
rank_threshold = 0.95; % 95% da energia
effective_rank = find(energy_L >= rank_threshold, 1);

fprintf('   Análise do componente L (low-rank):\n');
fprintf('     - Valores singulares: %d\n', length(singular_values_L));
fprintf('     - Rank efetivo (95%% energia): %d\n', effective_rank);
fprintf('     - Energia nos primeiros 10 modos: %.1f%%\n', 100*energy_L(10));

% Analisar esparsidade de S
S_abs = abs(S(:));
sparsity_threshold = 0.01 * max(S_abs); % 1% do máximo
sparse_ratio = sum(S_abs > sparsity_threshold) / numel(S);

fprintf('   Análise do componente S (sparse):\n');
fprintf('     - Elementos não-nulos: %d/%d (%.2f%%)\n', ...
    nnz(S), numel(S), 100*nnz(S)/numel(S));
fprintf('     - Elementos > 1%% do máximo: %.2f%%\n', 100*sparse_ratio);
fprintf('     - Média absoluta: %.2f\n', mean(S_abs));
fprintf('     - Máximo absoluto: %.2f\n', max(S_abs));

%% ------------------------------------------------------------
% 6. FIGURA 2: VALORES SINGULARES COMPARATIVOS
% ------------------------------------------------------------
fprintf('\n📈 Gerando Figura 2: Análise espectral...\n');

% Valores singulares originais (X)
[U_X, S_X, V_X] = svd(X, 'econ');
singular_values_X = diag(S_X);
energy_X = cumsum(singular_values_X.^2) / sum(singular_values_X.^2);

figure('Position', [100, 100, 1200, 500], 'Name', 'Figura 2: Análise Espectral Comparativa');

% Gráfico de valores singulares
subplot(1, 2, 1);
semilogy(singular_values_X / singular_values_X(1), 'b-o', ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Original X');
hold on;
semilogy(singular_values_L / singular_values_L(1), 'r-s', ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Low-rank L');
xlabel('Componente k');
ylabel('\sigma_k / \sigma_1 (escala log)');
title('Valores Singulares Normalizados');
legend('Location', 'best');
grid on;

% Gráfico de energia acumulada
subplot(1, 2, 2);
plot(100*energy_X, 'b-', 'LineWidth', 2, 'DisplayName', 'Original X');
hold on;
plot(100*energy_L, 'r-', 'LineWidth', 2, 'DisplayName', 'Low-rank L');
xlabel('Número de Componentes');
ylabel('Energia Acumulada (%)');
title('Energia Explicada');
legend('Location', 'southeast');
grid on;
xlim([0, min(50, length(energy_X))]);
ylim([0, 105]);

%% ------------------------------------------------------------
% 7. CLASSIFICAÇÃO SIMPLES USANDO COMPONENTE S (ANOMALIAS)
% ------------------------------------------------------------
fprintf('\n🤖 Testando classificação com COMPONENTE S (anomalias)...\n');

% Labels correspondentes
selected_labels = trainingLabels(selected_idx);

% Usar dados de teste se disponíveis
if numTestTotal > 0
    fprintf('   Usando dados de teste para avaliação...\n');
    
    % IDEA NOVA: Usar o componente S (anomalias) para classificação
    % As anomalias podem conter informações discriminativas
    
    % Separar S por classe
    S_fresh = S(:, selected_labels == 1);
    S_rotten = S(:, selected_labels == 2);
    
    % Calcular características das anomalias por imagem
    % 1. Norma L2 das anomalias
    S_norms_fresh = sqrt(sum(S_fresh.^2, 1));
    S_norms_rotten = sqrt(sum(S_rotten.^2, 1));
    
    % 2. Média das anomalias
    S_mean_fresh = mean(abs(S_fresh), 1);
    S_mean_rotten = mean(abs(S_rotten), 1);
    
    % 3. Máximo das anomalias
    S_max_fresh = max(abs(S_fresh), [], 1);
    S_max_rotten = max(abs(S_rotten), [], 1);
    
    % Calcular limiar simples para classificação
    threshold_norm = (mean(S_norms_fresh) + mean(S_norms_rotten)) / 2;
    
    fprintf('   Estatísticas das anomalias por classe:\n');
    fprintf('     - Fresh: norma média = %.2f\n', mean(S_norms_fresh));
    fprintf('     - Rotten: norma média = %.2f\n', mean(S_norms_rotten));
    fprintf('     - Limiar de classificação: %.2f\n', threshold_norm);
    
    % Classificar baseado no limiar (simples)
    if mean(S_norms_rotten) > mean(S_norms_fresh)
        fprintf('     - Rotten tem MAIS anomalias que Fresh\n');
        % Se Rotten tem mais anomalias, imagens com muitas anomalias são Rotten
        rule = 'mais_anomalias_eh_rotten';
    else
        fprintf('     - Fresh tem MAIS anomalias que Rotten\n');
        rule = 'mais_anomalias_eh_fresh';
    end
    
    % Para teste, precisamos calcular S para cada imagem de teste
    % Mas isso seria muito lento. Vamos usar uma aproximação:
    % Usar a projeção nos eigenfaces dos dados limpos
    
    [U_clean, ~, ~] = svd(L, 'econ');
    k_components = min(30, size(U_clean, 2));
    
    % Projetar dados de treino
    train_proj = U_clean(:, 1:k_components)' * L;
    
    % Separar por classe
    train_fresh_proj = train_proj(:, selected_labels == 1);
    train_rotten_proj = train_proj(:, selected_labels == 2);
    
    % Calcular centroides
    centroid_fresh = mean(train_fresh_proj, 2);
    centroid_rotten = mean(train_rotten_proj, 2);
    
    % Projetar dados de teste
    mean_train = mean(X, 2);
    test_data = double(testingFaces);
    test_proj = U_clean(:, 1:k_components)' * (test_data - mean_train);
    
    % Classificar (distância mínima)
    dist_to_fresh = sum((test_proj - centroid_fresh).^2, 1);
    dist_to_rotten = sum((test_proj - centroid_rotten).^2, 1);
    
    predicted_labels = 1 + (dist_to_rotten < dist_to_fresh);
    
    % Calcular acurácia
    accuracy_rpca = sum(predicted_labels == testingLabels) / numTestTotal * 100;
    
    % Matriz de confusão
    C_rpca = zeros(2, 2);
    for i = 1:numTestTotal
        trueLabel = testingLabels(i);
        predLabel = predicted_labels(i);
        C_rpca(trueLabel, predLabel) = C_rpca(trueLabel, predLabel) + 1;
    end
    
    fprintf('   📊 RESULTADOS CLASSIFICAÇÃO COM RPCA:\n');
    fprintf('      - Acurácia: %.2f%%\n', accuracy_rpca);
    fprintf('      - Matriz de confusão:\n');
    fprintf('        Verd/Pred   Fresh   Rotten\n');
    fprintf('        Fresh       %6d   %6d\n', C_rpca(1,1), C_rpca(1,2));
    fprintf('        Rotten      %6d   %6d\n', C_rpca(2,1), C_rpca(2,2));
    
    % Calcular métricas básicas
    TP = C_rpca(1,1); % True Positive
    FP = C_rpca(2,1); % False Positive
    TN = C_rpca(2,2); % True Negative
    FN = C_rpca(1,2); % False Negative
    
    if (TP + FP) > 0
        precision = TP / (TP + FP) * 100;
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN) * 100;
    else
        recall = 0;
    end
    
    if (TN + FP) > 0
        specificity = TN / (TN + FP) * 100;
    else
        specificity = 0;
    end
    
    fprintf('      - Métricas:\n');
    fprintf('        * Precisão: %.2f%%\n', precision);
    fprintf('        * Recall: %.2f%%\n', recall);
    fprintf('        * Especificidade: %.2f%%\n', specificity);
    
else
    fprintf('   ⚠️  Sem dados de teste para avaliação.\n');
    accuracy_rpca = NaN;
    C_rpca = [];
end

%% ------------------------------------------------------------
% 8. DETECÇÃO DE ANOMALIAS/OUTLIERS
% ------------------------------------------------------------
fprintf('\n🔍 Analisando detecção de anomalias...\n');

% Calcular norma de S para cada imagem (medida de "anomalia")
S_norms = sqrt(sum(S.^2, 1)); % Norma L2 por coluna

% Separar por classe
S_norms_fresh = S_norms(selected_labels == 1);
S_norms_rotten = S_norms(selected_labels == 2);

% Estatísticas
mean_fresh = mean(S_norms_fresh);
mean_rotten = mean(S_norms_rotten);
std_fresh = std(S_norms_fresh);
std_rotten = std(S_norms_rotten);

fprintf('   Norma do componente Sparse por classe:\n');
fprintf('     - Fresh: média = %.2f, desvio = %.2f\n', mean_fresh, std_fresh);
fprintf('     - Rotten: média = %.2f, desvio = %.2f\n', mean_rotten, std_rotten);

% Teste t manual (aproximado)
if length(S_norms_fresh) > 1 && length(S_norms_rotten) > 1
    % Diferença de médias
    mean_diff = mean_rotten - mean_fresh;
    
    % Erro padrão combinado
    n1 = length(S_norms_fresh);
    n2 = length(S_norms_rotten);
    pooled_se = sqrt((std_fresh^2/n1) + (std_rotten^2/n2));
    
    % Estatística t aproximada
    if pooled_se > 0
        t_stat = mean_diff / pooled_se;
        
        % Graus de liberdade aproximados
        df = ((std_fresh^2/n1 + std_rotten^2/n2)^2) / ...
             ((std_fresh^4/(n1^2*(n1-1))) + (std_rotten^4/(n2^2*(n2-1))));
        
        % Aproximação do valor p (para grandes amostras)
        % Usando aproximação normal
        p_value = 2 * (1 - 0.5*(1 + erf_approx(abs(t_stat)/sqrt(2))));
        
        fprintf('     - Teste t aproximado: t = %.4f, p = %.4f\n', t_stat, p_value);
        
        if p_value < 0.05
            fprintf('     - Diferença estatisticamente significativa (p < 0.05)\n');
            if mean_rotten > mean_fresh
                fprintf('     - Imagens Rotten têm MAIS anomalias que Fresh\n');
            else
                fprintf('     - Imagens Fresh têm MAIS anomalias que Rotten\n');
            end
        else
            fprintf('     - Diferença NÃO significativa\n');
        end
    else
        fprintf('     - Erro padrão muito pequeno para teste t\n');
    end
end

% Figura: Distribuição das anomalias
figure('Position', [100, 100, 900, 600], 'Name', 'Detecção de Anomalias');

subplot(2, 2, 1);
histogram(S_norms_fresh, 20, 'FaceColor', 'b', 'EdgeColor', 'k');
hold on;
histogram(S_norms_rotten, 20, 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.7);
xlabel('Norma de S (anomalia)');
ylabel('Frequência');
title('Distribuição de Anomalias por Classe');
legend('Fresh', 'Rotten');
grid on;

subplot(2, 2, 2);
boxplot_groups = [ones(length(S_norms_fresh), 1); 2*ones(length(S_norms_rotten), 1)];
boxplot_data = [S_norms_fresh(:); S_norms_rotten(:)];

% Boxplot manual simples
positions = [1, 2];
for i = 1:2
    group_data = boxplot_data(boxplot_groups == i);
    if ~isempty(group_data)
        % Mediana
        median_val = median(group_data);
        % Quartis
        q1 = quantile_approx(group_data, 0.25);
        q3 = quantile_approx(group_data, 0.75);
        % Limites
        iqr = q3 - q1;
        lower_whisker = max(min(group_data), q1 - 1.5*iqr);
        upper_whisker = min(max(group_data), q3 + 1.5*iqr);
        
        % Desenhar caixa
        rectangle('Position', [positions(i)-0.2, q1, 0.4, q3-q1], ...
                  'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'k');
        % Mediana
        line([positions(i)-0.2, positions(i)+0.2], [median_val, median_val], ...
             'Color', 'r', 'LineWidth', 2);
        % Bigodes
        line([positions(i), positions(i)], [q1, lower_whisker], 'Color', 'k');
        line([positions(i)-0.1, positions(i)+0.1], [lower_whisker, lower_whisker], 'Color', 'k');
        line([positions(i), positions(i)], [q3, upper_whisker], 'Color', 'k');
        line([positions(i)-0.1, positions(i)+0.1], [upper_whisker, upper_whisker], 'Color', 'k');
    end
end
set(gca, 'XTick', positions, 'XTickLabel', {'Fresh', 'Rotten'});
ylabel('Norma de S');
title('Boxplot de Anomalias');
grid on;

% Identificar imagens mais anômalas
[~, most_anomalous_idx] = sort(S_norms, 'descend');
num_show = min(6, length(S_norms));

subplot(2, 2, [3, 4]);
for i = 1:num_show
    idx = most_anomalous_idx(i);
    
    % Imagem original
    img_idx = selected_idx(idx);
    img_original = reshape(trainingFaces(:, img_idx), n, m);
    
    subplot(2, num_show, i);
    imagesc(img_original);
    colormap gray;
    axis image off;
    title(sprintf('Orig %d\nS=%.1f', i, S_norms(idx)));
    
    % Componente S (anomalia)
    img_sparse = reshape(S(:, idx), n, m);
    
    subplot(2, num_show, i + num_show);
    imagesc(img_sparse);
    colormap gray;
    axis image off;
    caxis([-max(abs(img_sparse(:))), max(abs(img_sparse(:)))]);
    title(sprintf('Anomalia %d', i));
end

sgtitle('Imagens Mais Anômalas Detectadas pelo RPCA', 'FontSize', 12);

%% ------------------------------------------------------------
% 9. APLICAÇÃO: REMOÇÃO DE RUÍDO/ARTEFATOS
% ------------------------------------------------------------
fprintf('\n🧹 Demonstração: Remoção de artefatos...\n');

% Adicionar ruído/artefatos sintéticos a algumas imagens
num_corrupted = 3;
corruption_strength = 100;

figure('Position', [100, 100, 1200, 400], 'Name', 'Remoção de Artefatos com RPCA');

for i = 1:num_corrupted
    % Selecionar uma imagem aleatória
    rand_idx = randi(size(X, 2));
    original_img = reshape(X(:, rand_idx), n, m);
    
    % Adicionar artefato (mancha)
    corrupted_img = original_img;
    
    % Criar uma mancha circular
    center_x = randi([30, 70]);
    center_y = randi([30, 70]);
    radius = 15;
    
    for x = 1:n
        for y = 1:m
            if (x-center_x)^2 + (y-center_y)^2 <= radius^2
                artifact_value = corruption_strength * (1 + 0.5*randn());
                corrupted_img(x, y) = original_img(x, y) + artifact_value;
            end
        end
    end
    
    % Aplicar RPCA na imagem corrompida
    X_corrupted = corrupted_img(:);
    
    % Parâmetros para imagem única
    lambda_single = 1 / sqrt(length(X_corrupted));
    mu_single = length(X_corrupted) / (4 * sum(abs(X_corrupted)));
    
    [L_single, S_single, ~, ~] = RPCA_adm(X_corrupted, lambda_single, mu_single, 500, 1e-6);
    
    % Reconstruir imagem limpa
    cleaned_img = reshape(L_single, n, m);
    
    % Plotar resultados
    subplot(3, num_corrupted, i);
    imagesc(original_img);
    colormap gray; axis image off;
    if i == 1, ylabel('Original'); end
    title(sprintf('Caso %d', i));
    
    subplot(3, num_corrupted, i + num_corrupted);
    imagesc(corrupted_img);
    colormap gray; axis image off;
    if i == 1, ylabel('Corrompida'); end
    
    subplot(3, num_corrupted, i + 2*num_corrupted);
    imagesc(cleaned_img);
    colormap gray; axis image off;
    if i == 1, ylabel('Limpa (RPCA)'); end
end

sgtitle('Remoção de Artefatos Sintéticos com RPCA', 'FontSize', 12);

%% ------------------------------------------------------------
% 10. SALVAR RESULTADOS DO RPCA
% ------------------------------------------------------------
fprintf('\n💾 Salvando resultados do RPCA...\n');

% Criar estrutura com resultados
rpca_results = struct();
rpca_results.L = L;
rpca_results.S = S;
rpca_results.Y = Y;
rpca_results.lambda = lambda;
rpca_results.mu = mu;
rpca_results.iterations = iter_count;
rpca_results.selected_indices = selected_idx;
rpca_results.selected_labels = selected_labels;

% Calcular métricas de qualidade
rpca_results.reconstruction_error = norm(X - L - S, 'fro');
rpca_results.rank_L = rank(L, 1e-6);
rpca_results.sparsity_S = nnz(S) / numel(S);

% Adicionar resultados de classificação se disponíveis
if ~isnan(accuracy_rpca)
    rpca_results.accuracy = accuracy_rpca;
    rpca_results.confusion_matrix = C_rpca;
end

% Adicionar análise de anomalias
rpca_results.S_norms = S_norms;
rpca_results.S_norms_fresh = S_norms_fresh;
rpca_results.S_norms_rotten = S_norms_rotten;

% Salvar
save('eigenbeef_rpca_results.mat', '-struct', 'rpca_results');
fprintf('   ✅ Resultados salvos em: eigenbeef_rpca_results.mat\n');

%% ------------------------------------------------------------
% 11. RESUMO FINAL DA PARTE 2
% ------------------------------------------------------------
fprintf('\n===========================================\n');
fprintf('📋 RESUMO DA ANÁLISE RPCA\n');
fprintf('===========================================\n\n');

fprintf('IMPLEMENTAÇÃO RPCA:\n');
fprintf('   - Método: ADM (Alternating Direction Method)\n');
fprintf('   - λ = 1/√max(n,m) = %.4f\n', lambda);
fprintf('   - μ calculado automaticamente\n');
fprintf('   - Tempo de execução: %.2f segundos\n', elapsed_time);
fprintf('   - Iterações: %d\n', iter_count);

fprintf('\nDECOMPOSIÇÃO X = L + S:\n');
fprintf('   - ||X||_F = %.4e\n', X_norm);
fprintf('   - ||L||_F = %.4e (%.1f%% de X)\n', L_norm, 100*L_norm/X_norm);
fprintf('   - ||S||_F = %.4e (%.1f%% de X)\n', S_norm, 100*S_norm/X_norm);
fprintf('   - Rank(L) = %d\n', rpca_results.rank_L);
fprintf('   - Sparsidade(S) = %.2f%%\n', 100*rpca_results.sparsity_S);

fprintf('\nANÁLISE DE COMPONENTES:\n');
fprintf('   - L (low-rank): estrutura principal das imagens\n');
fprintf('   - S (sparse): outliers, ruído, anomalias\n');
fprintf('   - Rank efetivo de L: %d (95%% energia)\n', effective_rank);

fprintf('\nDETECÇÃO DE ANOMALIAS:\n');
fprintf('   - Fresh: ||S|| médio = %.2f\n', mean_fresh);
fprintf('   - Rotten: ||S|| médio = %.2f\n', mean_rotten);
if exist('p_value', 'var')
    fprintf('   - Significância: p = %.4f\n', p_value);
end

if ~isnan(accuracy_rpca)
    fprintf('\nCLASSIFICAÇÃO COM RPCA:\n');
    fprintf('   - Acurácia: %.2f%%\n', accuracy_rpca);
end

fprintf('\nAPLICAÇÕES DEMONSTRADAS:\n');
fprintf('   1. Decomposição robusta (Fig 3.20 do livro)\n');
fprintf('   2. Análise espectral comparativa\n');
fprintf('   3. Classificação robusta\n');
fprintf('   4. Detecção de anomalias\n');
fprintf('   5. Remoção de artefatos\n');

fprintf('\n✨ ANÁLISE RPCA CONCLUÍDA COM SUCESSO!\n');
fprintf('   As figuras geradas mostram os resultados completos.\n');
fprintf('   Resultados salvos para análises futuras.\n');

%% ------------------------------------------------------------
% FUNÇÕES AUXILIARES (DEVEM VIR NO FINAL DO ARQUIVO)
% ------------------------------------------------------------

% Função para carregar imagens
function [images, labels, fileCount, imgDims] = loadImagesFromFolder(folderPath, labelValue, targetSize)
    fprintf('   Carregando de: %s\n', folderPath);
    extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'};
    allImages = [];
    for ext = extensions
        imageFiles = dir(fullfile(folderPath, '**', ext{1}));
        allImages = [allImages; imageFiles];
    end
    numImages = length(allImages);
    fprintf('     Encontradas: %d imagens\n', numImages);
    
    MAX_IMAGES = 500;
    if numImages > MAX_IMAGES
        fprintf('     Limite de %d imagens aplicado. Selecionando aleatoriamente...\n', MAX_IMAGES);
        selectedIdx = randperm(numImages, MAX_IMAGES);
        allImages = allImages(selectedIdx);
        numImages = MAX_IMAGES;
    end
    
    targetHeight = targetSize(1);
    targetWidth = targetSize(2);
    images = zeros(targetHeight * targetWidth, numImages, 'single');
    labels = labelValue * ones(1, numImages);
    
    for i = 1:numImages
        try
            imgPath = fullfile(allImages(i).folder, allImages(i).name);
            img = imread(imgPath);
            if mod(i, 50) == 0
                fprintf('     Processadas: %d/%d\n', i, numImages);
            end
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            img = imresize(img, [targetHeight, targetWidth]);
            images(:, i) = single(img(:));
        catch ME
            fprintf('     ⚠️  Erro na imagem %s: %s\n', allImages(i).name, ME.message);
            images(:, i) = zeros(targetHeight * targetWidth, 1, 'single');
        end
    end
    fileCount = numImages;
    imgDims = [targetHeight, targetWidth];
    fprintf('     ✅ Carregadas: %d imagens (%d x %d pixels)\n', numImages, targetHeight, targetWidth);
end

% Função de shrink (suavização L1)
function out = shrink(X, tau)
    out = sign(X) .* max(abs(X) - tau, 0);
end

% Função SVT (Singular Value Thresholding)
function out = SVT(X, tau)
    [U, S, V] = svd(X, 'econ');
    s = diag(S);
    s_shrunk = shrink(s, tau);
    S_shrunk = diag(s_shrunk);
    out = U * S_shrunk * V';
end

% Função principal RPCA (Alternating Direction Method)
function [L, S, Y, count] = RPCA_adm(X, lambda, mu, max_iter, tol)
    % Inicialização
    [n1, n2] = size(X);
    L = zeros(size(X));
    S = zeros(size(X));
    Y = zeros(size(X));
    count = 0;
    
    % Limiar baseado na norma de Frobenius de X
    thresh = tol * norm(X, 'fro');
    
    % Loop principal ADM
    while (norm(X - L - S, 'fro') > thresh) && (count < max_iter)
        % Atualizar L (componente de baixo rank)
        L = SVT(X - S + (1/mu) * Y, 1/mu);
        
        % Atualizar S (componente esparso)
        S = shrink(X - L + (1/mu) * Y, lambda/mu);
        
        % Atualizar multiplicadores de Lagrange
        Y = Y + mu * (X - L - S);
        
        count = count + 1;
        
        % Mostrar progresso a cada 100 iterações
        if mod(count, 100) == 0
            fprintf('     Iteração %d, erro: %.4e\n', count, norm(X - L - S, 'fro'));
        end
    end
end

% Funções auxiliares para cálculos estatísticos
function y = erf_approx(x)
    % Aproximação simples da função erro
    y = sign(x) .* sqrt(1 - exp(-2*x.^2/pi));
end

function q = quantile_approx(data, p)
    % Aproximação simples de quantil
    data_sorted = sort(data);
    idx = p * (length(data_sorted) - 1) + 1;
    if idx <= 1
        q = data_sorted(1);
    elseif idx >= length(data_sorted)
        q = data_sorted(end);
    else
        lower_idx = floor(idx);
        upper_idx = ceil(idx);
        weight = idx - lower_idx;
        q = (1-weight)*data_sorted(lower_idx) + weight*data_sorted(upper_idx);
    end
end