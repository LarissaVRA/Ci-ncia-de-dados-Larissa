%Exp. Cancer Mama
clear; clc; close all

% Ajustes de figura
set (0, "defaultfigureposition", [100 100 1600 800]);
set (0, "defaultaxesfontsize", 12);

% Carregar os dados
arquivo=fullfile("breast-cancer.csv");

% ---- 1) Lê o CSV inteiro como texto
fid = fopen(arquivo, "r");
raw = textscan(fid, "%s", "Delimiter", "\n");
fclose(fid);
raw = raw{1};

% ---- 2) Quebra cada linha em células (primeira coluna = grupo, resto = números)
n = numel(raw);
grp = cell(n,1);
obs = [];

idx=1;
for i = 2:n
    parts = strsplit(raw{i}, ",");  % separa por vírgula
    tipo = strtrim(parts{2});     % primeira coluna (Cancer/Comum)
     % verifica a última letra: C = Cancer, N = Normal
    if strcmp(tipo,"M")
        grp{idx} = "Maligno";
    elseif strcmp(tipo,"B")
        grp{idx} = "Benigno";
    else
        grp{i} = "Desconhecido";     % fallback, caso não tenha C/N
    end
    nums = str2double(parts(3:end));% resto vira números
    obs = [obs; nums];              % adiciona na matriz
    idx=idx+1;
end
grp(n,:)=[];
%------Padronização dos dados de obs (observação)
Media=mean(obs,1); %média de cada coluna
Variancia=std(obs,0,1); %desvio padrão em cada coluna
obs_padrao=(obs-Media)./Variancia;

% SVD
[U, S, V] = svd(obs_padrao, "econ"); %decompõe os genes em SVD
S = diag(S);   % transformar matriz S em vetor

% ---- FIGURA 1 ----
figure;
hold on;
for j = 1:size(obs,1)
    x = -V(:,1)' * obs_padrao(j,:)';   % Projeção no PC1
    y = -V(:,2)' * obs_padrao(j,:)';   % Projeção no PC2
%plotar com -V ou V? fazendo com -V, fica idêntico ao artigo.
    if strcmp(grp{j}, "Maligno")
        plot(x, y, "xr", "MarkerSize", 10, "LineWidth", 2);
    else
        plot(x, y, "+b", "MarkerSize", 10, "LineWidth", 2);
    end
end
%Adiciona pontos falsos somente para gerar a legenda
h1 = plot(NaN, NaN, "xr", "MarkerSize", 10, "LineWidth", 2);
h2 = plot(NaN, NaN, "+b", "MarkerSize", 10, "LineWidth", 2);

hold off;
grid on;
xlabel("PC1"); ylabel("PC2");

% Legenda
legend([h1, h2], {"Câncer de Mama Maligno", "Câncer de Mama Benigno"}, "location", "northeastoutside");

% ---- FIGURA 2 ----
figure;

% --- (a) Valores Singulares ---
subplot(1,2,1)
semilogy(S, 'k', 'LineWidth', 0.8) % apenas a curva em preto
xlabel('r')
ylabel('Singular value, \sigma_r')
title('(a) Valores Singulares')
grid on

% --- (b) Soma Cumulativa ---
subplot(1,2,2)
plot(cumsum(S)/sum(S), 'k', 'LineWidth', 0.8) % apenas a curva em preto
xlabel('r')
ylabel('Cumulative sum')
title('(b) Soma Cumulativa')
grid on