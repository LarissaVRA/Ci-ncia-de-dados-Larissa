%% =========================
%  SCRIPT PRINCIPAL
% =========================
clear
clc
% Teste com matrizes pequenas
testar_matriz_pequena();

% Teste com matrizes grandes
[resultados, tamanhos] = testar_matrizes_grandes();

% Plota resultados
plotar_resultados(resultados, tamanhos);

%% =========================
%  FUNÇÕES
% =========================

function A = gerar_matriz_simetrica_positiva_definida(n)
    % Gera uma matriz simétrica e positiva definida de ordem n
    A = rand(n,n);
    A = A' * A + n * eye(n);
end

function A = gerar_matriz_tridiagonal(n)
    % Gera uma matriz tridiagonal simétrica e positiva definida
    main_diag = 2 * ones(n,1);
    off_diag  = -1 * ones(n-1,1);

    A = diag(main_diag) + diag(off_diag,1) + diag(off_diag,-1);
end

function [S,T] = decomposicao_ST(A)
    % Decomposição ST: A = S * T
    % onde S é triangular inferior e T triangular superior

    try
        % Tenta Cholesky
        L = chol(A,'lower');
        S = L;
        T = L';
    catch
        % Fallback para decomposição conjugada (LDL^T)
        [P,D] = decomposicao_conjugada(A);
        S = P * sqrt(diag(D));
        T = S';
    end
end

function [T,S] = decomposicao_TS(A)
    % Decomposição TS: A = T * S
    [S_aux,T_aux] = decomposicao_ST(A);
    T = T_aux;
    S = S_aux;
end

function [P,D] = decomposicao_conjugada(A)
    % Decomposição Conjugada: A = P * D * P'
    % P triangular inferior com diagonal unitária
    % D vetor diagonal

    [n,m] = size(A);
    if n ~= m
        error('A matriz deve ser quadrada para decomposição conjugada');
    end

    P = zeros(n);
    D = zeros(n,1);

    % Inicializa diagonal de P
    for i = 1:n
        P(i,i) = 1;
    end

    for j = 1:n
        soma = 0;
        for k = 1:j-1
            soma = soma + P(j,k)^2 * D(k);
        end
        D(j) = A(j,j) - soma;

        for i = j+1:n
            soma = 0;
            for k = 1:j-1
                soma = soma + P(i,k) * P(j,k) * D(k);
            end
            if D(j) ~= 0
                P(i,j) = (A(i,j) - soma) / D(j);
            else
                P(i,j) = 0;
            end
        end
    end
end

function [erro,tempo,fatores] = verificar_decomposicao(A, decomp_func, nome)
    % Verifica precisão e tempo de uma decomposição

    try
        tic;
        fatores = decomp_func(A);
        tempo = toc;

        if strcmp(nome,'ST')
            S = fatores{1};
            T = fatores{2};
            A_rec = S * T;
        elseif strcmp(nome,'TS')
            T = fatores{1};
            S = fatores{2};
            A_rec = T * S;
        else
            P = fatores{1};
            D = fatores{2};
            A_rec = P * diag(D) * P';
        end

        erro = norm(A - A_rec) / norm(A);

    catch
        erro = inf;
        tempo = inf;
        fatores = [];
    end
end

function testar_matriz_pequena()
    disp(repmat('=',1,60))
    disp('TESTES COM MATRIZES PEQUENAS')
    disp(repmat('=',1,60))

    tamanhos = [3,4];

    for n = tamanhos
        fprintf('\n%s\n', repmat('=',1,50))
        fprintf('MATRIZ %dx%d\n', n, n)
        fprintf('%s\n', repmat('=',1,50))

        A = gerar_matriz_simetrica_positiva_definida(n);
        disp('Matriz A:')
        disp(A)

        decomposicoes = {
            'ST', @(X) num2cell(decomposicao_ST(X));
            'TS', @(X) num2cell(decomposicao_TS(X));
            'Conjugada', @(X) num2cell(decomposicao_conjugada(X))
        };

        for k = 1:size(decomposicoes,1)
            nome = decomposicoes{k,1};
            func = decomposicoes{k,2};

            fprintf('\n--- Decomposição %s ---\n', nome)
            [erro,tempo,fatores] = verificar_decomposicao(A, func, nome);

            if erro < inf
                fprintf('Erro relativo: %.2e\n', erro)
                fprintf('Tempo: %.6f segundos\n', tempo)

                disp('Fatores:')
                for i = 1:length(fatores)
                    fprintf('Fator %d:\n', i)
                    disp(fatores{i})
                end
            else
                disp('Falha na decomposição')
            end
        end
    end
end

function [resultados,tamanhos] = testar_matrizes_grandes()
    disp(' ')
    disp(repmat('=',1,60))
    disp('TESTES COM MATRIZES GRANDES')
    disp(repmat('=',1,60))

    tamanhos = [50,100,200];
    nomes = {'ST','TS','Conjugada'};

    for i = 1:length(nomes)
        resultados.(nomes{i}).erros  = [];
        resultados.(nomes{i}).tempos = [];
    end

    for n = tamanhos
        fprintf('\nMATRIZ %dx%d\n', n, n)

        A = gerar_matriz_tridiagonal(n);

        decomposicoes = {
            'ST', @(X) num2cell(decomposicao_ST(X));
            'TS', @(X) num2cell(decomposicao_TS(X));
            'Conjugada', @(X) num2cell(decomposicao_conjugada(X))
        };

        for k = 1:size(decomposicoes,1)
            nome = decomposicoes{k,1};
            func = decomposicoes{k,2};

            [erro,tempo,~] = verificar_decomposicao(A, func, nome);

            resultados.(nome).erros(end+1)  = erro;
            resultados.(nome).tempos(end+1) = tempo;

            fprintf('%s -> erro: %.2e | tempo: %.4f s\n', nome, erro, tempo)
        end
    end
end

function plotar_resultados(resultados, tamanhos)
    figure;

    nomes = fieldnames(resultados);

    cores = {'b','r','k'};          % azul, vermelho, preto
    marcadores = {'o','s','^'};     % círculo, quadrado, triângulo

    % -------- Gráfico de erro --------
    subplot(1,2,1)
    hold on
    for i = 1:length(nomes)
        semilogy(tamanhos, resultados.(nomes{i}).erros, ...
            'Color', cores{i}, ...
            'Marker', marcadores{i}, ...
            'LineWidth', 1.8, ...
            'MarkerSize', 8);
    end
    xlabel('Tamanho da Matriz (n)')
    ylabel('Erro Relativo')
    title('Precisão das Decomposições')
    legend(nomes, 'Location','best')
    grid on

    % -------- Gráfico de tempo --------
    subplot(1,2,2)
    hold on
    for i = 1:length(nomes)
        plot(tamanhos, resultados.(nomes{i}).tempos, ...
            'Color', cores{i}, ...
            'Marker', marcadores{i}, ...
            'LineWidth', 1.8, ...
            'MarkerSize', 8);
    end
    xlabel('Tamanho da Matriz (n)')
    ylabel('Tempo de Execução (s)')
    title('Tempo das Decomposições')
    legend(nomes, 'Location','best')
    grid on
end
