%% Pseudo-inversa usando decomposição SVD
% Utilizando a mesma matriz de Pascal da questão anterior (m x n)
clear 
clc
%% Dimensões
m = 15;
n = 10;

%% Gerar matriz de Pascal
A = gerar_matriz_pascal(m, n);

disp('Matriz A:');
disp(A);

%% Calcular pseudo-inversa usando SVD
try
    A_plus = pseudo_inversa_svd(A);
    A_plus_svd = A_plus;
    
    disp(' ');
    disp('PSEUDO-INVERSA A⁺:');
    fprintf('Dimensões: %dx%d\n', size(A_plus,1), size(A_plus,2));
    disp(A_plus);
    
    % Verificar propriedades
    verificar_propriedades(A, A_plus);
    
    % Comparar com pinv do MATLAB
    comparar_com_matlab(A, A_plus);

catch ME
    disp('Erro na decomposição SVD:');
    disp(ME.message);
end

%% ================= FUNÇÕES =================

function A = gerar_matriz_pascal(m, n)
% Gera uma matriz de Pascal de dimensão m x n
% A_ij = C(i+j-2, j-1)

A = zeros(m, n);

for i = 1:m
    for j = 1:n
        A(i, j) = nchoosek(i + j - 2, j - 1);
    end
end
end

% -------------------------------------------------------

function X = pseudo_inversa_svd(A)
% Calcula a pseudo-inversa usando decomposição SVD

[m, n] = size(A);

% Verifica posto da matriz
rank_A = rank(A);
if rank_A < n
    fprintf('Aviso: Matriz com rank %d < %d. Σ pode ser singular, mas o algoritmo continuará.\n', ...
        rank_A, n);
end

% Passo 1: Decomposição SVD fina A = U * Σ * V'
[U, S, V] = svd(A, 'econ');

% Verificar ortogonalidade
ortogonalidade_U = norm(U' * U - eye(n)) < 1e-10;
ortogonalidade_V = norm(V' * V - eye(n)) < 1e-10;

fprintf('U^T U ≈ I: %d\n', ortogonalidade_U);
fprintf('V^T V ≈ I: %d\n', ortogonalidade_V);

% Passo 2: Calcular A⁺ = V * Σ^{-1} * U'
singulares = diag(S);

% Inversa dos valores singulares
S_inv = diag(1 ./ singulares);

X = V * S_inv * U';
end

% -------------------------------------------------------

function verificar_propriedades(A, A_plus)
% Verifica as propriedades da pseudo-inversa de Moore-Penrose

disp(' ');
disp('VERIFICAÇÃO DAS PROPRIEDADES:');
disp('================================================');

propriedades = {
    'A * A⁺ * A ≈ A', A * A_plus * A, A;
    'A⁺ * A * A⁺ ≈ A⁺', A_plus * A * A_plus, A_plus;
    '(A * A⁺)^T = A * A⁺', (A * A_plus)', A * A_plus;
    '(A⁺ * A)^T = A⁺ * A', (A_plus * A)', A_plus * A
};

tolerancia = 1e-10;
todas_ok = true;

for i = 1:size(propriedades, 1)
    desc = propriedades{i, 1};
    mat1 = propriedades{i, 2};
    mat2 = propriedades{i, 3};
    
    erro = norm(mat1 - mat2);
    
    if erro < tolerancia
        status = '✅';
    else
        status = '❌';
        todas_ok = false;
    end
    
    fprintf('%d. %s %s\n', i, desc, status);
    fprintf('   Erro: %.2e\n', erro);
    fprintf('   Dimensões: %dx%d → %dx%d\n', ...
        size(mat1,1), size(mat1,2), size(mat2,1), size(mat2,2));
end

fprintf('\nTodas as propriedades satisfeitas: %d\n', todas_ok);
end

% -------------------------------------------------------

function comparar_com_matlab(A, A_plus_svd)
% Compara com a pseudo-inversa do MATLAB (pinv)

disp(' ');
disp('COMPARAÇÃO COM PINV DO MATLAB:');
disp('================================================');

A_plus_matlab = pinv(A);
diferenca = norm(A_plus_svd - A_plus_matlab);

fprintf('Diferença entre os métodos: %.2e\n', diferenca);
fprintf('Dimensões: SVD[%dx%d] vs pinv[%dx%d]\n', ...
    size(A_plus_svd,1), size(A_plus_svd,2), ...
    size(A_plus_matlab,1), size(A_plus_matlab,2));

if diferenca < 1e-12
    disp('✅ Os métodos produzem resultados equivalentes');
else
    disp('⚠️  Diferença significativa entre os métodos');
    disp('Submatriz 3x3 da diferença:');
    disp(A_plus_svd(1:3,1:3) - A_plus_matlab(1:3,1:3));
end
end
