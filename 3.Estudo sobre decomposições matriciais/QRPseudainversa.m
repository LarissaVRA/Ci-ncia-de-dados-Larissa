%% Pseudo-inversa usando decomposição QR
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

%% Analisar a matriz
posto_completo = analisar_matriz(A);

if ~posto_completo
    disp('⚠️  Aviso: Matriz não tem posto completo');
    disp('O algoritmo QR pode não funcionar corretamente');
end

%% Calcular pseudo-inversa usando QR
try
    disp(' ');
    disp('Calculando pseudo-inversa usando QR...');
    
    A_plus = pseudo_inversa_qr(A);
    
    disp(' ');
    disp('PSEUDO-INVERSA A⁺:');
    fprintf('Dimensões: %dx%d\n', size(A_plus,1), size(A_plus,2));
    disp(A_plus);
    
    % Verificar propriedades
    verificar_propriedades(A, A_plus);
    
    % Comparar com pinv do MATLAB
    comparar_com_matlab(A, A_plus);
    
    A_plus_QR = A_plus;

catch ME
    disp('Erro no método QR:');
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

function X = pseudo_inversa_qr(A)
% Calcula a pseudo-inversa usando decomposição QR

[m, n] = size(A);

% Verifica posto completo
if rank(A) < n
    error('A matriz não tem posto completo (rank < n). O método QR requer rank(A) = n.');
end

% Passo 1: Decomposição QR fina A = Q*R
[Q, R] = qr(A, 0);   % QR econômico

% Verificar ortogonalidade de Q
if norm(Q' * Q - eye(n)) > 1e-10
    warning('Q pode não ser perfeitamente ortogonal');
end

% Passo 2: A⁺ = R^{-1} Q^T
X = R \ Q';
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

function comparar_com_matlab(A, A_plus_qr)
% Compara com a pseudo-inversa do MATLAB (pinv)

disp(' ');
disp('COMPARAÇÃO COM PINV DO MATLAB:');
disp('================================================');

A_plus_matlab = pinv(A);
diferenca = norm(A_plus_qr - A_plus_matlab);

fprintf('Diferença entre os métodos: %.2e\n', diferenca);
fprintf('Dimensões: QR[%dx%d] vs pinv[%dx%d]\n', ...
    size(A_plus_qr,1), size(A_plus_qr,2), ...
    size(A_plus_matlab,1), size(A_plus_matlab,2));

if diferenca < 1e-10
    disp('✅ Os métodos produzem resultados equivalentes');
else
    disp('⚠️  Diferença significativa entre os métodos');
    disp('Submatriz 3x3 da diferença:');
    disp(A_plus_qr(1:3,1:3) - A_plus_matlab(1:3,1:3));
end
end

% -------------------------------------------------------

function posto_completo = analisar_matriz(A)
% Análise da matriz

[m, n] = size(A);
rk = rank(A);

if m == n
    cond_num = cond(A);
else
    cond_num = cond(A' * A);
end

disp(' ');
disp('ANÁLISE DA MATRIZ:');
disp('================================================');
fprintf('Dimensões: %dx%d\n', m, n);
fprintf('Posto (rank): %d\n', rk);
fprintf('Número de condição: %.2e\n', cond_num);
fprintf('Posto completo? %d\n', rk == n);

posto_completo = (rk == n);
end
