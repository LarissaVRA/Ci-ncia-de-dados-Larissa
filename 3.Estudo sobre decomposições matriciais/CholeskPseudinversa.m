%% Para esta primeira parte, iremos utilizar uma matriz de Pascal
% de dimensão m x n, aplicar o método de Cholesky para obter a
% pseudo-inversa e realizar algumas verificações.
clear
clc
%% Definir m e n
m = 15;
n = 10;

%% Gerar a matriz de Pascal
A = matriz_pascal(m, n);

disp('Matriz A:');
disp(A);

%% Calcular a pseudo-inversa usando Cholesky
try
    disp(' ');
    disp('Calculando pseudo-inversa usando Cholesky...');
    
    A_plus = pseudo_inversa_cholesky(A);
    
    disp('Pseudo-inversa A⁺:');
    disp(A_plus);
    
    % Verificar propriedades da pseudo-inversa
    verificar_propriedades(A, A_plus);
    
    % Comparar com pinv do MATLAB
    comparar_com_matlab(A, A_plus);
    
    A_plus_cholesky = A_plus;

catch ME
    disp('Erro na decomposição de Cholesky.');
    disp(ME.message);
    disp('A matriz A^T*A pode não ser positiva definida.');
end

%% ================= FUNÇÕES =================

function A = matriz_pascal(m, n)
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

function X = pseudo_inversa_cholesky(A)
% Calcula a pseudo-inversa de A usando decomposição de Cholesky

% Passo 1: M = A^T * A
M = A' * A;

% Passo 2: Fatoração de Cholesky: M = R' * R
[R, p] = chol(M);

if p ~= 0
    error('A matriz M = A^T*A não é positiva definida.');
end

% Passo 3: Resolver R' * R * X = A'
% Primeiro: R' * Y = A'
Y = R' \ A';

% Segundo: R * X = Y
X = R \ Y;
end

% -------------------------------------------------------

function verificar_propriedades(A, A_plus)
% Verifica as propriedades da pseudo-inversa de Moore-Penrose

disp(' ');
disp('Verificação das propriedades da pseudo-inversa:');
disp('================================================');

erro1 = norm(A * A_plus * A - A);
fprintf('1. ||A * A⁺ * A - A|| = %.2e\n', erro1);

erro2 = norm(A_plus * A * A_plus - A_plus);
fprintf('2. ||A⁺ * A * A⁺ - A⁺|| = %.2e\n', erro2);

erro3 = norm((A * A_plus)' - A * A_plus);
fprintf('3. ||(A * A⁺)^T - A * A⁺|| = %.2e\n', erro3);

erro4 = norm((A_plus * A)' - A_plus * A);
fprintf('4. ||(A⁺ * A)^T - A⁺ * A|| = %.2e\n', erro4);

tolerancia = 1e-10;
todas_ok = erro1 < tolerancia && erro2 < tolerancia && ...
           erro3 < tolerancia && erro4 < tolerancia;

fprintf('\nTodas as propriedades satisfeitas: %d\n', todas_ok);
end

% -------------------------------------------------------

function comparar_com_matlab(A, A_plus_chol)
% Compara com a função pinv do MATLAB

A_plus_matlab = pinv(A);
diferenca = norm(A_plus_chol - A_plus_matlab);

disp(' ');
disp('Comparação com pinv do MATLAB:');
disp('================================================');
fprintf('Diferença entre os métodos: %.2e\n', diferenca);

if diferenca < 1e-10
    disp('✅ Os métodos produzem resultados equivalentes');
else
    disp('⚠️  Os métodos produzem resultados diferentes');
end
end
