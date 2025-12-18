%% Experiment 2: Classical vs Modified Gram-Schmidt

% Passo 1: Criar matriz com valores singulares exponencialmente decrescentes
m = 80;
[U, ~] = qr(randn(m));
[V, ~] = qr(randn(m));
S = diag(2.^(-1:-1:-m));
A = U * S * V;

% Passo 2: Implementar Classical Gram-Schmidt (CGS)
function [Q, R] = clgs(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    for j = 1:n
        v = A(:, j);
        for i = 1:j-1
            R(i, j) = Q(:, i)' * A(:, j);
            v = v - R(i, j) * Q(:, i);
        end
        R(j, j) = norm(v);
        Q(:, j) = v / R(j, j);
    end
end

% Passo 3: Implementar Modified Gram-Schmidt (MGS)
function [Q, R] = mgs(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    V = A;
    for i = 1:n
        R(i, i) = norm(V(:, i));
        Q(:, i) = V(:, i) / R(i, i);
        for j = i+1:n
            R(i, j) = Q(:, i)' * V(:, j);
            V(:, j) = V(:, j) - R(i, j) * Q(:, i);
        end
    end
end

% Passo 4: Calcular as fatorações
[QC, RC] = clgs(A);
[QM, RM] = mgs(A);

% Passo 5: Plotar os r_jj (diagonais de R)
figure;
semilogy(1:m, diag(RC), 'bo', 'DisplayName', 'CGS');
hold on;
semilogy(1:m, diag(RM), 'rx', 'DisplayName', 'MGS');
hold off;
xlabel('j');
ylabel('r_{jj}');
title('Valores de r_{jj} para CGS (bolas) e MGS (cruz)');
legend;
grid on;