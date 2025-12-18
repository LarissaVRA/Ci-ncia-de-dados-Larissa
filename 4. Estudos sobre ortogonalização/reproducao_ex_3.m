%% Experiment 3: Loss of Orthogonality - Saída Detalhada
clear; clc; close all;
fprintf('=== EXPERIMENTO 3: PERDA DE ORTOGONALIDADE ===\n\n');

% Matriz do exemplo (quase colinear) - mesma do livro
A = [0.70000 0.70711; 0.70001 0.70711];
fprintf('Matriz A (mal condicionada, quase colinear):\n');
disp(A);
fprintf('Condicionamento estimado: cond(A) ≈ %.2e\n\n', cond(A));

% 1. QR via Householder (MATLAB built-in, estável)
[Q_house, R_house] = qr(A);
orth_error_house = norm(Q_house' * Q_house - eye(2), 'fro');
fprintf('--- Householder (qr do MATLAB) ---\n');
fprintf('Q =\n');
disp(Q_house);
fprintf('R =\n');
disp(R_house);
fprintf('Q^T * Q - I =\n');
disp(Q_house' * Q_house - eye(2));
fprintf('Erro de ortogonalidade ||Q^TQ - I||_F = %e\n\n', orth_error_house);

% 2. QR via Modified Gram-Schmidt (nossa implementação)
[Q_mgs, R_mgs] = modifiedGramSchmidt(A);
orth_error_mgs = norm(Q_mgs' * Q_mgs - eye(2), 'fro');
fprintf('--- Modified Gram-Schmidt (MGS) ---\n');
fprintf('Q =\n');
disp(Q_mgs);
fprintf('R =\n');
disp(R_mgs);
fprintf('Q^T * Q - I =\n');
disp(Q_mgs' * Q_mgs - eye(2));
fprintf('Erro de ortogonalidade ||Q^TQ - I||_F = %e\n\n', orth_error_mgs);

% 3. Comparação e interpretação
fprintf('=== INTERPRETAÇÃO DOS RESULTADOS ===\n');
fprintf('Precisão da máquina (eps) ≈ %.2e\n', eps);
fprintf('\n');
fprintf('1. Householder: erro = %.2e ≈ %.0f × eps\n', ...
    orth_error_house, orth_error_house/eps);
fprintf('   → Ortogonalidade preservada na precisão da máquina.\n');
fprintf('\n');
fprintf('2. MGS: erro = %.2e ≈ %.0f × eps\n', ...
    orth_error_mgs, orth_error_mgs/eps);
fprintf('   → Perda de ~%d dígitos de ortogonalidade.\n', ...
    round(log10(orth_error_mgs/orth_error_house)));
fprintf('\n');
fprintf('3. A matriz A é quase deficiente em posto (colunas quase paralelas).\n');
fprintf('   Isso amplifica erros de arredondamento no Gram-Schmidt.\n');
fprintf('\n');
fprintf('4. Conclusão: Mesmo MGS pode perder ortogonalidade significativa\n');
fprintf('   em problemas mal condicionados. Householder é preferível.\n');

% --- Implementação do Modified Gram-Schmidt (MGS) ---
function [Q, R] = modifiedGramSchmidt(A)
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