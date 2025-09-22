%Ovarian Cancer Data

% Ler os dados
obs = readmatrix("ovariancancer_obs.csv");
grp = readcell("ovariancancer_grp.csv");

% Decomposição SVD
[U,S,V] = svd(obs,'econ');
s=diag(S)
cumS = cumsum(s)/sum(s); % Soma cumulativa normalizada


% Calcular coordenadas nos três primeiros componentes
coords = obs * V(:,1:3);

% Separar grupos
idxCancer = strcmp(grp, 'Cancer');
idxNormal = ~idxCancer;

% Plotar em 3D
figure;
scatter3(coords(idxCancer,1), coords(idxCancer,2), coords(idxCancer,3), ...
    60, 'r', 'x', 'LineWidth', 2);
hold on;
scatter3(coords(idxNormal,1), coords(idxNormal,2), coords(idxNormal,3), ...
    60, 'b', 'o', 'LineWidth', 2);

% Aparência
grid on;
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend({'Cancer', 'Normal'});
title('Projeção 3D - Ovarian Cancer Dataset');
view(80,20);    % Ângulo de visão inicial
%rotate3d on;    % Permite rotacionar com o mouse

figure;

% --- (a) Valores Singulares ---
subplot(1,2,1)
semilogy(s, 'k', 'LineWidth', 0.8) % apenas a curva em preto
xlabel('r')
ylabel('Singular value, \sigma_r')
title('(a) Valores Singulares')
grid on

% --- (b) Soma Cumulativa ---
subplot(1,2,2)
plot(cumS, 'k', 'LineWidth', 0.8) % apenas a curva em preto
xlabel('r')
ylabel('Cumulative sum')
title('(b) Soma Cumulativa')
grid on