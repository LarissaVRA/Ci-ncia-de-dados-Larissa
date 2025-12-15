% Carregar e processar a imagem
% Se não tiver a imagem 'teste.jpg', use uma imagem de exemplo
try
    A = imread('teste.jpg');
    Al= double(rgb2gray(A));
catch
    % Usar uma imagem de exemplo se 'teste.jpg' não existir
%    A = imread('cameraman.tif');
    % Converter para RGB se for escala de cinza
    if size(A,3) == 1
        A = cat(3, A, A, A);
    end
end

B = double(rgb2gray(A)); % Convert RBG->gray, 256 bit->double

%% Wavelet decomposition (2 level)
n = 2; 
w = 'db1'; 
[C,S] = wavedec2(B,n,w);

% LEVEL 1
k = 1; % Definir k para o nível 1
A1 = appcoef2(C,S,w,1); % Approximation
[H1, V1, D1] = detcoef2('all',C,S,1); % Details - corrigido 'a' para 'all'

A1 = wcodemat(A1,128);
H1 = wcodemat(H1,128);
V1 = wcodemat(V1,128);
D1 = wcodemat(D1,128);

% LEVEL 2
k = 2; % Definir k para o nível 2
A2 = appcoef2(C,S,w,2); % Approximation - corrigido para nível 2
[H2, V2, D2] = detcoef2('all',C,S,2); % Details - corrigido 'a' para 'all'

A2 = wcodemat(A2,128);
H2 = wcodemat(H2,128);
V2 = wcodemat(V2,128);
D2 = wcodemat(D2,128);

% Montar a decomposição
dec2 = [A2 H2; V2 D2];
dec1 = [imresize(dec2,size(H1)) H1 ; V1 D1];

% Exibir o resultado
figure;
image(dec1);
colormap(gray(128));
title('Decomposição Wavelet - 2 Níveis');
axis off;

% Exibir também a imagem original para comparação
figure;
imagesc(Al);
colormap(gray);
axis off;
title('Imagem Original');