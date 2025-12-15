% Carregar e processar a imagem
try
    A = imread('teste.jpg');
catch
    % Usar uma imagem de exemplo se 'teste.jpg' não existir
    A = imread('cameraman.tif');
    % Converter para RGB se for escala de cinza
    if size(A,3) == 1
        A = cat(3, A, A, A);
    end
end

B = double(rgb2gray(A)); % Convert RBG->gray, 256 bit->double

% Wavelet decomposition (4 level)
[C,S] = wavedec2(B,4,'db1'); % Corrigido as aspas

Csort = sort(abs(C(:))); % Sort by magnitude

for keep = [.05 .01 .005 .001]
    thresh = Csort(floor((1-keep)*length(Csort)));
    ind = abs(C)>thresh;
    Cfilt = C.*ind; % Threshold small indices
    % Plot Reconstruction
    Arecon = uint8(waverec2(Cfilt,S,'db1')); % Corrigido as aspas
    figure, imagesc(uint8(Arecon))
    colormap(gray);
    title(['Reconstrução com ', num2str(keep*100), '% dos coeficientes']);
    axis off;
end