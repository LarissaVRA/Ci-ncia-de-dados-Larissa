clear
clc
close all
format long g;
A=imread('colagem35.jpg');
X=double(rgb2gray(A)); % Convert RBG->gray, 256 bit->double.
nx = size(X,1); ny = size(X,2);
% Take the SVD
[U,S,V] = svd(X);
% Approximate matrix with truncated SVD for various ranks r
r=[10 20 40 80 160 320 640]
subplot(2,4,1)
imagesc(X), axis off, colormap gray
title('Original')
for i=1:length(r) % Truncation value
Xapprox = U(:,1:r(i))*S(1:r(i),1:r(i))*V(:,1:r(i))'; % Approx. image
subplot(2,4,i+1)
imagesc(Xapprox), axis off, colormap(gray)
title(['r=',num2str(r(i),'%d')]);
erro(i)=norm(X-Xapprox, 'fro');
end
Erros=[transpose(r),transpose(erro)]

% Valores singulares
sigma = diag(S);
cumSigma = cumsum(sigma)/sum(sigma);

% R's de destaque (mude aqui os valores que quiser)
r_marks = [10 20 40 80 160 320 640];

figure

% --- (a) Singular values ---
subplot(1,2,1)
semilogy(sigma,'k') % curva em preto
hold on
semilogy(r_marks, sigma(r_marks), 'ro', 'MarkerFaceColor','r') % pontos vermelhos
hold off
xlabel('r')
ylabel('Singular value, \sigma_r')
title('(a)')
grid on

% --- (b) Cumulative sum ---
subplot(1,2,2)
plot(cumSigma,'k') % curva em preto
hold on
plot(r_marks, cumSigma(r_marks), 'ro','MarkerFaceColor','r') % pontos vermelhos
for k = 1:length(r_marks)
    text(r_marks(k)+20, cumSigma(r_marks(k)), ...
        ['r = ' num2str(r_marks(k))], ...
        'Color','r','FontSize',12)
end
hold off
xlabel('r')
ylabel('Cumulative sum')
title('(b)')
grid on
