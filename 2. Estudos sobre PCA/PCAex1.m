% EX1 noisy cloud of data - PCAex1

% Generate noisy cloud of data
xC = [2, 1]; % Center of data (mean)
sig = [2, .5]; % Principal axes

theta = pi/3; % Rotate cloud by pi/3
R = [cos(theta) sin(theta); % Rotation matrix
-sin(theta) cos(theta)];

nPoints = 10000; % Create 10,000 points
X = randn(nPoints,2)*diag(sig)*R + ones(nPoints,2)*diag(xC);
scatter(X(:,1),X(:,2),'k.','LineWidth',2) % Plot data

% Compute PCA and plot confidence intervals
Xavg = mean(X,1); % Compute mean
B = X - ones(nPoints,1)*Xavg; % Mean-subtracted Data
[U,S,V] = svd(B/sqrt(nPoints),'econ'); % PCA via SVD

theta = (0:.01:1)*2*pi;
Xstd = [cos(theta') sin(theta')]*S*V'; % 1std conf. interval
hold on, plot(Xavg(1)+Xstd(:,1),Xavg(2) + Xstd(:,2),'r-')
plot(Xavg(1)+2*Xstd(:,1),Xavg(2) + 2*Xstd(:,2),'r-')
plot(Xavg(1)+3*Xstd(:,1),Xavg(2) + 3*Xstd(:,2),'r-')

%plot grafico pg 52 fig1.14
