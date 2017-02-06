function demo_Theorem2
% To verity Theorem 2 in [1].
%
% [1] Li He, Nilanjan Ray and Hong Zhang. Error Bound of
% Nystrom-approximated NCut Eigenvectors and Its Application to Training
% Size Selection. To appear in Neurocomputing.
%
% Introduction:
% Theorem 2 in [1] gives the lower bound on Normalized Cuts Nystrom
% approximated eigenvectors.
%
% In this test, we generate the similarity matrix K, degree matrix D of
% data. We want to approximate the leading l eigenvectors V of this matrix:
%
%                   D^{-1/2}KD^{-1/2}
%
% We employ Nystrom to get V', the approximate eigenvectors to V.
% We then build matrix V'^TV, and calculate the singular values of V'^TV,
% titled s_i, i=1,...,l. Then, Theorem 2 should be satisfied.
%
% Source codes available at
% https://github.com/LiHeUA/
%
% Li He: lhe2@ualberta.ca

clc
close all

%% 0. Load Data and Build K, D, V
load Iris.mat; % Iris dataset, http://archive.ics.uci.edu/ml/datasets/Iris

% Number of data points
n = size(data,1);

% Classes number, l=3 in Iris
l = 3;

% Gaussian kernel parameter \tau
tau = .5;

% Euclidean distance
dis = pdist2(data,data);

% Similarity matrix K
K = exp(-dis.^2/2/tau^2);

% Degree matrix D
D = sum(K);

% D^{-1/2}
invD2 = sqrt(1./D);

% Normalized similarity matrix D^{-1/2}KD^{-1/2}
nK = diag(invD2)*K*diag(invD2);

% Ground truth eigenvectors V
[V,lambda] = eig(nK);
[lambda,idx] = sort(diag(lambda),'descend');
V = V(:,idx);
V = V(:,1:l);

% the l-th eigenvalue of nK
lambda_l = lambda(l);

% training size list
cList = round( [.1:.1:.9]*n );

% Expected sum of squared singular values and real values
ES = zeros(length(cList),1);
RS = zeros(length(cList),1);

%% 1. loop on different training size
for t=1:length(cList)
    
    c = cList(t); % training size
    
    %% 2. Expected \Sum_{i=1}^l { \Sigma_i^2 } from Theorem 2
    ES(t) = getExpectedSumSigma(n,c,K,lambda_l, l);
    
    %% 3. Real \Sum_{i=1}^l { \Sigma_i^2 } of Matrix V'^TV
    RS(t) = getRealSumSigma(n,c,K,l,V);
end

figure(2);
plot(cList,ES,'b*-');hold on;
plot(cList,RS,'r+-');
legend('Lower bound','Real sum of \sigma^2');
grid on;
xlabel('Traingin size c');
ylabel('sum of \sigma^2')
title(['Sum of \sigma^2 Test' 10 'Theorem 2 is correct if the blue line is lower than the red line']);

function ES = getExpectedSumSigma(n, c, K, lambda_l, l)
% Calculate the expected sum of squared singular values. Eq. (11) in [1]

% Degree matrix D
D = sum(K);

% D^{-1/2}
invD2 = sqrt(1./D);

% Normalized similarity matrix D^{-1/2}KD^{-1/2}
nK = diag(invD2)*K*diag(invD2);

% We assume \Delta=1, as described in [1]
Delta = 1;

% Eq. (11) in [1]
ES = lambda_l*l - lambda_l*( n*(n-c)/c/(n-1) * ( sum(1./D.^2)-norm(nK,'fro')^2/n ) )/Delta^2;
if ES<0
    ES = 0;
end


function RS = getRealSumSigma(n,c,K,l,V)
% Real sum of squared singular values.
% We sample a subset as training set, and calculate Nystrom approximated
% eigenvectors V' on that training set. Then, calculate the singular values
% s_i, i=1:l of matrix V'^TV, where V is the ground truth eigenvectors. 
% Then, RS = sum(s_i^2). We do this many times and return the average RS.

% max iterations
maxIter = 100;

%% 1. Test
% save sum of squared singular values in each trial
sumSingularVal = zeros(maxIter, 1);

% approximated eigenvectors
tildeV = zeros(n,l);

% loop in max iterations
for iter = 1:maxIter
    %% ramdomly pick up data points as training set, uniformly without replacement
    idx = randperm(n);
    idxTrain = idx(1:c);
    
    %% K_train, D_train, K_test_train, D_test
    K_train = K(idxTrain,idxTrain);
    D_train = sum(K_train);
    K_test_train = K(:,idxTrain);
    D_test = sum(K_test_train,2);
    
    % D_{train}^{-1/2} and D_{train}^{-1/2}*K_{train}*D_{train}^{-1/2}
    invD2_train = diag(1./sqrt(D_train));
    nK_train = invD2_train*K_train*invD2_train;
    
    % eigensystem of training set
    [vec_train,lambda] = eig(nK_train);
    [lambda, idx] = sort(diag(lambda),'descend');
    vec_train = vec_train(:,idx);
    vec_train = vec_train(:,1:l);
    
    %% Nystrom Approximation
    for i=1:l
        % approximated eigenvector
        tildeV(:,i) = sqrt(c/n)*diag(1./D_test.^.5) * K_test_train * diag(1./D_train.^.5) * vec_train(:,i) / lambda(i);
    end
    
    %% Sign Alignment. 
    % Since v or -v are both eigenvectors. So, require the approximated
    % eigenvectors tildeV pointing at the same direction as the ground 
    % truth.
    for i=1:l
        if dot( V(:,i), tildeV(:,i) ) < 0
            tildeV(:,i) = -tildeV(:,i);
        end
    end
    
    %% V'^TV and Its Singular Values
    A = tildeV'*V(:,1:l);
    [~, s, ~] = svd(A);
    
    sumSingularVal(iter) = sum(diag(s).^2);
end

% average value
RS = sum( sumSingularVal ) / maxIter;
