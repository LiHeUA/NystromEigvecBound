function demo_TrainingSizeSelection
% To verity Eq. (15) in [1].
%
% [1] Li He, Nilanjan Ray and Hong Zhang. Error Bound of
% Nystrom-approximated NCut Eigenvectors and Its Application to Training
% Size Selection. To appear in Neurocomputing.
%
% Introduction:
% Eq. (15) in [1] gives the upper bound on training set size if a tolerance
% of error l-e is given.
%
% In this test, given a user designed e, we follow Eq. (15) in [1] to
% estimate the training size c. Then, we verity the correctness of Eq. (15)
% whether or not the average sum of squared singular values from that c is
% larger than e.
%
% Eq. (15) works well if (a) sum(s^2)>e (correctness) and (b) c<n
% (usefulness).
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

% user designed e list, e\in [0,l]
eList = [.3:.1:1]*l;

% used in Eq. (15)
denominator = lambda_l*( sum(1./D.^2) - norm(nK,'fro')^2/n );

% we assume \Delta=1
Delta = 1;

% Suggested training size c from Eq. (15) in [1]
suggestedC = zeros(length(eList),1);

% Sum of squared singular values with training size suggestedC
RS = zeros(length(eList),1);

%% 1. loop on different tolerance
for t=1:length(eList)
    
    e = eList(t); % user desigend tolerance
    
    %% 2. Training Size Upper Bound from Eq. (15) [1]
    if lambda_l*l-e>0
        suggestedC(t) = n/( Delta^2*(lambda_l*l-e)/denominator + 1 );
    else
        suggestedC(t) = n;
    end
    if suggestedC(t)>n
        suggestedC(t) = n;
    end
    if suggestedC(t)<0
        suggestedC(t) = 0;
    end
    
    %% 3. Real \Sum_{i=1}^l { \Sigma_i^2 } of Matrix V'^TV with Training size from Eq. (15)
    RS(t) = getRealSumSigma(n,suggestedC(t),K,l,V);
end

figure(3);
subplot(2,1,1);plot(eList,eList,'b*-');hold on;
plot(eList,RS,'r+-');
legend('Required lower bound','Real sum of \sigma^2');
grid on;
xlabel('Required e');
ylabel('sum of \sigma^2')
title(['Training Size Selection Test' 10 'Eq. (15) is correct if the blue line is lower than the red and is useful if c<n']);
xlim([.5 3.5]);

subplot(2,1,2);bar(eList,suggestedC);grid on
xlabel('Required e');
ylabel('Suggested c');
title('Suggested Training Size');
xlim([.5 3.5]);
ylim([0 n]);

function RS = getRealSumSigma(n,c,K,l,V)
% Real sum of squared singular values.
% We sample a subset as training set, and calculate Nystrom approximated
% eigenvectors V' on that training set. Then, calculate the singular values
% s_i, i=1:l of the matrix V'^TV, where V is the ground truth eigenvectors.
% RS = sum(s_i^2). We do this many times and report the average RS.

% training size
c = round(c);
if c>n
    c = n;
end
if c<0
    c = 0;
end

% max iterations
maxIter = 100;

%% 1. Test
% save sum of squared singular values in each trial
sumSingularVal = zeros(maxIter, 1);

% apprimated eigenvectors
tildeV = zeros(n,l);

% loop in max iterations
for iter = 1:maxIter
    %% ramdomly pick up data points as training set
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
    
    [vec_train,lambda] = eig(nK_train);
    [lambda, idx] = sort(diag(lambda),'descend');
    vec_train = vec_train(:,idx);
    vec_train = vec_train(:,1:l);
    
    %% Nystrom approximation
    for i=1:l
        % approximated eigenvector
        tildeV(:,i) = sqrt(c/n)*diag(1./D_test.^.5) * K_test_train * diag(1./D_train.^.5) * vec_train(:,i) / lambda(i);
    end
    
    %% sign alignment. 
    % Since v or -v are both eigenvectors. So, ask the approximated
    % eigenvectors tildeV of the same direction as the ground truth
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

RS = sum( sumSingularVal ) / maxIter;
