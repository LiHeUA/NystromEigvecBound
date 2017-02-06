function demo_Theorem1
% To verity Theorem 1 in [1].
%
% [1] Li He, Nilanjan Ray and Hong Zhang. Error Bound of
% Nystrom-approximated NCut Eigenvectors and Its Application to Training
% Size Selection. To appear in Neurocomputing.
%
% Introduction:
% Theorem 1 in [1] gives the expected error of matrix multiplication
% approximation if sampling uniformly without replacement.
%
% In this test, we generate two matrices, A of p-by-n and B of n-by-q, and
% randomly sample c columns (or rows) from A (or B) and scale the sampled
% cols (rows) to form a couple new matrices, C of p-by-c and R of c-by-q. 
% We calculate the error by:
%
%               error = ||AB-CR||_F^2
%
% We have in total n!/(n-c)! possible permutations of c columns (rows)
% chosen from n columns (rows). We calculate in this test the average error
% among all permutations. It should be identical to Eq. (8) in [1].
%
% Variables:
%       A           p*n         One matrix, we want to approximate AB
%       B           n*q         One matrix, we want to approximate AB
%       c           scalar      Number of chosen columns (rows). We will
%                               sample at random without replacement c
%                               columns (rows) from A (B). See Theorem 1
%                               [1] for details.
%       C           p*c         A matrix formed by scaled chosen columns
%                               from A.
%       R           c*q         A matrix formed by scaled chosen rows from
%                               B.
%
% Source codes available at
% https://github.com/LiHeUA/
%
% Li He: lhe2@ualberta.ca

clc
close all

%% 0. Generate Two Matrices
% WARNING! Number of columns, n, should be small. We will loop on n!/(n-k)!
% permutations with varying k=1:n.
n = 7;  % number of columns

p = 20; % number of rows in A
q = 30; % number of cols in B

% generate two matrices A and B
A = rand(p,n);
B = rand(n,q);

EAB = zeros(1,n); % expected error on c=1:n
AE = zeros(1,n); % average on real errors with c=1:n

% we will randomly choose c columns (rows) from total n;
for c=1:n
    %% 1. Expected Error from Theorem 1 in [1]
    % Eq. (8) in [1]
    EAB(c) = getExpectedErrorAB(n,c,A,B);
    
    %% 2. Average on Real Error
    AE(c) = getAvgErrorAB(n,c,A,B);
end

%% 3. Display
figure(1); subplot(1,2,1); hold on; grid on
plot(1:n,AE,'r+-');title('Average on Real Errors');
xlabel('Number of chosen cols (rows)');
ylabel('Error');
figure(1); subplot(1,2,2); hold on; grid on
plot(1:n,EAB,'b+-');title('Expected Errors');
xlabel('Number of chosen cols (rows)');
ylabel('Error');

function EAB = getExpectedErrorAB(n,c,A,B)
% Calculate the expected error of AB-CR. Eq. (8) in [1]

%% 1. \sum_t=1^n { ||A^{(t)}B_{(t)}||_F^2 }
SAB = 0;
for i=1:n
    SAB = SAB + norm(A(:,i)*B(i,:),'fro')^2;
end
%% 2. Eq. (8)
EAB = n*(n-c)/(c*(n-1)) * ( SAB - norm(A*B,'fro')^2/n );

function AE = getAvgErrorAB(n,c,A,B)
% Calculate the average error of all n!/(n-c)! permutations. First get all
% combinations, then on each combination, get all its permutations.

%% 1. All Combinations
comb = nchoosek(1:n,c); % unordered combinations

%% 2. All Permutations
sumE = 0; % sum of error
scale = sqrt(n/c); % scale factor
AB = A*B; % ground truth matrix multiplication value
for i=1:size(comb,1)
    ordPerm = perms(comb(i,:)); % ordered permutations from one combination
    
    for j=1:size(ordPerm)
        p = ordPerm(j,:); % one permutation
        % get C and R
        C = A(:,p);
        R = B(p,:);
        
        % scale C and R
        C = C*scale;
        R = R*scale;
        
        % get error
        sumE = sumE + norm( AB - C*R, 'fro' )^2;
    end
end

%% 3. Average Error
AE = sumE*factorial(n-c)/factorial(n);