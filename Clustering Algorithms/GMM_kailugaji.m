function [label_new, iter_GMM, para_miu, para_sigma, NegativeLogLikelihood, fitness]=GMM_kailugaji(X, K, label_old)
% Input:
% K: number of cluster
% X: dataset, N*D
% label_old: initializing label. N*1
% Output:
% label_new: results of cluster. N*1
% iter_GMM: iterations
% Written by kailugaji. (wangrongrong1996@126.com)
format long 
%% initialization parameters
esp=1e-6;  % stopping criterion for iteration
max_iter=1000;    % maximum number of iterations 
beta=1e-4;  % a regularization coefficient of covariance matrix
fitness=zeros(max_iter,1);
[X_num, X_dim]=size(X);
para_sigma=zeros(X_dim, X_dim, K); % the covariance matrix
para_sigma_inv=zeros(X_dim, X_dim, K); % sigma^(-1)
para_miu=zeros(K, X_dim); % the mean
para_pi=zeros(1, K); % the mixing proportion
log_N_pdf=zeros(X_num, K);  % log pdf
%% initializing the mixing proportion, the mean and the covariance matrix
for k=1:K
    X_k=X(label_old==k, :); 
    para_pi(k)=size(X_k, 1)/X_num;  
    para_miu(k, :)=mean(X_k);  
    sample_cov=cov(X_k)+beta*eye(X_dim);
    para_sigma_inv(:, :, k)=inv(sample_cov);  %sigma^(-1)
end
%% Expectation maximization (EM) algorithm
for t=1:max_iter
    %% E-step
    for k=1:K
        % pdf of each cluster 
        X_miu=X-repmat(para_miu(k,:), X_num, 1);  % X-miu. X_num*X_dim
        exp_up=sum((X_miu*para_sigma_inv(:, :, k)).*X_miu, 2);  % (X-miu)'*sigma^(-1)*(X-miu)
        log_N_pdf(:,k)=log(para_pi(k))-0.5*X_dim*log(2*pi)+0.5*log(abs(det(para_sigma_inv(:, :, k))))-0.5*exp_up; % N*1
    end
    T = logsumexp(log_N_pdf,2);
    responsivity = exp(bsxfun(@minus,log_N_pdf,T)); % posterior probability
    responsivity(isnan(responsivity)==1) = 1;
    %% M-step
    R_k=sum(responsivity, 1);  % 1*K
    % update miu
    para_miu=(responsivity'*X)./repmat(R_k', 1, X_dim);
    % update sigma
    for k=1:K
        X_miu=X-repmat(para_miu(k, :), X_num, 1); % N*D
        temp_X_miu_r=X_miu.*repmat(sqrt(responsivity(:, k)), 1, X_dim); % N*D
        para_sigma(:, :, k)=(temp_X_miu_r'*temp_X_miu_r)/R_k(k);
        para_sigma(:, :, k)=para_sigma(:, :, k)+beta*eye(X_dim);
        para_sigma_inv(:, :, k)=inv(para_sigma(:, :, k));  % sigma^(-1)
    end
    % update pi
    para_pi=R_k/sum(R_k);
    %% Negative logLikelihood function
%     fitness(t)=-sum(sum(log_N_pdf));
    fitness(t)=sum(sum(responsivity.*log_N_pdf));
    %% stopping criterion for iteration
    if t>1 
        if abs(fitness(t)-fitness(t-1))<esp
            break;
        end
    end
end
iter_GMM=t;  % iterations
NegativeLogLikelihood=fitness(iter_GMM);
%% clustering
[~, label_new]=max(responsivity, [], 2);
