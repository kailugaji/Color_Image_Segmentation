% Demo Iris.data
% Written by kailugaji. (wangrongrong1996@126.com)
clear
clc
%% Setting the hyper-parameters
choose_norm=2; % Normalization methods, 0: no normalization, 1: z-score, 2: max-min
init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
repeat_num=10; % Repeat the experiment repeat_num times
addpath(genpath('.'));
%% Load data
data_load=dlmread('.\iris.data');
data=data_load(:, 1:end-1);
real_label=data_load(:, end);
K=length(unique(real_label)); % number of cluster
[N, ~]=size(data);
label_old=zeros(N, repeat_num);
s_1=0; 
%% Initialization & Normalization
data = normlization(data, choose_norm);
for i=1:repeat_num
    label_old(:, i)=init_methods(data, K, init);
end
%% Repeat the experiment repeat_num times
t0=cputime;
for i=1:repeat_num
    [label_new, iter_GMM]=GMM_kailugaji(data, K, label_old(:, i));
    iter_GMM_t(i)=iter_GMM;
    %% performanc indices
     [accuracy(i), RI(i), NMI(i)]=performance_index(real_label,label_new);
     s_1=s_1+iter_GMM_t(i);
     fprintf('Iteration %2d, the number of iterations: %2d, Accuary: %.8f\n', i, iter_GMM_t(i), accuracy(i));
end
run_time=cputime-t0;
%% Calculating evaluation indexes
repeat_num=length(find(accuracy~=0));
ave_iter_GMM=s_1/repeat_num; 
ave_run_time=run_time/repeat_num;
ave_acc_GMM=mean(accuracy); max_acc_GMM=max(accuracy); min_acc_GMM=min(accuracy);std_acc_GMM=std(accuracy);
ave_RI_GMM=mean(RI); max_RI_GMM=max(RI); min_RI_GMM=min(RI);std_RI_GMM=std(RI);
ave_NMI_GMM=mean(NMI); max_NMI_GMM=max(NMI); min_NMI_GMM=min(NMI);std_NMI_GMM=std(NMI);
fprintf('The average iteration number of the algorithm is: %.2f\nThe average running time is: %.5f\nThe average accuracy is: %.8f\nThe average rand index is: %.8f\nThe average normalized mutual information is: %.8f\n', ave_iter_GMM, ave_run_time, ave_acc_GMM, ave_RI_GMM, ave_NMI_GMM);
ACC=[ave_acc_GMM; std_acc_GMM; max_acc_GMM; min_acc_GMM];
ARI=[ave_RI_GMM; std_RI_GMM; max_RI_GMM; min_RI_GMM];
ANMI=[ave_NMI_GMM; std_NMI_GMM; max_NMI_GMM; min_NMI_GMM];
performance_indices=[ACC; ARI; ANMI; ave_iter_GMM; ave_run_time];
% performance_indices: 
% ave_acc_GMM
% std_acc_GMM
% max_acc_GMM
% min_acc_GMM
% ave_RI_GMM
% std_RI_GMM
% max_RI_GMM
% min_RI_GMM
% ave_NMI_GMM
% std_NMI_GMM
% max_NMI_GMM
% min_NMI_GMM
% ave_iter_GMM
% ave_run_time
save GMM_results performance_indices
rmpath(genpath('.'));
