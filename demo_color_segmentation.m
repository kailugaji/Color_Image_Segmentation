% Color Image Segmentation
% Written by kailugaji. (wangrongrong1996@126.com)
clear
clc
addpath(genpath('.'));
choose_algorithm=1; % choose clustering methods, 1: Fuzzy c-means clustering (FCM), 2: Fuzzy subspace clustering (FSC), 3: Maximum entropy clustering (MEC), 4: Gaussian mixture model (GMM)
K=2; % The num of cluster
init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
filename='.\Image data\1';
image_name=strcat(filename,'.jpg');
%label_name=strcat(filename,'.txt');
I=imread(image_name);
figure(1)
imshow(I);
title('Row Image');
% saveas(gcf,sprintf('Row image.jpg'),'bmp'); 
[mm, n, d]=size(I);
data=reshape(I, mm*n ,d);
%% Initialization
data=double(data);
label_old=init_methods(data, K, init);
%% Repeat the experiment repeat_num times
t0=cputime;
if choose_algorithm==1
    m=2; % fuzzy index
    [label,iter_num]=FCM_kailugaji(data, K, label_old, m);
elseif choose_algorithm==2
    tao=2; % an weighted index
    sigm=1e-5; % a weighted regularization parameter
    [label,iter_num]=FSC_kailugaji(data, K, label_old, tao, sigm);
elseif choose_algorithm==3
    gama=50; % a regularization parameter
    [label,iter_num]=MEC_kailugaji(data, K, label_old, gama);
elseif choose_algorithm==4
    [label, iter_num]=GMM_kailugaji(data, K, label_old);
end
run_time=cputime-t0;
label_update=reshape(label,mm, n);
img=label2rgb(label_update);
figure(2)
imshow(uint8(img));
title('FCM Result');
saveas(gcf,sprintf('Result_FCM.jpg'),'bmp'); 
rmpath(genpath('.'));
