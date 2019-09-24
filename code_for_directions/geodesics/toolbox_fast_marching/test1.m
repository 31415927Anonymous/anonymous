path(path, 'toolbox/');
path(path, 'data/');

% load the whole volume
load ../toolbox_fast_marching_data/brain1-crop-256.mat
% crop to retain only the central part
n = 100;
M = rescale( crop(M,n) );
% display some horizontal slices
slices = round(linspace(10,n-10,6));
Mlist = mat2cell( M(:,:,slices), n, n, ones(6,1));
clf; imageplot(Mlist);

delta = 5;
start_point = [91;15;delta];
W = abs(M-M(start_point(1),start_point(2),start_point(3)));
W = rescale(-W,.001,1);
% perform the front propagation
options.nb_iter_max = Inf;
[D,S] = perform_fast_marching(W, start_point, options);
% display the distance map
D1 = rescale(D); D1(D>.7) = 0;
clf; imageplot(D1,options);
alphamap('rampup');
colormap jet(256);