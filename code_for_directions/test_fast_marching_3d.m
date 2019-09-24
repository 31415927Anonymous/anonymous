% test for fast marching
%
%   Copyright (c) 2004 Gabriel Peyr?
%addpath('geodesics');


n = 100;

% gaussian weight (path will avoid center of the cube)
% x = -1:2/(n-1):1;
% [X,Y,Z] = meshgrid(x,x,x);
% sigma = 0.4;
% W = 1./(1 + exp( -(X.^2+Y.^2+Z.^2)/sigma^2 ) );




%k = 5;
%start_points = [n-k;k;k];
%end_points = [k;n-k;n-k];
%start_points = [round(ptsn(1,1));round(ptsn(1,2));round(ptsn(1,3))];
start_points = [1;1;1];
end_points = [86;54;49];

options.nb_iter_max = Inf;

tic
[D,S] = perform_fast_marching_3d(W, start_points, options);
toc
path = compute_geodesic(D,end_points);

% draw the path
figure;
plot_fast_marching_3d(D,S,path,start_points,end_points);