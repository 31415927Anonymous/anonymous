function [directions] = proess_directions(data)
delta = 0.5;
pts = data(:,1:3);
norms = data(:,4:6);
ptsn = pts*50+50*ones(size(pts));
W = zeros(100,100,100);
[N,~] = size(data);
for i = 1:N
    x = ptsn(i,1);
    x = min(98,max(x,2));
    y = ptsn(i,2);
    y = min(98,max(y,2));
    z = ptsn(i,3);
    z = min(98,max(z,2));
    W(floor(x-delta):ceil(x+delta),floor(y-delta):ceil(y+delta),floor(z-delta):ceil(z+delta))=...
        W(floor(x-delta):ceil(x+delta),floor(y-delta):ceil(y+delta),floor(z-delta):ceil(z+delta))+...
        ones(size(W(floor(x-delta):ceil(x+delta),floor(y-delta):ceil(y+delta),floor(z-delta):ceil(z+delta))));
end
W = W>0;
[x,y,z]=ind2sub(size(W),find(W));
%figure;scatter3(x,y,z);axis([0 100 0 100 0 100])

%% fast marching
W = W+0.01*ones(size(W));
start_points = [1;1;1];
options.nb_iter_max = Inf;
[D,~] = perform_fast_marching_3d(W, start_points, options);
color = D(sub2ind(size(D),x,y,z));
%figure;scatter3(x,y,z,4,color);axis([0 100 0 100 0 100])
F1 = scatteredInterpolant(x,y,z,color) ;
GD = F1(ptsn(:,1),ptsn(:,2),ptsn(:,3));
%figure;scatter3(ptsn(:,1),ptsn(:,2),ptsn(:,3),5,GD);axis([0 100 0 100 0 100])

%% calculate directions
[N,~] = size(ptsn);
KNN = 20;
neighborIds = knnsearch(ptsn,ptsn,'K', KNN);
pt_d = zeros(N,3);
pt_d2 = zeros(N,3);
for i = 1:N
    tind =  neighborIds(i,2:end)';
    tD = GD(tind)-GD(i);
    tpt = ptsn(tind,:)-repmat(ptsn(i,:),KNN-1,1);
    td = ((tpt'*tpt+10^(-2)*eye(3))\(tpt'*tD))';
    td = td-(norms(i,:)*td')*norms(i,:);
    pt_d(i,:) = td/norm(td,2);
    pt_d2(i,:) = cross(norms(i,:),pt_d(i,:));
end
directions = [pt_d,pt_d2];
%figure;quiver3(ptsn(:,1)',ptsn(:,2)',ptsn(:,3)',pt_d(:,1)'*0.1,pt_d(:,2)'*0.1,pt_d(:,3)'*0.1,1.5,'r');axis([0 100 0 100 0 100])



end

