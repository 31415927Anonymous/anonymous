function [pdata] = prezero(data)
[N,~] = size(data);
bias = sum(data(:,1:3))/N;
pdata = data;
pdata(:,1:3) = pdata(:,1:3)-repmat(bias(:,1:3),N,1);
maxp = max(pdata(:,1).^2+pdata(:,2).^2+pdata(:,3).^2);
maxp = sqrt(maxp);
tscale = 1/maxp;
pdata(:,1:3) = pdata(:,1:3)*tscale;







end