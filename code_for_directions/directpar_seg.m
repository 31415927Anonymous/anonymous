%for i = 1:698
warning('off');
addpath(genpath('geodesics'));

catalist = {'02691156','02773838','02954340','02958343','03001627','03261776','03467517','03624134','03636649','03642806','03790512','03797390','03948459','04099429','04225987','04379243'};
rawloc = 'data_seg/';


for j = 1:16      
    tic
    namelist = dir([rawloc catalist{j} '/*.txt']);
    txtname = {namelist.name};
    onlyname = strrep(txtname,'.txt','');
    [~,N] = size(onlyname);
    %N = 1;                        %for test 
    na = catalist{j};            
    mkdir('directions_seg/',na);
    vectors = cell(1,N);
    parfor i = 1:N      
        data = load([rawloc na '/' txtname{i}]);
        data = data(:,1:6);
        data = prezero(data);
        [directions] = proess_directions(data);
        vectors{i} = directions;
    end
    toc
    
    for i = 1:N
         vec = vectors{i};
         save(['directions_seg/' na '/' onlyname{i} '.mat'  ],'vec','-v6');
    end
end