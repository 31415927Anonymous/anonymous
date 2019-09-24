%for i = 1:698
warning('off');
addpath(genpath('geodesics'));
load totalnumber40.mat
load name40.mat
for j = 1:40
    tic
    N = total_num(j);
    %N = 1;    % for test
    na = name{j};
    mkdir('directions_class/',na);
    vectors = cell(1,N);
    parfor i = 1:N             
        name_num = num2str(i,'%04d');
        data = load(['data_class/' na '/' na '_' name_num '.txt']);
        data = prezero(data);
        [directions] = proess_directions(data);
        vectors{i} = directions;
    end
    toc
    for i = 1:N
         vec = vectors{i};
         name_num = num2str(i,'%04d');
         save(['directions_class/' na '/' na '_' name_num '.mat'  ],'vec','-v6');
    end
end