load('mt/adjust.mat');

% index of cellList cell
i = 1;
l = length(cellList);
for i=1:l,
    z = cellList{1, i};
    z(cellfun('isempty', z)) = [];
    cellList{1, i} = z;
    cl = length(z);
    cellListN(i) = cl;
end

save('mt/adjust-post.mat');

