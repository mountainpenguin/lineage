load('mt/alignment.mat');
save('mt/alignment-backup.mat', 'shiftframes');

shiftframes.x = -shiftframes.x;
shiftframes.y = -shiftframes.y;

save('mt/reverse.mat', 'shiftframes');

shiftframes.x = zeros(1, length(shiftframes.x));
shiftframes.y = zeros(1, length(shiftframes.y));

save('mt/alignment.mat', 'shiftframes');
