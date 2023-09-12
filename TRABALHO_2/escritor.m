loadedStructure = load('parameters_advanced.mat');
fid = fopen('parameters_advanced.txt', 'w');

structFields = fieldnames(loadedStructure);
