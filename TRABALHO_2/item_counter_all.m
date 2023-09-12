item_counter_hsv
item_counter_test_images_9
item_counter_test_images_7
item_counter_test_images_13
item_counter_test_images_12
item_counter_test_images_25

close all

% Load the MAT-file
loadedStructure = load('parameters_advanced.mat');

% Save the structure as a JSON file
jsonStr = savejson('', loadedStructure);
fid = fopen('parameters_advanced.txt', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);