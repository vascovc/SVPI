clear all
close all
clc

nn = 97746;

dificuldade = 1; % 1-hardest; 2-mid; 3 - easier

 lista = dir("Imagens de Referência/frame/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Imagens de Referência/noframe/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Seq39x/imagens/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("../svpi2023_TP2_img_*"+dificuldade+"_*.png");

num_files = size(lista,1);
matrix = zeros(num_files,18);
for i=1:num_files
    file = lista(i).name;
    num_seq = str2double(file(18:20));
    num_img = str2double(file(22:23));
    image = im2double(imread([lista(i).folder,'\',lista(i).name]));
    %imshow(image)

    obj_border = 0;
    obj_ok = 0;
    obj_frame = 0;
    bio = 0;
    cor = 0;
    elec = 0;
    explo = 0;
    info = 0;
    laser = 0;
    lit = 0;
    phone = 0;
    smoking = 0;
    tox = 0;
    wc = 0;
    wifi = 0;
    
    Z_hsv = rgb2hsv(image);
    Z = Z_hsv(:,:,3);
    Z = imadjust(Z);
    Z = autobin(Z);
    %Z = imbinarize(Z);
    figure
    imshow(Z)
%     %
%     imshow(Z)
%     figure
%     
%     %Z = bwmorph(Z,'close');
%     %Z = bwmorph(Z,'open');
%     imshow(Z)

    num_items = 21;
    mask = bwareaopen(Z, 100);
    [L,num] = bwlabel(mask);
    features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
    smallest_solidity = mink([features.Solidity]',num_items);
    frames_pictures = zeros(1,num_items);
    c=1;
    for counter=1:num
        x = features(counter).Centroid(1);
        y = features(counter).Centroid(2);
        text(x,y, {['Obj ' num2str(counter)], num2str(features(counter).Solidity)}, 'Color','r')
        if smallest_solidity(end) >= features(counter).Solidity
            frames_pictures(c) = counter;
            c = c+1;
        end
    end
%     
    figure
    imshow(Z.*0.5)
    hold on
    frame_results = struct;
    for j=1:size(frames_pictures,2)
        frame_results.circularity(j) = features(frames_pictures(j)).Circularity;
        frame_results.solidity(j) = features(frames_pictures(j)).Solidity;
        frame_results.eccentricity(j) = features(frames_pictures(j)).Eccentricity;
        rectangle('Position', features(frames_pictures(j)).BoundingBox, 'EdgeColor', 'r');
    end
    disp("Circularidade")
    disp(['min: ',num2str(min(frame_results.circularity))])
    disp(['max: ',num2str(max(frame_results.circularity))])
    disp("Solidez")
    disp(['min: ',num2str(min(frame_results.solidity))])
    disp(['max: ',num2str(max(frame_results.solidity))])
    disp("Eccentricity")
    disp(['min: ',num2str(min(frame_results.eccentricity))])
    disp(['max: ',num2str(max(frame_results.eccentricity))])
end

function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
  end
end