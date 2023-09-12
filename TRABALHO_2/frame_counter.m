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
    %%


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
    
    Z = rgb2gray(image);
    Z = imadjust(Z);
    Z = autobin(Z);
    %Z = imbinarize(Z,'adaptive');
    imshow(Z)
    figure
    
    %Z = bwmorph(Z,'close');
    %Z = bwmorph(Z,'open');
    imshow(Z)
    
    mask = bwareaopen(Z, 100); % Throw away blobs of 9 pixels or smaller.
    [L,num] = bwlabel(mask);
    features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
    for counter=1:num
        x = features(counter).Centroid(1);
        y = features(counter).Centroid(2);
        text(x,y, {['Obj ' num2str(counter)], num2str(features(counter).Circularity)}, 'Color','r')
    end
    
    figure
    imshow(Z.*0.5)
    hold on
    %frames_pictures = [1 2 3 4 16 17 18 19 26 27 28 29 36 37 38 39 47 48
    %49 51 57 58 59 60 68 69 70 71 82 83 84 85 99 100 103 104];%bin
    %adaptativa
    frames_pictures = [1 2 3 4 16 17 18 19 27 28 29 30 39 40 41 42 50 51 52 54 61 62 63 64 73 74 75 76 86 87 88 89 103 104 107 108];%outro metodo
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