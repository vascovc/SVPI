clear all
close all
clc
%function tp2_97746()
nn = 97746;

dificuldade = 1; % 1-hardest; 2-mid; 3 - easier

% lista = dir("Imagens de Referência/frame/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Imagens de Referência/noframe/svpi2023_TP2_img_*"+dificuldade+"_*.png");
 lista = dir("Seq39x/imagens/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("../svpi2023_TP2_img_*"+dificuldade+"_*.png");

num_files = size(lista,1);
matrix = zeros(num_files,18);
for i=1:num_files
    tic
    file = lista(i).name;
    num_seq = str2double(file(18:20));
    num_img = str2double(file(22:23));
    image = im2double(imread([lista(i).folder,'\',lista(i).name]));
    %imshow(image)
    figure
    [obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi] = func_identifier(image);
    matrix(i,:) =[nn,num_seq,num_img,obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi];
    %pause
    toc
end
writematrix(matrix,"tp2_97746.txt")
disp('finished')
%close all
%end

function [obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi] = func_identifier(image)
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

%imshow(Z)

%Z = bwareaopen(Z, 200);
%Z = imdilate(Z,ones(2));
%imshow(Z)

tolerance = 0.07;
tolerance_inf = 1-tolerance;
tolerance_sup = 1+tolerance;

mask = bwareaopen(Z, 200);
[L,n] = bwlabel(mask);
features = regionprops(mask,'all');
border_limits_solidity = [0.044261*tolerance_inf 0.08024*tolerance_sup];
border_idx_solidity = find([features.Solidity] > border_limits_solidity(1) & [features.Solidity]<border_limits_solidity(2));

mask = ismember(L, border_idx_solidity);
filled = imfill(mask, 'holes');
features_filled = regionprops(filled,'all');
border_idx_circularity = find([features_filled.Circularity] > 0.7);
obj_frame = length(border_idx_circularity);

Z = logical(Z-mask);

img = xor(filled,Z);
img = imfill(img,'holes');
img = imdilate(img,ones(3));
img = bwconvhull(img,'objects');
imshow(img)
%Z = imfill(Z,'holes');
%Z = imerode(Z,strel('disk', 1));
%Z = imerode(Z,ones(2));
%Z = imdilate(Z,ones(3));
%Z = imdilate(Z,strel('diamond', 3));
%Z = bwconvhull(Z,'objects');
%Z = bwmorph(Z,"bridge");
%Z = bwconvhull(Z,'objects');
%Z = imdilate(Z,ones(3));
%Z = imdilate(Z,strel('disk', 3));
% inner = imerode(Z,ones(3));
% %inner = imerode(Z,strel('disk', 2));
% %Z = imdilate(inner,strel('disk', 8));
% Z = imdilate(inner,ones(13));

% S = false(size(Z));
% S(1,:)=1;S(end,:)=1;
% S(:,1)=1;S(:,end)=1;
% S=and(S,Z);
% M = imreconstruct(S,Z);
% 
% subplot(1,2,1)
% [L,n] = bwlabel(M);
% obj_border = n;
% imshow(M)
% hold on
% features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
% for counter=1:n
%     x = features(counter).Centroid(1);
%     y = features(counter).Centroid(2);
%     text(x,y, {['Obj ' num2str(counter)]}, 'Color','r')
% end
% 
% N = and(Z,not(M));
% [L,n] = bwlabel(N);
% obj_ok = n;
% subplot(1,2,2)
% imshow(N)
% hold on
% features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
% for counter=1:n
%     x = features(counter).Centroid(1);
%     y = features(counter).Centroid(2);
%     text(x,y, {['Obj ' num2str(counter)], num2str(features(counter).Circularity)}, 'Color','r')
% end


end
function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
  end
end