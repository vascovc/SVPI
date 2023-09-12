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

Z_hsv = rgb2hsv(image);
Z = Z_hsv(:,:,3);
Z = imadjust(Z);
Z = autobin(Z);

%Z = imbinarize(Z,'adaptive');

%imshow(Z)

%Z = bwareaopen(Z, 200);
%Z = imdilate(Z,ones(2));
%imshow(Z)

tolerance = 0.2;
tolerance_inf = 1-tolerance;
tolerance_sup = 1+tolerance;

mask = bwareaopen(Z, 200);
[L,~] = bwlabel(mask);
features = regionprops(mask,'all');
border_limits_solidity = [0.044261*tolerance_inf 0.053804*tolerance_sup];
border_limits_circularity = [0.037822*tolerance_inf 0.046407*tolerance_sup];
border_idx_solidity = find([features.Solidity] > border_limits_solidity(1) & [features.Solidity]<border_limits_solidity(2));
border_idx_circularity = find([features.Circularity] > border_limits_circularity(1) & [features.Circularity]<border_limits_circularity(2));

mask = ismember(L, intersect(border_idx_solidity, border_idx_circularity));
% subplot(1,3,1)
% imshow(mask)
filled = imfill(mask, 'holes');
features_filled = regionprops(filled,'all');
border_idx_circularity = find([features_filled.Circularity] > 0.7);
obj_frame = length(border_idx_circularity);
Z = logical(Z-mask);

img = imdilate(Z,ones(3));
img = imfill(img,'holes');
Z = bwconvhull(img,'objects');
Z = bwmorph(Z,"bridge");
Z = bwareaopen(Z, 200);

S = false(size(Z));
S(1,:)=1;S(end,:)=1;
S(:,1)=1;S(:,end)=1;
S=and(S,Z);
M = imreconstruct(S,Z);

subplot(1,2,1)
[L,obj_border] = bwlabel(M);
%obj_border = n;
imshow(M)
hold on
features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
for counter=1:obj_border
    x = features(counter).Centroid(1);
    y = features(counter).Centroid(2);
    text(x,y, {['Obj ' num2str(counter)]}, 'Color','r')
end

Z_adp = Z;
Z_adp(1,:)=1;Z_adp(end,:)=1;
Z_adp(:,1)=1;Z_adp(:,end)=1;
N = imclearborder(Z_adp);
[L,obj_ok] = bwlabel(N);
subplot(1,2,2)
imshow(N)
hold on
features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity');
for counter=1:obj_ok
    x = features(counter).Centroid(1);
    y = features(counter).Centroid(2);
    text(x,y, {['Obj ' num2str(counter)], num2str(features(counter).Circularity)}, 'Color','r')
end


end
function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
  end
end