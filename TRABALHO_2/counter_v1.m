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
filled = imfill(mask, 'holes');
features_filled = regionprops(filled,'all');
border_idx_circularity = find([features_filled.Circularity] > 0.7);
obj_frame = length(border_idx_circularity);
Z_1 = logical(Z-mask);

img = imdilate(Z_1,ones(3));
img = imfill(img,'holes');
Z_1 = bwconvhull(img,'objects');
Z_1 = bwmorph(Z_1,"bridge");
Z_1 = bwareaopen(Z_1, 200);

S = false(size(Z));
S(1,:)=1;S(end,:)=1;
S(:,1)=1;S(:,end)=1;
S = and(S,Z_1);
M = imreconstruct(S,Z_1);

[~,obj_border] = bwlabel(M);

N = and(Z_1,not(M));
[L,obj_ok] = bwlabel(N);
L = and(L,Z);
imshow(L)
features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
% n = size([features.Circularity]',1);
% entropy_array = zeros(n,1);
% for ent = 1:n
%     entropy_array(ent) = entropy(features(ent).Image);
% end
Patts = [[features.Circularity]' [features.Solidity]' [features.Eccentricity]'];
%Patts = [[features.Circularity]' [features.Solidity]' [features.Eccentricity]' entropy_array];
noInfRows = ~any(isinf(Patts), 2);
Patts = Patts(noInfRows,:);

%
boxes = zeros(size([features.Circularity],2),4);
for b=1:size([features.Circularity],2)
    boxes(b,:) = features(b).BoundingBox;
end
boxes = boxes(noInfRows,:);
%

d_bio = zeros(size(Patts,1),size(Patts,2));
d_elet = d_bio;
d_corrosive = d_bio;
d_laser = d_bio;
d_toxic = d_bio;
d_explosive = d_bio;
%bio
load parameters.mat

for iter=1:3
  d_bio(:,iter) = mahal(Patts,s.bio(:,:,iter));
  d_bio(:,iter) = d_bio(:,iter)./max(d_bio(:,iter));

  d_elet(:,iter) = mahal(Patts,s.eletric(:,:,iter));
  d_elet(:,iter) = d_elet(:,iter)./max(d_elet(:,iter));
   
  d_corrosive(:,iter) = mahal(Patts,s.corrosive(:,:,iter));
  d_corrosive(:,iter) = d_corrosive(:,iter)./max(d_corrosive(:,iter));

  d_laser(:,iter) = mahal(Patts,s.laser(:,:,iter));
  d_laser(:,iter) = d_laser(:,iter)./max(d_laser(:,iter));
   
  d_toxic(:,iter) = mahal(Patts,s.toxic(:,:,iter));
  d_toxic(:,iter) = d_toxic(:,iter)./max(d_toxic(:,iter));

  d_explosive(:,iter) = mahal(Patts,s.explosive(:,:,iter));
  d_explosive(:,iter) = d_explosive(:,iter)./max(d_explosive(:,iter));
end

%
ind_bio = any(d_bio<0.0001,2);
ind_elet = any(d_elet<0.001,2);
ind_corr = any(d_corrosive<0.0001,2);
ind_laser = any(d_laser<0.001,2);
ind_toxic = any(d_toxic<0.001,2);
ind_explosive = any(d_explosive<0.0001,2);
for a=1:size(Patts,1)
    hold on
    if ind_bio(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'r');
    end
    if ind_elet(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'y');
    end
    if ind_corr(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'c');
    end
    if ind_laser(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'b');
    end
    if ind_toxic(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'm');
    end
    if ind_explosive(a)
        rectangle('Position', boxes(a,:), 'EdgeColor', 'w');
    end

end
%
bio = sum(any(d_bio         <0.0001,2));
elec = sum(any(d_elet       <0.001,2));
cor = sum(any(d_corrosive   <0.0001,2));
laser = sum(any(d_laser     <0.001,2));
tox = sum(any(d_toxic       <0.001,2));
explo = sum(any(d_explosive <0.0001,2));

end
function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
  end
end