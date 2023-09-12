clear all
close all
clc
%tp2_97746()
%function tp2_97746()
nn = 97746;

dificuldade = 1; % 1-hardest; 2-mid; 3 - easier

% lista = dir("Imagens de Referência/frame/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Imagens de Referência/noframe/svpi2023_TP2_img_*"+dificuldade+"_*.png");
lista = dir("Seq39x/imagens/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("../svpi2023_TP2_img_*"+dificuldade+"_*.png");

num_files = size(lista,1);
matrix = zeros(num_files,18);
parfor i=1:num_files
    tic
    file = lista(i).name;
    num_seq = str2double(file(18:20));
    num_img = str2double(file(22:23));
    image = im2double(imread([lista(i).folder,'\',lista(i).name]));
    %imshow(image)
    %figure(i)
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
% obj_border = 0;
% obj_ok = 0;
% obj_frame = 0;
% bio = 0;
% cor = 0;
% elec = 0;
% explo = 0;
% info = 0;
% laser = 0;
% lit = 0;
% phone = 0;
% smoking = 0;
% tox = 0;
% wc = 0;
% wifi = 0;

threshold_t = 500;
threshold_c = 600;
threshold_s = 400;
threshold_c_2 = 500;
threshold_s_2 = 500;
threshold_c_1 = 200;

Z_hsv = rgb2hsv(image);

Z = Z_hsv(:,:,3);
Z = imadjust(Z);
Z = autobin(Z);
%Z_show = bsxfun(@times,image,cast(Z, 'like', image));
% figure
% imshow(Z)
%Z_hsv_2 = autobin(imadjust(Z_hsv(:,:,2)));
Z_hsv_1 = autobin(imadjust(Z_hsv(:,:,1)));
% figure
% imshow(Z_hsv_2)
% Z = bsxfun(@times,image,cast(Z, 'like', image));
% Z = rgb2gray(Z);
% figure
% imshow(Z)

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
features_filled = regionprops(filled,'Circularity');
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
M = imdilate(M,ones(3));

[~,obj_border] = bwlabel(M);

Z_adp = Z_1;
Z_adp(1,:)=1;Z_adp(end,:)=1;
Z_adp(:,1)=1;Z_adp(:,end)=1;
N = imclearborder(Z_adp);
[L_numbered,obj_ok] = bwlabel(N);
%imshow(N)
%L_numbered = bwareaopen(L_numbered,50);

%
Z_show = bsxfun(@times,image,cast(N, 'like', image));
asfas = rgb2hsv(Z_show);
Z_hsv_2 = autobin(imadjust(asfas(:,:,2)));
%Z_hsv_1 = autobin(imadjust(asfas(:,:,1)));
%
L = and(L_numbered,Z);
L = bwareaopen(L, 300);
L_canal_2 = and(L_numbered,Z_hsv_2);
L_canal_2 = bwareaopen(L_canal_2, 300);
L_canal_1 = and(L_numbered,Z_hsv_1);
L_canal_1 = bwareaopen(L_canal_1, 300);

mask_stats = regionprops(N,'Circularity');
ff = [mask_stats.Circularity];
tr_lim = 0.78;
cir_lim = 0.945;
triangles_idx = find(ff<tr_lim);
squares_idx = find(ff>tr_lim & ff<cir_lim);
circles_idx = find(ff>cir_lim);
TRI = ismember(L_numbered,triangles_idx);
SQU = ismember(L_numbered,squares_idx);
CIR = ismember(L_numbered,circles_idx);
triangles = and(TRI,L);
squares = and(SQU,L);
circles = and(CIR,L);
%
% subplot(1,3,1)
% imshow(and(TRI,L))
% subplot(1,3,2)
% imshow(and(SQU,L))
% subplot(1,3,3)
% imshow(and(CIR,L))
%

features_triangles = regionprops(logical(triangles),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
features_squares   = regionprops(logical(squares),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
features_circles   = regionprops(logical(circles),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');

features_squares_2 = regionprops(logical(and(SQU,L_canal_2)),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
features_circles_2 = regionprops(logical(and(CIR,L_canal_2)),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');

features_circles_1 = regionprops(logical(and(CIR,L_canal_1)),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');

Patts_triangles = [[features_triangles.Circularity]' [features_triangles.Solidity]' [features_triangles.Eccentricity]' [features_triangles.Eccentricity]' [features_triangles.Eccentricity]' [features_triangles.Eccentricity]' [features_triangles.Eccentricity]'];
Patts_squares = [[features_squares.Circularity]' [features_squares.Solidity]' [features_squares.Eccentricity]' [features_squares.Eccentricity]' [features_squares.Eccentricity]' [features_squares.Eccentricity]' [features_squares.Eccentricity]'];
Patts_circles = [[features_circles.Circularity]' [features_circles.Solidity]' [features_circles.Eccentricity]' [features_circles.Eccentricity]' [features_circles.Eccentricity]' [features_circles.Eccentricity]' [features_circles.Eccentricity]'];

Patts_squares_2 = [[features_squares_2.Circularity]' [features_squares_2.Solidity]' [features_squares_2.Eccentricity]' [features_squares_2.Eccentricity]' [features_squares_2.Eccentricity]' [features_squares_2.Eccentricity]' [features_squares_2.Eccentricity]'];
Patts_circles_2 = [[features_circles_2.Circularity]' [features_circles_2.Solidity]' [features_circles_2.Eccentricity]' [features_circles_2.Eccentricity]' [features_circles_2.Eccentricity]' [features_circles_2.Eccentricity]' [features_circles_2.Eccentricity]'];

Patts_circles_1 = [[features_circles_1.Circularity]' [features_circles_1.Solidity]' [features_circles_1.Eccentricity]' [features_circles_1.Eccentricity]' [features_circles_1.Eccentricity]' [features_circles_1.Eccentricity]' [features_circles_1.Eccentricity]'];
for inter = 1:size(Patts_triangles,1)
    invariant_moments = feature_vec(features_triangles(inter).Image);
    Patts_triangles(inter,4) = log(invariant_moments(1));
    Patts_triangles(inter,5) = log(invariant_moments(2));
    Patts_triangles(inter,6) = log(invariant_moments(3));
    Patts_triangles(inter,7) = log(invariant_moments(4));
end
for inter = 1:size(Patts_squares,1)
    invariant_moments = feature_vec(features_squares(inter).Image);
    Patts_squares(inter,4) = log(invariant_moments(1));
    Patts_squares(inter,5) = log(invariant_moments(2));
    Patts_squares(inter,6) = log(invariant_moments(3));
    Patts_squares(inter,7) = log(invariant_moments(4));
end
for inter = 1:size(Patts_circles,1)
    invariant_moments = feature_vec(features_circles(inter).Image);
    Patts_circles(inter,4) = log(invariant_moments(1));
    Patts_circles(inter,5) = log(invariant_moments(2));
    Patts_circles(inter,6) = log(invariant_moments(3));
    Patts_circles(inter,7) = log(invariant_moments(4));
end

for inter = 1:size(Patts_squares_2,1)
    invariant_moments = feature_vec(features_squares_2(inter).Image);
    Patts_squares_2(inter,4) = log(invariant_moments(1));
    Patts_squares_2(inter,5) = log(invariant_moments(2));
    Patts_squares_2(inter,6) = log(invariant_moments(3));
    Patts_squares_2(inter,7) = log(invariant_moments(4));
end
for inter = 1:size(Patts_circles_2,1)
    invariant_moments = feature_vec(features_circles_2(inter).Image);
    Patts_circles_2(inter,4) = log(invariant_moments(1));
    Patts_circles_2(inter,5) = log(invariant_moments(2));
    Patts_circles_2(inter,6) = log(invariant_moments(3));
    Patts_circles_2(inter,7) = log(invariant_moments(4));
end
for inter = 1:size(Patts_circles_1,1)
    invariant_moments = feature_vec(features_circles_1(inter).Image);
    Patts_circles_1(inter,4) = log(invariant_moments(1));
    Patts_circles_1(inter,5) = log(invariant_moments(2));
    Patts_circles_1(inter,6) = log(invariant_moments(3));
    Patts_circles_1(inter,7) = log(invariant_moments(4));
end

noInfRows_t = ~any(isinf(Patts_triangles), 2);
Patts_triangles = Patts_triangles(noInfRows_t,:);
noInfRows_s = ~any(isinf(Patts_squares), 2);
Patts_squares = Patts_squares(noInfRows_s,:);
noInfRows_c = ~any(isinf(Patts_circles), 2);
Patts_circles = Patts_circles(noInfRows_c,:);

noInfRows_s_2 = ~any(isinf(Patts_squares_2), 2);
Patts_squares_2 = Patts_squares_2(noInfRows_s_2,:);
noInfRows_c_2 = ~any(isinf(Patts_circles_2), 2);
Patts_circles_2 = Patts_circles_2(noInfRows_c_2,:);

noInfRows_c_1 = ~any(isinf(Patts_circles_1), 2);
Patts_circles_1 = Patts_circles_1(noInfRows_c_1,:);

if size(Patts_triangles,1) == 0
    Patts_triangles = nan(1,7);
end
if size(Patts_squares,1) == 0
    Patts_squares = nan(1,7);
end
if size(Patts_circles,1) == 0
    Patts_circles = nan(1,7);
end
if size(Patts_squares_2,1) == 0
    Patts_squares_2 = nan(1,7);
end
if size(Patts_circles_2,1) == 0
    Patts_circles_2 = nan(1,7);
end
if size(Patts_circles_1,1) == 0
    Patts_circles_1 = nan(1,7);
end

%
% boxes_t = zeros(size([features_triangles.Circularity],2),4);
% x_t = zeros(size([features_triangles.Circularity],2),1);
% y_t = zeros(size([features_triangles.Circularity],2),1);
% boxes_s = zeros(size([features_squares.Circularity],2),4);
% boxes_s_2 = zeros(size([features_squares_2.Circularity],2),4);
% x_s = zeros(size([features_squares.Circularity],2),1);
% y_s = zeros(size([features_squares.Circularity],2),1);
% boxes_c = zeros(size([features_circles.Circularity],2),4);
% boxes_c_2 = zeros(size([features_circles_2.Circularity],2),4);
% boxes_c_1 = zeros(size([features_circles_1.Circularity],2),4);
% x_c = zeros(size([features_circles.Circularity],2),1);
% y_c = zeros(size([features_circles.Circularity],2),1);
% for b=1:size([features_triangles.Circularity],2)
%     boxes_t(b,:) = features_triangles(b).BoundingBox;
%     x_t(b) = features_triangles(b).Centroid(1);
%     y_t(b) = features_triangles(b).Centroid(2);
%     txt_t(b,:) = {num2str(features_triangles(b).Circularity),num2str(features_triangles(b).Solidity),num2str(features_triangles(b).Eccentricity)};
% end
% for b=1:size([features_squares.Circularity],2)
%     boxes_s(b,:) = features_squares(b).BoundingBox;
%     x_s(b) = features_squares(b).Centroid(1);
%     y_s(b) = features_squares(b).Centroid(2);
%     txt_s(b,:) = {num2str(features_squares(b).Circularity),num2str(features_squares(b).Solidity),num2str(features_squares(b).Eccentricity)};
% end
% for b=1:size([features_circles.Circularity],2)
%     boxes_c(b,:) = features_circles(b).BoundingBox;
%     x_c(b) = features_circles(b).Centroid(1);
%     y_c(b) = features_circles(b).Centroid(2);
%     txt_c(b,:) = {num2str(features_circles(b).Circularity),num2str(features_circles(b).Solidity),num2str(features_circles(b).Eccentricity)};
% end
% for b=1:size([features_circles_2.Circularity],2)
%     boxes_c_2(b,:) = features_circles_2(b).BoundingBox;
% end
% for b=1:size([features_circles_1.Circularity],2)
%     boxes_c_1(b,:) = features_circles_1(b).BoundingBox;
% end
% for b=1:size([features_squares_2.Circularity],2)
%     boxes_s_2(b,:) = features_squares_2(b).BoundingBox;
% end
% if ~isempty(noInfRows_t)
% boxes_t = boxes_t(noInfRows_t,:);
% x_t = x_t(noInfRows_t,:);
% y_t = y_t(noInfRows_t,:);
% txt_t = txt_t(noInfRows_t,:);
% end
% if ~isempty(noInfRows_s)
% boxes_s = boxes_s(noInfRows_s,:);
% x_s = x_s(noInfRows_s,:);
% y_s = y_s(noInfRows_s,:);
% txt_s = txt_s(noInfRows_s,:);
% end
% if ~isempty(noInfRows_c)
% boxes_c = boxes_c(noInfRows_c,:);
% x_c = x_c(noInfRows_c,:);
% y_c = y_c(noInfRows_c,:);
% txt_c = txt_c(noInfRows_c,:);
% end
% if ~isempty(noInfRows_c_2)
% boxes_c_2 = boxes_c_2(noInfRows_c_2,:);
% end
% if ~isempty(noInfRows_s_2)
% boxes_s_2 = boxes_s_2(noInfRows_s_2,:);
% end
% if ~isempty(noInfRows_c_1)
% boxes_c_1 = boxes_c_1(noInfRows_c_1,:);
% end
%

%load parameters.mat
load parameters_advanced.mat
num_triangles_analyzing = 11;
num_circles_analyzing = 8;
num_squares_analyzing = 15;

num_circles_analyzing_2 = 6;
num_squares_analyzing_2 = 2;

num_circles_analyzing_1 = 2;

dist_triangles = nan(size(Patts_triangles,1),num_triangles_analyzing);
dist_circles = nan(size(Patts_circles,1),num_circles_analyzing);
dist_squares = nan(size(Patts_squares,1),num_squares_analyzing);

dist_squares_2 = nan(size(Patts_squares_2,1),num_squares_analyzing_2);
dist_circles_2 = nan(size(Patts_circles_2,1),num_circles_analyzing_2);
dist_circles_1 = nan(size(Patts_circles_1,1),num_circles_analyzing_1);

%bio
bio_1_t = 1;
bio_2_c = 1;
bio_3_s = 1;
dist_triangles(:,bio_1_t) = mahal(Patts_triangles,s.bio(:,:,1));
dist_circles(:,bio_2_c) = mahal(Patts_circles,s.bio(:,:,2));
dist_squares(:,bio_3_s) = mahal(Patts_squares,s.bio(:,:,3));

%corrosive
corrosive_1_t = bio_1_t+1;
corrosive_2_t = corrosive_1_t+1;
corrosive_3_s = bio_3_s+1;
dist_triangles(:,corrosive_1_t) = mahal(Patts_triangles,s.corrosive(:,:,1));
dist_triangles(:,corrosive_2_t) = mahal(Patts_triangles,s.corrosive(:,:,2));
%dist_squares(:,corrosive_3_s) = mahal(Patts_squares,s.corrosive(:,:,1));
dist_squares(:,corrosive_3_s) = mahal(Patts_squares,s.corrosive(:,:,3));
%dist_squares(:,corrosive_3_s) = nan(size(Patts_squares,1),1);

%eletric
eletric_1_t = corrosive_2_t+1;
eletric_2_t = eletric_1_t+1;
eletric_3_t = eletric_2_t+1;
dist_triangles(:,eletric_1_t) = mahal(Patts_triangles,s.eletric(:,:,1));
dist_triangles(:,eletric_2_t) = mahal(Patts_triangles,s.eletric(:,:,2));
dist_triangles(:,eletric_3_t) = mahal(Patts_triangles,s.eletric(:,:,3));

%explosive
explosive_1_t = eletric_3_t+1;
explosive_2_c = bio_2_c+1;
explosive_3_t = explosive_1_t+1;
dist_triangles(:,explosive_1_t) = mahal(Patts_triangles,s.explosive(:,:,1));
dist_circles(:,explosive_2_c)   = mahal(Patts_circles  ,s.explosive(:,:,2));
dist_triangles(:,explosive_3_t) = mahal(Patts_triangles,s.explosive(:,:,3));

%laser
laser_1_t = explosive_3_t+1;
laser_2_t = laser_1_t+1;
laser_3_t = laser_2_t+1;
laser_3_s = corrosive_3_s+1;
dist_triangles(:,laser_1_t) = mahal(Patts_triangles,s.laser(:,:,1));
dist_triangles(:,laser_2_t) = mahal(Patts_triangles,s.laser(:,:,2));
dist_triangles(:,laser_3_t) = mahal(Patts_triangles,s.laser(:,:,3));
dist_squares(:,laser_3_s)   = mahal(Patts_squares  ,s.laser(:,:,3));

%info
info_1_c = explosive_2_c+1;
dist_circles(:,info_1_c) = mahal(Patts_circles,s.info(:,:,1));

%toxic
toxic_1_s = laser_3_s+1;
toxic_2_s = toxic_1_s+1;
toxic_3_s = toxic_2_s+1;
dist_squares(:,toxic_1_s) = mahal(Patts_squares,s.toxic(:,:,1));
dist_squares(:,toxic_2_s) = mahal(Patts_squares,s.toxic(:,:,2));
dist_squares(:,toxic_3_s) = mahal(Patts_squares,s.toxic(:,:,3));

%wc falta wc 3
wc_1_s = toxic_3_s+1;
wc_2_s = wc_1_s+1;
dist_squares(:,wc_1_s) = mahal(Patts_squares,s.wc(:,:,1));
dist_squares(:,wc_2_s) = mahal(Patts_squares,s.wc(:,:,2));

%wifi
wifi_2_s = wc_2_s+1;
wifi_3_s = wifi_2_s+1;
dist_squares(:,wifi_2_s)  = mahal(Patts_squares,s.wifi(:,:,2));
dist_squares(:,wifi_3_s) = mahal(Patts_squares,s.wifi(:,:,3));

%phone
phone_1_c = info_1_c+1;
phone_2_s = wifi_3_s+1;
dist_circles(:,phone_1_c)  = mahal(Patts_circles,s.phone(:,:,1));
dist_squares(:,phone_2_s) = mahal(Patts_squares,s.phone(:,:,2));

%lixo
lixo_1_c = phone_1_c+1;
lixo_3_s = phone_2_s+1;
dist_circles(:,lixo_1_c)  = mahal(Patts_circles,s.lixo(:,:,1));
dist_squares(:,lixo_3_s)  = mahal(Patts_squares,s.lixo(:,:,3));

%smoking
smoking_1_c = lixo_1_c+1;
smoking_2_c = smoking_1_c+1;
smoking_3_s = lixo_3_s+1;
dist_circles(:,smoking_1_c)  = mahal(Patts_circles,s.cigar(:,:,1));
dist_circles(:,smoking_2_c)  = mahal(Patts_circles,s.cigar(:,:,2));
dist_squares(:,smoking_3_s)  = mahal(Patts_squares,s.cigar(:,:,3));

%canal 2
%info 2 canal 2
info_2_s_2 = 1;
dist_squares_2(:,info_2_s_2) = mahal(Patts_squares_2,s.info(:,:,2));

%phone 3 canal 2
phone_3_c_2 = 1;
dist_circles_2(:,phone_3_c_2) = mahal(Patts_circles_2,s.phone(:,:,3));

%wc 3 canal 2
wc_3_c_2 = phone_3_c_2+1;
dist_circles_2(:,wc_3_c_2) = mahal(Patts_circles_2,s.wc(:,:,3));

%lixo 2 canal 2
lixo_2_c_2 = wc_3_c_2+1;
dist_circles_2(:,lixo_2_c_2) = mahal(Patts_circles_2,s.lixo(:,:,2));

%wifi 1 canal 2
wifi_1_s_2 = info_2_s_2+1;
dist_squares_2(:,wifi_1_s_2) = mahal(Patts_squares_2,s.wifi(:,:,1));

%canal 1
%info 3 canal 1
info_3_c_1 = 1;
dist_circles_1(:,info_3_c_1) = mahal(Patts_circles_1,s.info(:,:,3));

% lixo_1 extra canal 3
lixo_1_c_extra = smoking_2_c+1;
dist_circles(:,lixo_1_c_extra) = mahal(Patts_circles,s.lixo_extra(:,:,2));
%wifi 2 extra canal 3
wifi_2_s_extra = smoking_3_s+1;
dist_squares(:,wifi_2_s_extra) = mahal(Patts_squares,s.wifi_extra(:,:,1));
    
% wc_3 extra canal 2
wc_3_c_2_extra = lixo_2_c_2+1;
dist_circles_2(:,wc_3_c_2_extra) = mahal(Patts_circles_2,s.wc_extra_3_ch_2(:,:,1));

%corrosive 3 extra canal 3
corrosive_3_s_extra1 = wifi_2_s_extra+1;
dist_squares(:,corrosive_3_s_extra1) = mahal(Patts_squares,s.corrosive_3_extra(:,:,1));

% info 3 extra canal 1
info_3_c_1_extra = info_3_c_1+1;
dist_circles_1(:,info_3_c_1_extra) = mahal(Patts_circles_1,s.info_3_extra(:,:,1));

% phone 3 extra canal 2
phone_3_c_2_extra_1 = wc_3_c_2_extra+1;
phone_3_c_2_extra_2 = phone_3_c_2_extra_1+1;
dist_circles_2(:,phone_3_c_2_extra_1) = mahal(Patts_circles_2,s.phone_3_extra(:,:,1));
dist_circles_2(:,phone_3_c_2_extra_2) = mahal(Patts_circles_2,s.phone_3_extra(:,:,2));

% info 3 extra canal 1

%
%dist_triangles = dist_triangles./max(dist_triangles);
%dist_squares   = dist_squares./max(dist_squares);
%dist_circles   = dist_circles./max(dist_circles);

%dist_squares_2   = dist_squares_2./max(dist_squares_2);
%dist_circles_2   = dist_circles_2./max(dist_circles_2);
%dist_circles_1   = dist_circles_1./max(dist_circles_1);

minimos_triangles=transform_to_minims(dist_triangles,num_triangles_analyzing,threshold_t);
minimos_circles  =transform_to_minims(dist_circles,num_circles_analyzing,threshold_c);
minimos_squares  =transform_to_minims(dist_squares,num_squares_analyzing,threshold_s);

minimos_circles_2  =transform_to_minims(dist_circles_2,num_circles_analyzing_2,threshold_c_2);
minimos_squares_2  =transform_to_minims(dist_squares_2,num_squares_analyzing_2,threshold_s_2);

minimos_circles_1  =transform_to_minims(dist_circles_1,num_circles_analyzing_1,threshold_c_1);

elec = sum(sum(minimos_triangles(:,[eletric_1_t eletric_2_t eletric_3_t])));
cor = sum(sum(minimos_triangles(:,[corrosive_1_t corrosive_2_t]))) + sum(sum(minimos_squares(:,[corrosive_3_s corrosive_3_s_extra1])));
bio = sum(minimos_triangles(:,bio_1_t)) + sum(minimos_circles(:,bio_2_c)) + sum(minimos_squares(:,bio_3_s));
tox = sum(sum(minimos_squares(:,[toxic_1_s toxic_2_s toxic_3_s])));
laser = sum(sum(minimos_triangles(:,[laser_1_t laser_2_t laser_3_t]))) + sum(sum(minimos_squares(:,laser_3_s)));
wc = sum(sum(minimos_squares(:,[wc_1_s wc_2_s]))) + sum(sum(minimos_circles_2(:,[wc_3_c_2 wc_3_c_2_extra])));
wifi = sum(sum(minimos_squares(:,[wifi_2_s wifi_3_s wifi_2_s_extra]))) + sum(minimos_squares_2(:,wifi_1_s_2));
phone = sum(minimos_circles(:,phone_1_c))+sum(minimos_squares(:,phone_2_s)) + sum(sum(minimos_circles_2(:,[phone_3_c_2 phone_3_c_2_extra_1 phone_3_c_2_extra_2])));
lit = sum(sum(minimos_circles(:,[lixo_1_c lixo_1_c_extra]))) + sum(minimos_squares(:,lixo_3_s)) + sum(minimos_circles_2(:,lixo_2_c_2));
smoking = sum(sum(minimos_circles(:,[smoking_1_c smoking_2_c]))) + sum(minimos_squares(:,smoking_3_s));
explo = sum(sum(minimos_triangles(:,[explosive_1_t explosive_3_t]))) + sum(minimos_circles(:,explosive_2_c));
info = sum(minimos_circles(:,info_1_c)) + sum(minimos_squares_2(:,info_2_s_2)) + sum(sum(minimos_circles_1(:,[info_3_c_1 info_3_c_1_extra])));

% %
% subplot(1,3,1)
% imshow(triangles)
% if size(minimos_triangles,1) > 1
% for object=1:size(minimos_triangles,1)
%     hold on
%     text(x_t(object),y_t(object), txt_t(object,:), 'Color','r')
%     if sum(minimos_triangles(object,:)) == 1
%         for aj = 1:size(minimos_triangles,2)
%             if minimos_triangles(object,aj)
%                 if aj==eletric_1_t || aj==eletric_2_t || aj==eletric_3_t
%                     rectangle('Position', boxes_t(object,:),'EdgeColor','y','LineWidth',4);
%                 elseif aj==explosive_1_t || aj==explosive_3_t
%                     rectangle('Position', boxes_t(object,:),'EdgeColor','r','LineWidth',4);
%                 elseif aj==corrosive_1_t || aj==corrosive_2_t
%                     rectangle('Position', boxes_t(object,:),'EdgeColor','g','LineWidth',4);
%                 elseif aj==laser_1_t || aj==laser_2_t || aj==laser_3_t
%                     rectangle('Position', boxes_t(object,:),'EdgeColor','b','LineWidth',4);
%                 elseif aj==bio_1_t
%                     rectangle('Position', boxes_t(object,:),'EdgeColor','c','LineWidth',4);
% %                 else
% %                     rectangle('Position', boxes_t(object,:),'EdgeColor','w','LineStyle','-.','LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% end
% subplot(1,3,2)
% imshow(squares)
% if size(minimos_squares,1) > 1
% for object=1:size(minimos_squares,1)
%     hold on
%     text(x_s(object),y_s(object), txt_s(object,:), 'Color','r')
%     if sum(minimos_squares(object,:)) == 1
%         for aj = 1:size(minimos_squares,2)
%             if minimos_squares(object,aj)
%                 if aj==bio_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor','c','LineWidth',4);
%                 elseif aj==toxic_1_s||aj==toxic_2_s||aj==toxic_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor',[0.9290 0.6940 0.1250],'LineWidth',4);
%                 elseif aj == wifi_2_s || aj==wifi_3_s || aj == wifi_2_s_extra
%                     rectangle('Position', boxes_s(object,:),'EdgeColor','y','LineWidth',4);
%                 elseif aj == corrosive_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor','g','LineWidth',4);
%                 elseif aj == laser_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor','b','LineWidth',4);
%                 elseif aj == wc_1_s || wc_2_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor','m','LineWidth',4);
%                 elseif aj == phone_2_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor',[0.4940 0.1840 0.5560],'LineWidth',4);
%                 elseif aj == lixo_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor',[0.6350 0.0780 0.1840],'LineWidth',4);
%                 elseif aj == smoking_3_s
%                     rectangle('Position', boxes_s(object,:),'EdgeColor',[0 0.4470 0.7410],'LineWidth',4);
% %                 else
% %                     rectangle('Position', boxes_s(object,:),'EdgeColor','w','LineStyle','-.','LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% end
% if size(minimos_squares_2,1) > 1
% for object=1:size(minimos_squares_2,1)
%     hold on
%     if sum(minimos_squares_2(object,:)) == 1
%         for aj = 1:size(minimos_squares_2,2)
%             if minimos_squares_2(object,aj)
%                 if aj==info_2_s_2
%                     rectangle('Position', boxes_s_2(object,:),'EdgeColor','b','LineStyle','-.','LineWidth',4);
%                 elseif aj == wifi_1_s_2
%                     rectangle('Position', boxes_s_2(object,:),'EdgeColor','y','LineStyle','-.','LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% end
% 
% subplot(1,3,3)
% imshow(and(CIR,L))
% for object=1:size(minimos_circles,1)
%     hold on
%     text(x_c(object),y_c(object), txt_c(object,:), 'Color','r')
%     if sum(minimos_circles(object,:)) == 1
%         for aj = 1:size(minimos_circles,2)
%             if minimos_circles(object,aj)
%                 if aj==explosive_2_c
%                     rectangle('Position', boxes_c(object,:),'EdgeColor','r','LineWidth',4);
%                 elseif aj==info_1_c
%                     rectangle('Position', boxes_c(object,:),'EdgeColor','b','LineWidth',4);
%                 elseif aj==lixo_1_c || aj==lixo_1_c_extra
%                     rectangle('Position', boxes_c(object,:),'EdgeColor',[0.6350 0.0780 0.1840],'LineWidth',4);
%                 elseif aj==phone_1_c
%                     rectangle('Position', boxes_c(object,:),'EdgeColor',[0.4940 0.1840 0.5560],'LineWidth',4);
%                 elseif aj==smoking_1_c || aj==smoking_2_c
%                     rectangle('Position', boxes_c(object,:),'EdgeColor',[0 0.4470 0.7410],'LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% if size(minimos_circles_2,1)>1
% for object=1:size(minimos_circles_2,1)
%     hold on
%     if sum(minimos_circles_2(object,:)) == 1
%         for aj = 1:size(minimos_circles_2,2)
%             if minimos_circles_2(object,aj)
%                 if aj== info_2_s_2
%                     rectangle('Position', boxes_c_2(object,:),'EdgeColor','b','LineStyle',':','LineWidth',4);
%                 elseif aj== phone_3_c_2
%                     rectangle('Position', boxes_c_2(object,:),'EdgeColor',[0.4940 0.1840 0.5560],'LineStyle',':','LineWidth',4);
%                 elseif aj== wc_3_c_2 || aj == wc_3_c_2_extra
%                     rectangle('Position', boxes_c_2(object,:),'EdgeColor','m','LineStyle',':','LineWidth',4);
%                 elseif aj== lixo_2_c_2 || lixo_1_c_extra 
%                     rectangle('Position', boxes_c_2(object,:),'EdgeColor',[0.6350 0.0780 0.1840],'LineStyle',':','LineWidth',4);
% %                 else
% %                     rectangle('Position', boxes_c_2(object,:),'EdgeColor','w','LineStyle','-.','LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% end
% if size(minimos_circles_1,1) > 1
% for object=1:size(minimos_circles_1,1)
%     hold on
%     if sum(minimos_circles_1(object,:)) == 1
%         for aj = 1:size(minimos_circles_1,2)
%             if minimos_circles_1(object,aj)
%                 if aj== info_3_c_1
%                     rectangle('Position', boxes_c_1(object,:),'EdgeColor','b','LineStyle','-.','LineWidth',4);
% %                 else
% %                     rectangle('Position', boxes_c_1(object,:),'EdgeColor','w','LineStyle','-.','LineWidth',4);
%                 end
%             end
%         end
%     end
% end
% end

%



end
%end

function Z = autobin(A)
mask = graythresh(A);
Z = imbinarize(A,mask);
if mask < mean(Z(:))
    Z = 1-Z;
end
end

%funcao para ver o minimo de cada linha e se for maior que um threshold da
%0 e se for o valor da 1
function minimos_triangles=transform_to_minims(dist_triangles,num_triangles_analyzing,threshold)
    %threshold = 0.001;
    minimos_triangles = zeros(size(dist_triangles));
    for row=1:size(dist_triangles,1)
        minimo = dist_triangles(row,1);
        pos = 1;
        value = 1;
        for col = 2:num_triangles_analyzing
            if dist_triangles(row,col) < minimo
                minimo = dist_triangles(row,col);
                pos = col;
            end
        end
        if minimo > threshold
            value = 0;
        end
        minimos_triangles(row,:) = zeros(1,num_triangles_analyzing);
        minimos_triangles(row,pos) = value;
    end
end




function n_pq=cent_moment(p,q,A)

[m      n]=size(A);
moo=sum(sum(A));

m1o=0;
mo1=0;
for x=0:m-1
    for y=0:n-1
        m1o=m1o+(x)*A(x+1,y+1);
        mo1=mo1+(y)*A(x+1,y+1);
    end
end
xx=m1o/moo;
yy=mo1/moo;


mu_oo=moo;

mu_pq=0;
for ii=0:m-1
    x=ii-xx;
    for jj=0:n-1
        y=jj-yy;
        mu_pq=mu_pq+(x)^p*(y)^q*A(ii+1,jj+1);
    end
end

gamma=0.5*(p+q)+1;
n_pq=mu_pq/moo^(gamma);
end

function [M]=feature_vec(A)

% This function Calculates the Seven Invariant Moments for the image A
% the output of this function is a Vector M ; called the Feature vector
% the vector M is a column vector containing M1,M2,....M7

% First Moment
n20=cent_moment(2,0,A);
n02=cent_moment(0,2,A);
M1=n20+n02;

% Second Moment
n20=cent_moment(2,0,A);
n02=cent_moment(0,2,A);
n11=cent_moment(1,1,A);
M2=(n20-n02)^2+4*n11^2;

% Third Moment
n30=cent_moment(3,0,A);
n12=cent_moment(1,2,A);
n21=cent_moment(2,1,A);
n03=cent_moment(0,3,A);
M3=(n30-3*n12)^2+(3*n21-n03)^2;

% Fourth Moment
n30=cent_moment(3,0,A);
n12=cent_moment(1,2,A);
n21=cent_moment(2,1,A);
n03=cent_moment(0,3,A);
M4=(n30+n12)^2+(n21+n03)^2;

% Fifth Moment
n30=cent_moment(3,0,A);
n12=cent_moment(1,2,A);
n21=cent_moment(2,1,A);
n03=cent_moment(0,3,A);
M5=(n30-3*n21)*(n30+n12)*[(n30+n12)^2-3*(n21+n03)^2]+(3*n21-n03)*(n21+n03)*[3*(n30+n12)^2-(n21+n03)^2];

% Sixth Moment
n20=cent_moment(2,0,A);
n02=cent_moment(0,2,A);
n30=cent_moment(3,0,A);
n12=cent_moment(1,2,A);
n21=cent_moment(2,1,A);
n03=cent_moment(0,3,A);
n11=cent_moment(1,1,A);
M6=(n20-n02)*[(n30+n12)^2-(n21+n03)^2]+4*n11*(n30+n12)*(n21+n03);

% Seventh Moment
n30=cent_moment(3,0,A);
n12=cent_moment(1,2,A);
n21=cent_moment(2,1,A);
n03=cent_moment(0,3,A);
M7=(3*n21-n03)*(n30+n12)*[(n30+n12)^2-3*(n21+n03)^2]-(n30+3*n12)*(n21+n03)*[3*(n30+n12)^2-(n21+n03)^2];



% The vector M is a column vector containing M1,M2,....M7
M=[M1    M2     M3    M4     M5    M6    M7]';
%and this is the Feature vector
end