function nn = tp2_97746()
nn = 97746;

dificuldade = 1; % 1-hardest; 2-mid; 3 - easier

% lista = dir("Imagens de Referência/frame/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Imagens de Referência/noframe/svpi2023_TP2_img_*"+dificuldade+"_*.png");
lista = dir("Seq39x/imagens/svpi2023_TP2_img_*"+dificuldade+"_*.png");
%lista = dir("../svpi2023_TP2_img_*"+dificuldade+"_*.png");

num_files = size(lista,1);
matrix = zeros(num_files,18);
for i=1:num_files
    %tic
    file = lista(i).name;
    num_seq = str2double(file(18:20));
    num_img = str2double(file(22:23));
    image = im2double(imread([lista(i).folder,'\',lista(i).name]));
    %imshow(image)
    %figure(i)
    s = load_structure();
    [obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi] = func_identifier(image,s);
    matrix(i,:) =[nn,num_seq,num_img,obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi];
    %pause
    %toc
end
writematrix(matrix,"tp2_97746.txt")
%disp('finished')
end

function [obj_border,obj_ok,obj_frame,bio,cor,elec,explo,info,laser,lit,phone,smoking,tox,wc,wifi] = func_identifier(image,s)
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

features_circles_1 = regionprops(logical(and(L_numbered,L_canal_1)),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');

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
%load parameters_advanced.mat
num_triangles_analyzing = 11;
num_circles_analyzing = 8;
num_squares_analyzing = 15;

num_circles_analyzing_2 = 6;
num_squares_analyzing_2 = 2;

num_circles_analyzing_1 = 3;

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
info_extra_c_1_extra_2 = 3;
dist_circles_1(:,info_extra_c_1_extra_2) = mahal(Patts_circles_1,s.info_3_extra(:,:,2));

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
info = sum(minimos_circles(:,info_1_c)) + sum(minimos_squares_2(:,info_2_s_2)) + sum(sum(minimos_circles_1(:,[info_3_c_1 info_3_c_1_extra info_extra_c_1_extra_2])));

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


function s = load_structure()
s=struct;
s.bio(:,:,1) = [0.5260    0.7779    0.0422   -1.2847  -16.6120   -4.2360  -14.2857
    0.5115    0.7830    0.0601   -1.2919  -15.2144   -4.2601  -17.1065
    0.5141    0.7907    0.0328   -1.2982  -17.6515   -4.2750  -16.0421
    0.5100    0.7903    0.0839   -1.2964  -13.8867   -4.2773  -16.3736
    0.4967    0.7895    0.0616   -1.2973  -15.1276   -4.2781  -14.9090
    0.4916    0.7854    0.0466   -1.2836  -16.2169   -4.2302  -16.5398
    0.5084    0.7654    0.1345   -1.3121  -12.0166   -4.3047  -13.9426
    0.5106    0.7723    0.0705   -1.2993  -14.5854   -4.2758  -17.6397
    0.4541    0.7706    0.0650   -1.2972  -14.9079   -4.2673  -15.5649
    0.4497    0.7730    0.0700   -1.2965  -14.6101   -4.2731  -17.0668
    0.4147    0.7708    0.0611   -1.2927  -15.1515   -4.2604  -14.9015
    0.3951    0.7683    0.0787   -1.2855  -14.1197   -4.2351  -15.0748
    0.5312    0.7736    0.0844   -1.3020  -13.8698   -4.2989  -14.6107
    0.4914    0.7736    0.1329   -1.3005  -12.0406   -4.2913  -15.1666
    0.4699    0.7748    0.0554   -1.2952  -15.5428   -4.2704  -16.1996
    0.4453    0.7693    0.1154   -1.2910  -12.5911   -4.2574  -16.2625
    0.4391    0.7707    0.0700   -1.2914  -14.5996   -4.2594  -17.5885
    0.4333    0.7730    0.0671   -1.2855  -14.7592   -4.2394  -15.1525
    0.5086    0.7710    0.0460   -1.2981  -16.2945   -4.2716  -15.9304
    0.4805    0.7630    0.1321   -1.2841  -12.0334   -4.2224  -14.3873
    0.4764    0.7711    0.0722   -1.2950  -14.4854   -4.2798  -15.2858
    0.4570    0.7715    0.1248   -1.2908  -12.2759   -4.2512  -14.3304
    0.4368    0.7713    0.0933   -1.2934  -13.4506   -4.2696  -15.9443
    0.3983    0.7671    0.0371   -1.2807  -17.1249   -4.2233  -18.5913
    0.5242    0.7687    0.0861   -1.2960  -13.7766   -4.2708  -16.4917
    0.4996    0.7713    0.1118   -1.2971  -12.7326   -4.2651  -14.6909
    0.4702    0.7749    0.0379   -1.2985  -17.0680   -4.2853  -15.5914
    0.4555    0.7743    0.0883   -1.2947  -13.6779   -4.2634  -13.9379
    0.4426    0.7749    0.0645   -1.2964  -14.9375   -4.2664  -17.9681
    0.4368    0.7703    0.0567   -1.2820  -15.4258   -4.2218  -14.9510
    0.5214    0.7683    0.0973   -1.3017  -13.2976   -4.2886  -14.8822
    0.4993    0.7726    0.0587   -1.2950  -15.3134   -4.2684  -15.5821
    0.4718    0.7717    0.0732   -1.2933  -14.4261   -4.2634  -13.9943
    0.4344    0.7745    0.1147   -1.2990  -12.6316   -4.2794  -14.0876
    0.4228    0.7734    0.0919   -1.2938  -13.5140   -4.2603  -14.6621
    0.4198    0.7709    0.0635   -1.2860  -14.9840   -4.2317  -15.3326
    0.5396    0.7661    0.1297   -1.2967  -12.1307   -4.3031  -14.1738
    0.5026    0.7685    0.0735   -1.2817  -14.3844   -4.2140  -14.0737
    0.4755    0.7706    0.0803   -1.2876  -14.0442   -4.2379  -14.6789
    0.4314    0.7678    0.0984   -1.2854  -13.2205   -4.2325  -14.7537
    0.4206    0.7698    0.0808   -1.2890  -14.0230   -4.2465  -15.2451
    0.4074    0.7685    0.0839   -1.2804  -13.8548   -4.2169  -15.7356
    0.5475    0.7832    0.1654   -1.3085  -11.1728   -4.3408  -13.2009
    0.5120    0.7705    0.1743   -1.2905  -10.9249   -4.2300  -13.3240
    0.5026    0.7725    0.1612   -1.2875  -11.2354   -4.2394  -13.8347
    0.5040    0.7808    0.1728   -1.2994  -10.9784   -4.2779  -15.7416
    0.4860    0.7782    0.1565   -1.2940  -11.3693   -4.2688  -14.5198
    0.4791    0.7669    0.1715   -1.2803  -10.9700   -4.2123  -13.9277
    0.5213    0.7665    0.0573   -1.2836  -15.3847   -4.2178  -13.2027
    0.4972    0.7700    0.0981   -1.2869  -13.2355   -4.2341  -14.7132
    0.4779    0.7736    0.0325   -1.2954  -17.6798   -4.2711  -15.5806
    0.4467    0.7750    0.0900   -1.2928  -13.5972   -4.2744  -15.3959
    0.4219    0.7705    0.0761   -1.2875  -14.2568   -4.2437  -15.8281
    0.4012    0.7692    0.0681   -1.2803  -14.6885   -4.2187  -15.7507
    0.5248    0.7699    0.1011   -1.2975  -13.1359   -4.2911  -15.1084
    0.5103    0.7750    0.0714   -1.2976  -14.5344   -4.2832  -15.1283
    0.4800    0.7719    0.0280   -1.2922  -18.2708   -4.2551  -16.1660
    0.4551    0.7726    0.0763   -1.2953  -14.2607   -4.2649  -18.2272
    0.4267    0.7705    0.0415   -1.2912  -16.6937   -4.2510  -16.4639
    0.4401    0.7706    0.0583   -1.2825  -15.3163   -4.2217  -18.1015
    0.5006    0.7514    0.0858   -1.2818  -13.7652   -4.2366  -15.0401
    0.5017    0.7706    0.0720   -1.2983  -14.5038   -4.2880  -14.0365
    0.4736    0.7764    0.0722   -1.3043  -14.5004   -4.3107  -15.2247
    0.4569    0.7736    0.0788   -1.2952  -14.1315   -4.2678  -16.2889
    0.4419    0.7732    0.0654   -1.2927  -14.8785   -4.2611  -16.2893
    0.4135    0.7700    0.0500   -1.2823  -15.9339   -4.2245  -15.8291
    0.5122    0.7612    0.0872   -1.2978  -13.7297   -4.2681  -13.1234
    0.4953    0.7646    0.1018   -1.2842  -13.0834   -4.2356  -14.1815
    0.4604    0.7681    0.0794   -1.2915  -14.0946   -4.2491  -15.7449
    0.4586    0.7718    0.0897   -1.2959  -13.6165   -4.2635  -15.7575
    0.4420    0.7740    0.0637   -1.2938  -14.9813   -4.2574  -15.9135
    0.4245    0.7668    0.0374   -1.2834  -17.0999   -4.2213  -15.2795
    0.5230    0.7638    0.1496   -1.2923  -11.5472   -4.2554  -13.7250
    0.4952    0.7712    0.0997   -1.2899  -13.1793   -4.2482  -15.1232
    0.4675    0.7735    0.0711   -1.2948  -14.5437   -4.2632  -14.1668
    0.4267    0.7672    0.1138   -1.2904  -12.6476   -4.2470  -16.5940
    0.4010    0.7668    0.0667   -1.2877  -14.7872   -4.2393  -15.2887
    0.4189    0.7694    0.0519   -1.2849  -15.7854   -4.2348  -15.1935
    0.5355    0.7771    0.1569   -1.3045  -11.3774   -4.3190  -14.2840
    0.5098    0.7811    0.0852   -1.2926  -13.8145   -4.2514  -15.3912
    0.4775    0.7797    0.0968   -1.2945  -13.3057   -4.2533  -15.5160
    0.4410    0.7734    0.0663   -1.2951  -14.8240   -4.2794  -13.9293
    0.4407    0.7776    0.0697   -1.2926  -14.6232   -4.2632  -15.9334
    0.4124    0.7703    0.0722   -1.2846  -14.4618   -4.2342  -16.9998
    0.5266    0.7610    0.1450   -1.2954  -11.6787   -4.2609  -16.2103
    0.5105    0.7690    0.0961   -1.2902  -13.3244   -4.2298  -13.2188
    0.5055    0.7740    0.0544   -1.2917  -15.6154   -4.2598  -14.5806
    0.5058    0.7796    0.1208   -1.2997  -12.4269   -4.2812  -14.3519
    0.4872    0.7747    0.0833   -1.2940  -13.9064   -4.2652  -14.4453
    0.4816    0.7663    0.1175   -1.2807  -12.4982   -4.2203  -15.4410
];
s.bio(:,:,2) = [0.7511    0.7205    0.1027   -1.3663  -13.2129  -13.9059  -15.1997
    0.7445    0.7250    0.0832   -1.3732  -14.0698  -15.3519  -16.9328
    0.7312    0.7214    0.0511   -1.3636  -16.0095  -14.9775  -16.1765
    0.7294    0.7271    0.0638   -1.3742  -15.1373  -15.7424  -16.0550
    0.7030    0.7224    0.0558   -1.3662  -15.6617  -15.8620  -15.6479
    0.6781    0.7194    0.0541   -1.3598  -15.7686  -15.5377  -16.3385
    0.7295    0.7173    0.0760   -1.3676  -14.4208  -13.5665  -14.3798
    0.7123    0.7189    0.0455   -1.3654  -16.4781  -13.7851  -15.7617
    0.6689    0.7188    0.0635   -1.3642  -15.1342  -14.9986  -16.1794
    0.6317    0.7157    0.0361   -1.3609  -17.3915  -14.6327  -15.1305
    0.6178    0.7195    0.0282   -1.3681  -18.3925  -14.2972  -15.9315
    0.5855    0.7161    0.0699   -1.3611  -14.7454  -14.2985  -16.1674
    0.7440    0.7195    0.0871   -1.3699  -13.8795  -14.2412  -14.2734
    0.7236    0.7211    0.0698   -1.3682  -14.7653  -15.3185  -15.9472
    0.6700    0.7212    0.0407   -1.3680  -16.9229  -14.9158  -15.3323
    0.6262    0.7198    0.0518   -1.3680  -15.9599  -14.7155  -14.9184
    0.5903    0.7191    0.0519   -1.3674  -15.9497  -14.4535  -16.4454
    0.5887    0.7173    0.0666   -1.3626  -14.9431  -14.9764  -15.6915
    0.7364    0.7189    0.0703   -1.3699  -14.7401  -15.6887  -15.6683
    0.7020    0.7218    0.0782   -1.3702  -14.3125  -14.9440  -16.2512
    0.6956    0.7192    0.0394   -1.3669  -17.0568  -15.2226  -15.6459
    0.6558    0.7203    0.0695   -1.3693  -14.7849  -14.5016  -17.5481
    0.6241    0.7176    0.0316   -1.3662  -17.9420  -14.9129  -16.5681
    0.6048    0.7156    0.0743   -1.3612  -14.5022  -14.7470  -17.2344
    0.7527    0.7211    0.0682   -1.3702  -14.8631  -14.5631  -16.5876
    0.7228    0.7194    0.0429   -1.3660  -16.7114  -15.4653  -16.6983
    0.6832    0.7208    0.0518   -1.3677  -15.9568  -14.9602  -16.7704
    0.6755    0.7199    0.0469   -1.3670  -16.3541  -15.2632  -16.5920
    0.6351    0.7183    0.0409   -1.3646  -16.9028  -14.3601  -16.3753
    0.6209    0.7147    0.0748   -1.3595  -14.4704  -14.9169  -17.1519
    0.7454    0.7216    0.1000   -1.3712  -13.3303  -14.1556  -15.7146
    0.7112    0.7210    0.0488   -1.3692  -16.1993  -14.6546  -17.3792
    0.6658    0.7198    0.0612   -1.3671  -15.2915  -14.3768  -15.5562
    0.6521    0.7222    0.0659   -1.3709  -14.9994  -15.0038  -15.9406
    0.6229    0.7200    0.0585   -1.3683  -15.4719  -15.2132  -16.0114
    0.6192    0.7181    0.0685   -1.3630  -14.8307  -14.9749  -15.4132
    0.7426    0.7199    0.1119   -1.3706  -12.8756  -15.6862  -15.8311
    0.7183    0.7205    0.0549   -1.3663  -15.7232  -16.3530  -16.4462
    0.6682    0.7216    0.0358   -1.3694  -17.4476  -14.8600  -16.9388
    0.6415    0.7162    0.0660   -1.3597  -14.9741  -14.8944  -16.3945
    0.6048    0.7196    0.0521   -1.3678  -15.9408  -15.0434  -16.2887
    0.5959    0.7167    0.0337   -1.3602  -17.6725  -15.0068  -16.1112
    0.7479    0.7196    0.0727   -1.3665  -14.6001  -15.1010  -14.6973
    0.7405    0.7208    0.0660   -1.3641  -14.9819  -14.0029  -15.0483
    0.7279    0.7224    0.0686   -1.3668  -14.8337  -13.5575  -16.3626
    0.7305    0.7258    0.0462   -1.3713  -16.4248  -15.1768  -14.7307
    0.7003    0.7216    0.0475   -1.3647  -16.3034  -14.7441  -16.1539
    0.6784    0.7185    0.0652   -1.3591  -15.0206  -14.5209  -16.6233
    0.7467    0.7161    0.0823   -1.3590  -14.0871  -16.0544  -16.6979
    0.7205    0.7195    0.0764   -1.3657  -14.3992  -14.7053  -16.6481
    0.6594    0.7207    0.0562   -1.3692  -15.6392  -14.7558  -17.0163
    0.6359    0.7145    0.0811   -1.3573  -14.1408  -14.9565  -15.4676
    0.6197    0.7187    0.0617   -1.3657  -15.2581  -15.1929  -16.3474
    0.6134    0.7153    0.0754   -1.3588  -14.4375  -15.8532  -16.7966
    0.7510    0.7234    0.0563   -1.3767  -15.6437  -14.3612  -20.9221
    0.6981    0.7209    0.0618   -1.3670  -15.2519  -14.5342  -18.4242
    0.6503    0.7201    0.0630   -1.3665  -15.1751  -14.3309  -16.7665
    0.6317    0.7221    0.0600   -1.3727  -15.3821  -16.1851  -16.0877
    0.6061    0.7194    0.0626   -1.3665  -15.1980  -14.7616  -15.8471
    0.5931    0.7168    0.0875   -1.3615  -13.8439  -14.6282  -16.2247
    0.7424    0.7162    0.0706   -1.3639  -14.7112  -17.3802  -15.2180
    0.7233    0.7197    0.0758   -1.3648  -14.4284  -15.3111  -15.1915
    0.6680    0.7181    0.0627   -1.3651  -15.1927  -13.4520  -15.8010
    0.6612    0.7202    0.0679   -1.3677  -14.8749  -14.9046  -15.6704
    0.6251    0.7180    0.0497   -1.3655  -16.1231  -14.5353  -16.0132
    0.6033    0.7155    0.0770   -1.3596  -14.3545  -15.0105  -16.0667
    0.7546    0.7231    0.0816   -1.3700  -14.1434  -13.9780  -15.6116
    0.7051    0.7193    0.0595   -1.3659  -15.4039  -14.1383  -15.8501
    0.6859    0.7182    0.0535   -1.3646  -15.8255  -14.7217  -16.3810
    0.6681    0.7202    0.0336   -1.3686  -17.6950  -14.0564  -16.1477
    0.6289    0.7190    0.0480   -1.3671  -16.2616  -15.2196  -15.9073
    0.6124    0.7148    0.0717   -1.3590  -14.6406  -14.8618  -15.6644
    0.7433    0.7203    0.0894   -1.3688  -13.7730  -14.5481  -15.1133
    0.7123    0.7193    0.0555   -1.3657  -15.6770  -14.8658  -15.6931
    0.6768    0.7205    0.0360   -1.3668  -17.4142  -14.5649  -16.6000
    0.6634    0.7210    0.0657   -1.3690  -15.0093  -13.5008  -15.3640
    0.6250    0.7197    0.0537   -1.3667  -15.8144  -14.1074  -15.5722
    0.6221    0.7177    0.0589   -1.3614  -15.4351  -14.3218  -16.1039
    0.7424    0.7158    0.0874   -1.3613  -13.8500  -16.2440  -14.6834
    0.7180    0.7199    0.0719   -1.3665  -14.6437  -13.6448  -15.5755
    0.6784    0.7213    0.0658   -1.3694  -15.0024  -14.6047  -15.7532
    0.6563    0.7175    0.0510   -1.3609  -16.0086  -13.9709  -15.2507
    0.6157    0.7202    0.0452   -1.3681  -16.5043  -14.7027  -15.6210
    0.6149    0.7173    0.0513   -1.3617  -15.9852  -14.5448  -15.9248
    0.7482    0.7202    0.0876   -1.3722  -13.8637  -13.8089  -15.3485
    0.7373    0.7213    0.0874   -1.3699  -13.8662  -14.5096  -16.3121
    0.7295    0.7234    0.0407   -1.3691  -16.9270  -19.7580  -17.8726
    0.7305    0.7255    0.0565   -1.3730  -15.6210  -13.8766  -15.6702
    0.7017    0.7219    0.0440   -1.3669  -16.6146  -15.9388  -16.1544
    0.6707    0.7177    0.0631   -1.3601  -15.1518  -14.9012  -16.4757
];
s.bio(:,:,3) = [0.6863    0.8084    0.0947   -1.4470  -13.6982  -13.2164  -10.9711
    0.6722    0.8023    0.0897   -1.4364  -13.8965  -13.4954  -10.7419
    0.6668    0.8009    0.0912   -1.4332  -13.8215  -13.3034  -10.8179
    0.6646    0.8013    0.1021   -1.4344  -13.3734  -13.2923  -10.6966
    0.6626    0.8009    0.0983   -1.4336  -13.5244  -13.3293  -10.7263
    0.6585    0.7972    0.0982   -1.4271  -13.5150  -13.3773  -10.6115
    0.6595    0.7825    0.0927   -1.4340  -13.7583  -13.1465  -10.6452
    0.6444    0.7849    0.0898   -1.4309  -13.8797  -13.3722  -10.7124
    0.6330    0.7865    0.0979   -1.4337  -13.5406  -13.2857  -10.6516
    0.6355    0.7855    0.0963   -1.4301  -13.5985  -13.1625  -10.6266
    0.6068    0.7874    0.0969   -1.4340  -13.5821  -13.4445  -10.6822
    0.5606    0.7845    0.0906   -1.4284  -13.8403  -13.3076  -10.6366
    0.6563    0.7867    0.0974   -1.4389  -13.5696  -13.1486  -10.5496
    0.6454    0.7852    0.0971   -1.4319  -13.5673  -13.2071  -10.6523
    0.6392    0.7897    0.0966   -1.4360  -13.5978  -13.2354  -10.7147
    0.6323    0.7902    0.0951   -1.4360  -13.6615  -13.4888  -10.6946
    0.6094    0.7881    0.0919   -1.4332  -13.7923  -13.4420  -10.6821
    0.6093    0.7880    0.1035   -1.4279  -13.3043  -13.3418  -10.6392
    0.6379    0.7777    0.0964   -1.4331  -13.5990  -12.8198  -10.7618
    0.6101    0.7742    0.1145   -1.4301  -12.9016  -12.9205  -10.5266
    0.6145    0.7824    0.1064   -1.4306  -13.1967  -13.1754  -10.5963
    0.6041    0.7834    0.0949   -1.4319  -13.6593  -13.6562  -10.6878
    0.5882    0.7841    0.0999   -1.4308  -13.4519  -13.2939  -10.6452
    0.5605    0.7811    0.1127   -1.4264  -12.9601  -13.3423  -10.6083
    0.6350    0.7781    0.0948   -1.4368  -13.6725  -13.1303  -10.6719
    0.6228    0.7812    0.1031   -1.4353  -13.3326  -13.0182  -10.6231
    0.6196    0.7825    0.1003   -1.4339  -13.4412  -13.3151  -10.6320
    0.6070    0.7844    0.0910   -1.4334  -13.8317  -12.9171  -10.6589
    0.5916    0.7848    0.1002   -1.4347  -13.4474  -13.3301  -10.7030
    0.5625    0.7825    0.1118   -1.4282  -12.9959  -13.1981  -10.6021
    0.6497    0.7822    0.0955   -1.4322  -13.6359  -12.6784  -10.5193
    0.6451    0.7874    0.0843   -1.4353  -14.1426  -13.1842  -10.6971
    0.6376    0.7877    0.1055   -1.4328  -13.2368  -13.2723  -10.7055
    0.6340    0.7881    0.0740   -1.4322  -14.6598  -13.6355  -10.7741
    0.6038    0.7887    0.0975   -1.4330  -13.5529  -13.3222  -10.6870
    0.6028    0.7880    0.1017   -1.4292  -13.3764  -13.3403  -10.6629
    0.6735    0.7872    0.1447   -1.4376  -11.9732  -12.8260  -10.8166
    0.6627    0.7863    0.0935   -1.4286  -13.7143  -13.0419  -10.5533
    0.6482    0.7890    0.1094   -1.4334  -13.0935  -13.2895  -10.6538
    0.6464    0.7887    0.1191   -1.4343  -12.7531  -13.1176  -10.6244
    0.6083    0.7887    0.1016   -1.4332  -13.3884  -13.4901  -10.6568
    0.5705    0.7870    0.0906   -1.4277  -13.8369  -13.4097  -10.6295
    0.6807    0.7886    0.0924   -1.4371  -13.7766  -12.6145  -10.7993
    0.6545    0.7837    0.0983   -1.4285  -13.5127  -13.4134  -10.5508
    0.6462    0.7785    0.0907   -1.4256  -13.8294  -14.1362  -10.6906
    0.6597    0.7914    0.0964   -1.4321  -13.5977  -13.2498  -10.7191
    0.6524    0.7883    0.0968   -1.4314  -13.5790  -13.8756  -10.7338
    0.6437    0.7797    0.1007   -1.4221  -13.4029  -13.3987  -10.5717
    0.6563    0.7796    0.0882   -1.4311  -13.9517  -13.0866  -10.7411
    0.6438    0.7822    0.0895   -1.4310  -13.8943  -13.3120  -10.6031
    0.6306    0.7862    0.0936   -1.4346  -13.7231  -13.5862  -10.6792
    0.6310    0.7877    0.0974   -1.4343  -13.5591  -13.4451  -10.6466
    0.5984    0.7852    0.0954   -1.4323  -13.6405  -13.3653  -10.6504
    0.5556    0.7803    0.0882   -1.4257  -13.9402  -13.4041  -10.6044
    0.6403    0.7806    0.1215   -1.4298  -12.6633  -13.3549  -10.6737
    0.6304    0.7875    0.1034   -1.4367  -13.3243  -13.1559  -10.7102
    0.6164    0.7822    0.1047   -1.4326  -13.2685  -13.4933  -10.7319
    0.6123    0.7853    0.1001   -1.4341  -13.4517  -13.4250  -10.6590
    0.6027    0.7884    0.1023   -1.4333  -13.3620  -13.2956  -10.6995
    0.5935    0.7873    0.1059   -1.4291  -13.2134  -13.3423  -10.6460
    0.6427    0.7767    0.0965   -1.4311  -13.5916  -13.3806  -10.5809
    0.6317    0.7813    0.0983   -1.4330  -13.5200  -13.0595  -10.6251
    0.6214    0.7837    0.0891   -1.4317  -13.9142  -13.4389  -10.6479
    0.6107    0.7814    0.1016   -1.4271  -13.3778  -13.0063  -10.5713
    0.6050    0.7858    0.0984   -1.4332  -13.5175  -13.2460  -10.6725
    0.5712    0.7828    0.1155   -1.4269  -12.8611  -13.3426  -10.5968
    0.6404    0.7802    0.1005   -1.4382  -13.4416  -13.1014  -10.7522
    0.6201    0.7810    0.1063   -1.4324  -13.2055  -13.5608  -10.7291
    0.6124    0.7803    0.1019   -1.4345  -13.3782  -13.4161  -10.6716
    0.6095    0.7838    0.1102   -1.4333  -13.0621  -13.7732  -10.7202
    0.5998    0.7861    0.0933   -1.4343  -13.7351  -13.2745  -10.6959
    0.5663    0.7832    0.1085   -1.4274  -13.1148  -13.3587  -10.6113
    0.6365    0.7815    0.1119   -1.4326  -12.9996  -12.9254  -10.6223
    0.6217    0.7805    0.1044   -1.4310  -13.2754  -13.0974  -10.6712
    0.6253    0.7854    0.1010   -1.4324  -13.4125  -13.2156  -10.6453
    0.6202    0.7875    0.1055   -1.4344  -13.2408  -13.1520  -10.6372
    0.5996    0.7877    0.1045   -1.4322  -13.2730  -13.5249  -10.6930
    0.5899    0.7860    0.1059   -1.4273  -13.2124  -13.3489  -10.6080
    0.6701    0.7902    0.1609   -1.4359  -11.5392  -13.0200  -10.5764
    0.6540    0.7916    0.0958   -1.4332  -13.6235  -13.5465  -10.6825
    0.6439    0.7908    0.1029   -1.4326  -13.3379  -13.3873  -10.6317
    0.6421    0.7904    0.1009   -1.4322  -13.4132  -13.4627  -10.7859
    0.5831    0.7894    0.0935   -1.4316  -13.7207  -13.4984  -10.6386
    0.6068    0.7861    0.0950   -1.4265  -13.6443  -13.4587  -10.6207
    0.6712    0.7826    0.0992   -1.4343  -13.4861  -12.1991  -10.7172
    0.6563    0.7859    0.0929   -1.4332  -13.7486  -12.7409  -10.6654
    0.6448    0.7809    0.0998   -1.4278  -13.4520  -12.4788  -10.5060
    0.6575    0.7916    0.0980   -1.4326  -13.5326  -13.2900  -10.7470
    0.6484    0.7868    0.1017   -1.4300  -13.3806  -12.3922  -10.6257
    0.6431    0.7819    0.0970   -1.4255  -13.5616  -13.5787  -10.5482
];
s.corrosive(:,:,1) = [0.5320    0.7760    0.4680   -1.3460   -6.8819   -4.5967   -8.9708
    0.5304    0.8010    0.4695   -1.3806   -6.9379   -4.6875   -9.7554
    0.5158    0.7955    0.4725   -1.3669   -6.8821   -4.6482   -9.6289
    0.5128    0.7955    0.4701   -1.3677   -6.9063   -4.6491   -9.6406
    0.4958    0.7940    0.4718   -1.3607   -6.8767   -4.6282   -9.5709
    0.4948    0.7943    0.4721   -1.3587   -6.8695   -4.6219   -9.6196
    0.5387    0.7711    0.4787   -1.3664   -6.8215   -4.6546   -9.2406
    0.5089    0.7727    0.4671   -1.3524   -6.9047   -4.6073   -9.2289
    0.4777    0.7736    0.4737   -1.3651   -6.8665   -4.6546   -9.5321
    0.4492    0.7790    0.4717   -1.3639   -6.8834   -4.6454   -9.5379
    0.4308    0.7792    0.4711   -1.3631   -6.8880   -4.6332   -9.5546
    0.4082    0.7775    0.4725   -1.3592   -6.8666   -4.6191   -9.6103
    0.5376    0.7723    0.4717   -1.3641   -6.8832   -4.6433   -9.1308
    0.4984    0.7731    0.4822   -1.3536   -6.7635   -4.6200   -9.2573
    0.4645    0.7768    0.4669   -1.3595   -6.9204   -4.6291   -9.3913
    0.4521    0.7749    0.4633   -1.3573   -6.9515   -4.6077   -9.3542
    0.4390    0.7800    0.4690   -1.3626   -6.9065   -4.6385   -9.4843
    0.4340    0.7781    0.4707   -1.3595   -6.8842   -4.6262   -9.4736
    0.5199    0.7778    0.4793   -1.3617   -6.8065   -4.6316   -9.3121
    0.4958    0.7758    0.4655   -1.3521   -6.9191   -4.5983   -9.2134
    0.4697    0.7746    0.4704   -1.3553   -6.8783   -4.6028   -9.4728
    0.4647    0.7812    0.4809   -1.3676   -6.8042   -4.6550   -9.6030
    0.4507    0.7838    0.4689   -1.3662   -6.9147   -4.6443   -9.5211
    0.4301    0.7781    0.4722   -1.3597   -6.8709   -4.6248   -9.4839
    0.5105    0.7705    0.4784   -1.3539   -6.7992   -4.6100   -9.1829
    0.4913    0.7740    0.4697   -1.3573   -6.8897   -4.6186   -9.3111
    0.4761    0.7776    0.4729   -1.3601   -6.8643   -4.6247   -9.3575
    0.4586    0.7775    0.4737   -1.3659   -6.8686   -4.6528   -9.4242
    0.4428    0.7776    0.4763   -1.3645   -6.8416   -4.6412   -9.5089
    0.4280    0.7787    0.4720   -1.3587   -6.8704   -4.6240   -9.4998
    0.5219    0.7702    0.4563   -1.3656   -7.0342   -4.6491   -9.0689
    0.4962    0.7754    0.4671   -1.3590   -6.9177   -4.6305   -9.2448
    0.4686    0.7784    0.4680   -1.3621   -6.9156   -4.6287   -9.4610
    0.4520    0.7780    0.4632   -1.3684   -6.9743   -4.6588   -9.3698
    0.4250    0.7767    0.4669   -1.3636   -6.9290   -4.6345   -9.4403
    0.4369    0.7792    0.4682   -1.3626   -6.9147   -4.6332   -9.4719
    0.5205    0.7693    0.4773   -1.3558   -6.8129   -4.6020   -9.1366
    0.5221    0.7798    0.4813   -1.3659   -6.7966   -4.6703   -9.4410
    0.4744    0.7744    0.4797   -1.3585   -6.7966   -4.6283   -9.4401
    0.4462    0.7794    0.4752   -1.3681   -6.8585   -4.6506   -9.6426
    0.4123    0.7820    0.4738   -1.3664   -6.8688   -4.6427   -9.5299
    0.4045    0.7782    0.4725   -1.3638   -6.8759   -4.6408   -9.6174
    0.5496    0.7986    0.4845   -1.3894   -6.8122   -4.7387   -9.6189
    0.5175    0.7864    0.4820   -1.3745   -6.8072   -4.6469   -9.7038
    0.5146    0.7870    0.4782   -1.3689   -6.8318   -4.6595   -9.5598
    0.5088    0.7839    0.4830   -1.3605   -6.7700   -4.6301   -9.6348
    0.4958    0.7872    0.4839   -1.3661   -6.7727   -4.6561   -9.6177
    0.4898    0.7766    0.4891   -1.3563   -6.7044   -4.6190   -9.6370
    0.5502    0.7808    0.4681   -1.3655   -6.9197   -4.6631   -9.2027
    0.5161    0.7802    0.4701   -1.3628   -6.8965   -4.6312   -9.3122
    0.4675    0.7761    0.4704   -1.3530   -6.8742   -4.6133   -9.3551
    0.4528    0.7866    0.4621   -1.3733   -6.9943   -4.6724   -9.5141
    0.4231    0.7830    0.4696   -1.3658   -6.9072   -4.6500   -9.5109
    0.1493    0.7808    0.4716   -1.3622   -6.8815   -4.6322   -9.5249
    0.5321    0.7684    0.4623   -1.3579   -6.9608   -4.6491   -9.1971
    0.5023    0.7716    0.4666   -1.3525   -6.9096   -4.6243   -9.2939
    0.4734    0.7748    0.4658   -1.3587   -6.9300   -4.6236   -9.3718
    0.4550    0.7836    0.4686   -1.3702   -6.9262   -4.6713   -9.5517
    0.4416    0.7774    0.4637   -1.3638   -6.9602   -4.6397   -9.4872
    0.4392    0.7835    0.4675   -1.3626   -6.9218   -4.6361   -9.5351
    0.5198    0.7700    0.4789   -1.3652   -6.8165   -4.6645   -9.1684
    0.5063    0.7727    0.4684   -1.3521   -6.8912   -4.6170   -9.2748
    0.4729    0.7789    0.4693   -1.3662   -6.9109   -4.6545   -9.3835
    0.4634    0.7803    0.4756   -1.3642   -6.8476   -4.6418   -9.4201
    0.4396    0.7772    0.4735   -1.3629   -6.8643   -4.6372   -9.5106
    0.4207    0.7715    0.4697   -1.3459   -6.8672   -4.5841   -9.6744
    0.5248    0.7730    0.4623   -1.3736   -6.9922   -4.6930   -9.0806
    0.5075    0.7767    0.4670   -1.3609   -6.9219   -4.6362   -9.4305
    0.4661    0.7683    0.4725   -1.3560   -6.8602   -4.6196   -9.3154
    0.1651    0.7806    0.4608   -1.3705   -7.0016   -4.6597   -9.4966
    0.4531    0.7818    0.4694   -1.3671   -6.9126   -4.6515   -9.5055
    0.1550    0.7774    0.4708   -1.3594   -6.8834   -4.6271   -9.4784
    0.5279    0.7717    0.4797   -1.3642   -6.8070   -4.6681   -9.2446
    0.4964    0.7779    0.4784   -1.3648   -6.8210   -4.6496   -9.4328
    0.4818    0.7825    0.4735   -1.3692   -6.8766   -4.6620   -9.5093
    0.4713    0.7820    0.4748   -1.3666   -6.8594   -4.6473   -9.5869
    0.4383    0.7816    0.4701   -1.3661   -6.9031   -4.6477   -9.5382
    0.4461    0.7827    0.4737   -1.3634   -6.8640   -4.6395   -9.5655
    0.5208    0.7780    0.4640   -1.3557   -6.9396   -4.6184   -9.0359
    0.5096    0.7820    0.4630   -1.3597   -6.9581   -4.6233   -9.1931
    0.4690    0.7825    0.4595   -1.3646   -7.0022   -4.6370   -9.3726
    0.4571    0.7895    0.4665   -1.3709   -6.9472   -4.6659   -9.5719
    0.4345    0.7874    0.4633   -1.3670   -6.9706   -4.6495   -9.4824
    0.4100    0.7844    0.4662   -1.3647   -6.9376   -4.6484   -9.4568
    0.5224    0.7690    0.4590   -1.3597   -6.9962   -4.6213   -8.8537
    0.5175    0.7727    0.4603   -1.3553   -6.9756   -4.6165   -9.4160
    0.5013    0.7723    0.4682   -1.3634   -6.9161   -4.6177   -9.5347
    0.5011    0.7816    0.4680   -1.3668   -6.9246   -4.6378   -9.5071
    0.4889    0.7826    0.4612   -1.3668   -6.9908   -4.6387   -9.4764
    0.4886    0.7793    0.4820   -1.3641   -6.7865   -4.6475   -9.6671

];
s.corrosive(:,:,2) = [0.5316    0.7715    0.3985   -1.3497   -7.5980   -4.5943   -8.4542
    0.5231    0.7777    0.4195   -1.3527   -7.3813   -4.6001   -8.6206
    0.5102    0.7780    0.4219   -1.3502   -7.3516   -4.5987   -8.6668
    0.5060    0.7753    0.4162   -1.3473   -7.4055   -4.5791   -8.5677
    0.4835    0.7719    0.4204   -1.3447   -7.3569   -4.5808   -8.5920
    0.4861    0.7731    0.4305   -1.3388   -7.2403   -4.5702   -8.5457
    0.5470    0.7500    0.4301   -1.3640   -7.2929   -4.6816   -8.4807
    0.5132    0.7553    0.4280   -1.3441   -7.2756   -4.5897   -8.4857
    0.4726    0.7486    0.4323   -1.3462   -7.2367   -4.6040   -8.5294
    0.4449    0.7564    0.4321   -1.3470   -7.2399   -4.5905   -8.6339
    0.4285    0.7545    0.4271   -1.3449   -7.2882   -4.5916   -8.5405
    0.4018    0.7530    0.4283   -1.3424   -7.2701   -4.5836   -8.5765
    0.2883    0.7603    0.4205   -1.3765   -7.4172   -4.6806   -8.4895
    0.4994    0.7516    0.4165   -1.3428   -7.3934   -4.5609   -8.4243
    0.4609    0.7519    0.4146   -1.3410   -7.4096   -4.5655   -8.4733
    0.4349    0.7526    0.4210   -1.3441   -7.3493   -4.5623   -8.6730
    0.4144    0.7572    0.4237   -1.3461   -7.3256   -4.5776   -8.5864
    0.4172    0.7563    0.4257   -1.3442   -7.3010   -4.5721   -8.5951
    0.5159    0.7516    0.4143   -1.3487   -7.4265   -4.5878   -8.3018
    0.4761    0.7507    0.4148   -1.3452   -7.4161   -4.5693   -8.3020
    0.4627    0.7613    0.4302   -1.3469   -7.2590   -4.5719   -8.5936
    0.4409    0.7590    0.4272   -1.3535   -7.3034   -4.6165   -8.4863
    0.4240    0.7583    0.4290   -1.3501   -7.2780   -4.5958   -8.6241
    0.4177    0.7566    0.4323   -1.3414   -7.2275   -4.5721   -8.5303
    0.5121    0.7616    0.4273   -1.3684   -7.3302   -4.6662   -8.5573
    0.4917    0.7543    0.4224   -1.3477   -7.3410   -4.6148   -8.4904
    0.4573    0.7513    0.4416   -1.3369   -7.1238   -4.5662   -8.4782
    0.4462    0.7592    0.4267   -1.3487   -7.2992   -4.6159   -8.6248
    0.4312    0.7561    0.4367   -1.3447   -7.1887   -4.5952   -8.6395
    0.4175    0.7538    0.4386   -1.3393   -7.1596   -4.5751   -8.5769
    0.5180    0.7522    0.4291   -1.3644   -7.3039   -4.6385   -8.5277
    0.4898    0.7575    0.4407   -1.3495   -7.1578   -4.5931   -8.5222
    0.4717    0.7585    0.4247   -1.3536   -7.3291   -4.6151   -8.5696
    0.4651    0.7590    0.4298   -1.3447   -7.2589   -4.5765   -8.7043
    0.4368    0.7549    0.4249   -1.3475   -7.3158   -4.5951   -8.5740
    0.4407    0.7568    0.4247   -1.3444   -7.3117   -4.5782   -8.5752
    0.5393    0.7541    0.4117   -1.3701   -7.4966   -4.6979   -8.3868
    0.4972    0.7531    0.4380   -1.3467   -7.1793   -4.5977   -8.5622
    0.4695    0.7543    0.4363   -1.3499   -7.2037   -4.6268   -8.5464
    0.4403    0.7549    0.3926   -1.3490   -7.6629   -4.5849   -8.5191
    0.4143    0.7556    0.4341   -1.3494   -7.2247   -4.5976   -8.6612
    0.4022    0.7550    0.4365   -1.3464   -7.1943   -4.5863   -8.6184
    0.5599    0.7712    0.4617   -1.3657   -6.9815   -4.7092   -8.5086
    0.5206    0.7626    0.4174   -1.3558   -7.4099   -4.6185   -8.4620
    0.5036    0.7661    0.4120   -1.3442   -7.4436   -4.5687   -8.6387
    0.5038    0.7724    0.4274   -1.3595   -7.3140   -4.6283   -8.6038
    0.4882    0.7739    0.4360   -1.3595   -7.2256   -4.6283   -8.6990
    0.4821    0.7585    0.4171   -1.3397   -7.3815   -4.5659   -8.4765
    0.5280    0.7568    0.4637   -1.3656   -6.9618   -4.6712   -8.4805
    0.4984    0.7567    0.4273   -1.3468   -7.2888   -4.6038   -8.5076
    0.4739    0.7611    0.4284   -1.3486   -7.2815   -4.6110   -8.5584
    0.4463    0.7563    0.4376   -1.3407   -7.1722   -4.5630   -8.5719
    0.4310    0.7598    0.4306   -1.3469   -7.2553   -4.5893   -8.6008
    0.4028    0.7582    0.4318   -1.3433   -7.2367   -4.5840   -8.5750
    0.2843    0.7574    0.4144   -1.3716   -7.4716   -4.6854   -8.4010
    0.5046    0.7592    0.4170   -1.3517   -7.4059   -4.5970   -8.5004
    0.4758    0.7623    0.4285   -1.3516   -7.2863   -4.6086   -8.6276
    0.4465    0.7600    0.4379   -1.3454   -7.1788   -4.5883   -8.6558
    0.4267    0.7621    0.4290   -1.3512   -7.2808   -4.6130   -8.6166
    0.4357    0.7646    0.4270   -1.3494   -7.2980   -4.5988   -8.5709
    0.5301    0.7590    0.4595   -1.3730   -7.0176   -4.7071   -8.6584
    0.4945    0.7598    0.4261   -1.3573   -7.3218   -4.6293   -8.4786
    0.4627    0.7532    0.4350   -1.3474   -7.2118   -4.6119   -8.5679
    0.4472    0.7604    0.4199   -1.3488   -7.3704   -4.5757   -8.5215
    0.4263    0.7541    0.4322   -1.3472   -7.2403   -4.5904   -8.6415
    0.4275    0.7530    0.4315   -1.3374   -7.2271   -4.5705   -8.5416
    0.5266    0.7513    0.4381   -1.3635   -7.2106   -4.6483   -8.6058
    0.5014    0.7508    0.4402   -1.3450   -7.1535   -4.5852   -8.5531
    0.4889    0.7590    0.4350   -1.3628   -7.2423   -4.6603   -8.6066
    0.4616    0.7521    0.4317   -1.3411   -7.2323   -4.5642   -8.5006
    0.4324    0.7516    0.4289   -1.3493   -7.2783   -4.6032   -8.5986
    0.4162    0.7504    0.4369   -1.3421   -7.1819   -4.5827   -8.5885
    0.5367    0.7580    0.4455   -1.3722   -7.1536   -4.6748   -8.5742
    0.5128    0.7630    0.4336   -1.3619   -7.2548   -4.6343   -8.6175
    0.4771    0.7636    0.4266   -1.3562   -7.3153   -4.6200   -8.5926
    0.4285    0.7566    0.4382   -1.3506   -7.1851   -4.6036   -8.7881
    0.4186    0.7571    0.4297   -1.3513   -7.2732   -4.6052   -8.6211
    0.4298    0.7598    0.4305   -1.3478   -7.2584   -4.5970   -8.6238
    0.5353    0.7795    0.4666   -1.3643   -6.9310   -4.6596   -8.7427
    0.5123    0.7783    0.4449   -1.3668   -7.1501   -4.6828   -8.6369
    0.4493    0.7548    0.4324   -1.3519   -7.2471   -4.6250   -8.5285
    0.4325    0.7594    0.4465   -1.3523   -7.1055   -4.6174   -8.6603
    0.4222    0.7587    0.4402   -1.3570   -7.1786   -4.6414   -8.6043
    0.4205    0.7596    0.4363   -1.3505   -7.2046   -4.6088   -8.5557
    0.5417    0.7602    0.4172   -1.3635   -7.4256   -4.6150   -8.4092
    0.5219    0.7675    0.4017   -1.3665   -7.5987   -4.6422   -8.4348
    0.5044    0.7620    0.4236   -1.3443   -7.3227   -4.5811   -8.5372
    0.4971    0.7590    0.4036   -1.3481   -7.5413   -4.5579   -8.4459
    0.4795    0.7603    0.4158   -1.3422   -7.3998   -4.5573   -8.4388
    0.4790    0.7593    0.4274   -1.3412   -7.2774   -4.5751   -8.5348

];
s.corrosive(:,:,3) = [0.0846    0.6744    0.8127   -0.9767   -3.3668   -3.4790   -6.5522
    0.0777    0.6723    0.8121   -0.9638   -3.3460   -3.4518   -6.6176
    0.0779    0.6847    0.8122   -0.9843   -3.3864   -3.5217   -6.6324
    0.0750    0.6784    0.8125   -0.9711   -3.3577   -3.4688   -6.5799
    0.0744    0.6821    0.8125   -0.9770   -3.3694   -3.4942   -6.6201
    0.0743    0.6827    0.8114   -0.9722   -3.3681   -3.4841   -6.6080
    0.0844    0.6491    0.8118   -0.9790   -3.3783   -3.4902   -6.6148
    0.0864    0.6580    0.8130   -0.9784   -3.3685   -3.4897   -6.5900
    0.0808    0.6607    0.8116   -0.9743   -3.3707   -3.4761   -6.5926
    0.0779    0.6577    0.8123   -0.9697   -3.3564   -3.4681   -6.5994
    0.0745    0.6607    0.8117   -0.9769   -3.3754   -3.4848   -6.5962
    0.0695    0.6568    0.8117   -0.9677   -3.3570   -3.4553   -6.5530
    0.0831    0.6486    0.8127   -0.9681   -3.3497   -3.4679   -6.5904
    0.0853    0.6527    0.8125   -0.9766   -3.3686   -3.4826   -6.5762
    0.0790    0.6593    0.8129   -0.9773   -3.3672   -3.4929   -6.6268
    0.0654    0.6525    0.8117   -0.9710   -3.3636   -3.4672   -6.6038
    0.0721    0.6609    0.8132   -0.9775   -3.3652   -3.4968   -6.6095
    0.0635    0.6539    0.8123   -0.9654   -3.3480   -3.4532   -6.5635
    0.0831    0.6521    0.8144   -0.9754   -3.3515   -3.4889   -6.5856
    0.0755    0.6567    0.8124   -0.9744   -3.3652   -3.4806   -6.6100
    0.0715    0.6583    0.8131   -0.9726   -3.3561   -3.4726   -6.5674
    0.0702    0.6594    0.8147   -0.9757   -3.3508   -3.4849   -6.6043
    0.0670    0.6610    0.8130   -0.9721   -3.3563   -3.4763   -6.5853
    0.0650    0.6566    0.8131   -0.9652   -3.3419   -3.4473   -6.5501
    0.0854    0.6529    0.8129   -0.9819   -3.3756   -3.4846   -6.5419
    0.0745    0.6543    0.8125   -0.9783   -3.3715   -3.5020   -6.6055
    0.0720    0.6563    0.8119   -0.9728   -3.3657   -3.4744   -6.5819
    0.0692    0.6563    0.8132   -0.9701   -3.3503   -3.4740   -6.5669
    0.0673    0.6598    0.8124   -0.9725   -3.3614   -3.4791   -6.5771
    0.0669    0.6540    0.8126   -0.9559   -3.3270   -3.4198   -6.4898
    0.0836    0.6527    0.8118   -0.9782   -3.3766   -3.4839   -6.6160
    0.0776    0.6633    0.8140   -0.9823   -3.3687   -3.5133   -6.6038
    0.0708    0.6588    0.8119   -0.9777   -3.3757   -3.4891   -6.6151
    0.0680    0.6587    0.8136   -0.9701   -3.3477   -3.4600   -6.5799
    0.0656    0.6619    0.8125   -0.9747   -3.3650   -3.4834   -6.5981
    0.0657    0.6589    0.8113   -0.9697   -3.3638   -3.4611   -6.5700
    0.0837    0.6556    0.8101   -0.9950   -3.4225   -3.5425   -6.6819
    0.0860    0.6619    0.8124   -0.9855   -3.3871   -3.5397   -6.6550
    0.0725    0.6584    0.8136   -0.9730   -3.3538   -3.4816   -6.5792
    0.0678    0.6597    0.8123   -0.9775   -3.3722   -3.4937   -6.5789
    0.0655    0.6618    0.8127   -0.9768   -3.3674   -3.4889   -6.5879
    0.0618    0.6570    0.8116   -0.9681   -3.3586   -3.4564   -6.5564
    0.0950    0.6792    0.8066   -1.0039   -3.4658   -3.6593   -6.8993
    0.0775    0.6644    0.8121   -0.9666   -3.3518   -3.4040   -6.4257
    0.0756    0.6683    0.8143   -0.9675   -3.3371   -3.4534   -6.5226
    0.0736    0.6665    0.8136   -0.9672   -3.3420   -3.4692   -6.5872
    0.0745    0.6719    0.8116   -0.9797   -3.3814   -3.5223   -6.6616
    0.0727    0.6622    0.8128   -0.9598   -3.3331   -3.4205   -6.5050
    0.1039    0.6597    0.8149   -0.9873   -3.3715   -3.5372   -6.6561
    0.0766    0.6608    0.8130   -0.9756   -3.3626   -3.4852   -6.6143
    0.0790    0.6605    0.8127   -0.9777   -3.3694   -3.4893   -6.6061
    0.0765    0.6621    0.8115   -0.9798   -3.3825   -3.4966   -6.6283
    0.0721    0.6614    0.8122   -0.9759   -3.3693   -3.4830   -6.5904
    0.0694    0.6594    0.8122   -0.9729   -3.3635   -3.4719   -6.5844
    0.1019    0.6533    0.8144   -0.9728   -3.3467   -3.4894   -6.5474
    0.0848    0.6570    0.8141   -0.9716   -3.3470   -3.4734   -6.5580
    0.0794    0.6587    0.8112   -0.9749   -3.3747   -3.4747   -6.5847
    0.0765    0.6576    0.8120   -0.9699   -3.3588   -3.4573   -6.5342
    0.0730    0.6608    0.8128   -0.9737   -3.3607   -3.4757   -6.5907
    0.0722    0.6589    0.8128   -0.9664   -3.3464   -3.4566   -6.5638
    0.0898    0.6460    0.8146   -0.9622   -3.3234   -3.4704   -6.5953
    0.0848    0.6574    0.8117   -0.9811   -3.3835   -3.5093   -6.6350
    0.0806    0.6575    0.8116   -0.9687   -3.3592   -3.4695   -6.5879
    0.0809    0.6620    0.8143   -0.9733   -3.3492   -3.4869   -6.5975
    0.0759    0.6604    0.8121   -0.9731   -3.3645   -3.4786   -6.6048
    0.0748    0.6578    0.8124   -0.9635   -3.3435   -3.4450   -6.5473
    0.0836    0.6519    0.8111   -0.9859   -3.3972   -3.5113   -6.5833
    0.0757    0.6575    0.8103   -0.9748   -3.3809   -3.4732   -6.5999
    0.0710    0.6543    0.8121   -0.9691   -3.3569   -3.4652   -6.5792
    0.0699    0.6609    0.8138   -0.9715   -3.3493   -3.4718   -6.5745
    0.0669    0.6593    0.8123   -0.9723   -3.3618   -3.4736   -6.5978
    0.0653    0.6561    0.8126   -0.9617   -3.3383   -3.4340   -6.5230
    0.0828    0.6499    0.8100   -0.9767   -3.3867   -3.4674   -6.5805
    0.0849    0.6560    0.8134   -0.9691   -3.3470   -3.4705   -6.5595
    0.0715    0.6602    0.8109   -0.9792   -3.3856   -3.4986   -6.6099
    0.0675    0.6575    0.8084   -0.9743   -3.3945   -3.4534   -6.5641
    0.0657    0.6611    0.8119   -0.9736   -3.3673   -3.4737   -6.5828
    0.0653    0.6575    0.8119   -0.9645   -3.3488   -3.4428   -6.5488
    0.1023    0.6638    0.8160   -0.9659   -3.3205   -3.4435   -6.5000
    0.0769    0.6680    0.8124   -0.9779   -3.3719   -3.4985   -6.5765
    0.0812    0.6637    0.8108   -0.9743   -3.3764   -3.4879   -6.6242
    0.0779    0.6685    0.8122   -0.9821   -3.3820   -3.5138   -6.6220
    0.0742    0.6650    0.8118   -0.9803   -3.3811   -3.5067   -6.6262
    0.0730    0.6652    0.8112   -0.9729   -3.3709   -3.4795   -6.5867
    0.0932    0.6443    0.8095   -0.9582   -3.3536   -3.4256   -6.4631
    0.0788    0.6693    0.8143   -0.9816   -3.3652   -3.4908   -6.5793
    0.0746    0.6499    0.8157   -0.9529   -3.2981   -3.3870   -6.4520
    0.0737    0.6607    0.8115   -0.9692   -3.3616   -3.4611   -6.5943
    0.0721    0.6630    0.8128   -0.9717   -3.3570   -3.4664   -6.5557
    0.0712    0.6573    0.8120   -0.9637   -3.3465   -3.4351   -6.5343
];
s.eletric(:,:,1) = [0.5766    0.8471    0.3512   -1.4597   -8.3628   -4.8925  -10.1732
    0.5648    0.8573    0.3363   -1.4579   -8.5444   -4.8885  -10.0811
    0.5634    0.8608    0.3395   -1.4616   -8.5116   -4.9032  -10.0396
    0.5566    0.8590    0.3459   -1.4566   -8.4225   -4.8831  -10.0229
    0.5442    0.8614    0.3369   -1.4593   -8.5403   -4.8937   -9.9925
    0.5421    0.8597    0.3432   -1.4522   -8.4466   -4.8726   -9.9530
    0.5550    0.8196    0.3656   -1.4442   -8.1598   -4.8407   -9.9920
    0.5493    0.8344    0.3432   -1.4447   -8.4309   -4.8421   -9.9279
    0.5093    0.8389    0.3397   -1.4532   -8.4922   -4.8618  -10.0091
    0.5062    0.8431    0.3217   -1.4548   -8.7255   -4.8760  -10.0751
    0.4724    0.8385    0.3467   -1.4524   -8.4038   -4.8705   -9.9898
    0.4581    0.8404    0.3424   -1.4502   -8.4533   -4.8625   -9.9579
    0.5663    0.8337    0.3331   -1.4552   -8.5782   -4.8803   -9.8545
    0.5440    0.8384    0.3436   -1.4558   -8.4483   -4.8725   -9.9326
    0.5224    0.8399    0.3344   -1.4547   -8.5625   -4.8761  -10.0217
    0.5101    0.8451    0.3271   -1.4567   -8.6594   -4.8836   -9.9793
    0.4932    0.8444    0.3337   -1.4555   -8.5724   -4.8796   -9.9740
    0.4824    0.8457    0.3418   -1.4501   -8.4606   -4.8635   -9.9224
    0.5599    0.8393    0.3521   -1.4522   -8.3372   -4.8601  -10.0305
    0.5222    0.8365    0.3482   -1.4536   -8.3873   -4.8646   -9.9638
    0.5138    0.8394    0.3388   -1.4562   -8.5097   -4.8897  -10.0287
    0.5023    0.8441    0.3452   -1.4513   -8.4197   -4.8698   -9.9391
    0.4847    0.8415    0.3438   -1.4539   -8.4427   -4.8782   -9.9883
    0.4599    0.8434    0.3404   -1.4511   -8.4792   -4.8684   -9.9543
    0.5778    0.8379    0.3419   -1.4580   -8.4735   -4.9015   -9.9394
    0.5501    0.8437    0.3262   -1.4546   -8.6668   -4.8747   -9.9458
    0.5173    0.8470    0.3348   -1.4605   -8.5689   -4.9018  -10.0192
    0.5066    0.8471    0.3304   -1.4578   -8.6195   -4.8811  -10.1759
    0.4823    0.8449    0.3373   -1.4569   -8.5294   -4.8790  -10.0136
    0.4589    0.8444    0.3412   -1.4506   -8.4690   -4.8611  -10.0016
    0.5688    0.8327    0.3386   -1.4569   -8.5126   -4.8877   -9.8490
    0.5431    0.8438    0.3359   -1.4568   -8.5470   -4.8834  -10.0485
    0.5125    0.8442    0.3398   -1.4575   -8.5001   -4.8854   -9.9651
    0.5092    0.8448    0.3310   -1.4593   -8.6145   -4.9000  -10.0468
    0.4898    0.8452    0.3423   -1.4560   -8.4654   -4.8819  -10.0553
    0.4869    0.8480    0.3397   -1.4533   -8.4929   -4.8745  -10.0523
    0.5808    0.8376    0.3007   -1.4606   -9.0197   -4.9102   -9.7670
    0.5499    0.8402    0.3438   -1.4487   -8.4316   -4.8502  -10.0670
    0.5134    0.8423    0.3371   -1.4535   -8.5257   -4.8611   -9.9992
    0.4793    0.8405    0.3412   -1.4528   -8.4724   -4.8705   -9.9712
    0.4650    0.8415    0.3396   -1.4554   -8.4987   -4.8758  -10.0376
    0.4498    0.8423    0.3440   -1.4508   -8.4344   -4.8650   -9.9648
    0.5866    0.8485    0.3330   -1.4647   -8.5988   -4.9335   -9.8270
    0.5645    0.8413    0.3561   -1.4467   -8.2780   -4.8380  -10.0871
    0.5507    0.8455    0.3535   -1.4528   -8.3216   -4.8678   -9.9909
    0.5531    0.8535    0.3416   -1.4586   -8.4798   -4.8845   -9.9214
    0.5345    0.8505    0.3453   -1.4571   -8.4311   -4.8866   -9.9606
    0.5308    0.8402    0.3590   -1.4460   -8.2429   -4.8474   -9.9461
    0.5724    0.8361    0.3356   -1.4433   -8.5230   -4.8285   -9.9239
    0.5416    0.8418    0.3370   -1.4532   -8.5260   -4.8717  -10.0493
    0.5243    0.8484    0.3229   -1.4576   -8.7153   -4.8880  -10.0439
    0.4927    0.8444    0.3440   -1.4532   -8.4387   -4.8860   -9.9810
    0.4716    0.8449    0.3308   -1.4548   -8.6084   -4.8762   -9.9732
    0.4465    0.8449    0.3382   -1.4529   -8.5102   -4.8743   -9.9663
    0.5575    0.8301    0.3256   -1.4490   -8.6618   -4.8547   -9.8888
    0.5448    0.8443    0.3281   -1.4568   -8.6466   -4.8858  -10.0053
    0.5151    0.8447    0.3279   -1.4518   -8.6389   -4.8545   -9.9051
    0.4988    0.8456    0.3305   -1.4553   -8.6127   -4.8783  -10.0024
    0.4771    0.8438    0.3284   -1.4511   -8.6321   -4.8564   -9.9276
    0.4877    0.8463    0.3312   -1.4508   -8.5956   -4.8579   -9.9305
    0.5536    0.8259    0.3409   -1.4557   -8.4810   -4.8925   -9.9314
    0.5379    0.8448    0.3453   -1.4591   -8.4349   -4.8954  -10.0730
    0.5172    0.8410    0.3413   -1.4567   -8.4792   -4.8959  -10.0153
    0.5009    0.8457    0.3358   -1.4581   -8.5515   -4.8891  -10.0090
    0.4832    0.8408    0.3402   -1.4545   -8.4890   -4.8727  -10.0090
    0.4571    0.8433    0.3400   -1.4518   -8.4864   -4.8664   -9.9771
    0.5603    0.8314    0.3335   -1.4551   -8.5736   -4.8709   -9.8786
    0.5365    0.8386    0.3441   -1.4515   -8.4335   -4.8616   -9.9189
    0.5116    0.8392    0.3379   -1.4539   -8.5164   -4.8686  -10.0496
    0.5045    0.8461    0.3398   -1.4577   -8.4997   -4.8815  -10.1176
    0.4782    0.8443    0.3357   -1.4526   -8.5417   -4.8641   -9.9804
    0.4580    0.8415    0.3422   -1.4500   -8.4547   -4.8556   -9.9582
    0.5701    0.8354    0.3242   -1.4644   -8.7111   -4.9202  -10.0273
    0.5464    0.8428    0.3308   -1.4529   -8.6038   -4.8719  -10.0492
    0.5110    0.8423    0.3355   -1.4548   -8.5480   -4.8728  -10.0852
    0.4686    0.8397    0.3173   -1.4508   -8.7757   -4.8614   -9.9366
    0.4470    0.8389    0.3303   -1.4522   -8.6091   -4.8653   -9.9869
    0.4622    0.8423    0.3332   -1.4519   -8.5722   -4.8641   -9.9802
    0.5640    0.8459    0.2888   -1.4554   -9.1787   -4.8660   -9.8606
    0.5652    0.8559    0.3353   -1.4584   -8.5575   -4.8869   -9.9867
    0.5300    0.8527    0.3252   -1.4545   -8.6800   -4.8677   -9.9782
    0.5041    0.8476    0.3324   -1.4576   -8.5930   -4.8922  -10.0457
    0.4822    0.8529    0.3258   -1.4585   -8.6801   -4.8879  -10.0013
    0.4637    0.8452    0.3335   -1.4508   -8.5662   -4.8675   -9.8889
    0.5791    0.8285    0.3532   -1.4522   -8.3235   -4.8678  -10.0080
    0.5583    0.8366    0.3403   -1.4496   -8.4779   -4.8564  -10.0791
    0.5531    0.8447    0.3410   -1.4579   -8.4858   -4.8920  -10.0495
    0.5472    0.8476    0.3497   -1.4557   -8.3737   -4.8783   -9.9431
    0.5328    0.8470    0.3388   -1.4565   -8.5108   -4.8835   -9.9994
    0.5298    0.8409    0.3491   -1.4504   -8.3709   -4.8649   -9.9366

];
s.eletric(:,:,2) = [0.6342    0.8652    0.3174   -1.5147   -8.9004   -5.1445  -10.6292
    0.6049    0.8724    0.2998   -1.5035   -9.1184   -5.0570  -10.8852
    0.5920    0.8736    0.3052   -1.4962   -9.0303   -5.0311  -10.7787
    0.5862    0.8750    0.3059   -1.5037   -9.0354   -5.0725  -10.6860
    0.5732    0.8717    0.3112   -1.4996   -8.9553   -5.0584  -10.7380
    0.5702    0.8733    0.3190   -1.4922   -8.8359   -5.0288  -10.7031
    0.6128    0.8458    0.3189   -1.4938   -8.8388   -5.0192  -10.8210
    0.5940    0.8614    0.3117   -1.4985   -8.9452   -5.0368  -10.7408
    0.5397    0.8526    0.3093   -1.4915   -8.9646   -5.0215  -10.8746
    0.5132    0.8545    0.3417   -1.4903   -8.5418   -5.0249  -10.9893
    0.4918    0.8576    0.3147   -1.4998   -8.9085   -5.0525  -10.8628
    0.4736    0.8553    0.3215   -1.4899   -8.7982   -5.0173  -10.6929
    0.6229    0.8486    0.2994   -1.4918   -9.1003   -5.0350  -10.5785
    0.5738    0.8577    0.3110   -1.4877   -8.9333   -4.9986  -10.8428
    0.5361    0.8576    0.3174   -1.4910   -8.8553   -5.0145  -10.7373
    0.5107    0.8587    0.2988   -1.4919   -9.1105   -5.0120  -10.7814
    0.4853    0.8585    0.3094   -1.4942   -8.9688   -5.0182  -10.7611
    0.4893    0.8593    0.3190   -1.4896   -8.8308   -5.0089  -10.8830
    0.5957    0.8528    0.2947   -1.4939   -9.1698   -5.0179  -10.6861
    0.5762    0.8582    0.3075   -1.4953   -8.9958   -5.0208  -10.7144
    0.5609    0.8630    0.3038   -1.4994   -9.0561   -5.0390  -10.8834
    0.5424    0.8590    0.3156   -1.4957   -8.8879   -5.0430  -10.7628
    0.5292    0.8613    0.3040   -1.4974   -9.0495   -5.0372  -10.8384
    0.5056    0.8618    0.3162   -1.4934   -8.8763   -5.0193  -10.8240
    0.6044    0.8579    0.3385   -1.5020   -8.6027   -5.0821  -11.1099
    0.5872    0.8610    0.3131   -1.4907   -8.9107   -5.0080  -10.7352
    0.5392    0.8601    0.3008   -1.4997   -9.0981   -5.0413  -10.7201
    0.5450    0.8654    0.3068   -1.5021   -9.0206   -5.0602  -10.9618
    0.5143    0.8585    0.3202   -1.4958   -8.8280   -5.0278  -10.8994
    0.4987    0.8583    0.3156   -1.4940   -8.8854   -5.0207  -10.8427
    0.6279    0.8528    0.3144   -1.4965   -8.9041   -5.0616  -10.7482
    0.5903    0.8608    0.3143   -1.5014   -8.9168   -5.0785  -10.8559
    0.5439    0.8613    0.3219   -1.4966   -8.8064   -5.0385  -10.8196
    0.5127    0.8589    0.2897   -1.4913   -9.2380   -5.0092  -10.6253
    0.4986    0.8598    0.3164   -1.4950   -8.8766   -5.0309  -10.7856
    0.5055    0.8601    0.3149   -1.4954   -8.8979   -5.0345  -10.8014
    0.5991    0.8414    0.3200   -1.4922   -8.8206   -5.0204  -10.9022
    0.5823    0.8569    0.3055   -1.5027   -9.0380   -5.0643  -10.8193
    0.5471    0.8569    0.3178   -1.5020   -8.8711   -5.0632  -10.9268
    0.5127    0.8579    0.3374   -1.4971   -8.6089   -5.0413  -10.8727
    0.4855    0.8549    0.3109   -1.4998   -8.9597   -5.0535  -10.8228
    0.4554    0.8547    0.3103   -1.4944   -8.9574   -5.0281  -10.8210
    0.6149    0.8672    0.3097   -1.5032   -8.9811   -5.0505  -10.6067
    0.5875    0.8556    0.3039   -1.4928   -9.0411   -5.0079  -10.5822
    0.5848    0.8648    0.3124   -1.4962   -8.9316   -5.0359  -10.7495
    0.5764    0.8719    0.3208   -1.5005   -8.8294   -5.0415  -10.9071
    0.5664    0.8717    0.3095   -1.4995   -8.9780   -5.0470  -10.8464
    0.5566    0.8606    0.3216   -1.4902   -8.7988   -5.0141  -10.6932
    0.6299    0.8665    0.3254   -1.5029   -8.7719   -5.0943  -10.6584
    0.5984    0.8716    0.3122   -1.5061   -8.9544   -5.0679  -10.9830
    0.5444    0.8595    0.3116   -1.4927   -8.9353   -5.0352  -10.8279
    0.5247    0.8555    0.3482   -1.4885   -8.4574   -4.9991  -10.7409
    0.4952    0.8593    0.3193   -1.4920   -8.8318   -5.0203  -10.9155
    0.4772    0.8607    0.3226   -1.4914   -8.7881   -5.0240  -10.7643
    0.6242    0.8489    0.3181   -1.4951   -8.8522   -5.0450  -10.8112
    0.5618    0.8569    0.2984   -1.4972   -9.1266   -5.0395  -10.8544
    0.5413    0.8543    0.3117   -1.4925   -8.9346   -5.0062  -10.9493
    0.5095    0.8553    0.2884   -1.4943   -9.2630   -5.0206  -10.7700
    0.4910    0.8535    0.3057   -1.4939   -9.0190   -5.0215  -10.8271
    0.4983    0.8564    0.3155   -1.4915   -8.8817   -5.0208  -10.7639
    0.6082    0.8580    0.3251   -1.4990   -8.7675   -5.0775  -10.9548
    0.5754    0.8618    0.3253   -1.5004   -8.7688   -5.0514  -10.8380
    0.5568    0.8600    0.3292   -1.4977   -8.7139   -5.0378  -10.7825
    0.5134    0.8562    0.3297   -1.4937   -8.6998   -5.0140  -10.9996
    0.5164    0.8610    0.3274   -1.4956   -8.7339   -5.0367  -10.8944
    0.4825    0.8566    0.3256   -1.4898   -8.7446   -5.0117  -10.7901
    0.6032    0.8525    0.3199   -1.4963   -8.8313   -5.0488  -10.9483
    0.5828    0.8637    0.3145   -1.4990   -8.9092   -5.0305  -10.7074
    0.5583    0.8595    0.3241   -1.5007   -8.7866   -5.0621  -10.9663
    0.5453    0.8646    0.3269   -1.5004   -8.7496   -5.0362  -11.1756
    0.5116    0.8594    0.3206   -1.4985   -8.8273   -5.0474  -10.8933
    0.4895    0.8565    0.3131   -1.4937   -8.9174   -5.0254  -10.8028
    0.6279    0.8539    0.3258   -1.5031   -8.7671   -5.0901  -10.9146
    0.5719    0.8577    0.3283   -1.4906   -8.7116   -5.0091  -10.7647
    0.5301    0.8575    0.3211   -1.4957   -8.8158   -5.0364  -10.8793
    0.4922    0.8551    0.2966   -1.4910   -9.1387   -5.0074  -10.5064
    0.4851    0.8604    0.3196   -1.4970   -8.8383   -5.0366  -10.8087
    0.4881    0.8605    0.3182   -1.4951   -8.8526   -5.0311  -10.8317
    0.6169    0.8597    0.3100   -1.4929   -8.9555   -5.0317  -10.8152
    0.5834    0.8658    0.3111   -1.4947   -8.9464   -5.0295  -10.8400
    0.5385    0.8641    0.3096   -1.4960   -8.9689   -5.0299  -10.7840
    0.5088    0.8659    0.3162   -1.4973   -8.8833   -5.0193  -10.8257
    0.4958    0.8634    0.3073   -1.4954   -9.0001   -5.0227  -10.7170
    0.4497    0.8635    0.3119   -1.4936   -8.9337   -5.0202  -10.7234
    0.6240    0.8555    0.3180   -1.5019   -8.8673   -5.0694  -10.8452
    0.6080    0.8722    0.3407   -1.4936   -8.5599   -5.0221  -10.8361
    0.5953    0.8748    0.3059   -1.5019   -9.0324   -5.0580  -10.9227
    0.5810    0.8670    0.3181   -1.4946   -8.8536   -5.0207  -10.8261
    0.5677    0.8705    0.3106   -1.4990   -8.9617   -5.0414  -10.8152
    0.5583    0.8599    0.3246   -1.4900   -8.7584   -5.0097  -10.7604
];
s.eletric(:,:,3) = [0.5945    0.8566    0.4103   -1.4604   -7.6927   -4.9051  -11.3307
    0.5719    0.8567    0.4171   -1.4524   -7.6059   -4.8696  -11.5080
    0.5615    0.8538    0.4239   -1.4508   -7.5323   -4.8704  -11.3877
    0.5476    0.8512    0.4210   -1.4496   -7.5600   -4.8502  -11.4644
    0.5427    0.8570    0.4209   -1.4524   -7.5669   -4.8680  -11.3428
    0.5350    0.8537    0.4223   -1.4449   -7.5371   -4.8459  -11.3345
    0.5679    0.8353    0.4044   -1.4522   -7.7396   -4.8585  -11.9727
    0.5494    0.8375    0.4185   -1.4524   -7.5918   -4.8710  -11.6284
    0.5051    0.8376    0.4166   -1.4521   -7.6108   -4.8586  -11.5495
    0.4948    0.8411    0.4037   -1.4577   -7.7594   -4.8785  -11.3940
    0.4571    0.8340    0.4172   -1.4505   -7.6021   -4.8652  -11.3829
    0.4488    0.8386    0.4203   -1.4458   -7.5601   -4.8475  -11.2512
    0.5535    0.8323    0.4261   -1.4546   -7.5154   -4.8838  -11.4853
    0.5356    0.8322    0.4247   -1.4480   -7.5178   -4.8516  -11.4258
    0.5148    0.8417    0.4159   -1.4527   -7.6196   -4.8621  -11.4212
    0.4936    0.8394    0.4288   -1.4524   -7.4854   -4.8694  -11.3084
    0.4626    0.8355    0.4170   -1.4526   -7.6077   -4.8687  -11.4306
    0.4765    0.8397    0.4230   -1.4466   -7.5331   -4.8498  -11.3455
    0.5820    0.8416    0.4416   -1.4538   -7.3559   -4.8995  -11.8400
    0.5427    0.8422    0.4164   -1.4469   -7.6019   -4.8414  -11.3259
    0.5091    0.8388    0.4133   -1.4479   -7.6378   -4.8477  -11.6436
    0.5028    0.8426    0.4149   -1.4468   -7.6191   -4.8411  -11.4331
    0.4883    0.8371    0.4256   -1.4475   -7.5084   -4.8555  -11.6009
    0.4542    0.8423    0.4197   -1.4442   -7.5628   -4.8452  -11.4120
    0.5814    0.8404    0.4205   -1.4589   -7.5830   -4.9015  -11.6959
    0.5434    0.8372    0.4278   -1.4502   -7.4906   -4.8701  -11.3533
    0.5132    0.8418    0.4169   -1.4538   -7.6117   -4.8832  -11.4847
    0.4931    0.8351    0.4164   -1.4475   -7.6037   -4.8497  -11.1693
    0.4740    0.8392    0.4207   -1.4515   -7.5672   -4.8677  -11.3919
    0.4637    0.8413    0.4210   -1.4461   -7.5529   -4.8479  -11.3111
    0.5707    0.8272    0.4069   -1.4444   -7.6975   -4.8410  -11.3888
    0.5462    0.8389    0.4206   -1.4504   -7.5656   -4.8620  -11.2519
    0.5170    0.8379    0.4236   -1.4481   -7.5303   -4.8562  -11.3712
    0.4990    0.8409    0.4394   -1.4470   -7.3664   -4.8440  -11.4185
    0.4629    0.8401    0.4170   -1.4479   -7.5984   -4.8506  -11.3966
    0.4808    0.8397    0.4215   -1.4446   -7.5452   -4.8406  -11.3572
    0.5717    0.8289    0.4007   -1.4561   -7.7872   -4.8808  -11.2844
    0.5494    0.8375    0.4289   -1.4443   -7.4671   -4.8317  -11.6041
    0.5194    0.8400    0.4178   -1.4503   -7.5943   -4.8610  -11.4669
    0.4902    0.8424    0.4116   -1.4504   -7.6606   -4.8512  -11.3656
    0.4600    0.8387    0.4159   -1.4488   -7.6120   -4.8494  -11.3700
    0.4412    0.8372    0.4200   -1.4462   -7.5641   -4.8482  -11.3067
    0.5978    0.8446    0.4268   -1.4569   -7.5135   -4.8953  -11.0042
    0.5595    0.8316    0.4412   -1.4413   -7.3368   -4.8220  -11.1478
    0.5526    0.8456    0.4290   -1.4500   -7.4780   -4.8609  -11.2840
    0.5447    0.8435    0.4272   -1.4487   -7.4947   -4.8537  -11.1336
    0.5285    0.8447    0.4272   -1.4505   -7.4978   -4.8620  -11.1684
    0.5249    0.8327    0.4324   -1.4420   -7.4273   -4.8356  -11.1760
    0.5798    0.8359    0.4376   -1.4499   -7.3891   -4.8636  -11.6100
    0.5431    0.8405    0.4097   -1.4471   -7.6733   -4.8463  -11.3150
    0.5271    0.8414    0.4198   -1.4465   -7.5663   -4.8543  -11.2652
    0.4895    0.8451    0.4079   -1.4528   -7.7051   -4.8609  -11.3842
    0.4733    0.8373    0.4195   -1.4463   -7.5692   -4.8500  -11.3452
    0.4482    0.8381    0.4230   -1.4433   -7.5271   -4.8390  -11.2550
    0.5665    0.8270    0.4256   -1.4445   -7.5005   -4.8512  -11.1275
    0.5451    0.8313    0.4182   -1.4475   -7.5852   -4.8511  -11.2891
    0.5077    0.8397    0.4159   -1.4511   -7.6162   -4.8632  -11.1491
    0.5033    0.8346    0.4302   -1.4476   -7.4607   -4.8491  -11.2735
    0.4835    0.8394    0.4200   -1.4472   -7.5661   -4.8473  -11.3721
    0.4843    0.8415    0.4202   -1.4458   -7.5609   -4.8434  -11.2310
    0.5766    0.8408    0.4325   -1.4469   -7.4345   -4.8758  -11.4184
    0.5445    0.8446    0.4161   -1.4535   -7.6185   -4.8780  -11.4304
    0.5173    0.8351    0.4192   -1.4511   -7.5817   -4.8703  -11.2866
    0.5052    0.8391    0.4197   -1.4448   -7.5638   -4.8484  -11.4995
    0.4873    0.8395    0.4203   -1.4486   -7.5658   -4.8615  -11.5214
    0.4503    0.8382    0.4196   -1.4436   -7.5626   -4.8439  -11.2970
    0.5677    0.8265    0.4335   -1.4461   -7.4227   -4.8503  -11.1711
    0.5476    0.8375    0.4193   -1.4495   -7.5776   -4.8519  -11.8178
    0.5245    0.8397    0.4196   -1.4525   -7.5800   -4.8758  -11.6089
    0.5013    0.8341    0.4246   -1.4400   -7.5033   -4.8191  -11.3191
    0.4893    0.8399    0.4229   -1.4514   -7.5437   -4.8663  -11.4644
    0.4667    0.8402    0.4214   -1.4458   -7.5490   -4.8485  -11.3784
    0.5688    0.8296    0.4238   -1.4443   -7.5187   -4.8451  -11.2372
    0.5557    0.8414    0.4179   -1.4458   -7.5847   -4.8459  -11.2283
    0.5011    0.8371    0.4078   -1.4524   -7.7050   -4.8653  -11.2792
    0.4864    0.8407    0.4266   -1.4486   -7.5002   -4.8533  -11.5194
    0.4544    0.8388    0.4153   -1.4495   -7.6196   -4.8529  -11.3458
    0.4613    0.8406    0.4209   -1.4461   -7.5544   -4.8482  -11.3380
    0.5752    0.8461    0.4272   -1.4461   -7.4874   -4.8593  -11.7762
    0.5689    0.8586    0.4169   -1.4526   -7.6086   -4.8761  -11.2946
    0.5159    0.8511    0.4299   -1.4535   -7.4759   -4.8893  -11.4075
    0.4886    0.8520    0.4107   -1.4496   -7.6688   -4.8659  -11.3237
    0.4605    0.8474    0.4257   -1.4531   -7.5187   -4.8824  -11.3934
    0.4239    0.8445    0.4210   -1.4478   -7.5571   -4.8591  -11.3120
    0.5899    0.8465    0.4197   -1.4586   -7.5902   -4.8995  -11.1198
    0.5627    0.8376    0.4177   -1.4521   -7.5989   -4.8686  -11.4119
    0.5573    0.8423    0.4177   -1.4503   -7.5956   -4.8633  -11.1952
    0.5385    0.8359    0.4234   -1.4512   -7.5383   -4.8651  -11.1827
    0.5274    0.8362    0.4224   -1.4510   -7.5484   -4.8681  -11.1690
    0.5203    0.8310    0.4218   -1.4443   -7.5412   -4.8426  -11.0850
];  
s.explosive(:,:,1) = [0.1439    0.7479    0.1525   -1.2319  -11.3486   -4.1366   -8.0488
    0.4984    0.7512    0.1366   -1.2194  -11.7695   -4.1466   -8.2124
    0.4946    0.7556    0.1135   -1.2238  -12.5247   -4.1448   -8.1667
    0.4834    0.7489    0.1183   -1.2168  -12.3436   -4.1055   -7.9140
    0.4744    0.7559    0.1006   -1.2243  -13.0092   -4.1411   -8.1148
    0.4651    0.7476    0.1154   -1.2062  -12.4243   -4.0688   -7.8461
    0.1441    0.7185    0.1184   -1.2099  -12.3238   -4.0766   -7.7329
    0.1230    0.7319    0.1361   -1.2151  -11.7747   -4.1057   -7.8715
    0.1034    0.7366    0.1091   -1.2189  -12.6747   -4.1306   -7.8994
    0.4279    0.7358    0.1549   -1.2147  -11.2508   -4.0998   -7.9001
    0.4123    0.7333    0.1177   -1.2133  -12.3569   -4.0978   -7.9074
    0.3862    0.7298    0.0999   -1.2074  -13.0071   -4.0810   -7.8537
    0.1452    0.7194    0.1577   -1.2090  -11.1670   -4.0727   -7.8799
    0.1211    0.7308    0.1185   -1.2165  -12.3347   -4.1081   -7.8544
    0.4547    0.7351    0.1306   -1.2189  -11.9485   -4.1107   -7.9406
    0.4305    0.7318    0.1128   -1.2112  -12.5241   -4.0882   -7.8538
    0.4084    0.7348    0.1236   -1.2160  -12.1650   -4.1011   -7.9182
    0.4071    0.7358    0.1013   -1.2126  -12.9584   -4.0890   -7.8976
    0.1374    0.7250    0.1474   -1.2086  -11.4376   -4.0716   -7.7478
    0.4664    0.7282    0.1268   -1.2100  -12.0494   -4.0749   -7.7482
    0.4491    0.7332    0.1229   -1.2194  -12.1965   -4.1175   -7.9030
    0.4226    0.7308    0.0977   -1.2130  -13.1055   -4.0927   -7.8642
    0.4161    0.7321    0.1075   -1.2138  -12.7230   -4.0959   -7.8800
    0.4014    0.7294    0.1342   -1.2093  -11.8216   -4.0819   -7.7905
    0.1417    0.7179    0.0957   -1.2060  -13.1756   -4.0619   -7.8436
    0.1207    0.7349    0.1152   -1.2176  -12.4533   -4.1113   -7.9341
    0.1050    0.7308    0.1334   -1.2130  -11.8507   -4.0954   -7.8702
    0.4309    0.7317    0.1183   -1.2111  -12.3340   -4.0905   -7.9574
    0.4204    0.7339    0.1318   -1.2154  -11.9045   -4.1009   -7.8980
    0.3959    0.7328    0.1168   -1.2140  -12.3883   -4.1004   -7.8782
    0.1425    0.7227    0.0732   -1.2116  -14.2596   -4.0910   -7.8409
    0.1178    0.7274    0.0961   -1.2128  -13.1724   -4.0777   -7.8441
    0.1087    0.7314    0.1498   -1.2180  -11.3923   -4.1165   -7.9141
    0.4112    0.7337    0.1128   -1.2115  -12.5239   -4.0791   -7.9111
    0.4001    0.7339    0.1240   -1.2131  -12.1468   -4.0876   -7.8601
    0.3967    0.7331    0.1139   -1.2099  -12.4826   -4.0813   -7.8486
    0.1483    0.7265    0.1707   -1.2203  -10.8683   -4.1269   -7.9774
    0.1231    0.7275    0.0763   -1.2183  -14.1078   -4.1212   -7.8478
    0.1020    0.7328    0.0966   -1.2133  -13.1514   -4.1052   -7.8641
    0.4177    0.7307    0.1097   -1.2065  -12.6260   -4.0734   -7.8259
    0.3966    0.7311    0.0955   -1.2130  -13.1988   -4.0954   -7.8608
    0.3916    0.7323    0.1199   -1.2085  -12.2726   -4.0793   -7.8440
    0.1408    0.7417    0.1548   -1.2215  -11.2662   -4.1169   -7.9378
    0.4812    0.7353    0.1905   -1.2148  -10.4109   -4.0897   -7.8904
    0.4785    0.7380    0.1611   -1.2106  -11.0837   -4.0802   -7.8584
    0.4726    0.7367    0.1265   -1.2089  -12.0580   -4.0693   -7.9179
    0.4570    0.7418    0.0554   -1.2152  -15.3848   -4.0960   -7.9284
    0.4575    0.7346    0.1776   -1.2033  -10.6736   -4.0537   -7.8013
    0.1383    0.7289    0.1075   -1.2178  -12.7310   -4.1035   -7.8751
    0.1264    0.7345    0.1393   -1.2103  -11.6715   -4.0844   -7.8823
    0.1133    0.7350    0.1058   -1.2125  -12.7842   -4.0929   -7.8915
    0.4116    0.7354    0.1524   -1.2153  -11.3172   -4.1005   -7.8921
    0.4039    0.7368    0.1140   -1.2175  -12.4960   -4.1057   -7.9217
    0.3831    0.7321    0.1313   -1.2086  -11.9060   -4.0808   -7.8589
    0.1427    0.7257    0.0660   -1.2149  -14.6797   -4.0975   -7.9118
    0.1184    0.7321    0.1149   -1.2180  -12.4634   -4.1120   -7.8974
    0.4427    0.7331    0.1188   -1.2150  -12.3233   -4.0963   -7.8784
    0.4169    0.7345    0.1423   -1.2119  -11.5888   -4.0908   -7.8483
    0.3989    0.7320    0.1186   -1.2130  -12.3253   -4.0925   -7.8782
    0.4065    0.7322    0.1194   -1.2098  -12.2937   -4.0807   -7.8586
    0.1433    0.7290    0.0889   -1.2213  -13.4983   -4.1020   -7.8907
    0.1196    0.7305    0.0998   -1.2156  -13.0247   -4.1008   -7.8556
    0.4554    0.7323    0.0925   -1.2117  -13.3215   -4.0825   -7.9024
    0.4379    0.7360    0.1382   -1.2221  -11.7258   -4.1181   -7.9113
    0.4138    0.7297    0.1046   -1.2123  -12.8310   -4.0860   -7.8672
    0.4072    0.7329    0.1205   -1.2102  -12.2548   -4.0780   -7.8509
    0.1373    0.7252    0.1468   -1.2226  -11.4841   -4.1233   -7.9726
    0.1183    0.7314    0.1573   -1.2220  -11.2020   -4.1330   -7.9353
    0.1086    0.7287    0.1843   -1.2152  -10.5476   -4.0991   -7.9521
    0.4306    0.7305    0.1447   -1.2084  -11.5139   -4.0675   -7.8487
    0.4042    0.7288    0.1554   -1.2137  -11.2359   -4.0969   -7.9211
    0.3942    0.7285    0.1437   -1.2107  -11.5466   -4.0855   -7.8641
    0.1337    0.7228    0.1333   -1.2120  -11.8524   -4.0775   -7.8582
    0.1187    0.7304    0.1442   -1.2129  -11.5365   -4.0911   -7.9229
    0.1058    0.7282    0.1476   -1.2134  -11.4435   -4.0972   -7.8972
    0.4236    0.7325    0.1465   -1.2157  -11.4783   -4.1080   -7.9512
    0.4097    0.7377    0.1387   -1.2213  -11.7110   -4.1249   -7.9618
    0.4107    0.7349    0.1304   -1.2113  -11.9395   -4.0852   -7.8804
    0.1388    0.7277    0.1267   -1.1945  -12.0212   -4.0269   -7.7502
    0.1172    0.7376    0.1107   -1.2002  -12.5783   -4.0305   -7.8478
    0.4482    0.7354    0.1344   -1.2128  -11.8199   -4.0819   -7.9077
    0.4254    0.7413    0.0885   -1.2153  -13.5081   -4.1045   -7.9250
    0.3905    0.7367    0.1092   -1.2128  -12.6585   -4.0863   -7.9117
    0.4074    0.7351    0.1250   -1.2045  -12.0962   -4.0613   -7.8253
    0.2002    0.6774    0.3579   -1.1199   -7.6016   -3.5963   -6.4789
    0.4922    0.7379    0.2249   -1.2199   -9.7426   -4.1414   -8.0600
    0.4861    0.7441    0.1640   -1.2213  -11.0325   -4.1262   -7.9531
    0.4730    0.7378    0.1974   -1.2124  -10.2625   -4.0807   -7.9082
    0.4566    0.7389    0.1581   -1.2141  -11.1667   -4.0830   -7.8720
    0.4552    0.7333    0.1310   -1.2033  -11.9072   -4.0557   -7.7868
];
s.explosive(:,:,2) = [0.8474    0.8136    0.2558   -1.5032   -9.7783  -15.3957   -9.9101
    0.8268    0.8107    0.2288   -1.4964  -10.2259  -16.0691  -10.0781
    0.8202    0.8125    0.2358   -1.4982  -10.1045  -18.0486   -9.9844
    0.8194    0.8126    0.2485   -1.4949   -9.8817  -16.3257   -9.9830
    0.7871    0.8118    0.2474   -1.4954   -9.9019  -16.6154   -9.9898
    0.7635    0.8079    0.2407   -1.4876   -9.9991  -17.4120   -9.9349
    0.8403    0.8055    0.2435   -1.4935   -9.9627  -16.3837   -9.9317
    0.8053    0.8086    0.2418   -1.4938   -9.9923  -16.3441   -9.9601
    0.7389    0.8072    0.2520   -1.4943   -9.8240  -15.6929   -9.9475
    0.7315    0.8078    0.2492   -1.4948   -9.8710  -15.5658   -9.9293
    0.6883    0.8071    0.2457   -1.4940   -9.9282  -16.8338   -9.9473
    0.6760    0.8053    0.2453   -1.4899   -9.9268  -16.2048   -9.9364
    0.8477    0.8106    0.2400   -1.4994  -10.0337  -16.2707   -9.8443
    0.8208    0.8097    0.2442   -1.4932   -9.9504  -15.6772   -9.9653
    0.7471    0.8059    0.2451   -1.4910   -9.9324  -17.0313   -9.9220
    0.7089    0.8060    0.2412   -1.4906   -9.9971  -15.3836   -9.9586
    0.6673    0.8049    0.2434   -1.4923   -9.9636  -15.9610   -9.9407
    0.6848    0.8056    0.2421   -1.4915   -9.9834  -15.9622   -9.9484
    0.8492    0.8132    0.2269   -1.5041  -10.2749  -16.4289   -9.9295
    0.7869    0.8081    0.2514   -1.4949   -9.8341  -17.2718   -9.9453
    0.7660    0.8067    0.2429   -1.4940   -9.9755  -16.2860   -9.9662
    0.7435    0.8057    0.2438   -1.4904   -9.9520  -16.0256   -9.9248
    0.7056    0.8072    0.2432   -1.4953   -9.9716  -16.0486   -9.9445
    0.6789    0.8044    0.2488   -1.4893   -9.8671  -16.1524   -9.9484
    0.8250    0.8046    0.2451   -1.4924   -9.9346  -15.7379   -9.9262
    0.7983    0.8065    0.2394   -1.4942  -10.0347  -16.8663   -9.9373
    0.7765    0.8075    0.2456   -1.4954   -9.9323  -20.2454   -9.9465
    0.7578    0.8060    0.2456   -1.4920   -9.9247  -17.6617   -9.9296
    0.7313    0.8067    0.2404   -1.4947  -10.0184  -16.0341   -9.9716
    0.6964    0.8031    0.2483   -1.4884   -9.8733  -16.3261   -9.9507
    0.8452    0.8102    0.2356   -1.4975  -10.1075  -14.4057   -9.9448
    0.8262    0.8098    0.2440   -1.4947   -9.9567  -15.8586   -9.9243
    0.7343    0.8058    0.2490   -1.4913   -9.8670  -16.3718   -9.9446
    0.7174    0.8086    0.2506   -1.4948   -9.8474  -16.7092   -9.9709
    0.6638    0.8058    0.2440   -1.4935   -9.9546  -16.0809   -9.9573
    0.6893    0.8069    0.2441   -1.4914   -9.9485  -16.3233   -9.9499
    0.8434    0.8089    0.2251   -1.5001  -10.3000  -14.5583   -9.9003
    0.8203    0.8113    0.2328   -1.4998  -10.1618  -17.1266   -9.9136
    0.7452    0.8074    0.2434   -1.4952   -9.9690  -16.1900   -9.9199
    0.7310    0.8068    0.2428   -1.4916   -9.9711  -15.3120   -9.9387
    0.7024    0.8070    0.2410   -1.4937  -10.0058  -16.2301   -9.9418
    0.6790    0.8064    0.2386   -1.4906  -10.0416  -16.0122   -9.9344
    0.8411    0.8091    0.2402   -1.5010  -10.0341  -17.0214   -9.7951
    0.8374    0.8133    0.2414   -1.4991  -10.0102  -16.3541  -10.1233
    0.8225    0.8122    0.2474   -1.4981   -9.9068  -16.6966   -9.9814
    0.8218    0.8122    0.2405   -1.4939  -10.0159  -16.3721   -9.9758
    0.7848    0.8103    0.2429   -1.4938   -9.9742  -16.3626   -9.9878
    0.7598    0.8068    0.2391   -1.4871  -10.0262  -15.8789   -9.9232
    0.8330    0.8074    0.2409   -1.4957  -10.0125  -15.1201   -9.9051
    0.8128    0.8062    0.2394   -1.4929  -10.0320  -15.5735   -9.9291
    0.7500    0.8067    0.2442   -1.4948   -9.9547  -15.8657   -9.9349
    0.7307    0.8070    0.2418   -1.4959   -9.9971  -15.9427   -9.9130
    0.7043    0.8079    0.2435   -1.4957   -9.9682  -15.9425   -9.9258
    0.6893    0.8053    0.2401   -1.4911  -10.0161  -15.8879   -9.9321
    0.8281    0.8066    0.2376   -1.4958  -10.0682  -16.7324   -9.9165
    0.8218    0.8101    0.2491   -1.4971   -9.8764  -17.6201   -9.9488
    0.7393    0.8057    0.2432   -1.4929   -9.9679  -15.6700   -9.9619
    0.7215    0.8082    0.2396   -1.4954  -10.0334  -17.7381   -9.9401
    0.6824    0.8052    0.2453   -1.4920   -9.9308  -16.9113   -9.9691
    0.6849    0.8060    0.2458   -1.4904   -9.9181  -16.1281   -9.9606
    0.8390    0.8073    0.2492   -1.4982   -9.8768  -16.3035   -9.9152
    0.7982    0.8072    0.2442   -1.4934   -9.9517  -14.7370   -9.9411
    0.7711    0.8077    0.2392   -1.4937  -10.0380  -14.7143   -9.9419
    0.7382    0.8064    0.2449   -1.4911   -9.9357  -15.7086   -9.8995
    0.6954    0.8057    0.2445   -1.4934   -9.9470  -16.0267   -9.9535
    0.6799    0.8043    0.2486   -1.4892   -9.8696  -16.5087   -9.9573
    0.8363    0.8093    0.2477   -1.5014   -9.9085  -17.2403   -9.9437
    0.7986    0.8066    0.2403   -1.4944  -10.0190  -15.6201   -9.9653
    0.7658    0.8060    0.2504   -1.4921   -9.8458  -14.6991   -9.9544
    0.7363    0.8058    0.2470   -1.4910   -9.8992  -15.1869   -9.9134
    0.7068    0.8070    0.2438   -1.4943   -9.9600  -16.0068   -9.9594
    0.6888    0.8034    0.2491   -1.4880   -9.8595  -15.7305   -9.9326
    0.8404    0.8097    0.2402   -1.4992  -10.0312  -15.5968   -9.9244
    0.8121    0.8107    0.2492   -1.4957   -9.8721  -15.4523   -9.9194
    0.7505    0.8069    0.2485   -1.4928   -9.8785  -15.3706   -9.9545
    0.7282    0.8092    0.2510   -1.4946   -9.8405  -14.5792   -9.9703
    0.6697    0.8052    0.2480   -1.4926   -9.8862  -15.1346   -9.9557
    0.6870    0.8056    0.2472   -1.4898   -9.8935  -15.4111   -9.9564
    0.8385    0.8100    0.2389   -1.5008  -10.0553  -13.9626   -9.8550
    0.8174    0.8092    0.2422   -1.4973   -9.9925  -15.6734   -9.9808
    0.7485    0.8064    0.2423   -1.4948   -9.9873  -16.5499   -9.9468
    0.7288    0.8080    0.2470   -1.4947   -9.9071  -16.9121   -9.9432
    0.7029    0.8073    0.2423   -1.4951   -9.9873  -15.9740   -9.9275
    0.6821    0.8057    0.2413   -1.4905   -9.9949  -15.9450   -9.9457
    0.8295    0.8026    0.2489   -1.4860   -9.8567  -15.1935   -9.8950
    0.8256    0.8087    0.2445   -1.4951   -9.9496  -16.6092   -9.9604
    0.8154    0.8105    0.2498   -1.4971   -9.8646  -17.0047   -9.9396
    0.8136    0.8103    0.2415   -1.4920   -9.9947  -15.5238   -9.9867
    0.7870    0.8096    0.2453   -1.4942   -9.9351  -16.4013   -9.9461
    0.7455    0.8062    0.2405   -1.4882  -10.0040  -16.2713   -9.9130

];
s.explosive(:,:,3) = [0.1219    0.7275    0.3712   -1.3192   -7.8444   -4.3119  -10.0566
    0.0515    0.7152    0.3829   -1.2818   -7.6366   -4.1910  -10.8709
    0.0379    0.7182    0.3716   -1.2869   -7.7767   -4.2089  -11.5610
    0.0322    0.7191    0.3598   -1.2820   -7.9052   -4.1874  -10.6115
    0.0314    0.7189    0.3731   -1.2853   -7.7562   -4.2069  -11.0310
    0.0333    0.6933    0.3461   -1.2394   -7.9855   -3.9928  -13.6106
    0.1005    0.7026    0.4044   -1.3082   -7.4523   -4.2792   -9.8138
    0.0677    0.7081    0.3684   -1.2933   -7.8257   -4.2252  -10.3656
    0.0504    0.7018    0.3636   -1.2861   -7.8682   -4.2106  -10.6769
    0.0417    0.7076    0.3710   -1.2822   -7.7745   -4.2024  -10.5715
    0.0395    0.7067    0.3751   -1.2851   -7.7330   -4.2021  -10.6183
    0.0361    0.6737    0.4133   -1.2211   -7.1844   -3.9140   -9.0681
    0.0956    0.7027    0.3952   -1.2967   -7.5292   -4.2476  -10.2153
    0.0528    0.7023    0.3735   -1.2795   -7.7393   -4.1860  -11.0801
    0.0448    0.7043    0.3735   -1.2882   -7.7567   -4.2207  -10.7566
    0.0387    0.7047    0.3664   -1.2855   -7.8338   -4.2035  -11.5978
    0.0394    0.7073    0.3704   -1.2862   -7.7895   -4.2100  -10.8862
    0.0373    0.6996    0.3689   -1.2706   -7.7754   -4.1604  -11.0061
    0.1116    0.7138    0.4047   -1.3064   -7.4453   -4.2554   -9.8556
    0.0549    0.7036    0.3701   -1.2821   -7.7837   -4.1986  -10.7179
    0.0501    0.7055    0.3826   -1.2915   -7.6602   -4.2241  -10.4505
    0.0483    0.6775    0.4303   -1.2360   -7.0364   -3.9677   -9.0804
    0.0468    0.6830    0.4241   -1.2411   -7.1111   -3.9837   -9.2405
    0.0391    0.6720    0.4262   -1.2222   -7.0512   -3.9265   -9.3967
    0.0993    0.7152    0.3676   -1.3154   -7.8783   -4.2848  -10.2922
    0.0653    0.7000    0.3699   -1.2842   -7.7901   -4.1908  -10.9008
    0.0538    0.7072    0.3654   -1.2890   -7.8527   -4.2150  -10.8751
    0.0383    0.7060    0.3617   -1.2909   -7.8999   -4.2279  -11.0375
    0.0343    0.7080    0.3775   -1.2902   -7.7150   -4.2309  -10.7017
    0.0370    0.6716    0.4223   -1.2195   -7.0864   -3.9182   -9.3963
    0.0996    0.7055    0.3659   -1.3161   -7.9005   -4.2876   -9.7690
    0.0569    0.7038    0.3722   -1.2909   -7.7775   -4.2244  -10.5518
    0.0549    0.7077    0.3724   -1.2924   -7.7783   -4.2261  -10.8826
    0.0468    0.7076    0.3848   -1.2901   -7.6323   -4.2371  -10.9216
    0.0448    0.7072    0.3774   -1.2890   -7.7143   -4.2187  -10.7626
    0.0346    0.7024    0.3775   -1.2755   -7.6864   -4.1797  -10.7971
    0.1114    0.7138    0.3515   -1.3204   -8.0803   -4.2907   -9.9937
    0.0579    0.7094    0.3871   -1.3004   -7.6271   -4.2573  -10.3951
    0.0586    0.7065    0.3855   -1.2948   -7.6341   -4.2363  -10.4899
    0.0549    0.7082    0.3816   -1.2927   -7.6740   -4.2328  -10.7797
    0.0492    0.7089    0.3734   -1.2903   -7.7628   -4.2255  -10.6717
    0.0364    0.7020    0.3714   -1.2807   -7.7670   -4.1943  -10.7040
    0.0981    0.7112    0.3902   -1.2999   -7.5903   -4.2395   -9.8245
    0.0492    0.7003    0.3921   -1.2712   -7.5134   -4.1472  -11.2296
    0.0394    0.7154    0.3750   -1.2955   -7.7546   -4.2396  -10.6208
    0.0332    0.7167    0.3746   -1.2899   -7.7480   -4.2199  -10.5108
    0.0337    0.7184    0.3742   -1.2926   -7.7584   -4.2291  -10.6007
    0.0325    0.6812    0.3587   -1.2397   -7.8335   -3.9961  -12.6547
    0.1060    0.7085    0.3676   -1.2982   -7.8444   -4.2496   -9.9988
    0.0645    0.7100    0.3801   -1.2884   -7.6820   -4.2246  -10.8263
    0.0461    0.7086    0.3717   -1.2864   -7.7744   -4.2220  -10.6471
    0.0389    0.7098    0.3712   -1.2934   -7.7943   -4.2464  -11.1344
    0.0373    0.7146    0.3801   -1.2962   -7.6981   -4.2545  -10.7521
    0.0356    0.6758    0.4177   -1.2237   -7.1431   -3.9271   -9.1600
    0.0994    0.6969    0.3913   -1.2882   -7.5548   -4.2530  -10.1857
    0.0587    0.7056    0.3673   -1.2882   -7.8283   -4.2329  -10.8648
    0.0468    0.7010    0.3775   -1.2779   -7.6906   -4.1836  -10.7683
    0.0411    0.7097    0.3725   -1.2924   -7.7774   -4.2386  -10.7794
    0.0362    0.7049    0.3667   -1.2830   -7.8263   -4.2008  -10.8852
    0.0399    0.6780    0.4145   -1.2242   -7.1778   -3.9277   -9.2470
    0.1088    0.7133    0.3825   -1.3328   -7.7422   -4.3668  -10.0314
    0.0662    0.7047    0.3787   -1.2883   -7.6974   -4.2238  -10.7159
    0.0470    0.7015    0.3722   -1.2863   -7.7679   -4.2291  -10.9239
    0.0414    0.7075    0.3821   -1.2903   -7.6634   -4.2269  -10.7718
    0.0379    0.7036    0.3785   -1.2872   -7.6977   -4.2133  -10.8615
    0.0287    0.6962    0.3876   -1.2686   -7.5581   -4.1623  -11.0067
    0.0869    0.6995    0.3789   -1.3021   -7.7217   -4.2639   -9.7486
    0.0557    0.7060    0.3757   -1.2934   -7.7419   -4.2294  -10.7261
    0.0427    0.7014    0.3756   -1.2897   -7.7361   -4.2242  -10.5934
    0.0391    0.7078    0.3560   -1.2879   -7.9617   -4.2191  -10.8566
    0.0394    0.7094    0.3753   -1.2895   -7.7391   -4.2238  -10.3941
    0.0359    0.6946    0.3788   -1.2713   -7.6625   -4.1679  -11.1066
    0.1072    0.7066    0.3969   -1.3099   -7.5372   -4.2980  -10.2413
    0.0528    0.7075    0.3934   -1.2900   -7.5368   -4.2267  -10.3633
    0.0431    0.7096    0.3738   -1.2906   -7.7589   -4.2184  -10.4949
    0.0384    0.7060    0.3871   -1.2883   -7.6029   -4.2210  -10.3778
    0.0405    0.6824    0.4158   -1.2406   -7.1969   -3.9828   -9.1718
    0.0382    0.6803    0.4191   -1.2331   -7.1471   -3.9600   -9.2217
    0.1043    0.7207    0.3643   -1.3146   -7.9154   -4.3007  -10.3413
    0.0617    0.7122    0.3635   -1.2952   -7.8876   -4.2283  -10.5491
    0.0491    0.6851    0.4068   -1.2434   -7.2976   -3.9891   -9.2019
    0.0434    0.7129    0.3683   -1.2918   -7.8246   -4.2222  -10.5573
    0.0416    0.7135    0.3634   -1.2940   -7.8867   -4.2285  -10.5468
    0.0326    0.7060    0.3707   -1.2784   -7.7701   -4.1863  -10.7708
    0.0982    0.7030    0.3745   -1.3028   -7.7737   -4.2412  -10.2408
    0.0525    0.6998    0.3809   -1.2787   -7.6535   -4.1776  -10.7638
    0.0413    0.6746    0.3937   -1.2382   -7.4300   -3.9482   -9.2488
    0.0337    0.7045    0.3539   -1.2839   -7.9791   -4.1895  -10.9860
    0.0327    0.7072    0.3593   -1.2817   -7.9109   -4.1886  -10.8522
    0.0307    0.6990    0.3881   -1.2665   -7.5489   -4.1536  -10.7449
];
s.info(:,:,1) = [0.8518    0.8123    0.3863   -1.5337   -8.1027  -12.8090  -11.8619
    0.8486    0.8255    0.3764   -1.5564   -8.2600  -13.9346  -12.1409
    0.8276    0.8161    0.3885   -1.5377   -8.0865  -12.9465  -11.7498
    0.8232    0.8180    0.3867   -1.5387   -8.1087  -12.7600  -11.7335
    0.7962    0.8206    0.3835   -1.5456   -8.1583  -13.3089  -11.9841
    0.7689    0.8166    0.3864   -1.5375   -8.1099  -12.8230  -11.8174
    0.8464    0.8163    0.3770   -1.5436   -8.2275  -13.9872  -11.7813
    0.8323    0.8167    0.3835   -1.5437   -8.1538  -13.1721  -11.8241
    0.7696    0.8155    0.3888   -1.5418   -8.0917  -12.8912  -11.8828
    0.7329    0.8159    0.3854   -1.5434   -8.1322  -12.6864  -11.7541
    0.7065    0.8142    0.3876   -1.5418   -8.1044  -12.8334  -11.8543
    0.7068    0.8140    0.3878   -1.5399   -8.0984  -12.8176  -11.8156
    0.8360    0.8143    0.3926   -1.5434   -8.0517  -13.1584  -11.8337
    0.7817    0.8137    0.3909   -1.5426   -8.0696  -12.7247  -11.8535
    0.7464    0.8155    0.3818   -1.5431   -8.1721  -13.0614  -11.8148
    0.7187    0.8163    0.3901   -1.5426   -8.0790  -12.8582  -11.8667
    0.6803    0.8149    0.3871   -1.5422   -8.1115  -13.1277  -11.8540
    0.6823    0.8144    0.3877   -1.5416   -8.1031  -12.9634  -11.8671
    0.8554    0.8178    0.3805   -1.5430   -8.1863  -12.7231  -11.9100
    0.8132    0.8164    0.3852   -1.5412   -8.1306  -13.0267  -11.8761
    0.7590    0.8143    0.3860   -1.5415   -8.1216  -13.0014  -11.8959
    0.7496    0.8168    0.3833   -1.5437   -8.1563  -12.7504  -11.8296
    0.7237    0.8157    0.3857   -1.5419   -8.1264  -12.9206  -11.8813
    0.6777    0.8121    0.3930   -1.5397   -8.0411  -12.7622  -11.8539
    0.8482    0.8133    0.3893   -1.5414   -8.0839  -14.2032  -11.8531
    0.8045    0.8154    0.3852   -1.5411   -8.1295  -12.8684  -11.8157
    0.7745    0.8143    0.3877   -1.5411   -8.1026  -13.0407  -11.8770
    0.7678    0.8171    0.3829   -1.5429   -8.1593  -12.8950  -11.8518
    0.7430    0.8154    0.3875   -1.5418   -8.1061  -12.8990  -11.8419
    0.6985    0.8129    0.3921   -1.5395   -8.0499  -12.8184  -11.8495
    0.8384    0.8115    0.3882   -1.5400   -8.0942  -12.3378  -11.8011
    0.8240    0.8162    0.3871   -1.5424   -8.1115  -12.7218  -11.8602
    0.7759    0.8160    0.3912   -1.5419   -8.0651  -12.8996  -11.8426
    0.7402    0.8173    0.3884   -1.5428   -8.0973  -12.8904  -11.8526
    0.7201    0.8159    0.3875   -1.5420   -8.1063  -12.9315  -11.8546
    0.7001    0.8148    0.3900   -1.5406   -8.0762  -12.9767  -11.8845
    0.8468    0.8158    0.3878   -1.5446   -8.1070  -12.7361  -11.9192
    0.8247    0.8168    0.3899   -1.5431   -8.0810  -12.4401  -11.8209
    0.7685    0.8146    0.3893   -1.5417   -8.0851  -12.8753  -11.8279
    0.7403    0.8140    0.3886   -1.5421   -8.0936  -12.4740  -11.8537
    0.7235    0.8133    0.3897   -1.5413   -8.0805  -12.7304  -11.8400
    0.6886    0.8126    0.3894   -1.5397   -8.0805  -12.9995  -11.8176
    0.8522    0.8150    0.3863   -1.5378   -8.1106  -13.5672  -11.6044
    0.8424    0.8168    0.3895   -1.5372   -8.0745  -12.7889  -11.7305
    0.8303    0.8230    0.3821   -1.5483   -8.1793  -12.9948  -12.0711
    0.8253    0.8186    0.3878   -1.5391   -8.0968  -13.4663  -11.8399
    0.7950    0.8170    0.3843   -1.5393   -8.1363  -13.0748  -11.8102
    0.7636    0.8151    0.3886   -1.5373   -8.0848  -12.6862  -11.7594
    0.8318    0.8119    0.3893   -1.5420   -8.0852  -12.2084  -11.7482
    0.8252    0.8163    0.3876   -1.5424   -8.1052  -13.1907  -11.8866
    0.7490    0.8138    0.3886   -1.5412   -8.0919  -13.1442  -11.8800
    0.7147    0.8129    0.3874   -1.5408   -8.1051  -13.2435  -11.9145
    0.6465    0.8118    0.3865   -1.5422   -8.1174  -13.2684  -11.8454
    0.6749    0.8121    0.3882   -1.5394   -8.0935  -13.0716  -11.8381
    0.8399    0.8131    0.3899   -1.5411   -8.0776  -11.9979  -11.8484
    0.8084    0.8145    0.3890   -1.5418   -8.0891  -13.0331  -11.9216
    0.7629    0.8146    0.3840   -1.5410   -8.1435  -12.9038  -11.9225
    0.7265    0.8169    0.3892   -1.5428   -8.0892  -13.0460  -11.8573
    0.7154    0.8149    0.3879   -1.5408   -8.0997  -12.8462  -11.9029
    0.7110    0.8151    0.3905   -1.5407   -8.0700  -12.9087  -11.8949
    0.8461    0.8157    0.3810   -1.5432   -8.1807  -12.8038  -11.8525
    0.8231    0.8162    0.3850   -1.5431   -8.1361  -12.6294  -11.8309
    0.7580    0.8136    0.3893   -1.5408   -8.0837  -13.0439  -11.8904
    0.7599    0.8173    0.3831   -1.5430   -8.1580  -12.9024  -11.8793
    0.7250    0.8146    0.3879   -1.5417   -8.1013  -12.8879  -11.8509
    0.6969    0.8125    0.3907   -1.5391   -8.0646  -12.8470  -11.8551
    0.8531    0.8145    0.3859   -1.5435   -8.1268  -13.9053  -11.9002
    0.8201    0.8163    0.3840   -1.5421   -8.1456  -12.9144  -11.8266
    0.7503    0.8139    0.3847   -1.5417   -8.1372  -12.4758  -11.8829
    0.7510    0.8167    0.3828   -1.5436   -8.1624  -12.9314  -11.8633
    0.7015    0.8140    0.3853   -1.5414   -8.1294  -12.7004  -11.8675
    0.6701    0.8117    0.3928   -1.5394   -8.0423  -12.9326  -11.8651
    0.8317    0.8131    0.3903   -1.5400   -8.0703  -12.6290  -11.7989
    0.8064    0.8144    0.3860   -1.5431   -8.1255  -12.7018  -11.8016
    0.7705    0.8155    0.3910   -1.5428   -8.0693  -12.8642  -11.8158
    0.7354    0.8148    0.3873   -1.5412   -8.1067  -12.8588  -11.8412
    0.7105    0.8151    0.3886   -1.5418   -8.0940  -12.8936  -11.8557
    0.6840    0.8140    0.3907   -1.5414   -8.0690  -12.8035  -11.8403
    0.8455    0.8142    0.3911   -1.5447   -8.0707  -12.6998  -11.8028
    0.8215    0.8166    0.3820   -1.5441   -8.1721  -13.2357  -11.9569
    0.7536    0.8166    0.3857   -1.5454   -8.1329  -13.0366  -11.8797
    0.7204    0.8168    0.3849   -1.5455   -8.1428  -13.0186  -11.8465
    0.6947    0.8158    0.3852   -1.5449   -8.1377  -13.2351  -11.8811
    0.6892    0.8150    0.3858   -1.5423   -8.1259  -13.0136  -11.8698
    0.8423    0.8100    0.3886   -1.5371   -8.0832  -13.2299  -11.6234
    0.8383    0.8141    0.3895   -1.5379   -8.0757  -12.8578  -11.7199
    0.8314    0.8185    0.3863   -1.5431   -8.1215  -12.8616  -11.7303
    0.8127    0.8161    0.3886   -1.5388   -8.0880  -12.3711  -11.7234
    0.7882    0.8157    0.3860   -1.5398   -8.1182  -12.8487  -11.7883
    0.7628    0.8148    0.3872   -1.5374   -8.1006  -12.8045  -11.8038

];
s.info(:,:,2) = [0.8175    0.8649    0.3261   -1.5779   -8.9142  -16.4299  -13.5377
    0.8064    0.8661    0.3236   -1.5810   -8.9530  -16.2735  -13.3794
    0.8051    0.8699    0.3188   -1.5865   -9.0269  -16.7616  -13.7942
    0.7975    0.8682    0.3209   -1.5828   -8.9919  -16.2080  -13.6480
    0.7879    0.8685    0.3227   -1.5851   -8.9739  -16.2767  -13.5194
    0.7848    0.8663    0.3264   -1.5810   -8.9175  -15.9224  -13.4086
    0.7913    0.8462    0.3289   -1.5840   -8.8903  -14.6327  -13.2335
    0.7818    0.8548    0.3242   -1.5831   -8.9504  -15.9829  -13.3235
    0.7537    0.8544    0.3250   -1.5849   -8.9430  -14.9635  -13.2790
    0.7563    0.8577    0.3231   -1.5854   -8.9686  -14.8514  -13.2608
    0.7057    0.8550    0.3241   -1.5849   -8.9551  -15.0990  -13.2848
    0.6686    0.8549    0.3224   -1.5830   -8.9736  -15.3420  -13.2890
    0.7816    0.8517    0.3291   -1.5828   -8.8849  -14.4066  -13.5267
    0.7750    0.8564    0.3226   -1.5852   -8.9745  -15.1410  -13.2902
    0.7566    0.8573    0.3224   -1.5855   -8.9784  -14.7256  -13.2451
    0.7291    0.8578    0.3196   -1.5859   -9.0163  -15.2342  -13.2808
    0.7134    0.8575    0.3223   -1.5852   -8.9790  -15.0294  -13.2285
    0.7162    0.8596    0.3244   -1.5845   -8.9504  -15.5771  -13.3586
    0.7723    0.8471    0.3220   -1.5857   -8.9832  -15.1657  -13.5997
    0.7589    0.8500    0.3216   -1.5845   -8.9866  -15.2869  -13.4000
    0.7385    0.8527    0.3251   -1.5838   -8.9401  -15.6266  -13.3954
    0.7279    0.8531    0.3266   -1.5816   -8.9159  -14.9612  -13.3348
    0.7020    0.8544    0.3222   -1.5847   -8.9795  -15.2529  -13.3815
    0.6764    0.8541    0.3283   -1.5830   -8.8962  -14.9697  -13.3060
    0.7794    0.8493    0.3192   -1.5888   -9.0264  -14.7684  -13.2326
    0.7605    0.8514    0.3246   -1.5845   -8.9469  -14.9909  -13.3296
    0.7549    0.8530    0.3255   -1.5832   -8.9332  -14.8155  -13.3235
    0.7445    0.8528    0.3260   -1.5810   -8.9225  -15.5896  -13.3545
    0.7218    0.8544    0.3230   -1.5837   -8.9667  -15.4504  -13.3393
    0.6807    0.8541    0.3285   -1.5828   -8.8935  -15.1596  -13.2849
    0.7847    0.8510    0.3290   -1.5826   -8.8862  -14.5089  -13.1711
    0.7764    0.8572    0.3215   -1.5870   -8.9923  -15.3748  -13.4478
    0.7499    0.8564    0.3242   -1.5837   -8.9519  -15.3175  -13.3104
    0.7320    0.8577    0.3239   -1.5850   -8.9573  -16.5223  -13.3386
    0.7126    0.8572    0.3251   -1.5841   -8.9400  -15.2554  -13.3350
    0.7062    0.8586    0.3255   -1.5847   -8.9367  -15.2128  -13.3292
    0.8021    0.8543    0.3248   -1.5851   -8.9451  -15.2120  -13.3532
    0.7879    0.8562    0.3228   -1.5841   -8.9705  -14.0983  -13.1792
    0.7610    0.8570    0.3203   -1.5857   -9.0063  -15.7533  -13.3316
    0.7507    0.8586    0.3231   -1.5845   -8.9673  -15.6094  -13.3303
    0.7116    0.8569    0.3228   -1.5857   -8.9732  -15.4469  -13.3639
    0.6913    0.8569    0.3223   -1.5838   -8.9758  -15.5351  -13.3144
    0.8156    0.8658    0.3315   -1.5854   -8.8595  -13.9608  -14.0464
    0.7914    0.8657    0.3272   -1.5770   -8.8984  -14.6658  -13.0529
    0.7980    0.8695    0.3221   -1.5891   -8.9899  -13.9326  -12.8930
    0.7959    0.8707    0.3182   -1.5888   -9.0407  -15.4816  -13.3167
    0.7885    0.8698    0.3208   -1.5886   -9.0055  -15.6205  -13.3356
    0.7656    0.8610    0.3287   -1.5784   -8.8826  -14.2766  -13.0807
    0.8002    0.8522    0.3190   -1.5869   -9.0247  -20.4096  -13.4026
    0.7838    0.8552    0.3231   -1.5850   -8.9681  -14.7571  -13.2716
    0.7616    0.8570    0.3230   -1.5857   -8.9712  -15.2779  -13.3103
    0.7614    0.8583    0.3231   -1.5855   -8.9692  -15.8477  -13.2864
    0.7220    0.8571    0.3234   -1.5850   -8.9643  -15.2360  -13.3256
    0.6870    0.8564    0.3219   -1.5838   -8.9813  -15.2056  -13.2578
    0.7788    0.8487    0.3288   -1.5819   -8.8873  -14.7305  -13.3838
    0.7716    0.8542    0.3240   -1.5845   -8.9555  -15.6562  -13.2712
    0.7484    0.8563    0.3223   -1.5849   -8.9790  -15.7946  -13.4443
    0.7353    0.8567    0.3214   -1.5848   -8.9894  -15.6025  -13.4112
    0.7052    0.8567    0.3237   -1.5852   -8.9611  -15.6470  -13.3764
    0.7113    0.8591    0.3242   -1.5850   -8.9539  -15.2754  -13.3119
    0.7785    0.8495    0.3231   -1.5863   -8.9706  -15.6557  -13.1628
    0.7582    0.8511    0.3228   -1.5838   -8.9696  -14.6407  -13.3175
    0.7396    0.8549    0.3223   -1.5874   -8.9835  -14.8704  -13.2131
    0.7326    0.8559    0.3231   -1.5851   -8.9677  -15.2830  -13.1975
    0.7066    0.8553    0.3218   -1.5855   -8.9862  -15.2853  -13.2748
    0.6798    0.8539    0.3290   -1.5832   -8.8876  -15.6331  -13.3054
    0.7801    0.8461    0.3228   -1.5806   -8.9624  -14.7855  -13.2869
    0.7627    0.8509    0.3259   -1.5858   -8.9336  -15.2457  -13.2930
    0.7394    0.8523    0.3225   -1.5842   -8.9746  -15.9054  -13.3782
    0.7374    0.8536    0.3255   -1.5810   -8.9296  -15.1506  -13.4991
    0.7218    0.8560    0.3233   -1.5851   -8.9655  -15.5196  -13.2786
    0.6826    0.8532    0.3292   -1.5815   -8.8819  -15.0233  -13.3091
    0.7769    0.8496    0.3281   -1.5823   -8.8969  -14.4938  -13.0868
    0.7646    0.8548    0.3221   -1.5867   -8.9844  -15.6456  -13.2931
    0.7547    0.8569    0.3220   -1.5854   -8.9831  -15.5076  -13.2685
    0.7386    0.8583    0.3223   -1.5866   -8.9815  -14.7147  -13.3564
    0.7206    0.8576    0.3226   -1.5851   -8.9753  -15.2769  -13.3050
    0.7100    0.8585    0.3240   -1.5853   -8.9566  -15.0582  -13.3190
    0.8043    0.8556    0.3249   -1.5827   -8.9397  -15.0664  -13.3837
    0.7955    0.8613    0.3220   -1.5841   -8.9800  -15.5182  -13.3210
    0.7692    0.8564    0.3218   -1.5847   -8.9847  -14.9765  -13.3173
    0.7557    0.8581    0.3231   -1.5861   -8.9706  -15.4115  -13.2403
    0.7203    0.8565    0.3238   -1.5842   -8.9571  -15.5412  -13.3956
    0.6809    0.8573    0.3225   -1.5826   -8.9713  -15.5687  -13.2977
    0.8042    0.8525    0.3259   -1.5844   -8.9298  -14.5278  -13.2828
    0.7977    0.8615    0.3253   -1.5794   -8.9274  -14.6708  -12.9984
    0.7883    0.8638    0.3244   -1.5827   -8.9465  -13.2613  -13.5303
    0.7914    0.8626    0.3205   -1.5822   -8.9970  -15.4300  -13.1731
    0.7801    0.8618    0.3230   -1.5837   -8.9672  -13.8980  -13.3877
    0.7593    0.8568    0.3264   -1.5797   -8.9143  -14.8956  -13.1694

];
s.info(:,:,3) = [.8572    0.8188    0.3732   -1.5409   -8.2654  -13.7010  -13.7602
    0.8393    0.8207    0.3617   -1.5396   -8.3979  -15.1836  -14.5715
    0.8037    0.8165    0.3667   -1.5338   -8.3278  -18.2833  -14.2481
    0.8028    0.8185    0.3641   -1.5350   -8.3601  -14.9099  -14.0462
    0.7789    0.8201    0.3587   -1.5371   -8.4281  -15.1962  -14.3390
    0.7555    0.8176    0.3657   -1.5351   -8.3412  -15.1358  -14.2713
    0.8576    0.8211    0.3636   -1.5446   -8.3845  -14.7971  -15.5025
    0.8222    0.8192    0.3712   -1.5419   -8.2913  -13.9047  -14.4351
    0.7799    0.8188    0.3576   -1.5425   -8.4523  -14.7183  -14.4637
    0.7362    0.8200    0.3585   -1.5443   -8.4457  -14.5305  -13.8987
    0.6869    0.8175    0.3636   -1.5423   -8.3810  -14.9181  -14.3266
    0.6778    0.8159    0.3598   -1.5388   -8.4185  -15.2076  -14.2612
    0.8532    0.8157    0.3598   -1.5358   -8.4118  -14.2988  -14.4508
    0.8112    0.8178    0.3563   -1.5415   -8.4661  -15.0194  -14.3163
    0.7685    0.8187    0.3584   -1.5421   -8.4423  -13.9503  -14.4106
    0.7091    0.8166    0.3634   -1.5399   -8.3779  -15.3378  -14.5021
    0.7006    0.8184    0.3605   -1.5423   -8.4178  -14.4092  -14.3788
    0.6834    0.8169    0.3611   -1.5408   -8.4068  -14.5308  -14.4105
    0.8359    0.8127    0.3592   -1.5365   -8.4209  -14.8785  -14.8570
    0.7960    0.8157    0.3673   -1.5395   -8.3320  -15.0525  -14.0433
    0.7492    0.8179    0.3614   -1.5406   -8.4036  -14.4759  -14.5000
    0.7360    0.8165    0.3625   -1.5379   -8.3846  -14.4094  -14.5404
    0.7056    0.8174    0.3619   -1.5412   -8.3991  -15.0215  -14.3028
    0.6945    0.8148    0.3683   -1.5384   -8.3180  -14.7330  -14.3100
    0.8556    0.8170    0.3650   -1.5422   -8.3631  -13.8578  -14.4151
    0.8115    0.8179    0.3622   -1.5426   -8.3976  -15.9110  -14.4930
    0.7504    0.8163    0.3619   -1.5404   -8.3971  -14.7138  -14.2776
    0.7244    0.8168    0.3611   -1.5407   -8.4070  -15.8629  -14.8479
    0.7053    0.8164    0.3621   -1.5409   -8.3960  -15.1884  -14.3270
    0.6963    0.8144    0.3677   -1.5373   -8.3223  -14.9645  -14.3530
    0.8444    0.8152    0.3633   -1.5358   -8.3709  -14.4200  -14.1797
    0.8098    0.8181    0.3626   -1.5430   -8.3937  -14.5089  -14.0721
    0.7646    0.8181    0.3632   -1.5421   -8.3852  -14.8735  -14.3841
    0.7265    0.8172    0.3651   -1.5394   -8.3576  -14.6437  -14.4567
    0.6850    0.8166    0.3628   -1.5407   -8.3869  -14.6374  -14.3922
    0.6927    0.8166    0.3643   -1.5399   -8.3677  -14.5039  -14.3805
    0.8605    0.8191    0.3794   -1.5402   -8.1932  -15.3012  -14.8548
    0.8088    0.8155    0.3614   -1.5407   -8.4030  -15.7898  -14.4451
    0.7735    0.8187    0.3634   -1.5407   -8.3795  -14.4350  -14.2843
    0.7282    0.8169    0.3667   -1.5400   -8.3395  -14.9240  -14.3766
    0.6827    0.8164    0.3628   -1.5405   -8.3870  -14.5779  -14.3366
    0.6956    0.8160    0.3626   -1.5375   -8.3835  -14.5367  -14.3828
    0.8495    0.8155    0.3522   -1.5351   -8.5016  -16.1372  -14.5000
    0.8358    0.8200    0.3555   -1.5379   -8.4682  -14.0266  -14.4471
    0.8086    0.8171    0.3633   -1.5341   -8.3679  -14.3417  -14.0718
    0.8020    0.8188    0.3645   -1.5344   -8.3541  -14.1698  -14.1044
    0.7752    0.8163    0.3699   -1.5322   -8.2867  -14.3492  -14.0589
    0.7539    0.8157    0.3628   -1.5325   -8.3709  -14.3364  -14.0591
    0.8510    0.8153    0.3601   -1.5400   -8.4164  -15.4753  -14.4083
    0.8199    0.8187    0.3693   -1.5415   -8.3120  -14.3190  -14.3240
    0.7741    0.8185    0.3634   -1.5423   -8.3836  -14.4224  -14.4706
    0.7272    0.8176    0.3653   -1.5406   -8.3574  -14.5641  -14.4998
    0.6976    0.8162    0.3640   -1.5415   -8.3750  -14.4722  -14.2232
    0.6837    0.8159    0.3643   -1.5384   -8.3648  -14.7208  -14.3820
    0.8453    0.8153    0.3572   -1.5396   -8.4505  -14.9847  -13.9490
    0.8008    0.8182    0.3605   -1.5417   -8.4160  -14.1693  -14.3371
    0.7660    0.8172    0.3622   -1.5415   -8.3954  -14.6346  -14.4323
    0.7157    0.8167    0.3650   -1.5398   -8.3592  -14.7563  -14.4752
    0.7041    0.8166    0.3625   -1.5407   -8.3909  -14.5174  -14.3990
    0.6941    0.8165    0.3665   -1.5403   -8.3431  -14.8740  -14.4442
    0.8400    0.8164    0.3704   -1.5396   -8.2956  -16.1445  -14.0413
    0.7968    0.8174    0.3623   -1.5415   -8.3941  -14.4900  -14.4703
    0.7650    0.8174    0.3609   -1.5414   -8.4114  -15.1874  -14.2717
    0.7121    0.8174    0.3619   -1.5434   -8.4032  -15.0133  -14.4624
    0.6979    0.8160    0.3637   -1.5406   -8.3768  -14.7270  -14.4824
    0.6691    0.8147    0.3688   -1.5380   -8.3113  -14.9550  -14.3501
    0.8453    0.8175    0.3536   -1.5440   -8.5028  -15.2175  -14.0885
    0.8181    0.8161    0.3638   -1.5391   -8.3715  -14.7398  -14.3681
    0.7631    0.8177    0.3625   -1.5407   -8.3901  -15.1848  -14.4194
    0.7311    0.8179    0.3652   -1.5427   -8.3623  -14.5165  -14.0066
    0.6935    0.8169    0.3610   -1.5415   -8.4098  -15.1605  -14.3574
    0.6988    0.8148    0.3682   -1.5383   -8.3193  -14.9514  -14.3224
    0.8529    0.8203    0.3666   -1.5444   -8.3491  -14.0621  -14.3271
    0.8100    0.8189    0.3613   -1.5397   -8.4029  -14.5613  -14.4271
    0.7702    0.8186    0.3604   -1.5412   -8.4163  -14.0298  -14.4492
    0.7247    0.8167    0.3654   -1.5395   -8.3540  -15.2394  -14.5926
    0.6959    0.8173    0.3625   -1.5410   -8.3906  -14.5373  -14.4474
    0.7012    0.8169    0.3650   -1.5402   -8.3597  -14.4437  -14.4026
    0.8444    0.8159    0.3706   -1.5378   -8.2889  -15.3318  -14.2551
    0.8151    0.8192    0.3588   -1.5413   -8.4355  -14.8123  -14.3976
    0.7726    0.8189    0.3648   -1.5401   -8.3620  -14.6439  -14.4831
    0.7224    0.8179    0.3650   -1.5405   -8.3611  -14.3143  -14.3657
    0.6823    0.8173    0.3633   -1.5412   -8.3817  -14.8465  -14.4145
    0.6561    0.8161    0.3654   -1.5379   -8.3504  -14.9300  -14.4555
    0.8538    0.8158    0.3642   -1.5341   -8.3568  -14.7464  -14.5544
    0.8431    0.8224    0.3629   -1.5454   -8.3949  -13.8048  -14.6136
    0.8087    0.8193    0.3664   -1.5384   -8.3399  -15.0634  -14.0755
    0.8001    0.8182    0.3646   -1.5344   -8.3535  -14.3672  -14.0807
    0.7731    0.8204    0.3641   -1.5415   -8.3728  -15.4115  -14.4571
    0.7550    0.8184    0.3632   -1.5373   -8.3753  -14.6635  -14.3328

];
s.laser(:,:,1) = [.5470    0.8037    0.0997   -1.3171  -13.2315   -4.4856  -10.9062
    0.5312    0.8077    0.0671   -1.3047  -14.7970   -4.4542  -10.8450
    0.4454    0.8101    0.0685   -1.3063  -14.7196   -4.4566  -10.9312
    0.4380    0.8062    0.1048   -1.2964  -12.9899   -4.4248  -10.9082
    0.4278    0.8079    0.1052   -1.2988  -12.9783   -4.4311  -10.8606
    0.4266    0.8049    0.0914   -1.2887  -13.5233   -4.4014  -10.9568
    0.5289    0.7810    0.1277   -1.3070  -12.2137   -4.4535  -11.0845
    0.5193    0.7899    0.0534   -1.2990  -15.7039   -4.4253  -11.1884
    0.4604    0.7863    0.1029   -1.2932  -13.0576   -4.4053  -11.1351
    0.4607    0.7912    0.0833   -1.2995  -13.9182   -4.4332  -11.0317
    0.4225    0.7883    0.0826   -1.2958  -13.9473   -4.4197  -11.1943
    0.3638    0.7881    0.0972   -1.2899  -13.2786   -4.4035  -11.0354
    0.5307    0.7813    0.1084   -1.2913  -12.8423   -4.4181  -10.5002
    0.4836    0.7827    0.1367   -1.2862  -11.8994   -4.3869  -10.8663
    0.4639    0.7855    0.1034   -1.2892  -13.0285   -4.4027  -11.1196
    0.4752    0.7903    0.1215   -1.2911  -12.3848   -4.4138  -11.0602
    0.4589    0.7897    0.1041   -1.2908  -13.0049   -4.4092  -11.0405
    0.4484    0.7909    0.1106   -1.2842  -12.7508   -4.3890  -10.9680
    0.5261    0.7975    0.1243   -1.3134  -12.3379   -4.4680  -10.9631
    0.4693    0.7839    0.1424   -1.2924  -11.7463   -4.4042  -11.0173
    0.4479    0.7865    0.0882   -1.2955  -13.6829   -4.4273  -11.0986
    0.4352    0.7936    0.1162   -1.2955  -12.5739   -4.4221  -11.1877
    0.4222    0.7903    0.1100   -1.2965  -12.7969   -4.4264  -11.1653
    0.3650    0.7896    0.0976   -1.2877  -13.2588   -4.3994  -11.0890
    0.5469    0.7951    0.1237   -1.3155  -12.3611   -4.4957  -10.8930
    0.4947    0.7904    0.1124   -1.2925  -12.6998   -4.4123  -11.0041
    0.4590    0.7908    0.1043   -1.2923  -13.0008   -4.4212  -11.1327
    0.4544    0.7920    0.0674   -1.2946  -14.7596   -4.4178  -11.2445
    0.3983    0.7906    0.0868   -1.2937  -13.7434   -4.4111  -11.0678
    0.3598    0.7911    0.0925   -1.2893  -13.4790   -4.3997  -11.2370
    0.5330    0.7910    0.0837   -1.3173  -13.9329   -4.4916  -10.8667
    0.5030    0.7907    0.0563   -1.2960  -15.4801   -4.4253  -11.0915
    0.4798    0.7912    0.0684   -1.2953  -14.7028   -4.4228  -10.9834
    0.4438    0.7907    0.0895   -1.2958  -13.6242   -4.4322  -11.1846
    0.4572    0.7927    0.0446   -1.2960  -16.4135   -4.4250  -11.1929
    0.4248    0.7925    0.0665   -1.2876  -14.8008   -4.3995  -11.0910
    0.5506    0.7941    0.1850   -1.3179  -10.7364   -4.5191  -10.4338
    0.5135    0.7889    0.0744   -1.2897  -14.3553   -4.3980  -11.2526
    0.4775    0.7914    0.1113   -1.2960  -12.7479   -4.4164  -11.3307
    0.4464    0.7890    0.1358   -1.2948  -11.9439   -4.4229  -11.1242
    0.4326    0.7873    0.1136   -1.2918  -12.6577   -4.4089  -11.0894
    0.4168    0.7887    0.1107   -1.2879  -12.7532   -4.3981  -10.9638
    0.5571    0.8059    0.1773   -1.3232  -10.9194   -4.5303  -10.3473
    0.4413    0.7841    0.1791   -1.2787  -10.7900   -4.3574  -11.0679
    0.4356    0.7944    0.1823   -1.2957  -10.7524   -4.4202  -11.0443
    0.4339    0.7958    0.1876   -1.2905  -10.6255   -4.4073  -10.8624
    0.4210    0.7984    0.1760   -1.2983  -10.9014   -4.4341  -10.8576
    0.4164    0.7870    0.1886   -1.2836  -10.5899   -4.3813  -11.0562
    0.5449    0.7959    0.1219   -1.3065  -12.4028   -4.4513  -10.5352
    0.5032    0.7914    0.1268   -1.2985  -12.2281   -4.4300  -11.1815
    0.4830    0.7950    0.1052   -1.2979  -12.9767   -4.4342  -11.0672
    0.4584    0.7960    0.0994   -1.3031  -13.2161   -4.4550  -11.0951
    0.4370    0.7922    0.1010   -1.2956  -13.1379   -4.4215  -10.9362
    0.3552    0.7906    0.1152   -1.2897  -12.5976   -4.4080  -10.9095
    0.5165    0.7778    0.0953   -1.2885  -13.3560   -4.3999  -10.5598
    0.5046    0.7907    0.0800   -1.2954  -14.0734   -4.4273  -11.0131
    0.4757    0.7900    0.0988   -1.2876  -13.2091   -4.3869  -10.8220
    0.4619    0.7928    0.1347   -1.2954  -11.9784   -4.4237  -11.0818
    0.4409    0.7902    0.0960   -1.2902  -13.3300   -4.3965  -10.8197
    0.4464    0.7921    0.1011   -1.2883  -13.1191   -4.3938  -10.8988
    0.5225    0.7795    0.0759   -1.3058  -14.3026   -4.4695  -10.7957
    0.5035    0.7919    0.0861   -1.2980  -13.7826   -4.4339  -11.2226
    0.4837    0.7883    0.1030   -1.2961  -13.0581   -4.4352  -11.1383
    0.4649    0.7918    0.0881   -1.2956  -13.6874   -4.4256  -11.0837
    0.4549    0.7890    0.1079   -1.2950  -12.8721   -4.4184  -11.1304
    0.3756    0.7876    0.1051   -1.2841  -12.9544   -4.3871  -11.0350
    0.5301    0.7873    0.1380   -1.3077  -11.9022   -4.4591  -10.9142
    0.5055    0.7914    0.1191   -1.2996  -12.4827   -4.4287  -10.8308
    0.4768    0.7851    0.0925   -1.2890  -13.4782   -4.3982  -11.1646
    0.4699    0.7897    0.0965   -1.2915  -13.3144   -4.4077  -11.3156
    0.4466    0.7902    0.1130   -1.2897  -12.6754   -4.3995  -11.0909
    0.4273    0.7878    0.0993   -1.2865  -13.1871   -4.3880  -11.1392
    0.5400    0.7912    0.0994   -1.3209  -13.2506   -4.5119  -11.0204
    0.5130    0.7926    0.0841   -1.2983  -13.8801   -4.4324  -10.8917
    0.4626    0.7885    0.0960   -1.2925  -13.3336   -4.4088  -11.1787
    0.4264    0.7885    0.1218   -1.2943  -12.3823   -4.4192  -10.9737
    0.4042    0.7867    0.0956   -1.2929  -13.3538   -4.4119  -10.9524
    0.4122    0.7888    0.0930   -1.2895  -13.4587   -4.4019  -10.9914
    0.5329    0.7992    0.1789   -1.3089  -10.8546   -4.4559  -10.8120
    0.5244    0.7987    0.1144   -1.2901  -12.6244   -4.4085  -11.0174
    0.4937    0.7966    0.0900   -1.2884  -13.5860   -4.3961  -10.9830
    0.4707    0.7958    0.0805   -1.3000  -14.0597   -4.4449  -11.0821
    0.4502    0.7977    0.1036   -1.2944  -13.0347   -4.4205  -11.0376
    0.4295    0.7911    0.0869   -1.2877  -13.7261   -4.4010  -10.9340
    0.5487    0.7870    0.1733   -1.3120  -10.9900   -4.4737  -10.9230
    0.4384    0.7826    0.0592   -1.2864  -15.2600   -4.3908  -10.9011
    0.4367    0.7939    0.0737   -1.3010  -14.4134   -4.4427  -11.4043
    0.4320    0.7944    0.1461   -1.2934  -11.6458   -4.4116  -10.9493
    0.4215    0.7950    0.0305   -1.2977  -17.9405   -4.4321  -11.0080
    0.4151    0.7861    0.1466   -1.2844  -11.6148   -4.3884  -11.0272

];
s.laser(:,:,2) = [.1974    0.7616    0.1994   -1.2364  -10.2682   -4.1971  -11.5227
    0.0805    0.7805    0.2000   -1.2661  -10.3147   -4.2784  -11.9883
    0.0493    0.7813    0.1724   -1.2679  -10.9243   -4.2874  -12.4585
    0.0472    0.7828    0.1755   -1.2648  -10.8460   -4.2801  -12.7215
    0.0448    0.7811    0.1820   -1.2622  -10.6911   -4.2726  -12.1373
    0.0437    0.7779    0.1844   -1.2437  -10.6026   -4.2181  -12.8023
    0.1557    0.7522    0.1562   -1.2629  -11.3115   -4.2786  -12.4939
    0.1124    0.7668    0.1784   -1.2623  -10.7743   -4.2741  -12.1318
    0.0642    0.7651    0.1901   -1.2658  -10.5228   -4.2742  -12.1018
    0.0561    0.7693    0.2018   -1.2679  -10.2836   -4.2882  -12.5056
    0.0535    0.7665    0.1899   -1.2655  -10.5257   -4.2731  -12.6871
    0.0459    0.7608    0.1828   -1.2523  -10.6546   -4.2405  -12.5532
    0.1888    0.7527    0.1403   -1.2495  -11.7196   -4.2300  -11.6929
    0.0860    0.7584    0.1929   -1.2487  -10.4286   -4.2425  -11.6617
    0.0727    0.7636    0.1891   -1.2598  -10.5308   -4.2627  -12.4869
    0.0546    0.7666    0.1884   -1.2601  -10.5463   -4.2702  -12.3501
    0.0523    0.7638    0.1865   -1.2547  -10.5786   -4.2469  -12.3730
    0.0466    0.7656    0.1829   -1.2511  -10.6509   -4.2380  -12.2655
    0.2183    0.7627    0.1825   -1.2624  -10.6804   -4.2643  -11.9394
    0.1244    0.7689    0.1883   -1.2643  -10.5568   -4.2831  -12.4964
    0.0740    0.7722    0.2042   -1.2736  -10.2457   -4.3068  -12.9023
    0.0611    0.7685    0.2068   -1.2626  -10.1730   -4.2765  -11.6624
    0.0566    0.7697    0.1826   -1.2673  -10.6896   -4.2924  -12.2692
    0.0476    0.7652    0.1906   -1.2532  -10.4855   -4.2482  -12.1237
    0.2272    0.7567    0.2146   -1.2487   -9.9917   -4.2358  -12.9568
    0.0930    0.7638    0.1958   -1.2501  -10.3692   -4.2433  -12.1064
    0.0550    0.7657    0.1876   -1.2590  -10.5616   -4.2639  -12.4836
    0.0505    0.7671    0.1711   -1.2550  -10.9278   -4.2528  -12.5582
    0.0481    0.7670    0.1909   -1.2573  -10.4886   -4.2583  -12.5954
    0.0423    0.7649    0.1908   -1.2482  -10.4719   -4.2311  -12.5014
    0.2104    0.7615    0.1934   -1.2630  -10.4441   -4.2756  -12.0942
    0.0866    0.7590    0.1713   -1.2551  -10.9228   -4.2675  -11.8391
    0.0683    0.7674    0.1613   -1.2699  -11.1975   -4.2988  -11.8972
    0.0595    0.7687    0.1895   -1.2608  -10.5253   -4.2628  -12.5525
    0.0581    0.7703    0.1690   -1.2669  -11.0017   -4.2870  -12.0468
    0.0570    0.7677    0.1715   -1.2573  -10.9232   -4.2596  -12.1630
    0.1738    0.7555    0.1575   -1.2485  -11.2515   -4.2466  -12.6878
    0.0875    0.7619    0.2045   -1.2491  -10.1908   -4.2367  -12.5506
    0.0583    0.7623    0.1893   -1.2556  -10.5187   -4.2503  -12.2960
    0.0501    0.7666    0.2084   -1.2566  -10.1287   -4.2529  -13.1490
    0.0476    0.7650    0.1854   -1.2602  -10.6124   -4.2636  -12.8875
    0.0447    0.7634    0.1965   -1.2462  -10.3481   -4.2242  -12.5763
    0.2094    0.7779    0.2038   -1.2786  -10.2628   -4.3623  -17.0597
    0.0675    0.7590    0.2294   -1.2470   -9.7147   -4.2252  -12.3626
    0.0524    0.7643    0.2036   -1.2569  -10.2236   -4.2585  -13.3562
    0.0461    0.7716    0.2197   -1.2615   -9.9222   -4.2724  -13.7508
    0.0450    0.7739    0.2133   -1.2615  -10.0431   -4.2789  -13.5569
    0.0429    0.7575    0.2122   -1.2409  -10.0235   -4.2080  -12.6831
    0.1843    0.7566    0.1765   -1.2567  -10.8038   -4.2588  -11.3208
    0.0924    0.7664    0.1787   -1.2506  -10.7419   -4.2389  -12.1811
    0.0644    0.7716    0.2071   -1.2626  -10.1658   -4.2714  -12.0826
    0.0569    0.7710    0.1786   -1.2622  -10.7692   -4.2620  -12.4814
    0.0533    0.7709    0.1908   -1.2645  -10.5048   -4.2772  -12.4506
    0.0473    0.7638    0.1910   -1.2503  -10.4727   -4.2362  -12.5272
    0.1737    0.7564    0.1856   -1.2531  -10.5921   -4.2639  -11.9729
    0.1137    0.7676    0.2138   -1.2640  -10.0378   -4.2739  -12.3542
    0.0643    0.7658    0.1968   -1.2601  -10.3696   -4.2712  -12.3530
    0.0553    0.7693    0.2086   -1.2634  -10.1380   -4.2811  -12.1380
    0.0557    0.7702    0.1928   -1.2670  -10.4678   -4.2939  -12.2440
    0.0473    0.7657    0.1912   -1.2543  -10.4759   -4.2532  -12.0174
    0.1917    0.7451    0.2033   -1.2380  -10.1916   -4.2139  -11.8577
    0.1059    0.7562    0.1669   -1.2575  -11.0333   -4.2630  -12.6388
    0.0684    0.7618    0.2064   -1.2585  -10.1725   -4.2727  -12.0921
    0.0517    0.7651    0.1794   -1.2603  -10.7474   -4.2706  -12.0939
    0.0512    0.7618    0.1926   -1.2580  -10.4540   -4.2703  -12.3093
    0.0457    0.7624    0.1925   -1.2475  -10.4350   -4.2280  -12.4157
    0.1977    0.7434    0.1765   -1.2398  -10.7699   -4.2113  -11.4582
    0.0852    0.7585    0.2018   -1.2506  -10.2468   -4.2428  -13.0745
    0.0641    0.7608    0.1652   -1.2620  -11.0862   -4.2675  -12.2664
    0.0508    0.7615    0.2000   -1.2590  -10.3020   -4.2574  -12.4901
    0.0493    0.7620    0.1824   -1.2601  -10.6800   -4.2625  -12.4000
    0.0479    0.7596    0.1912   -1.2481  -10.4622   -4.2263  -12.2022
    0.1480    0.7487    0.1727   -1.2454  -10.8701   -4.2366  -11.3407
    0.0798    0.7601    0.2035   -1.2567  -10.2262   -4.2542  -11.9505
    0.0612    0.7671    0.1943   -1.2586  -10.4181   -4.2693  -12.3214
    0.0533    0.7679    0.2010   -1.2618  -10.2857   -4.2790  -12.5384
    0.0456    0.7650    0.1941   -1.2590  -10.4245   -4.2619  -12.7738
    0.0454    0.7646    0.1918   -1.2515  -10.4571   -4.2408  -12.4587
    0.1708    0.7627    0.2227   -1.2492   -9.8399   -4.2504  -11.6699
    0.0730    0.7656    0.1792   -1.2519  -10.7346   -4.2579  -12.5350
    0.0526    0.7717    0.1884   -1.2650  -10.5564   -4.3024  -13.4169
    0.0461    0.7606    0.1865   -1.2511  -10.5696   -4.2310  -11.5645
    0.0467    0.7626    0.1899   -1.2581  -10.5120   -4.2677  -12.2370
    0.0415    0.7602    0.1954   -1.2472  -10.3726   -4.2310  -12.3805
    0.1952    0.7564    0.2084   -1.2506  -10.1148   -4.2619  -13.2074
    0.0882    0.7660    0.1800   -1.2679  -10.7484   -4.3018  -14.0429
    0.0616    0.7648    0.1848   -1.2597  -10.6248   -4.2721  -12.3469
    0.0496    0.7682    0.2022   -1.2592  -10.2581   -4.2591  -12.4881
    0.0472    0.7654    0.1829   -1.2570  -10.6609   -4.2631  -13.1414
    0.0467    0.7601    0.2022   -1.2491  -10.2373   -4.2340  -12.8642

];
s.laser(:,:,3) = [.1692    0.7598    0.0875   -1.2332  -13.5900   -4.1969  -12.3273
    0.0930    0.7773    0.0457   -1.2648  -16.2562   -4.2974  -14.2466
    0.0648    0.7941    0.1195   -1.2888  -12.4454   -4.3535  -14.9672
    0.0478    0.7848    0.0345   -1.2686  -17.3907   -4.2955  -15.2863
    0.0463    0.7872    0.1111   -1.2718  -12.7051   -4.3085  -17.2721
    0.0446    0.7793    0.0829   -1.2577  -13.8552   -4.2587  -16.0078
    0.5254    0.7488    0.0941   -1.2512  -13.3309   -4.2497  -12.2028
    0.4068    0.7496    0.1187   -1.2499  -12.3966   -4.2417  -13.1286
    0.0658    0.7598    0.0773   -1.2677  -14.1564   -4.2921  -14.6981
    0.0557    0.7593    0.1141   -1.2572  -12.5704   -4.2536  -12.2210
    0.0481    0.7605    0.0623   -1.2621  -15.0126   -4.2670  -13.5687
    0.0430    0.7594    0.0706   -1.2536  -14.4893   -4.2510  -13.0732
    0.5135    0.7473    0.1146   -1.2526  -12.5422   -4.2516  -13.0404
    0.4727    0.7505    0.1001   -1.2502  -13.0818   -4.2399  -13.0176
    0.0636    0.7656    0.0823   -1.2758  -13.9203   -4.3229  -14.0959
    0.0476    0.7605    0.1438   -1.2548  -11.6321   -4.2459  -14.1913
    0.0491    0.7611    0.0444   -1.2641  -16.3694   -4.2868  -13.7661
    0.0457    0.7647    0.0376   -1.2568  -17.0208   -4.2593  -14.9255
    0.5037    0.7524    0.1302   -1.2470  -12.0147   -4.2486  -13.2091
    0.1054    0.7603    0.0610   -1.2544  -15.0781   -4.2414  -13.7987
    0.0692    0.7685    0.1279   -1.2722  -12.1402   -4.3194  -13.6649
    0.0555    0.7648    0.0875   -1.2641  -13.6491   -4.2666  -13.4738
    0.0500    0.7621    0.0688   -1.2643  -14.6147   -4.2798  -14.8269
    0.0508    0.7642    0.0815   -1.2565  -13.9192   -4.2565  -14.6356
    0.4921    0.7491    0.0759   -1.2419  -14.1781   -4.2234  -13.2962
    0.3827    0.7547    0.0888   -1.2555  -13.5735   -4.2506  -14.0427
    0.0813    0.7617    0.1171   -1.2623  -12.4771   -4.2780  -15.9091
    0.0605    0.7681    0.0517   -1.2684  -15.7674   -4.2816  -13.7041
    0.0572    0.7607    0.0839   -1.2657  -13.8250   -4.2782  -16.2025
    0.0517    0.7596    0.0707   -1.2550  -14.4865   -4.2474  -13.8186
    0.4902    0.7438    0.1710   -1.2397  -10.8988   -4.2326  -13.4720
    0.4549    0.7587    0.0751   -1.2574  -14.2528   -4.2706  -16.7549
    0.0714    0.7634    0.0916   -1.2622  -13.4651   -4.2802  -14.4833
    0.0465    0.7662    0.1399   -1.2649  -11.7639   -4.2970  -15.5138
    0.0457    0.7629    0.0901   -1.2622  -13.5284   -4.2737  -14.3050
    0.0429    0.7620    0.0894   -1.2567  -13.5500   -4.2608  -15.5511
    0.5360    0.7598    0.2035   -1.2704  -10.2512   -4.3320  -12.2386
    0.0960    0.7535    0.0914   -1.2371  -13.4208   -4.2059  -14.1387
    0.0637    0.7581    0.1124   -1.2589  -12.6351   -4.2623  -13.4989
    0.0524    0.7666    0.0893   -1.2679  -13.5753   -4.3006  -14.3009
    0.0483    0.7648    0.0977   -1.2702  -13.2219   -4.2991  -14.8637
    0.0441    0.7624    0.0630   -1.2585  -14.9556   -4.2645  -15.1862
    0.1845    0.7669    0.2134   -1.2505  -10.0169   -4.2561  -13.3905
    0.0765    0.7634    0.1581   -1.2424  -11.2228   -4.2232  -13.9049
    0.0562    0.7668    0.1106   -1.2516  -12.6829   -4.2402  -13.2323
    0.0467    0.7773    0.0855   -1.2674  -13.7516   -4.2903  -14.0134
    0.0456    0.7772    0.0858   -1.2648  -13.7295   -4.2826  -14.3604
    0.0439    0.7725    0.1261   -1.2569  -12.1652   -4.2580  -13.9839
    0.5174    0.7545    0.1513   -1.2431  -11.4028   -4.2437  -12.7545
    0.4322    0.7685    0.1096   -1.2619  -12.7393   -4.2768  -15.4250
    0.4658    0.7692    0.0925   -1.2688  -13.4385   -4.2991  -14.3337
    0.0580    0.7686    0.1161   -1.2690  -12.5252   -4.3108  -13.1032
    0.0541    0.7681    0.0695   -1.2696  -14.5879   -4.3008  -14.8645
    0.0442    0.7637    0.0955   -1.2577  -13.2855   -4.2603  -15.2511
    0.4933    0.7542    0.1044   -1.2588  -12.9286   -4.2725  -14.1490
    0.4506    0.7684    0.0481   -1.2662  -16.0569   -4.3028  -15.3723
    0.0691    0.7662    0.0716   -1.2618  -14.4523   -4.2725  -14.2668
    0.0481    0.7727    0.1042   -1.2693  -12.9596   -4.3045  -13.5531
    0.0458    0.7695    0.0626   -1.2658  -14.9954   -4.2851  -15.1798
    0.0450    0.7689    0.0422   -1.2600  -16.5692   -4.2663  -15.2049
    0.5058    0.7518    0.1080   -1.2411  -12.7590   -4.2352  -13.6974
    0.4811    0.7555    0.0959   -1.2515  -13.2578   -4.2509  -13.9779
    0.0651    0.7650    0.0886   -1.2706  -13.6126   -4.3073  -18.8532
    0.0507    0.7665    0.0569   -1.2713  -15.3895   -4.3015  -14.6944
    0.0486    0.7649    0.0669   -1.2684  -14.7346   -4.2951  -14.6393
    0.0417    0.7601    0.0938   -1.2553  -13.3551   -4.2577  -15.1703
    0.4915    0.7473    0.1154   -1.2384  -12.4861   -4.2332  -14.6977
    0.4747    0.7553    0.1069   -1.2477  -12.8131   -4.2314  -17.9223
    0.0965    0.7630    0.0734   -1.2697  -14.3645   -4.3107  -13.4882
    0.0573    0.7639    0.0734   -1.2586  -14.3436   -4.2532  -13.7681
    0.0554    0.7629    0.0523   -1.2656  -15.7200   -4.2868  -14.4975
    0.0445    0.7592    0.0490   -1.2585  -15.9666   -4.2581  -14.2122
    0.5286    0.7503    0.0760   -1.2485  -14.1832   -4.2437  -14.6585
    0.4485    0.7575    0.1085   -1.2559  -12.7695   -4.2495  -13.8873
    0.4286    0.7699    0.0979   -1.2742  -13.2204   -4.3098  -14.3698
    0.0524    0.7704    0.1454   -1.2704  -11.6195   -4.3045  -15.4735
    0.0511    0.7684    0.0563   -1.2690  -15.4313   -4.2945  -14.9243
    0.0471    0.7673    0.0738   -1.2624  -14.3323   -4.2720  -16.2176
    0.4993    0.7708    0.1457   -1.2500  -11.5671   -4.2762  -13.8792
    0.4721    0.7727    0.0707   -1.2656  -14.5094   -4.2950  -14.8562
    0.0605    0.7642    0.1013   -1.2539  -13.0438   -4.2483  -13.8209
    0.0618    0.7710    0.1138   -1.2646  -12.5945   -4.2963  -12.8955
    0.0571    0.7671    0.0551   -1.2622  -15.5046   -4.2733  -12.7741
    0.0531    0.7667    0.0857   -1.2483  -13.7017   -4.2329  -14.1518
    0.1597    0.7577    0.1729   -1.2555  -10.8845   -4.3034  -14.3235
    0.0881    0.7595    0.1462   -1.2465  -11.5471   -4.2149  -13.6556
    0.0599    0.7764    0.1091   -1.2746  -12.7841   -4.3137  -13.8261
    0.0482    0.7736    0.0911   -1.2668  -13.4940   -4.2859  -14.2181
    0.0460    0.7719    0.0691   -1.2639  -14.5959   -4.2671  -13.3443
    0.0452    0.7682    0.1072   -1.2580  -12.8218   -4.2630  -15.7330

];
s.lixo(:,:,1) = [.7217    0.6903    0.3453   -1.2955   -8.1072  -11.0164  -10.8937
    0.6986    0.6923    0.3612   -1.2990   -7.9226  -10.6901  -10.6418
    0.6864    0.6871    0.3547   -1.2829   -7.9676  -10.6899  -11.0863
    0.7012    0.6900    0.3574   -1.2856   -7.9413  -11.1317  -10.8006
    0.6591    0.6891    0.3551   -1.2843   -7.9665  -10.8670  -10.6902
    0.6466    0.6855    0.3574   -1.2787   -7.9279  -10.9605  -10.8013
    0.7143    0.6878    0.3550   -1.2944   -7.9868  -10.7885  -10.6689
    0.6818    0.6875    0.3541   -1.2874   -7.9839  -10.9392  -10.8536
    0.6337    0.6866    0.3565   -1.2884   -7.9573  -11.1207  -10.7597
    0.6065    0.6839    0.3601   -1.2853   -7.9085  -11.1115  -10.7200
    0.5942    0.6867    0.3579   -1.2889   -7.9421  -11.0876  -10.7540
    0.5708    0.6844    0.3567   -1.2844   -7.9469  -11.0103  -10.7212
    0.7117    0.6853    0.3650   -1.2861   -7.8520  -10.6244  -10.6054
    0.6967    0.6866    0.3598   -1.2869   -7.9145  -11.0656  -10.7492
    0.6359    0.6852    0.3546   -1.2863   -7.9758  -11.0252  -10.7614
    0.6094    0.6870    0.3535   -1.2901   -7.9969  -11.2551  -10.8062
    0.5699    0.6843    0.3583   -1.2857   -7.9307  -11.1423  -10.7820
    0.5820    0.6847    0.3567   -1.2843   -7.9462  -11.1405  -10.8025
    0.6984    0.6832    0.3635   -1.2848   -7.8660  -11.0729  -10.5531
    0.6716    0.6860    0.3520   -1.2878   -8.0103  -10.9822  -10.7770
    0.6677    0.6893    0.3543   -1.2906   -7.9886  -11.1891  -10.7619
    0.6366    0.6884    0.3541   -1.2897   -7.9891  -11.1458  -10.8891
    0.5965    0.6867    0.3559   -1.2896   -7.9669  -11.1645  -10.8746
    0.5977    0.6833    0.3578   -1.2819   -7.9294  -11.0320  -10.7951
    0.7213    0.6904    0.3570   -1.2929   -7.9594  -11.0390  -10.7442
    0.6685    0.6841    0.3595   -1.2857   -7.9159  -11.3126  -10.7563
    0.6662    0.6884    0.3560   -1.2880   -7.9623  -10.9322  -10.7920
    0.6490    0.6859    0.3584   -1.2836   -7.9249  -11.0470  -10.8618
    0.6218    0.6879    0.3572   -1.2891   -7.9506  -11.1341  -10.7838
    0.6088    0.6845    0.3585   -1.2835   -7.9241  -11.0972  -10.8261
    0.7194    0.6884    0.3627   -1.2859   -7.8783  -10.7455  -10.6703
    0.6973    0.6887    0.3518   -1.2883   -8.0140  -11.1192  -10.8087
    0.6369    0.6863    0.3571   -1.2858   -7.9447  -11.1005  -10.8113
    0.6145    0.6876    0.3545   -1.2885   -7.9816  -11.0887  -10.8168
    0.5642    0.6860    0.3561   -1.2878   -7.9615  -11.1514  -10.8470
    0.5752    0.6857    0.3612   -1.2855   -7.8951  -11.0096  -10.8463
    0.7068    0.6820    0.3494   -1.2801   -8.0259  -10.8045  -10.9919
    0.6975    0.6882    0.3595   -1.2868   -7.9182  -10.9112  -10.9289
    0.6348    0.6863    0.3594   -1.2870   -7.9200  -11.2306  -10.7577
    0.6171    0.6820    0.3590   -1.2762   -7.9032  -11.2896  -10.6751
    0.5944    0.6873    0.3549   -1.2881   -7.9764  -11.0675  -10.8021
    0.5752    0.6844    0.3583   -1.2836   -7.9259  -11.1134  -10.7340
    0.7137    0.6864    0.3543   -1.2837   -7.9738  -10.5928  -10.5003
    0.7067    0.6902    0.3588   -1.2907   -7.9346  -11.3242  -10.7992
    0.7029    0.6926    0.3562   -1.2944   -7.9728  -11.1807  -10.8519
    0.6992    0.6901    0.3569   -1.2875   -7.9504  -11.1052  -10.7647
    0.6776    0.6916    0.3543   -1.2924   -7.9923  -11.0954  -10.8048
    0.6428    0.6809    0.3596   -1.2698   -7.8835  -11.0142  -10.4812
    0.7213    0.6895    0.3516   -1.2916   -8.0219  -11.0580  -10.8543
    0.6837    0.6863    0.3521   -1.2873   -8.0080  -11.1745  -10.6873
    0.6315    0.6864    0.3581   -1.2886   -7.9389  -11.0602  -10.7423
    0.6185    0.6856    0.3558   -1.2870   -7.9631  -11.0754  -10.6861
    0.5885    0.6879    0.3561   -1.2901   -7.9651  -11.0577  -10.7516
    0.5900    0.6826    0.3539   -1.2764   -7.9647  -11.0035  -10.5454
    0.7117    0.6844    0.3645   -1.2843   -7.8531  -11.3603  -11.0417
    0.6862    0.6879    0.3549   -1.2893   -7.9785  -11.0142  -10.7301
    0.6316    0.6853    0.3555   -1.2859   -7.9647  -11.0531  -10.8239
    0.6246    0.6864    0.3528   -1.2870   -7.9989  -10.9485  -10.7859
    0.5808    0.6857    0.3566   -1.2881   -7.9555  -10.9991  -10.8191
    0.5991    0.6856    0.3525   -1.2857   -8.0005  -11.1190  -10.7692
    0.7180    0.6892    0.3631   -1.2911   -7.8838  -11.0383  -10.7803
    0.6767    0.6863    0.3546   -1.2871   -7.9772  -10.9249  -10.7469
    0.6557    0.6884    0.3548   -1.2904   -7.9820  -10.9870  -10.7880
    0.6177    0.6878    0.3561   -1.2900   -7.9650  -11.0913  -10.8841
    0.5832    0.6862    0.3546   -1.2895   -7.9824  -10.9812  -10.7920
    0.5793    0.6812    0.3594   -1.2754   -7.8967  -11.0250  -10.6415
    0.7089    0.6856    0.3667   -1.2856   -7.8301  -11.5874  -10.8455
    0.6815    0.6879    0.3535   -1.2907   -7.9978  -11.1929  -10.8744
    0.6421    0.6877    0.3554   -1.2918   -7.9776  -11.1639  -10.8340
    0.6197    0.6868    0.3529   -1.2882   -8.0005  -11.2759  -10.9061
    0.5944    0.6870    0.3544   -1.2893   -7.9850  -11.1362  -10.8483
    0.5920    0.6806    0.3616   -1.2739   -7.8672  -11.0418  -10.6649
    0.7192    0.6890    0.3637   -1.2918   -7.8775  -10.9153  -10.8212
    0.7049    0.6911    0.3598   -1.2918   -7.9244  -11.1266  -10.8439
    0.6332    0.6890    0.3566   -1.2923   -7.9644  -11.3396  -10.9161
    0.6076    0.6895    0.3565   -1.2944   -7.9700  -11.3422  -10.8523
    0.5520    0.6866    0.3567   -1.2901   -7.9578  -11.2046  -10.8254
    0.5843    0.6865    0.3594   -1.2879   -7.9217  -11.2052  -10.8023
    0.7017    0.6855    0.3642   -1.2916   -7.8723  -11.2947  -10.7677
    0.6917    0.6878    0.3581   -1.2862   -7.9332  -11.0436  -10.8356
    0.6335    0.6871    0.3615   -1.2907   -7.9027  -11.1142  -10.7839
    0.6253    0.6854    0.3610   -1.2859   -7.8983  -10.9597  -10.8352
    0.5922    0.6853    0.3582   -1.2877   -7.9357  -11.1309  -10.7768
    0.5770    0.6810    0.3616   -1.2749   -7.8696  -11.0470  -10.5793
    0.7109    0.6855    0.3481   -1.2878   -8.0573  -10.9669  -10.8773
    0.7020    0.6855    0.3468   -1.2799   -8.0583  -11.0448  -10.8827
    0.6893    0.6884    0.3532   -1.2897   -7.9995  -10.9971  -10.8087
    0.6960    0.6885    0.3567   -1.2845   -7.9474  -11.2089  -10.8381
    0.6626    0.6870    0.3573   -1.2858   -7.9428  -11.1128  -10.8193
    0.6405    0.6805    0.3607   -1.2690   -7.8686  -11.0532  -10.5091

];
s.lixo(:,:,2) = [.7165    0.6856    0.3463   -1.2754   -8.0543  -10.9969  -10.6941
    0.7048    0.6867    0.3476   -1.2736   -8.0357  -11.0094  -10.4255
    0.7024    0.6889    0.3461   -1.2753   -8.0575  -11.1018  -10.6957
    0.6915    0.6888    0.3447   -1.2734   -8.0713  -11.0204  -10.5770
    0.6759    0.6922    0.3428   -1.2831   -8.1139  -11.1290  -10.5379
    0.6504    0.6855    0.3453   -1.2677   -8.0519  -11.1018  -10.4656
    0.7172    0.6871    0.3539   -1.2784   -7.9688  -10.9335  -10.5774
    0.6974    0.6879    0.3420   -1.2773   -8.1126  -11.3653  -10.4782
    0.6411    0.6854    0.3454   -1.2741   -8.0637  -11.3053  -10.4839
    0.6197    0.6846    0.3479   -1.2736   -8.0317  -11.3337  -10.4660
    0.5919    0.6855    0.3470   -1.2755   -8.0462  -11.2533  -10.5197
    0.5784    0.6836    0.3449   -1.2704   -8.0622  -11.1976  -10.4588
    0.7140    0.6868    0.3455   -1.2756   -8.0652  -10.9792  -10.8010
    0.7049    0.6877    0.3466   -1.2745   -8.0495  -11.2437  -10.6399
    0.6460    0.6863    0.3451   -1.2755   -8.0705  -11.1009  -10.5854
    0.6337    0.6879    0.3438   -1.2782   -8.0921  -11.3083  -10.4922
    0.5960    0.6868    0.3451   -1.2773   -8.0740  -11.1161  -10.5660
    0.6129    0.6864    0.3434   -1.2739   -8.0880  -11.1596  -10.4956
    0.6991    0.6813    0.3276   -1.2687   -8.2765  -11.2245  -10.3366
    0.6767    0.6845    0.3471   -1.2732   -8.0408  -11.0639  -10.4536
    0.6587    0.6856    0.3459   -1.2747   -8.0587  -11.2531  -10.4828
    0.6343    0.6873    0.3430   -1.2770   -8.0993  -11.3565  -10.6336
    0.6142    0.6861    0.3430   -1.2765   -8.0977  -11.2519  -10.5405
    0.5909    0.6830    0.3466   -1.2696   -8.0404  -11.2422  -10.5496
    0.7065    0.6832    0.3491   -1.2707   -8.0116  -11.4522  -10.4555
    0.6920    0.6858    0.3454   -1.2745   -8.0639  -11.1635  -10.5594
    0.6500    0.6867    0.3422   -1.2788   -8.1127  -11.1996  -10.5234
    0.6323    0.6841    0.3467   -1.2718   -8.0427  -11.1113  -10.5454
    0.6070    0.6849    0.3449   -1.2765   -8.0742  -11.3470  -10.5007
    0.5733    0.6822    0.3483   -1.2711   -8.0215  -11.1628  -10.5386
    0.7154    0.6879    0.3456   -1.2797   -8.0718  -11.3917  -10.5481
    0.6986    0.6869    0.3456   -1.2735   -8.0604  -11.1164  -10.5760
    0.6475    0.6859    0.3420   -1.2753   -8.1080  -11.0705  -10.4737
    0.6469    0.6869    0.3416   -1.2747   -8.1116  -11.1380  -10.4867
    0.6092    0.6864    0.3414   -1.2764   -8.1177  -11.1297  -10.5274
    0.6118    0.6860    0.3485   -1.2742   -8.0253  -11.0725  -10.5431
    0.7171    0.6871    0.3420   -1.2765   -8.1099  -10.9952  -10.4985
    0.6956    0.6867    0.3460   -1.2739   -8.0555  -11.2369  -10.5463
    0.6281    0.6865    0.3445   -1.2759   -8.0779  -11.5184  -10.4623
    0.6116    0.6857    0.3460   -1.2738   -8.0551  -11.7176  -10.5041
    0.5877    0.6858    0.3443   -1.2745   -8.0777  -11.2390  -10.4754
    0.5815    0.6838    0.3472   -1.2698   -8.0328  -11.2332  -10.4211
    0.7144    0.6884    0.3521   -1.2793   -7.9919  -11.1783  -10.5093
    0.7077    0.6882    0.3369   -1.2735   -8.1686  -11.0902  -10.6357
    0.6973    0.6889    0.3426   -1.2763   -8.1022  -11.4678  -10.4302
    0.6958    0.6885    0.3443   -1.2740   -8.0775  -11.1207  -10.4672
    0.6735    0.6887    0.3459   -1.2757   -8.0604  -11.3062  -10.5221
    0.6448    0.6844    0.3454   -1.2666   -8.0485  -11.2135  -10.4506
    0.7126    0.6851    0.3457   -1.2768   -8.0649  -11.1715  -10.5244
    0.6925    0.6865    0.3412   -1.2749   -8.1177  -11.2799  -10.4755
    0.6320    0.6851    0.3436   -1.2760   -8.0891  -11.3152  -10.5180
    0.6092    0.6850    0.3466   -1.2756   -8.0513  -11.2624  -10.4250
    0.5890    0.6859    0.3430   -1.2763   -8.0973  -11.2607  -10.5549
    0.5758    0.6832    0.3451   -1.2711   -8.0611  -11.2109  -10.5083
    0.7137    0.6871    0.3419   -1.2784   -8.1147  -11.5530  -10.6077
    0.6949    0.6888    0.3471   -1.2775   -8.0498  -11.4430  -10.5688
    0.6168    0.6854    0.3424   -1.2749   -8.1018  -11.1359  -10.5162
    0.6183    0.6874    0.3420   -1.2773   -8.1120  -11.0979  -10.4910
    0.5738    0.6865    0.3458   -1.2779   -8.0661  -11.2451  -10.5379
    0.5834    0.6859    0.3436   -1.2754   -8.0889  -11.2267  -10.5222
    0.7083    0.6865    0.3352   -1.2789   -8.1999  -11.3801  -10.5259
    0.6889    0.6854    0.3472   -1.2730   -8.0396  -11.1427  -10.5118
    0.6530    0.6885    0.3456   -1.2801   -8.0737  -11.1811  -10.4863
    0.6333    0.6869    0.3434   -1.2757   -8.0920  -11.2530  -10.6125
    0.6041    0.6856    0.3462   -1.2762   -8.0573  -11.1527  -10.5542
    0.5872    0.6830    0.3467   -1.2692   -8.0382  -11.1265  -10.5170
    0.7021    0.6844    0.3472   -1.2787   -8.0504  -11.1868  -10.4915
    0.6913    0.6880    0.3447   -1.2774   -8.0791  -11.1609  -10.4136
    0.6577    0.6862    0.3416   -1.2775   -8.1172  -11.2210  -10.4978
    0.6438    0.6854    0.3435   -1.2735   -8.0854  -11.3510  -10.5348
    0.6194    0.6859    0.3443   -1.2766   -8.0824  -11.0893  -10.5420
    0.5940    0.6827    0.3472   -1.2697   -8.0325  -11.1747  -10.5215
    0.7150    0.6869    0.3430   -1.2757   -8.0964  -11.2063  -10.5906
    0.6944    0.6862    0.3458   -1.2733   -8.0563  -11.3312  -10.5754
    0.6260    0.6848    0.3417   -1.2724   -8.1061  -11.1299  -10.4626
    0.6019    0.6867    0.3426   -1.2763   -8.1022  -11.3199  -10.4792
    0.5765    0.6849    0.3459   -1.2738   -8.0563  -11.1187  -10.5176
    0.5768    0.6840    0.3467   -1.2712   -8.0414  -11.2254  -10.5191
    0.7131    0.6878    0.3388   -1.2800   -8.1572  -11.5132  -10.5184
    0.6872    0.6872    0.3451   -1.2762   -8.0718  -11.2915  -10.5041
    0.6487    0.6860    0.3420   -1.2755   -8.1082  -11.0216  -10.4813
    0.6263    0.6863    0.3454   -1.2754   -8.0663  -10.8898  -10.5782
    0.5885    0.6856    0.3432   -1.2751   -8.0929  -11.0891  -10.5161
    0.5812    0.6832    0.3456   -1.2711   -8.0553  -11.1851  -10.4889
    0.7084    0.6824    0.3412   -1.2697   -8.1063  -11.0588  -10.5148
    0.7058    0.6873    0.3417   -1.2743   -8.1094  -11.0833  -10.6090
    0.6907    0.6902    0.3373   -1.2821   -8.1803  -10.9856  -10.5400
    0.6926    0.6870    0.3459   -1.2712   -8.0513  -11.3325  -10.5700
    0.6701    0.6869    0.3468   -1.2732   -8.0446  -11.1409  -10.5897
    0.6343    0.6819    0.3469   -1.2627   -8.0228  -11.1815  -10.4582

];
s.lixo(:,:,3) = [.7016    0.7734    0.2618   -1.4395   -9.5557  -12.9519   -8.6551
    0.6939    0.7712    0.2661   -1.4380   -9.4858  -12.6209   -8.6456
    0.6964    0.7768    0.2528   -1.4455   -9.7137  -12.6355   -8.7111
    0.6883    0.7722    0.2658   -1.4373   -9.4890  -12.6113   -8.6644
    0.6876    0.7776    0.2664   -1.4481   -9.5018  -12.6441   -8.7778
    0.6746    0.7716    0.2689   -1.4382   -9.4423  -12.7942   -8.6655
    0.6898    0.7580    0.2872   -1.4440   -9.1806  -13.0075   -8.6803
    0.6779    0.7652    0.2581   -1.4483   -9.6330  -12.4490   -8.7189
    0.6578    0.7646    0.2645   -1.4464   -9.5267  -12.6980   -8.7060
    0.6599    0.7646    0.2619   -1.4431   -9.5623  -12.9228   -8.7031
    0.6289    0.7651    0.2655   -1.4453   -9.5093  -12.6382   -8.7053
    0.5926    0.7630    0.2651   -1.4404   -9.5059  -12.6164   -8.6650
    0.6697    0.7592    0.2629   -1.4394   -9.5385  -12.7666   -8.6878
    0.6690    0.7637    0.2728   -1.4452   -9.3959  -12.8573   -8.6842
    0.6554    0.7658    0.2624   -1.4463   -9.5602  -12.6357   -8.6930
    0.6408    0.7644    0.2585   -1.4423   -9.6150  -12.6012   -8.6867
    0.6302    0.7661    0.2641   -1.4445   -9.5304  -12.7360   -8.6975
    0.6321    0.7660    0.2640   -1.4437   -9.5307  -12.8418   -8.6978
    0.6648    0.7584    0.2620   -1.4486   -9.5704  -13.2384   -8.7194
    0.6589    0.7611    0.2660   -1.4467   -9.5047  -12.8105   -8.7152
    0.6469    0.7613    0.2657   -1.4433   -9.5017  -12.9049   -8.7080
    0.6375    0.7627    0.2655   -1.4429   -9.5050  -12.8018   -8.7330
    0.6287    0.7642    0.2646   -1.4446   -9.5216  -12.7377   -8.7161
    0.5897    0.7602    0.2661   -1.4402   -9.4895  -12.7222   -8.6596
    0.6701    0.7571    0.2714   -1.4458   -9.4195  -12.6584   -8.6997
    0.6537    0.7608    0.2632   -1.4460   -9.5464  -13.0440   -8.7037
    0.6389    0.7611    0.2681   -1.4440   -9.4661  -12.7088   -8.7035
    0.6336    0.7628    0.2611   -1.4428   -9.5737  -12.7132   -8.6866
    0.6208    0.7637    0.2628   -1.4449   -9.5515  -12.7262   -8.7011
    0.5956    0.7613    0.2685   -1.4401   -9.4533  -12.5935   -8.6659
    0.6692    0.7598    0.2638   -1.4430   -9.5305  -12.6773   -8.6844
    0.6673    0.7649    0.2709   -1.4468   -9.4289  -12.4648   -8.6977
    0.6491    0.7653    0.2623   -1.4446   -9.5588  -12.6772   -8.7017
    0.6343    0.7650    0.2618   -1.4443   -9.5664  -12.4530   -8.7106
    0.6290    0.7662    0.2634   -1.4450   -9.5423  -12.7706   -8.7064
    0.6088    0.7656    0.2674   -1.4431   -9.4754  -12.7425   -8.6933
    0.6934    0.7644    0.2752   -1.4409   -9.3518  -12.8895   -8.6892
    0.6723    0.7631    0.2674   -1.4465   -9.4820  -13.3262   -8.7172
    0.6560    0.7640    0.2682   -1.4459   -9.4682  -12.6820   -8.7057
    0.6542    0.7642    0.2711   -1.4449   -9.4219  -12.7846   -8.6947
    0.6185    0.7638    0.2646   -1.4447   -9.5219  -12.6680   -8.7046
    0.5869    0.7627    0.2696   -1.4410   -9.4370  -12.9359   -8.6721
    0.6878    0.7561    0.2686   -1.4296   -9.4296  -11.7291   -8.6764
    0.6671    0.7542    0.2665   -1.4425   -9.4889  -12.4663   -8.7021
    0.6882    0.7674    0.2679   -1.4509   -9.4835  -12.8118   -8.7015
    0.6862    0.7675    0.2619   -1.4445   -9.5653  -12.5004   -8.6560
    0.6773    0.7663    0.2624   -1.4457   -9.5590  -12.6215   -8.7298
    0.6617    0.7573    0.2688   -1.4366   -9.4416  -12.6883   -8.5984
    0.6909    0.7603    0.2646   -1.4466   -9.5252  -12.5644   -8.6949
    0.6738    0.7626    0.2639   -1.4442   -9.5324  -12.6609   -8.7016
    0.6593    0.7652    0.2637   -1.4468   -9.5412  -12.5517   -8.7069
    0.6553    0.7648    0.2627   -1.4456   -9.5534  -12.3459   -8.7113
    0.6246    0.7658    0.2624   -1.4468   -9.5610  -12.6416   -8.7185
    0.5876    0.7630    0.2622   -1.4409   -9.5522  -12.5884   -8.6699
    0.6733    0.7582    0.2653   -1.4427   -9.5065  -12.4138   -8.6390
    0.6610    0.7642    0.2658   -1.4448   -9.5036  -12.5897   -8.7024
    0.6556    0.7649    0.2658   -1.4442   -9.5028  -12.8415   -8.6922
    0.6439    0.7664    0.2567   -1.4445   -9.6481  -12.8593   -8.6977
    0.6236    0.7665    0.2644   -1.4454   -9.5264  -12.4533   -8.6994
    0.6197    0.7660    0.2633   -1.4426   -9.5389  -12.6010   -8.6944
    0.6648    0.7599    0.2642   -1.4511   -9.5411  -12.6886   -8.6977
    0.6484    0.7616    0.2616   -1.4461   -9.5732  -12.5052   -8.6973
    0.6474    0.7637    0.2633   -1.4453   -9.5440  -12.8335   -8.7094
    0.6384    0.7650    0.2612   -1.4461   -9.5797  -12.3955   -8.6776
    0.6233    0.7644    0.2636   -1.4451   -9.5388  -12.6780   -8.7055
    0.5936    0.7614    0.2685   -1.4406   -9.4541  -12.6480   -8.6608
    0.6582    0.7542    0.2627   -1.4408   -9.5438  -12.5148   -8.6940
    0.6547    0.7592    0.2575   -1.4423   -9.6296  -12.4855   -8.6886
    0.6458    0.7634    0.2644   -1.4482   -9.5331  -12.7093   -8.7262
    0.6398    0.7654    0.2667   -1.4478   -9.4964  -12.7078   -8.7270
    0.6204    0.7639    0.2607   -1.4462   -9.5878  -12.7025   -8.7146
    0.5980    0.7611    0.2672   -1.4403   -9.4734  -12.5126   -8.6591
    0.6804    0.7631    0.2699   -1.4490   -9.4490  -12.8899   -8.7291
    0.6616    0.7641    0.2661   -1.4449   -9.4988  -12.5469   -8.6924
    0.6578    0.7665    0.2639   -1.4489   -9.5419  -12.5082   -8.7148
    0.6417    0.7646    0.2667   -1.4443   -9.4884  -12.7841   -8.7002
    0.6306    0.7650    0.2649   -1.4456   -9.5194  -12.4738   -8.7001
    0.6201    0.7661    0.2679   -1.4437   -9.4693  -12.5454   -8.7012
    0.6954    0.7653    0.2662   -1.4431   -9.4936  -13.0114   -8.6382
    0.6775    0.7692    0.2670   -1.4459   -9.4871  -12.7746   -8.7370
    0.6580    0.7679    0.2698   -1.4492   -9.4507  -12.9146   -8.7460
    0.6695    0.7688    0.2734   -1.4480   -9.3932  -12.5538   -8.7344
    0.6314    0.7700    0.2700   -1.4494   -9.4479  -12.6521   -8.7462
    0.5901    0.7687    0.2701   -1.4460   -9.4406  -12.9209   -8.7197
    0.6911    0.7623    0.2604   -1.4458   -9.5897  -12.4968   -8.7035
    0.6873    0.7629    0.2678   -1.4469   -9.4765  -12.6211   -8.7396
    0.6744    0.7602    0.2754   -1.4440   -9.3542  -11.6557   -8.7149
    0.6825    0.7652    0.2629   -1.4398   -9.5393  -12.5872   -8.6626
    0.6678    0.7625    0.2671   -1.4422   -9.4778  -11.8376   -8.6602
    0.6584    0.7563    0.2660   -1.4337   -9.4785  -12.4814   -8.5905

];

s.phone(:,:,1) = [.8443    0.8091    0.3426   -1.5318   -8.6132  -11.5036   -9.8689
    0.8319    0.8109    0.3459   -1.5313   -8.5715  -11.7436   -9.8724
    0.8198    0.8079    0.3443   -1.5254   -8.5793  -11.9660   -9.8113
    0.8135    0.8110    0.3443   -1.5299   -8.5891  -11.8858   -9.8494
    0.7845    0.8088    0.3453   -1.5280   -8.5727  -11.8954   -9.8040
    0.7586    0.8071    0.3421   -1.5248   -8.6052  -11.8646   -9.8150
    0.8194    0.8063    0.3453   -1.5299   -8.5762  -12.5565   -9.8302
    0.8027    0.8079    0.3396   -1.5293   -8.6453  -12.3436   -9.8242
    0.7404    0.8060    0.3396   -1.5277   -8.6421  -11.9051   -9.8107
    0.7175    0.8046    0.3421   -1.5268   -8.6098  -11.9114   -9.8220
    0.6977    0.8048    0.3425   -1.5271   -8.6055  -12.1053   -9.8184
    0.6634    0.8034    0.3448   -1.5248   -8.5721  -11.9902   -9.8155
    0.8341    0.8073    0.3457   -1.5307   -8.5723  -11.9966   -9.8375
    0.8074    0.8099    0.3454   -1.5328   -8.5809  -12.0661   -9.8380
    0.7447    0.8072    0.3433   -1.5299   -8.6009  -11.8616   -9.8288
    0.7138    0.8078    0.3407   -1.5305   -8.6346  -12.0128   -9.8377
    0.6678    0.8057    0.3430   -1.5291   -8.6035  -12.0166   -9.8340
    0.6633    0.8041    0.3450   -1.5254   -8.5710  -11.9365   -9.8140
    0.8346    0.8103    0.3423   -1.5332   -8.6190  -11.8739   -9.8516
    0.7936    0.8073    0.3466   -1.5294   -8.5592  -12.0480   -9.8326
    0.7851    0.8074    0.3419   -1.5282   -8.6146  -11.9474   -9.8252
    0.7360    0.8057    0.3454   -1.5280   -8.5718  -12.1044   -9.8429
    0.7060    0.8045    0.3429   -1.5280   -8.6023  -11.8145   -9.8102
    0.6821    0.8030    0.3463   -1.5258   -8.5565  -11.9304   -9.8084
    0.8440    0.8088    0.3457   -1.5305   -8.5718  -12.2369   -9.8572
    0.7978    0.8076    0.3377   -1.5306   -8.6724  -11.8941   -9.8066
    0.7604    0.8074    0.3423   -1.5300   -8.6140  -12.0136   -9.8189
    0.7568    0.8073    0.3440   -1.5303   -8.5934  -11.9684   -9.8329
    0.7114    0.8054    0.3420   -1.5285   -8.6149  -11.9079   -9.8228
    0.6968    0.8028    0.3449   -1.5259   -8.5736  -11.9059   -9.8022
    0.8270    0.8082    0.3463   -1.5309   -8.5655  -12.0645   -9.8433
    0.7858    0.8059    0.3406   -1.5263   -8.6275  -12.3211   -9.8578
    0.7454    0.8059    0.3437   -1.5276   -8.5916  -12.0220   -9.8246
    0.7321    0.8046    0.3464   -1.5251   -8.5527  -11.8566   -9.8364
    0.7011    0.8055    0.3432   -1.5272   -8.5965  -11.8794   -9.8245
    0.6950    0.8045    0.3420   -1.5245   -8.6061  -11.9665   -9.8174
    0.8310    0.8064    0.3485   -1.5277   -8.5327  -11.5117   -9.8336
    0.8047    0.8075    0.3425   -1.5271   -8.6055  -11.8946   -9.8277
    0.7449    0.8064    0.3444   -1.5286   -8.5850  -11.8794   -9.7897
    0.7222    0.8034    0.3442   -1.5230   -8.5764  -11.8986   -9.8046
    0.6746    0.8039    0.3477   -1.5263   -8.5393  -11.9462   -9.8125
    0.6720    0.8035    0.3429   -1.5235   -8.5930  -11.9954   -9.8133
    0.8356    0.8051    0.3471   -1.5269   -8.5474  -11.9398   -9.8416
    0.8331    0.8122    0.3458   -1.5321   -8.5741  -12.2601   -9.8641
    0.8112    0.8083    0.3432   -1.5276   -8.5975  -12.2272   -9.8337
    0.8172    0.8094    0.3441   -1.5273   -8.5862  -11.9782   -9.8307
    0.7814    0.8085    0.3426   -1.5279   -8.6062  -11.9838   -9.8325
    0.7576    0.8061    0.3406   -1.5245   -8.6234  -12.0296   -9.8071
    0.8452    0.8102    0.3413   -1.5337   -8.6331  -12.1431   -9.8709
    0.8096    0.8072    0.3476   -1.5295   -8.5466  -11.9264   -9.8481
    0.7367    0.8058    0.3440   -1.5283   -8.5888  -11.8384   -9.8173
    0.7173    0.8044    0.3440   -1.5272   -8.5869  -12.0186   -9.8013
    0.6883    0.8050    0.3434   -1.5284   -8.5966  -11.8043   -9.8224
    0.6842    0.8040    0.3445   -1.5259   -8.5778  -11.8469   -9.8191
    0.8384    0.8078    0.3374   -1.5313   -8.6768  -12.1397   -9.8343
    0.7835    0.8088    0.3463   -1.5293   -8.5631  -11.9950   -9.8579
    0.7302    0.8067    0.3446   -1.5301   -8.5849  -12.0695   -9.8119
    0.7052    0.8040    0.3469   -1.5263   -8.5493  -11.9225   -9.7841
    0.6822    0.8048    0.3451   -1.5268   -8.5726  -11.9143   -9.8116
    0.6671    0.8035    0.3505   -1.5247   -8.5021  -11.9710   -9.8138
    0.8357    0.8063    0.3427   -1.5297   -8.6074  -11.7228   -9.8010
    0.8111    0.8091    0.3425   -1.5298   -8.6111  -11.8080   -9.8305
    0.7543    0.8049    0.3449   -1.5277   -8.5769  -12.4301   -9.8183
    0.7483    0.8046    0.3458   -1.5260   -8.5628  -11.9931   -9.8422
    0.7046    0.8034    0.3425   -1.5270   -8.6056  -11.9903   -9.8061
    0.6873    0.8030    0.3469   -1.5248   -8.5460  -11.9881   -9.8197
    0.8457    0.8106    0.3384   -1.5287   -8.6587  -12.1412   -9.8247
    0.7908    0.8086    0.3360   -1.5314   -8.6949  -12.1802   -9.8247
    0.7782    0.8058    0.3423   -1.5264   -8.6068  -11.9318   -9.8144
    0.7511    0.8062    0.3429   -1.5283   -8.6035  -11.9308   -9.8047
    0.7118    0.8051    0.3410   -1.5279   -8.6253  -12.0256   -9.8153
    0.6953    0.8033    0.3444   -1.5249   -8.5770  -11.9715   -9.8038
    0.8345    0.8090    0.3445   -1.5327   -8.5908  -12.0078   -9.8384
    0.7957    0.8066    0.3410   -1.5282   -8.6258  -12.1714   -9.8300
    0.7588    0.8064    0.3450   -1.5281   -8.5761  -12.1417   -9.8261
    0.7436    0.8068    0.3459   -1.5291   -8.5672  -12.2411   -9.8150
    0.7066    0.8053    0.3424   -1.5276   -8.6074  -12.2072   -9.8153
    0.6954    0.8045    0.3426   -1.5243   -8.5984  -12.1372   -9.8017
    0.8328    0.8050    0.3388   -1.5286   -8.6539  -11.8531   -9.8386
    0.8056    0.8073    0.3458   -1.5285   -8.5672  -12.1954   -9.8552
    0.7568    0.8058    0.3420   -1.5263   -8.6106  -12.0359   -9.8389
    0.7298    0.8065    0.3408   -1.5275   -8.6267  -12.2381   -9.8104
    0.6781    0.8046    0.3415   -1.5264   -8.6171  -12.0210   -9.8196
    0.6818    0.8029    0.3424   -1.5234   -8.5997  -12.0216   -9.7967
    0.8373    0.8068    0.3311   -1.5321   -8.7583  -12.2641   -9.8286
    0.8260    0.8086    0.3483   -1.5299   -8.5396  -11.9657   -9.8225
    0.8130    0.8081    0.3462   -1.5290   -8.5631  -11.6884   -9.8091
    0.8143    0.8095    0.3436   -1.5298   -8.5969  -12.0319   -9.8136
    0.7840    0.8080    0.3443   -1.5284   -8.5862  -11.8723   -9.8288
    0.7515    0.8053    0.3445   -1.5251   -8.5766  -11.9785   -9.8017

];
s.phone(:,:,2) = [.3421    0.6148    0.6016   -1.5390   -6.0966   -8.3853  -10.9756
    0.3336    0.6158    0.6028   -1.5378   -6.0853   -8.4100  -10.7342
    0.3272    0.6130    0.6046   -1.5384   -6.0720   -8.4462  -10.8760
    0.3280    0.6142    0.6014   -1.5377   -6.0966   -8.3396  -10.7882
    0.3166    0.6120    0.6026   -1.5383   -6.0879   -8.3528  -10.8320
    0.3152    0.6162    0.6045   -1.5369   -6.0696   -8.3608  -10.8054
    0.3394    0.6122    0.6020   -1.5353   -6.0861   -8.4010  -10.6651
    0.3272    0.6149    0.6065   -1.5347   -6.0492   -8.3401  -10.9183
    0.3129    0.6164    0.6006   -1.5383   -6.1039   -8.3551  -10.8822
    0.2895    0.6148    0.6003   -1.5363   -6.1028   -8.3244  -10.9150
    0.2795    0.6146    0.6001   -1.5379   -6.1076   -8.3616  -10.8685
    0.2755    0.6149    0.6019   -1.5372   -6.0916   -8.3453  -10.8545
    0.3380    0.6199    0.6001   -1.5373   -6.1056   -8.3618  -11.1131
    0.3263    0.6219    0.5976   -1.5404   -6.1324   -8.3104  -11.0790
    0.2969    0.6174    0.5989   -1.5373   -6.1161   -8.3258  -10.9614
    0.2806    0.6157    0.5969   -1.5428   -6.1437   -8.3406  -10.9145
    0.2722    0.6147    0.6019   -1.5386   -6.0945   -8.3611  -10.9343
    0.2740    0.6137    0.6008   -1.5373   -6.1010   -8.3693  -10.7939
    0.3396    0.6148    0.5996   -1.5398   -6.1145   -8.4093  -10.9936
    0.3283    0.6143    0.6009   -1.5348   -6.0944   -8.3100  -10.8917
    0.3045    0.6124    0.6001   -1.5377   -6.1071   -8.3546  -10.8711
    0.2986    0.6152    0.5980   -1.5379   -6.1243   -8.3826  -10.8893
    0.2846    0.6117    0.6010   -1.5372   -6.0986   -8.3442  -10.9049
    0.2793    0.6118    0.5995   -1.5379   -6.1121   -8.3603  -10.8810
    0.3476    0.6189    0.6032   -1.5402   -6.0858   -8.2766  -10.9343
    0.3306    0.6164    0.5988   -1.5420   -6.1258   -8.3530  -10.9828
    0.3087    0.6102    0.6000   -1.5382   -6.1086   -8.3317  -10.9072
    0.3040    0.6126    0.6010   -1.5390   -6.1027   -8.3704  -10.8631
    0.2872    0.6118    0.6004   -1.5391   -6.1074   -8.3332  -10.9360
    0.2774    0.6122    0.6012   -1.5371   -6.0974   -8.3374  -10.8702
    0.3435    0.6171    0.5958   -1.5405   -6.1465   -8.3131  -10.7610
    0.3173    0.6170    0.6028   -1.5361   -6.0813   -8.3912  -10.9306
    0.2980    0.6191    0.6016   -1.5387   -6.0970   -8.3829  -10.7317
    0.2756    0.6153    0.6036   -1.5391   -6.0812   -8.4568  -10.8385
    0.2654    0.6148    0.6008   -1.5384   -6.1031   -8.3987  -10.8966
    0.2685    0.6118    0.6008   -1.5365   -6.0987   -8.3755  -10.8542
    0.3429    0.6188    0.6029   -1.5425   -6.0929   -8.4203  -10.9998
    0.3336    0.6222    0.5974   -1.5427   -6.1387   -8.4018  -10.8987
    0.3023    0.6162    0.6021   -1.5399   -6.0953   -8.3579  -10.8462
    0.2872    0.6188    0.6014   -1.5416   -6.1044   -8.4342  -10.9161
    0.2769    0.6167    0.6008   -1.5400   -6.1062   -8.3670  -10.8605
    0.2757    0.6152    0.6012   -1.5385   -6.0997   -8.3692  -10.8755
    0.3447    0.6164    0.5938   -1.5417   -6.1655   -8.4946  -10.9540
    0.3401    0.6196    0.6012   -1.5388   -6.1000   -8.3095  -10.8804
    0.3337    0.6196    0.6036   -1.5394   -6.0821   -8.4147  -11.0006
    0.3261    0.6163    0.6026   -1.5379   -6.0873   -8.4211  -10.8689
    0.3136    0.6168    0.6021   -1.5388   -6.0932   -8.3660  -10.9209
    0.3140    0.6180    0.6033   -1.5394   -6.0849   -8.3618  -10.8968
    0.3419    0.6172    0.6001   -1.5402   -6.1111   -8.3595  -10.8650
    0.3246    0.6153    0.6003   -1.5376   -6.1048   -8.3730  -10.8904
    0.3090    0.6139    0.6001   -1.5400   -6.1117   -8.3666  -10.8477
    0.2909    0.6124    0.5989   -1.5366   -6.1147   -8.3715  -10.9129
    0.2866    0.6148    0.6004   -1.5392   -6.1079   -8.3554  -10.9429
    0.2779    0.6119    0.6002   -1.5389   -6.1086   -8.3659  -10.9092
    0.3432    0.6228    0.6005   -1.5407   -6.1085   -8.3345  -10.9057
    0.3197    0.6182    0.5993   -1.5368   -6.1115   -8.2986  -11.0481
    0.3028    0.6177    0.6011   -1.5371   -6.0976   -8.3673  -10.7491
    0.2856    0.6178    0.6013   -1.5369   -6.0957   -8.4707  -10.9887
    0.2680    0.6141    0.6005   -1.5371   -6.1027   -8.3614  -10.8888
    0.2755    0.6130    0.5975   -1.5386   -6.1304   -8.3786  -10.8460
    0.3442    0.6174    0.6007   -1.5399   -6.1052   -8.4252  -10.9449
    0.3258    0.6166    0.6005   -1.5376   -6.1036   -8.3485  -10.8342
    0.3032    0.6133    0.6006   -1.5367   -6.1010   -8.4074  -10.8660
    0.3013    0.6141    0.6014   -1.5376   -6.0965   -8.3887  -10.8835
    0.2855    0.6120    0.6009   -1.5378   -6.1011   -8.3898  -10.8219
    0.2808    0.6111    0.6010   -1.5375   -6.0998   -8.3827  -10.7985
    0.3396    0.6160    0.5970   -1.5436   -6.1432   -8.3856  -11.1211
    0.3266    0.6161    0.6002   -1.5370   -6.1044   -8.3438  -10.7092
    0.3105    0.6159    0.6022   -1.5397   -6.0939   -8.4211  -10.8289
    0.3076    0.6158    0.6041   -1.5379   -6.0748   -8.3503  -10.7999
    0.2963    0.6144    0.6019   -1.5386   -6.0944   -8.3705  -10.8363
    0.2744    0.6138    0.6013   -1.5371   -6.0960   -8.3639  -10.7878
    0.3447    0.6192    0.6033   -1.5394   -6.0832   -8.3742  -10.6295
    0.3253    0.6173    0.6038   -1.5372   -6.0757   -8.3183  -10.8265
    0.3051    0.6163    0.6041   -1.5367   -6.0723   -8.3878  -10.7668
    0.2909    0.6146    0.6023   -1.5385   -6.0911   -8.3367  -11.0794
    0.2807    0.6145    0.6023   -1.5383   -6.0903   -8.3892  -10.9188
    0.2742    0.6116    0.6036   -1.5365   -6.0765   -8.3493  -10.8086
    0.3434    0.6195    0.6004   -1.5403   -6.1093   -8.3096  -10.9760
    0.3320    0.6191    0.5999   -1.5402   -6.1131   -8.3277  -10.9831
    0.3019    0.6142    0.6019   -1.5394   -6.0962   -8.4188  -10.8661
    0.2793    0.6170    0.6020   -1.5412   -6.0985   -8.3279  -10.8881
    0.2733    0.6128    0.6027   -1.5376   -6.0856   -8.3782  -10.9327
    0.2721    0.6144    0.6008   -1.5381   -6.1025   -8.3437  -10.8675
    0.3437    0.6127    0.5922   -1.5395   -6.1743   -8.4610  -10.8376
    0.3364    0.6171    0.6060   -1.5368   -6.0569   -8.2951  -10.8171
    0.3271    0.6145    0.6054   -1.5372   -6.0627   -8.3413  -10.9652
    0.3248    0.6160    0.6071   -1.5385   -6.0521   -8.3515  -10.8715
    0.3138    0.6142    0.6046   -1.5380   -6.0709   -8.3486  -10.8095
    0.3117    0.6162    0.6050   -1.5382   -6.0680   -8.3400  -10.8910

];
s.phone(:,:,3) = [.4561    0.4369    0.3279   -0.8490   -7.4330   -8.6720   -7.1527
    0.4471    0.4356    0.3349   -0.8464   -7.3395   -8.8820   -7.0957
    0.4421    0.4353    0.3393   -0.8468   -7.2850   -8.8032   -7.1242
    0.4416    0.4401    0.3397   -0.8572   -7.3011   -8.6958   -7.2130
    0.4270    0.4371    0.3391   -0.8529   -7.2995   -8.6841   -7.2326
    0.4115    0.4330    0.3453   -0.8424   -7.2019   -8.7937   -7.2001
    0.4496    0.4309    0.3347   -0.8411   -7.3311   -8.8616   -7.1210
    0.4363    0.4323    0.3495   -0.8446   -7.1540   -8.6712   -7.1726
    0.4076    0.4322    0.3400   -0.8442   -7.2712   -8.9905   -7.1634
    0.3948    0.4316    0.3410   -0.8436   -7.2569   -8.9585   -7.1632
    0.3778    0.4330    0.3362   -0.8473   -7.3252   -8.8153   -7.1637
    0.3596    0.4287    0.3499   -0.8372   -7.1351   -8.6949   -7.1298
    0.4479    0.4316    0.3412   -0.8469   -7.2604   -8.6000   -7.1993
    0.4447    0.4348    0.3358   -0.8500   -7.3346   -8.9016   -7.1334
    0.4069    0.4343    0.3378   -0.8504   -7.3106   -8.6826   -7.2226
    0.3928    0.4341    0.3353   -0.8499   -7.3418   -8.8939   -7.1208
    0.3741    0.4328    0.3411   -0.8476   -7.2636   -8.6908   -7.1966
    0.3781    0.4304    0.3436   -0.8401   -7.2177   -8.6373   -7.1215
    0.4448    0.4345    0.3542   -0.8541   -7.1157   -8.7426   -7.2068
    0.4299    0.4333    0.3324   -0.8502   -7.3784   -8.7221   -7.1902
    0.4174    0.4328    0.3383   -0.8463   -7.2965   -8.7954   -7.2041
    0.4034    0.4343    0.3425   -0.8500   -7.2509   -8.6768   -7.1726
    0.3877    0.4330    0.3393   -0.8483   -7.2877   -8.8187   -7.2136
    0.3738    0.4291    0.3434   -0.8403   -7.2211   -8.6415   -7.1404
    0.4371    0.4315    0.3523   -0.8461   -7.1223   -9.0036   -7.1623
    0.4338    0.4322    0.3380   -0.8461   -7.2999   -8.7680   -7.1570
    0.4136    0.4328    0.3370   -0.8490   -7.3184   -8.6972   -7.1690
    0.4004    0.4332    0.3398   -0.8477   -7.2803   -8.5181   -7.1418
    0.3797    0.4316    0.3395   -0.8470   -7.2824   -8.9471   -7.1917
    0.3689    0.4300    0.3384   -0.8425   -7.2876   -8.7070   -7.1607
    0.4471    0.4303    0.3470   -0.8470   -7.1893   -8.6817   -7.1278
    0.4441    0.4345    0.3461   -0.8472   -7.2007   -8.7332   -7.2028
    0.4151    0.4343    0.3495   -0.8520   -7.1693   -8.5348   -7.1924
    0.4103    0.4342    0.3416   -0.8510   -7.2640   -8.6134   -7.1383
    0.3902    0.4332    0.3479   -0.8491   -7.1834   -8.6116   -7.1896
    0.3877    0.4307    0.3403   -0.8414   -7.2610   -8.5734   -7.1630
    0.4526    0.4337    0.3353   -0.8502   -7.3410   -8.6436   -7.1958
    0.4396    0.4318    0.3367   -0.8426   -7.3085   -8.7537   -7.1835
    0.4020    0.4334    0.3451   -0.8488   -7.2172   -9.1530   -7.1586
    0.3881    0.4325    0.3446   -0.8495   -7.2245   -9.1576   -7.1463
    0.3776    0.4343    0.3423   -0.8510   -7.2557   -8.7309   -7.1785
    0.3648    0.4302    0.3419   -0.8419   -7.2426   -8.7901   -7.1635
    0.4519    0.4343    0.3317   -0.8499   -7.3868   -8.9615   -7.3070
    0.4431    0.4327    0.3535   -0.8429   -7.1029   -8.3297   -7.2328
    0.4433    0.4357    0.3561   -0.8516   -7.0884   -8.7041   -7.1582
    0.4405    0.4354    0.3339   -0.8466   -7.3525   -8.6541   -7.1878
    0.4239    0.4346    0.3419   -0.8482   -7.2558   -8.7577   -7.2408
    0.4043    0.4313    0.3482   -0.8413   -7.1638   -8.7440   -7.1637
    0.4505    0.4322    0.3561   -0.8455   -7.0755   -8.7777   -7.2366
    0.4360    0.4332    0.3437   -0.8474   -7.2308   -8.6491   -7.1768
    0.3961    0.4328    0.3429   -0.8473   -7.2406   -8.7446   -7.2075
    0.3760    0.4304    0.3434   -0.8439   -7.2278   -8.7784   -7.1756
    0.3759    0.4328    0.3450   -0.8479   -7.2161   -8.9034   -7.2012
    0.3562    0.4293    0.3415   -0.8404   -7.2448   -8.8178   -7.1310
    0.4507    0.4330    0.3624   -0.8489   -7.0079   -9.0260   -7.1380
    0.4399    0.4336    0.3390   -0.8487   -7.2927   -8.8171   -7.2327
    0.3835    0.4322    0.3444   -0.8453   -7.2181   -8.6185   -7.1963
    0.3882    0.4348    0.3469   -0.8520   -7.2009   -8.6840   -7.2563
    0.3596    0.4323    0.3431   -0.8475   -7.2386   -8.7709   -7.1920
    0.3627    0.4301    0.3470   -0.8409   -7.1775   -8.7507   -7.1567
    0.4418    0.4323    0.3437   -0.8490   -7.2340   -8.8376   -7.2341
    0.4401    0.4337    0.3459   -0.8469   -7.2030   -8.7845   -7.2051
    0.4184    0.4343    0.3396   -0.8502   -7.2875   -8.5009   -7.1380
    0.3985    0.4351    0.3434   -0.8513   -7.2422   -8.7357   -7.2293
    0.3762    0.4320    0.3394   -0.8463   -7.2828   -8.7131   -7.1952
    0.3734    0.4301    0.3428   -0.8414   -7.2303   -8.6333   -7.1542
    0.4417    0.4299    0.3393   -0.8435   -7.2775   -8.5094   -7.2318
    0.4347    0.4320    0.3410   -0.8446   -7.2591   -8.6834   -7.2138
    0.4153    0.4328    0.3418   -0.8473   -7.2549   -8.8149   -7.1986
    0.4063    0.4333    0.3479   -0.8445   -7.1734   -9.0552   -7.2661
    0.3861    0.4312    0.3371   -0.8446   -7.3080   -8.6183   -7.1718
    0.3783    0.4302    0.3362   -0.8417   -7.3140   -8.7548   -7.1855
    0.4451    0.4286    0.3375   -0.8377   -7.2883   -8.7652   -7.1618
    0.4438    0.4368    0.3361   -0.8561   -7.3434   -8.6977   -7.2006
    0.3945    0.4344    0.3421   -0.8511   -7.2589   -8.7241   -7.2099
    0.3877    0.4350    0.3407   -0.8511   -7.2760   -8.7626   -7.2649
    0.3720    0.4328    0.3382   -0.8475   -7.2994   -8.7105   -7.1803
    0.3705    0.4310    0.3354   -0.8417   -7.3231   -8.7492   -7.1618
    0.4504    0.4340    0.3484   -0.8496   -7.1774   -9.1955   -7.2465
    0.4321    0.4322    0.3410   -0.8484   -7.2666   -8.6851   -7.2312
    0.4032    0.4314    0.3530   -0.8464   -7.1156   -8.3772   -7.1800
    0.3933    0.4312    0.3466   -0.8446   -7.1896   -8.3826   -7.1943
    0.3743    0.4321    0.3429   -0.8471   -7.2405   -8.5958   -7.1740
    0.3658    0.4299    0.3450   -0.8417   -7.2034   -8.7208   -7.1642
    0.4485    0.4309    0.3534   -0.8442   -7.1061   -8.6613   -7.1886
    0.4452    0.4331    0.3366   -0.8450   -7.3145   -8.7118   -7.2110
    0.4362    0.4340    0.3495   -0.8468   -7.1590   -8.5225   -7.2352
    0.4385    0.4366    0.3312   -0.8528   -7.3989   -8.8502   -7.1750
    0.4238    0.4346    0.3397   -0.8497   -7.2853   -8.7755   -7.2041
    0.4055    0.4311    0.3467   -0.8414   -7.1821   -8.7770   -7.1778

];

s.cigar(:,:,1) = [.9514    0.9121    0.1958   -1.7173  -11.3037  -13.1233  -13.9398
    0.9340    0.9082    0.2034   -1.7057  -11.1257  -13.2310  -13.5891
    0.9212    0.9085    0.1988   -1.7100  -11.2281  -13.6148  -14.3822
    0.9071    0.9049    0.2021   -1.7017  -11.1451  -13.2300  -13.8201
    0.8789    0.9075    0.1985   -1.7076  -11.2303  -13.5124  -14.1101
    0.8479    0.9021    0.2128   -1.6996  -10.9288  -13.3530  -13.8146
    0.9216    0.9052    0.2068   -1.7105  -11.0664  -13.5400  -13.9605
    0.8958    0.9059    0.2134   -1.7076  -10.9328  -13.5184  -13.9940
    0.8351    0.9012    0.2123   -1.7030  -10.9463  -13.4256  -13.9798
    0.8049    0.9012    0.2033   -1.7051  -11.1275  -13.7379  -14.1425
    0.7849    0.9010    0.2073   -1.7033  -11.0435  -13.5380  -13.9925
    0.7545    0.8991    0.2081   -1.7012  -11.0245  -13.5642  -13.8684
    0.9346    0.9054    0.2011   -1.7087  -11.1770  -14.0574  -14.0713
    0.9047    0.9071    0.1987   -1.7087  -11.2284  -13.4553  -13.9705
    0.8255    0.9032    0.2074   -1.7048  -11.0442  -13.4417  -13.9395
    0.8011    0.9025    0.2049   -1.7041  -11.0937  -13.7900  -13.9544
    0.7508    0.9012    0.2043   -1.7043  -11.1047  -13.7859  -13.9739
    0.7426    0.9003    0.2051   -1.7026  -11.0866  -13.5503  -13.8624
    0.9378    0.9045    0.2145   -1.7061  -10.9097  -13.2984  -13.8549
    0.8946    0.9061    0.1989   -1.7094  -11.2240  -13.6941  -13.9659
    0.8833    0.9043    0.2093   -1.7052  -11.0087  -13.6135  -14.0107
    0.8353    0.9022    0.2030   -1.7041  -11.1312  -13.5095  -14.0389
    0.7914    0.8997    0.2091   -1.7031  -11.0088  -13.6280  -13.9029
    0.7661    0.8976    0.2031   -1.7001  -11.1216  -13.4831  -13.8337
    0.9475    0.9071    0.2046   -1.7059  -11.1017  -13.8972  -13.8990
    0.8776    0.9041    0.2115   -1.7075  -10.9695  -13.7175  -14.0522
    0.8519    0.9030    0.2077   -1.7038  -11.0361  -13.3734  -14.0913
    0.8398    0.9008    0.2046   -1.7014  -11.0940  -13.2729  -13.8308
    0.7995    0.9007    0.2098   -1.7029  -10.9944  -13.6814  -13.8982
    0.7760    0.8962    0.2086   -1.6985  -11.0094  -13.2992  -13.8496
    0.9357    0.9058    0.2128   -1.7077  -10.9450  -13.3572  -14.0102
    0.8926    0.9072    0.2077   -1.7093  -11.0470  -13.6419  -13.9777
    0.8359    0.9018    0.2036   -1.7041  -11.1184  -13.5842  -13.7072
    0.8054    0.9026    0.2031   -1.7047  -11.1295  -13.4002  -13.7624
    0.7752    0.9014    0.2117   -1.7043  -10.9592  -13.5909  -13.8123
    0.7728    0.9000    0.2099   -1.7009  -10.9880  -13.5076  -13.8363
    0.9332    0.9050    0.2006   -1.7076  -11.1857  -13.2856  -13.8534
    0.9011    0.9076    0.2055   -1.7087  -11.0904  -13.1002  -14.0052
    0.8386    0.9031    0.2068   -1.7055  -11.0581  -13.5567  -13.9304
    0.8110    0.9012    0.2083   -1.7033  -11.0232  -13.7302  -13.9098
    0.7604    0.9011    0.2053   -1.7043  -11.0854  -13.2955  -13.9060
    0.7512    0.8988    0.2134   -1.6995  -10.9177  -13.4585  -13.8385
    0.9481    0.9118    0.1996   -1.7195  -11.2304  -14.1024  -14.5013
    0.9351    0.9086    0.2081   -1.7058  -11.0331  -14.0604  -13.7440
    0.9133    0.9057    0.2091   -1.7050  -11.0117  -13.6863  -13.7653
    0.9138    0.9040    0.2068   -1.7013  -11.0505  -13.3261  -13.9834
    0.8780    0.9044    0.2055   -1.7033  -11.0785  -13.7314  -13.9076
    0.8442    0.9012    0.2164   -1.6994  -10.8599  -13.3482  -13.8020
    0.9443    0.9065    0.2197   -1.7081  -10.8140  -13.7032  -14.0030
    0.9137    0.9062    0.2011   -1.7093  -11.1796  -13.5813  -13.9758
    0.8264    0.9022    0.2045   -1.7059  -11.1042  -13.5407  -13.9173
    0.8114    0.9013    0.2047   -1.7057  -11.0996  -13.4155  -13.9963
    0.7879    0.9010    0.2048   -1.7041  -11.0959  -13.4914  -13.9201
    0.7761    0.8983    0.2076   -1.7007  -11.0335  -13.4895  -13.9287
    0.9387    0.9051    0.2165   -1.7086  -10.8761  -13.1975  -14.1111
    0.8792    0.9071    0.2033   -1.7083  -11.1328  -13.8548  -13.8411
    0.8274    0.9023    0.2086   -1.7045  -11.0195  -13.3559  -13.8240
    0.7970    0.9025    0.2053   -1.7060  -11.0897  -13.3676  -13.9348
    0.7727    0.9012    0.2055   -1.7036  -11.0796  -13.3997  -13.8668
    0.7538    0.9006    0.2005   -1.7027  -11.1780  -13.5435  -13.8310
    0.9306    0.9048    0.2082   -1.7098  -11.0380  -13.2047  -13.6484
    0.8994    0.9050    0.2096   -1.7054  -11.0017  -13.6305  -13.9187
    0.8476    0.9018    0.2089   -1.7045  -11.0136  -13.5585  -13.9839
    0.8323    0.9018    0.2025   -1.7042  -11.1418  -13.4612  -13.8492
    0.7841    0.8997    0.2072   -1.7038  -11.0471  -13.6434  -13.9216
    0.7608    0.8973    0.2038   -1.6995  -11.1050  -13.3099  -13.8286
    0.9483    0.9089    0.2110   -1.7070  -10.9776  -13.5156  -13.9362
    0.8828    0.9068    0.2140   -1.7101  -10.9264  -13.4395  -13.8139
    0.8683    0.9031    0.2041   -1.7050  -11.1102  -13.7839  -14.0658
    0.8436    0.9027    0.1977   -1.7056  -11.2416  -13.6417  -13.9357
    0.7889    0.8997    0.2053   -1.7030  -11.0829  -13.2869  -13.9433
    0.7781    0.8973    0.2058   -1.6991  -11.0643  -13.6515  -13.8274
    0.9339    0.9054    0.2140   -1.7071  -10.9206  -13.4366  -14.1843
    0.8973    0.9039    0.2144   -1.7052  -10.9096  -13.2000  -13.8812
    0.8546    0.9028    0.2035   -1.7045  -11.1217  -13.4328  -13.8631
    0.8245    0.9019    0.2030   -1.7031  -11.1296  -13.7166  -13.9303
    0.7783    0.9010    0.2072   -1.7035  -11.0453  -13.5378  -13.8707
    0.7715    0.9002    0.2093   -1.7011  -11.0001  -13.5552  -13.9322
    0.9370    0.9068    0.2140   -1.7088  -10.9245  -12.9691  -13.6264
    0.9032    0.9052    0.2115   -1.7066  -10.9674  -13.6288  -13.9436
    0.8473    0.9017    0.2094   -1.7018  -10.9985  -13.2546  -13.7753
    0.8123    0.9005    0.2055   -1.7005  -11.0734  -13.1951  -13.8234
    0.7553    0.9003    0.2098   -1.7027  -10.9942  -13.3582  -13.7824
    0.7587    0.8983    0.2146   -1.6995  -10.8938  -13.4638  -13.8221
    0.9351    0.9018    0.2323   -1.7028  -10.5749  -14.0734  -13.5870
    0.9241    0.9091    0.1978   -1.7124  -11.2535  -13.9008  -14.2028
    0.9090    0.9024    0.2140   -1.7003  -10.9084  -13.0059  -13.8326
    0.9055    0.9033    0.2133   -1.7016  -10.9237  -13.6999  -13.8286
    0.8803    0.9027    0.2077   -1.7017  -11.0320  -13.2282  -13.7569
    0.8425    0.8989    0.2165   -1.6969  -10.8529  -13.3537  -13.8599

];
s.cigar(:,:,2) = [.8927    0.8563    0.3030   -1.6240   -9.3160  -12.4566  -13.6802
    0.8770    0.8540    0.3188   -1.6203   -9.0949  -13.9806  -13.6369
    0.8706    0.8589    0.3015   -1.6305   -9.3503  -14.2124  -14.0467
    0.8596    0.8568    0.2991   -1.6262   -9.3747  -13.6378  -13.7750
    0.8356    0.8586    0.3019   -1.6319   -9.3470  -13.7889  -13.9737
    0.8021    0.8510    0.3115   -1.6204   -9.1930  -13.5053  -13.9712
    0.8745    0.8598    0.3019   -1.6339   -9.3513  -14.5873  -14.4138
    0.8486    0.8563    0.3160   -1.6274   -9.1467  -14.2141  -14.0190
    0.7913    0.8528    0.3151   -1.6267   -9.1567  -13.9530  -13.8945
    0.7500    0.8511    0.3101   -1.6264   -9.2240  -14.5340  -14.0992
    0.7291    0.8511    0.3083   -1.6258   -9.2465  -14.1794  -14.1273
    0.6928    0.8487    0.3138   -1.6229   -9.1667  -14.0518  -13.9132
    0.8877    0.8586    0.3106   -1.6322   -9.2281  -14.2555  -14.3413
    0.8605    0.8575    0.3042   -1.6297   -9.3109  -14.4631  -13.7295
    0.7932    0.8545    0.3055   -1.6277   -9.2893  -13.8251  -14.0958
    0.7419    0.8528    0.3049   -1.6274   -9.2973  -14.5124  -13.9825
    0.6986    0.8511    0.3049   -1.6268   -9.2958  -14.2756  -14.0257
    0.6968    0.8490    0.3080   -1.6230   -9.2454  -14.0975  -13.7517
    0.8784    0.8576    0.3151   -1.6321   -9.1681  -13.8933  -13.9716
    0.8329    0.8564    0.2994   -1.6287   -9.3761  -13.9573  -13.5875
    0.8263    0.8540    0.3070   -1.6281   -9.2695  -13.9313  -14.0805
    0.7749    0.8509    0.3043   -1.6251   -9.3011  -14.1109  -13.8908
    0.7388    0.8492    0.3071   -1.6261   -9.2646  -13.9944  -14.1185
    0.7130    0.8457    0.3046   -1.6206   -9.2881  -13.8883  -13.8016
    0.8976    0.8598    0.3075   -1.6306   -9.2671  -14.1659  -13.6220
    0.8604    0.8564    0.3063   -1.6274   -9.2778  -14.1250  -13.7766
    0.8113    0.8552    0.3088   -1.6289   -9.2460  -13.8064  -14.1007
    0.8013    0.8533    0.3059   -1.6275   -9.2838  -13.6558  -14.0337
    0.7526    0.8508    0.3069   -1.6262   -9.2669  -14.1344  -14.0529
    0.7331    0.8434    0.3090   -1.6166   -9.2192  -13.8981  -13.4589
    0.8899    0.8615    0.3110   -1.6339   -9.2262  -13.7643  -14.1358
    0.8414    0.8569    0.3146   -1.6295   -9.1697  -14.0513  -13.8610
    0.7901    0.8533    0.3109   -1.6273   -9.2150  -14.1276  -13.9593
    0.7730    0.8527    0.3132   -1.6275   -9.1841  -14.0260  -13.9480
    0.7385    0.8509    0.3148   -1.6255   -9.1591  -13.9975  -13.9115
    0.7326    0.8478    0.3119   -1.6201   -9.1867  -14.2472  -13.5125
    0.8831    0.8562    0.2925   -1.6262   -9.4677  -13.3202  -13.6861
    0.8549    0.8575    0.2994   -1.6271   -9.3717  -13.7909  -13.8600
    0.7909    0.8541    0.3033   -1.6276   -9.3194  -13.8347  -14.1010
    0.7628    0.8516    0.3032   -1.6264   -9.3187  -14.0243  -14.1657
    0.7126    0.8501    0.3058   -1.6253   -9.2800  -13.8380  -13.9374
    0.7075    0.8494    0.3094   -1.6228   -9.2268  -13.9746  -13.7619
    0.8923    0.8583    0.3050   -1.6280   -9.2961  -13.9046  -13.5266
    0.8807    0.8573    0.3033   -1.6238   -9.3117  -14.7235  -13.9030
    0.8653    0.8587    0.3080   -1.6315   -9.2627  -14.5509  -13.9921
    0.8614    0.8559    0.3023   -1.6260   -9.3293  -13.8302  -14.0674
    0.8295    0.8547    0.3079   -1.6252   -9.2517  -14.2821  -14.2197
    0.8029    0.8503    0.3138   -1.6202   -9.1618  -13.7821  -13.9645
    0.8998    0.8629    0.3081   -1.6350   -9.2674  -14.5662  -14.0849
    0.8583    0.8570    0.3040   -1.6289   -9.3113  -13.9360  -13.7986
    0.7810    0.8535    0.3004   -1.6273   -9.3587  -13.8863  -14.0638
    0.7573    0.8506    0.3064   -1.6262   -9.2743  -13.7599  -14.2405
    0.7340    0.8511    0.3049   -1.6257   -9.2936  -13.8948  -13.9347
    0.7277    0.8485    0.3096   -1.6227   -9.2230  -13.6802  -13.8128
    0.8905    0.8577    0.3178   -1.6311   -9.1294  -13.3123  -14.4705
    0.8316    0.8587    0.2997   -1.6302   -9.3739  -14.4981  -14.0949
    0.7709    0.8536    0.3046   -1.6280   -9.3028  -13.8936  -14.3385
    0.7458    0.8526    0.3035   -1.6288   -9.3184  -13.5935  -14.0219
    0.7177    0.8519    0.3037   -1.6271   -9.3127  -13.9049  -14.0856
    0.7027    0.8491    0.3057   -1.6233   -9.2772  -13.9218  -13.9049
    0.8920    0.8605    0.3096   -1.6351   -9.2474  -13.6341  -14.4032
    0.8634    0.8591    0.2961   -1.6303   -9.4250  -13.6018  -13.9311
    0.7915    0.8518    0.3099   -1.6266   -9.2271  -14.1264  -14.1852
    0.7843    0.8539    0.3023   -1.6281   -9.3341  -14.0496  -14.2591
    0.7403    0.8501    0.3048   -1.6260   -9.2958  -14.3397  -14.0312
    0.7160    0.8468    0.3020   -1.6206   -9.3235  -13.7497  -13.8005
    0.9020    0.8643    0.3013   -1.6344   -9.3600  -14.0215  -14.0437
    0.8407    0.8576    0.3049   -1.6293   -9.3000  -13.9420  -13.9024
    0.8148    0.8532    0.3088   -1.6270   -9.2430  -14.4306  -13.9770
    0.7937    0.8524    0.3012   -1.6266   -9.3461  -14.0807  -14.0745
    0.7504    0.8511    0.3068   -1.6254   -9.2665  -13.6735  -13.9036
    0.7243    0.8453    0.3076   -1.6182   -9.2412  -14.1732  -13.6569
    0.8940    0.8586    0.3120   -1.6286   -9.2021  -13.5504  -13.8023
    0.8476    0.8560    0.3138   -1.6286   -9.1777  -13.9143  -13.9157
    0.8042    0.8559    0.3111   -1.6305   -9.2185  -13.9094  -14.4329
    0.7843    0.8522    0.3099   -1.6266   -9.2276  -14.7469  -13.9389
    0.7395    0.8515    0.3104   -1.6262   -9.2195  -14.2042  -13.9675
    0.7375    0.8484    0.3129   -1.6206   -9.1742  -14.2861  -13.5255
    0.8925    0.8605    0.3125   -1.6339   -9.2064  -13.7084  -14.0927
    0.8543    0.8567    0.3121   -1.6283   -9.2003  -14.3291  -14.2402
    0.8103    0.8553    0.3091   -1.6287   -9.2417  -14.0340  -14.1693
    0.7846    0.8529    0.3008   -1.6268   -9.3525  -13.6893  -13.9904
    0.7306    0.8514    0.3081   -1.6255   -9.2492  -13.9348  -14.0189
    0.7289    0.8496    0.3097   -1.6229   -9.2227  -13.9161  -14.0031
    0.8872    0.8541    0.3144   -1.6247   -9.1625  -15.1459  -13.7633
    0.8778    0.8588    0.3025   -1.6305   -9.3361  -14.3264  -13.6651
    0.8625    0.8554    0.3045   -1.6252   -9.2984  -13.3776  -14.6124
    0.8595    0.8535    0.3005   -1.6237   -9.3506  -14.6066  -13.6032
    0.8308    0.8548    0.3048   -1.6259   -9.2957  -13.6899  -14.2769
    0.7930    0.8486    0.3139   -1.6191   -9.1585  -13.7446  -13.8403

];
s.cigar(:,:,3) = [.7791    0.9086    0.2615   -1.6556   -9.9920  -12.2139  -13.4578
    0.7623    0.9001    0.2737   -1.6422   -9.7772  -12.1136  -12.7862
    0.7616    0.9029    0.2677   -1.6478   -9.8798  -12.1601  -13.0813
    0.7580    0.9028    0.2695   -1.6477   -9.8518  -12.1234  -12.9954
    0.7561    0.9040    0.2669   -1.6490   -9.8955  -12.2097  -13.0818
    0.7525    0.8987    0.2750   -1.6414   -9.7563  -12.0831  -12.9177
    0.7559    0.8792    0.2792   -1.6441   -9.6978  -12.3775  -12.9968
    0.7445    0.8863    0.2690   -1.6480   -9.8603  -12.3092  -12.9743
    0.7322    0.8879    0.2706   -1.6479   -9.8361  -12.2039  -12.9295
    0.7280    0.8890    0.2698   -1.6472   -9.8468  -12.2490  -12.9728
    0.6938    0.8885    0.2711   -1.6478   -9.8281  -12.1701  -12.9608
    0.6497    0.8882    0.2738   -1.6455   -9.7829  -12.1663  -13.0645
    0.7398    0.8828    0.2666   -1.6469   -9.8956  -12.1224  -12.9943
    0.7306    0.8875    0.2709   -1.6492   -9.8341  -12.1852  -12.9973
    0.7189    0.8875    0.2672   -1.6466   -9.8855  -12.1124  -13.0506
    0.7119    0.8900    0.2681   -1.6484   -9.8750  -12.1391  -13.1410
    0.6899    0.8885    0.2691   -1.6477   -9.8590  -12.2203  -12.9768
    0.6829    0.8914    0.2672   -1.6470   -9.8874  -12.0871  -13.0216
    0.7254    0.8760    0.2738   -1.6453   -9.7808  -12.1877  -13.1372
    0.7153    0.8818    0.2689   -1.6451   -9.8560  -12.2916  -13.0599
    0.7066    0.8837    0.2715   -1.6462   -9.8190  -12.1915  -12.9822
    0.6989    0.8852    0.2686   -1.6458   -9.8617  -12.0152  -13.0659
    0.6763    0.8860    0.2712   -1.6460   -9.8232  -12.1612  -13.0620
    0.6514    0.8842    0.2703   -1.6433   -9.8308  -12.1067  -13.0481
    0.7183    0.8778    0.2722   -1.6482   -9.8110  -12.0380  -13.1484
    0.7090    0.8820    0.2687   -1.6486   -9.8664  -12.2129  -12.9727
    0.7134    0.8844    0.2693   -1.6488   -9.8581  -12.1106  -12.9997
    0.6990    0.8854    0.2699   -1.6477   -9.8469  -12.3532  -12.9676
    0.6826    0.8868    0.2707   -1.6480   -9.8350  -12.1525  -13.0140
    0.6479    0.8849    0.2700   -1.6451   -9.8390  -12.1693  -12.9704
    0.7378    0.8837    0.2683   -1.6485   -9.8717  -12.2178  -12.9939
    0.7315    0.8884    0.2690   -1.6481   -9.8607  -12.1173  -12.9634
    0.7262    0.8892    0.2699   -1.6479   -9.8474  -12.1660  -12.9640
    0.7212    0.8889    0.2726   -1.6461   -9.8011  -12.1254  -13.0800
    0.7018    0.8910    0.2684   -1.6481   -9.8708  -12.1312  -12.9729
    0.6856    0.8902    0.2698   -1.6458   -9.8433  -12.1382  -13.0423
    0.7652    0.8894    0.2724   -1.6511   -9.8138  -11.9696  -12.8015
    0.7415    0.8899    0.2595   -1.6480  -10.0097  -12.0188  -13.0833
    0.7295    0.8898    0.2679   -1.6489   -9.8801  -12.1934  -12.9158
    0.7261    0.8883    0.2663   -1.6476   -9.9017  -12.1456  -13.0123
    0.6994    0.8901    0.2714   -1.6480   -9.8238  -12.1695  -12.9574
    0.6518    0.8893    0.2729   -1.6449   -9.7946  -12.1140  -13.0449
    0.7820    0.9023    0.2599   -1.6563  -10.0188  -12.3392  -13.1067
    0.7362    0.8766    0.2728   -1.6425   -9.7907  -11.9063  -12.8908
    0.7379    0.8831    0.2708   -1.6440   -9.8256  -12.0955  -12.9329
    0.7532    0.8930    0.2695   -1.6477   -9.8527  -12.1507  -12.9905
    0.7463    0.8895    0.2680   -1.6473   -9.8743  -12.1446  -12.9714
    0.7395    0.8831    0.2739   -1.6420   -9.7733  -12.0897  -13.0450
    0.7549    0.8773    0.2661   -1.6445   -9.8982  -12.1582  -13.2507
    0.7486    0.8865    0.2685   -1.6501   -9.8730  -12.2853  -12.9506
    0.7281    0.8872    0.2693   -1.6479   -9.8556  -12.1309  -13.0167
    0.7247    0.8884    0.2689   -1.6470   -9.8601  -12.1889  -13.1321
    0.6903    0.8881    0.2686   -1.6480   -9.8662  -12.1493  -12.9912
    0.6476    0.8879    0.2719   -1.6458   -9.8124  -12.1390  -13.0425
    0.7376    0.8825    0.2690   -1.6482   -9.8601  -12.0391  -12.9521
    0.7286    0.8871    0.2703   -1.6470   -9.8392  -12.0613  -12.9703
    0.7196    0.8894    0.2679   -1.6488   -9.8795  -12.1544  -13.0526
    0.7115    0.8900    0.2710   -1.6468   -9.8273  -12.2152  -12.9428
    0.6914    0.8884    0.2701   -1.6471   -9.8421  -12.1467  -13.0493
    0.6874    0.8909    0.2693   -1.6462   -9.8528  -12.1561  -13.0711
    0.7306    0.8770    0.2648   -1.6467   -9.9234  -12.1586  -13.1546
    0.7189    0.8822    0.2702   -1.6465   -9.8390  -12.1652  -13.0409
    0.7121    0.8870    0.2688   -1.6496   -9.8667  -12.0544  -13.0077
    0.7028    0.8855    0.2706   -1.6471   -9.8337  -12.2764  -13.0302
    0.6838    0.8876    0.2673   -1.6480   -9.8865  -12.1007  -13.0115
    0.6513    0.8854    0.2687   -1.6444   -9.8582  -12.1442  -13.0239
    0.7267    0.8765    0.2733   -1.6465   -9.7911  -12.1850  -13.1150
    0.7169    0.8822    0.2719   -1.6472   -9.8151  -11.9481  -13.1444
    0.7142    0.8853    0.2669   -1.6490   -9.8957  -12.1864  -13.0318
    0.7033    0.8880    0.2680   -1.6492   -9.8792  -12.0224  -12.9335
    0.6864    0.8877    0.2691   -1.6481   -9.8595  -12.1626  -12.9984
    0.6542    0.8853    0.2700   -1.6446   -9.8378  -12.1195  -13.0013
    0.7483    0.8846    0.2631   -1.6490   -9.9542  -12.1870  -12.9911
    0.7294    0.8868    0.2703   -1.6477   -9.8404  -12.2410  -13.0213
    0.7254    0.8880    0.2684   -1.6467   -9.8675  -12.2338  -13.0251
    0.7203    0.8887    0.2666   -1.6485   -9.8985  -12.1909  -13.0241
    0.6896    0.8889    0.2680   -1.6473   -9.8744  -12.1200  -12.9908
    0.6870    0.8906    0.2698   -1.6452   -9.8428  -12.1543  -13.0797
    0.7628    0.8890    0.2428   -1.6456  -10.2795  -12.0612  -13.2241
    0.7454    0.8928    0.2619   -1.6474   -9.9710  -12.1349  -12.9411
    0.7306    0.8908    0.2703   -1.6481   -9.8413  -12.1637  -12.9368
    0.7298    0.8921    0.2681   -1.6484   -9.8759  -12.1166  -12.9527
    0.7040    0.8931    0.2709   -1.6474   -9.8301  -12.1177  -12.9799
    0.6541    0.8927    0.2752   -1.6448   -9.7596  -12.1148  -13.0282
    0.7595    0.8820    0.2662   -1.6463   -9.9006  -12.4086  -13.5302
    0.7491    0.8837    0.2764   -1.6454   -9.7428  -11.8186  -12.9684
    0.7439    0.8873    0.2739   -1.6437   -9.7766  -12.1178  -13.1119
    0.7429    0.8888    0.2711   -1.6471   -9.8267  -12.1746  -13.0350
    0.7388    0.8876    0.2718   -1.6462   -9.8135  -12.2075  -13.0959
    0.7337    0.8841    0.2753   -1.6435   -9.7562  -12.0247  -13.0274

];

s.toxic(:,:,1) = [.4651    0.7240    0.6250   -1.5026   -5.8343   -6.9941   -7.5252
    0.4334    0.7323    0.5934   -1.5150   -6.1158   -7.2795   -7.6331
    0.4199    0.7237    0.5886   -1.5116   -6.1480   -7.1868   -7.5423
    0.4235    0.7331    0.5826   -1.5225   -6.2196   -7.3494   -7.6445
    0.4087    0.7306    0.5919   -1.5169   -6.1323   -7.2343   -7.6059
    0.3996    0.7241    0.6014   -1.5090   -6.0395   -7.1253   -7.5263
    0.4614    0.7334    0.5840   -1.5270   -6.2147   -7.3295   -7.6886
    0.4144    0.7205    0.5886   -1.5131   -6.1511   -7.2461   -7.5847
    0.3997    0.7240    0.5883   -1.5118   -6.1511   -7.2138   -7.5900
    0.3707    0.7210    0.5969   -1.5115   -6.0807   -7.1939   -7.5598
    0.3636    0.7213    0.5904   -1.5147   -6.1405   -7.3017   -7.6348
    0.3562    0.7187    0.5918   -1.5086   -6.1167   -7.2066   -7.5512
    0.4576    0.7250    0.5979   -1.5139   -6.0747   -7.2301   -7.5883
    0.4335    0.7297    0.5981   -1.5161   -6.0793   -7.2418   -7.6316
    0.4067    0.7285    0.5911   -1.5187   -6.1419   -7.3199   -7.6391
    0.3870    0.7261    0.5959   -1.5149   -6.0959   -7.3735   -7.6121
    0.3690    0.7261    0.5904   -1.5169   -6.1446   -7.2873   -7.6269
    0.3617    0.7256    0.5883   -1.5160   -6.1600   -7.2542   -7.6015
    0.4510    0.7234    0.5853   -1.5188   -6.1882   -7.3313   -7.5988
    0.4136    0.7275    0.5891   -1.5153   -6.1508   -7.2510   -7.6020
    0.3858    0.7232    0.5965   -1.5129   -6.0862   -7.1806   -7.5736
    0.3836    0.7230    0.5857   -1.5158   -6.1813   -7.3801   -7.6330
    0.3678    0.7238    0.5896   -1.5152   -6.1479   -7.2369   -7.6030
    0.3471    0.7192    0.5873   -1.5125   -6.1614   -7.2093   -7.5665
    0.4649    0.7265    0.5943   -1.5146   -6.1059   -7.1553   -7.5440
    0.4385    0.7311    0.5949   -1.5140   -6.1014   -7.2947   -7.6448
    0.4081    0.7259    0.5940   -1.5121   -6.1055   -7.2388   -7.6053
    0.3939    0.7248    0.5884   -1.5153   -6.1575   -7.2376   -7.5620
    0.3868    0.7275    0.5900   -1.5165   -6.1472   -7.2708   -7.6292
    0.3602    0.7204    0.5890   -1.5126   -6.1476   -7.2801   -7.5697
    0.4549    0.7241    0.5792   -1.5173   -6.2355   -7.3136   -7.5717
    0.4301    0.7204    0.5909   -1.5109   -6.1279   -7.2097   -7.5707
    0.4019    0.7216    0.5958   -1.5155   -6.0970   -7.2683   -7.5871
    0.3709    0.7228    0.5846   -1.5142   -6.1869   -7.2764   -7.5811
    0.3585    0.7218    0.5915   -1.5155   -6.1328   -7.2327   -7.5893
    0.3532    0.7194    0.5910   -1.5126   -6.1308   -7.2642   -7.5652
    0.4655    0.7309    0.5951   -1.5142   -6.0981   -7.2694   -7.5677
    0.4345    0.7279    0.5912   -1.5168   -6.1370   -7.2348   -7.5977
    0.3869    0.7245    0.5892   -1.5154   -6.1513   -7.2390   -7.6032
    0.3666    0.7244    0.5952   -1.5157   -6.1030   -7.2443   -7.6113
    0.3545    0.7227    0.5943   -1.5120   -6.1027   -7.2274   -7.5735
    0.3589    0.7204    0.5964   -1.5099   -6.0817   -7.2160   -7.5414
    0.4524    0.7227    0.5894   -1.5232   -6.1632   -7.2880   -7.6272
    0.4225    0.7285    0.5961   -1.5235   -6.1107   -7.3121   -7.6438
    0.4257    0.7273    0.5998   -1.5169   -6.0677   -7.2568   -7.5895
    0.4150    0.7279    0.5994   -1.5183   -6.0737   -7.2901   -7.5880
    0.4058    0.7290    0.5938   -1.5175   -6.1182   -7.2613   -7.6215
    0.3952    0.7224    0.5975   -1.5079   -6.0689   -7.2080   -7.5347
    0.5191    0.7289    0.6163   -1.5133   -5.9253   -7.3724   -7.5961
    0.4281    0.7237    0.5881   -1.5142   -6.1575   -7.2786   -7.6077
    0.3982    0.7223    0.5945   -1.5157   -6.1086   -7.2221   -7.6111
    0.3798    0.7200    0.5870   -1.5125   -6.1636   -7.2594   -7.5633
    0.3641    0.7208    0.5939   -1.5165   -6.1156   -7.3083   -7.6416
    0.3594    0.7193    0.5948   -1.5128   -6.1008   -7.2400   -7.5684
    0.4426    0.7221    0.5953   -1.5054   -6.0794   -7.1792   -7.5557
    0.4241    0.7285    0.5860   -1.5228   -6.1921   -7.3041   -7.6627
    0.4091    0.7255    0.5899   -1.5196   -6.1537   -7.3200   -7.6386
    0.3979    0.7287    0.5927   -1.5154   -6.1225   -7.2096   -7.6252
    0.3751    0.7280    0.5910   -1.5180   -6.1424   -7.2582   -7.6389
    0.3679    0.7251    0.5922   -1.5159   -6.1283   -7.2603   -7.6063
    0.4612    0.7278    0.5942   -1.5085   -6.0944   -7.0742   -7.5605
    0.4322    0.7282    0.5859   -1.5161   -6.1791   -7.3454   -7.6406
    0.3995    0.7266    0.5789   -1.5155   -6.2361   -7.2349   -7.6225
    0.3821    0.7238    0.5875   -1.5127   -6.1600   -7.2235   -7.5789
    0.3614    0.7247    0.5812   -1.5170   -6.2205   -7.2880   -7.6341
    0.3571    0.7228    0.5841   -1.5143   -6.1912   -7.2461   -7.5987
    0.4630    0.7299    0.5729   -1.5215   -6.2962   -7.2013   -7.6107
    0.4059    0.7254    0.5856   -1.5180   -6.1853   -7.2538   -7.6172
    0.3989    0.7282    0.5909   -1.5189   -6.1443   -7.3067   -7.6273
    0.3903    0.7243    0.5825   -1.5123   -6.2000   -7.2156   -7.6156
    0.3750    0.7261    0.5881   -1.5164   -6.1623   -7.2866   -7.6089
    0.3549    0.7192    0.5874   -1.5130   -6.1613   -7.2722   -7.5810
    0.4450    0.7212    0.5944   -1.5214   -6.1187   -7.2029   -7.6238
    0.4245    0.7217    0.5953   -1.5127   -6.0950   -7.1792   -7.5690
    0.4065    0.7252    0.5885   -1.5148   -6.1561   -7.2178   -7.5863
    0.3983    0.7207    0.5847   -1.5151   -6.1876   -7.2338   -7.5753
    0.3793    0.7205    0.5887   -1.5140   -6.1531   -7.2133   -7.5583
    0.3680    0.7211    0.5872   -1.5138   -6.1651   -7.2248   -7.5549
    0.4502    0.7285    0.5865   -1.5251   -6.1909   -7.4055   -7.7500
    0.4189    0.7193    0.5905   -1.5117   -6.1324   -7.2175   -7.5220
    0.3924    0.7256    0.5885   -1.5139   -6.1535   -7.2700   -7.5815
    0.3685    0.7201    0.5937   -1.5123   -6.1083   -7.2076   -7.5436
    0.3548    0.7217    0.5904   -1.5148   -6.1406   -7.2814   -7.5937
    0.3602    0.7203    0.5942   -1.5093   -6.0984   -7.2267   -7.5447
    0.4578    0.7270    0.5757   -1.5257   -6.2812   -7.3193   -7.6958
    0.4374    0.7292    0.5979   -1.5176   -6.0842   -7.2547   -7.6256
    0.4211    0.7260    0.6004   -1.5109   -6.0506   -7.1319   -7.5381
    0.4142    0.7342    0.5883   -1.5212   -6.1703   -7.3208   -7.6330
    0.3996    0.7250    0.5919   -1.5122   -6.1228   -7.2359   -7.5810
    0.3876    0.7216    0.5947   -1.5091   -6.0938   -7.2373   -7.5352

];
s.toxic(:,:,2) = [.5274    0.6091    0.0883   -0.9975  -13.0798  -14.4847  -11.9640
    0.5127    0.6018    0.0955   -0.9806  -12.7323  -16.1460  -12.9067
    0.5077    0.6015    0.0799   -0.9808  -13.4512  -15.7098  -13.9261
    0.5033    0.6011    0.0746   -0.9789  -13.7203  -15.7715  -12.4518
    0.5024    0.6016    0.0771   -0.9807  -13.5915  -20.9717  -13.0084
    0.4996    0.5986    0.0733   -0.9736  -13.7835  -16.3550  -13.3826
    0.5092    0.5862    0.0515   -0.9737  -15.1967  -12.9409  -13.6570
    0.4984    0.5888    0.0663   -0.9723  -14.1814  -14.5645  -13.5334
    0.4847    0.5897    0.0485   -0.9738  -15.4361  -16.6395  -13.8103
    0.4773    0.5891    0.0738   -0.9733  -13.7515  -16.1433  -13.5931
    0.4495    0.5889    0.0779   -0.9725  -13.5330  -20.3456  -13.9456
    0.4196    0.5884    0.0777   -0.9690  -13.5380  -19.8432  -13.8880
    0.4951    0.5876    0.0589   -0.9746  -14.6614  -16.3080  -15.6890
    0.4896    0.5876    0.0674   -0.9715  -14.1135  -16.1597  -13.4530
    0.4758    0.5886    0.0727   -0.9718  -13.8118  -18.2798  -13.7496
    0.4752    0.5918    0.0775   -0.9752  -13.5629  -16.5226  -13.9338
    0.4613    0.5893    0.0773   -0.9718  -13.5654  -17.5229  -13.7238
    0.4532    0.5899    0.0730   -0.9683  -13.7853  -21.3574  -13.5732
    0.4916    0.5822    0.0738   -0.9728  -13.7540  -14.5026  -14.5492
    0.4802    0.5847    0.0663   -0.9712  -14.1804  -16.7809  -13.9401
    0.4755    0.5872    0.0773   -0.9725  -13.5679  -16.7181  -13.9788
    0.4581    0.5881    0.0826   -0.9733  -13.3020  -16.3585  -13.6248
    0.4432    0.5888    0.0760   -0.9744  -13.6358  -18.8178  -13.8821
    0.4301    0.5870    0.0782   -0.9695  -13.5154  -18.8802  -13.7687
    0.4855    0.5815    0.0616   -0.9710  -14.4704  -13.1874  -13.2024
    0.4741    0.5845    0.0674   -0.9704  -14.1120  -18.2050  -13.7653
    0.4740    0.5866    0.0812   -0.9723  -13.3667  -17.0757  -13.8309
    0.4601    0.5877    0.0850   -0.9721  -13.1819  -15.6173  -13.5948
    0.4519    0.5880    0.0876   -0.9716  -13.0625  -17.8933  -13.7560
    0.4297    0.5867    0.0840   -0.9680  -13.2222  -17.6134  -13.8799
    0.4925    0.5847    0.0619   -0.9712  -14.4530  -16.5037  -13.6331
    0.4830    0.5873    0.0866   -0.9709  -13.1055  -16.5927  -13.5550
    0.4782    0.5882    0.0687   -0.9728  -14.0418  -17.3120  -13.9169
    0.4744    0.5919    0.0760   -0.9758  -13.6423  -16.0981  -14.1587
    0.4577    0.5892    0.0795   -0.9712  -13.4483  -16.9355  -14.0124
    0.4501    0.5887    0.0867   -0.9678  -13.0976  -19.0465  -13.8853
    0.5092    0.5892    0.1362   -0.9737  -11.2899  -12.4855  -14.0361
    0.4928    0.5871    0.0684   -0.9736  -14.0598  -14.1180  -14.3282
    0.4838    0.5891    0.0742   -0.9719  -13.7270  -18.6697  -14.1587
    0.4807    0.5898    0.0705   -0.9722  -13.9322  -18.1831  -13.5682
    0.4571    0.5883    0.0728   -0.9722  -13.8037  -16.9855  -13.8452
    0.4317    0.5880    0.0838   -0.9686  -13.2327  -18.6512  -13.5751
    0.5113    0.5905    0.0946   -0.9780  -12.7669  -12.2548  -13.3520
    0.4989    0.5847    0.1043   -0.9621  -12.3394  -16.4052  -12.9914
    0.5001    0.5907    0.0916   -0.9729  -12.8844  -16.7087  -13.0555
    0.4916    0.5879    0.0777   -0.9680  -13.5352  -13.9056  -13.6314
    0.5000    0.5947    0.0817   -0.9797  -13.3582  -19.5770  -13.3871
    0.4867    0.5860    0.0748   -0.9663  -13.6864  -13.9909  -13.3793
    0.5041    0.5842    0.0921   -0.9727  -12.8615  -14.4015  -14.8424
    0.4928    0.5867    0.0650   -0.9715  -14.2581  -16.1006  -13.6535
    0.4858    0.5879    0.0698   -0.9730  -13.9776  -16.3134  -13.9487
    0.4804    0.5896    0.0776   -0.9729  -13.5515  -16.2026  -14.2724
    0.4561    0.5889    0.0733   -0.9725  -13.7766  -18.4053  -13.7687
    0.4300    0.5882    0.0773   -0.9692  -13.5613  -19.1349  -13.7368
    0.4940    0.5851    0.0739   -0.9712  -13.7443  -14.3591  -13.7099
    0.4816    0.5860    0.0657   -0.9679  -14.2076  -15.5631  -14.0904
    0.4833    0.5887    0.0676   -0.9721  -14.1005  -20.3589  -13.6382
    0.4750    0.5914    0.0797   -0.9740  -13.4443  -15.4582  -13.1931
    0.4606    0.5894    0.0795   -0.9709  -13.4522  -19.7238  -13.8570
    0.4578    0.5889    0.0725   -0.9671  -13.8143  -20.4617  -13.8459
    0.4847    0.5809    0.0480   -0.9702  -15.4718  -16.6291  -13.3877
    0.4770    0.5849    0.0552   -0.9714  -14.9129  -18.0718  -13.9110
    0.4684    0.5865    0.0810   -0.9725  -13.3799  -16.2330  -13.8994
    0.4586    0.5879    0.0763   -0.9731  -13.6209  -17.3474  -13.7560
    0.4423    0.5882    0.0707   -0.9737  -13.9250  -17.4335  -13.8071
    0.4269    0.5866    0.0806   -0.9691  -13.3897  -17.4903  -13.7501
    0.4911    0.5797    0.0719   -0.9662  -13.8428  -18.9568  -13.9988
    0.4785    0.5847    0.0547   -0.9706  -14.9446  -17.1456  -13.6939
    0.4757    0.5861    0.0777   -0.9718  -13.5455  -15.7665  -13.5577
    0.4655    0.5874    0.0841   -0.9732  -13.2267  -16.3265  -13.8721
    0.4542    0.5874    0.0824   -0.9719  -13.3068  -16.9962  -13.6333
    0.4352    0.5868    0.0873   -0.9693  -13.0686  -19.7649  -13.5778
    0.4926    0.5860    0.0563   -0.9729  -14.8399  -16.5465  -14.2623
    0.4869    0.5884    0.0859   -0.9730  -13.1429  -15.6596  -14.2737
    0.4857    0.5891    0.0785   -0.9724  -13.5058  -18.6606  -13.7279
    0.4754    0.5916    0.0674   -0.9773  -14.1241  -17.5701  -13.4391
    0.4624    0.5900    0.0777   -0.9719  -13.5416  -16.1583  -13.9817
    0.4574    0.5893    0.0779   -0.9684  -13.5243  -19.6216  -13.7766
    0.5033    0.5904    0.0248   -0.9724  -18.1215  -14.5583  -12.6350
    0.4946    0.5896    0.0843   -0.9710  -13.2126  -14.7325  -12.9221
    0.4778    0.5866    0.0811   -0.9686  -13.3658  -17.0898  -13.5982
    0.4760    0.5882    0.0788   -0.9689  -13.4807  -16.9150  -13.6136
    0.4572    0.5873    0.0795   -0.9681  -13.4422  -18.1084  -13.6813
    0.4269    0.5875    0.0808   -0.9655  -13.3728  -17.5040  -13.7729
    0.5055    0.5844    0.1068   -0.9684  -12.2592  -12.2281  -13.8520
    0.5013    0.5869    0.1007   -0.9709  -12.4995  -11.4515  -15.3292
    0.4974    0.5919    0.0824   -0.9765  -13.3172  -14.8038  -12.9903
    0.4902    0.5881    0.0718   -0.9689  -13.8544  -14.2927  -13.7182
    0.4908    0.5905    0.0774   -0.9752  -13.5679  -14.7957  -13.1284
    0.4876    0.5861    0.0842   -0.9677  -13.2117  -21.0647  -13.3407

];
s.toxic(:,:,3) = [.6115    0.7135    0.5282   -1.3883   -6.4154   -8.5321   -8.0096
    0.6096    0.7175    0.5119   -1.3875   -6.5586   -8.5721   -8.0070
    0.6015    0.7131    0.5159   -1.3848   -6.5172   -8.4590   -7.9648
    0.5976    0.7132    0.5365   -1.3926   -6.3510   -8.3905   -8.0159
    0.5982    0.7158    0.5246   -1.3899   -6.4500   -8.4558   -7.9954
    0.5943    0.7106    0.5274   -1.3829   -6.4119   -8.3856   -7.9621
    0.6027    0.6970    0.5284   -1.3930   -6.4224   -8.5457   -7.9953
    0.5911    0.7015    0.5337   -1.3928   -6.3763   -8.3701   -8.0229
    0.5786    0.7022    0.5330   -1.3919   -6.3804   -8.3378   -8.0028
    0.5734    0.7020    0.5314   -1.3922   -6.3949   -8.3577   -7.9999
    0.5426    0.7021    0.5319   -1.3916   -6.3899   -8.3397   -7.9984
    0.5152    0.7005    0.5335   -1.3870   -6.3666   -8.3105   -7.9779
    0.5829    0.6963    0.5329   -1.3852   -6.3672   -8.2804   -7.9279
    0.5759    0.7001    0.5320   -1.3900   -6.3857   -8.3137   -7.9851
    0.5711    0.7024    0.5332   -1.3926   -6.3805   -8.3240   -8.0049
    0.5637    0.7028    0.5315   -1.3889   -6.3878   -8.3593   -7.9758
    0.5400    0.7017    0.5324   -1.3898   -6.3813   -8.3359   -7.9907
    0.5395    0.7022    0.5348   -1.3864   -6.3538   -8.2822   -7.9886
    0.5759    0.6933    0.5316   -1.3924   -6.3930   -8.3638   -8.0088
    0.5624    0.6966    0.5323   -1.3909   -6.3846   -8.3282   -7.9944
    0.5592    0.6984    0.5298   -1.3894   -6.4033   -8.4016   -7.9875
    0.5525    0.6992    0.5316   -1.3906   -6.3906   -8.3463   -7.9865
    0.5394    0.7008    0.5317   -1.3910   -6.3900   -8.3644   -7.9932
    0.5139    0.6970    0.5373   -1.3860   -6.3315   -8.2785   -7.9664
    0.5760    0.6942    0.5279   -1.3903   -6.4214   -8.4013   -7.9725
    0.5599    0.6970    0.5332   -1.3914   -6.3778   -8.3098   -8.0016
    0.5596    0.6993    0.5339   -1.3911   -6.3714   -8.3227   -8.0031
    0.5512    0.7003    0.5348   -1.3922   -6.3659   -8.3182   -8.0144
    0.5330    0.6999    0.5347   -1.3902   -6.3624   -8.2845   -8.0040
    0.5116    0.6973    0.5390   -1.3856   -6.3155   -8.2546   -7.9718
    0.5828    0.6966    0.5341   -1.3892   -6.3647   -8.2697   -7.9477
    0.5755    0.7010    0.5344   -1.3917   -6.3680   -8.2541   -8.0360
    0.5718    0.7019    0.5330   -1.3923   -6.3812   -8.3057   -7.9878
    0.5653    0.7041    0.5320   -1.3931   -6.3920   -8.3301   -7.9992
    0.5428    0.7023    0.5335   -1.3911   -6.3748   -8.3001   -8.0031
    0.5420    0.7023    0.5336   -1.3878   -6.3670   -8.3149   -7.9878
    0.6035    0.7007    0.5361   -1.3962   -6.3619   -8.1917   -8.0082
    0.5834    0.7024    0.5291   -1.3918   -6.4144   -8.4023   -7.9904
    0.5722    0.7010    0.5343   -1.3915   -6.3680   -8.3074   -8.0036
    0.5708    0.7008    0.5346   -1.3911   -6.3651   -8.3370   -7.9826
    0.5495    0.7013    0.5344   -1.3914   -6.3673   -8.3230   -7.9998
    0.5089    0.6997    0.5350   -1.3878   -6.3548   -8.3144   -7.9818
    0.6074    0.7044    0.5415   -1.3973   -6.3171   -8.2677   -8.0554
    0.5914    0.6992    0.5255   -1.3882   -6.4392   -8.2323   -7.9473
    0.5937    0.7017    0.5384   -1.3886   -6.3268   -8.2925   -8.0067
    0.5865    0.7005    0.5328   -1.3914   -6.3814   -8.2963   -8.0053
    0.5928    0.7074    0.5275   -1.3932   -6.4319   -8.4196   -8.0333
    0.5810    0.6960    0.5366   -1.3837   -6.3331   -8.2604   -7.9568
    0.5943    0.6937    0.5344   -1.3895   -6.3634   -8.2453   -7.9778
    0.5830    0.6970    0.5360   -1.3866   -6.3437   -8.2823   -7.9936
    0.5796    0.7001    0.5333   -1.3897   -6.3735   -8.3177   -7.9849
    0.5758    0.7013    0.5317   -1.3922   -6.3925   -8.3569   -7.9908
    0.5441    0.7019    0.5321   -1.3915   -6.3878   -8.3202   -7.9910
    0.5118    0.6991    0.5342   -1.3866   -6.3598   -8.2874   -7.9677
    0.5834    0.6967    0.5350   -1.3895   -6.3576   -8.3497   -7.9547
    0.5790    0.7022    0.5323   -1.3934   -6.3897   -8.2892   -7.9948
    0.5725    0.7028    0.5308   -1.3936   -6.4034   -8.3131   -8.0026
    0.5659    0.7039    0.5319   -1.3945   -6.3956   -8.3206   -8.0157
    0.5497    0.7040    0.5310   -1.3943   -6.4024   -8.3462   -8.0106
    0.5441    0.7025    0.5340   -1.3885   -6.3651   -8.3243   -7.9855
    0.5749    0.6922    0.5308   -1.3890   -6.3938   -8.3220   -7.9864
    0.5697    0.6977    0.5339   -1.3916   -6.3719   -8.3237   -7.9916
    0.5567    0.6990    0.5302   -1.3919   -6.4049   -8.3809   -8.0018
    0.5499    0.7006    0.5321   -1.3913   -6.3869   -8.3481   -7.9874
    0.5365    0.7014    0.5319   -1.3920   -6.3903   -8.3219   -7.9891
    0.5101    0.6973    0.5368   -1.3859   -6.3352   -8.2717   -7.9650
    0.5760    0.6905    0.5376   -1.3886   -6.3331   -8.2161   -8.0044
    0.5566    0.6952    0.5331   -1.3898   -6.3757   -8.3146   -7.9829
    0.5567    0.6975    0.5308   -1.3909   -6.3978   -8.3848   -7.9794
    0.5489    0.6990    0.5331   -1.3900   -6.3756   -8.2653   -7.9870
    0.5341    0.6994    0.5315   -1.3909   -6.3913   -8.3579   -7.9896
    0.5127    0.6967    0.5369   -1.3862   -6.3349   -8.2822   -7.9681
    0.5837    0.6975    0.5321   -1.3927   -6.3898   -8.3179   -8.0053
    0.5774    0.7020    0.5330   -1.3932   -6.3832   -8.2949   -7.9996
    0.5737    0.7026    0.5334   -1.3914   -6.3757   -8.3063   -7.9927
    0.5643    0.7029    0.5317   -1.3907   -6.3895   -8.3125   -8.0046
    0.5460    0.7021    0.5323   -1.3915   -6.3858   -8.3287   -8.0003
    0.5377    0.7033    0.5340   -1.3901   -6.3680   -8.3047   -7.9918
    0.5964    0.6938    0.5110   -1.3856   -6.5631   -8.4221   -7.9409
    0.5803    0.6978    0.5294   -1.3853   -6.3990   -8.2636   -7.9653
    0.5731    0.6999    0.5260   -1.3906   -6.4394   -8.3355   -7.9891
    0.5672    0.6999    0.5284   -1.3895   -6.4166   -8.3332   -7.9842
    0.5579    0.7034    0.5279   -1.3906   -6.4225   -8.3212   -7.9872
    0.5305    0.7021    0.5283   -1.3872   -6.4120   -8.2933   -7.9664
    0.6040    0.6972    0.5270   -1.3833   -6.4158   -8.2542   -7.9672
    0.5946    0.6968    0.5292   -1.3851   -6.3998   -8.6347   -7.9481
    0.5884    0.7004    0.5337   -1.3862   -6.3626   -8.3144   -7.9339
    0.5852    0.7015    0.5327   -1.3923   -6.3838   -8.4338   -7.9981
    0.5837    0.7014    0.5357   -1.3886   -6.3507   -8.2546   -7.9883
    0.5735    0.6948    0.5366   -1.3823   -6.3297   -8.2920   -7.9589

];

s.wc(:,:,1) = [.7518    0.8273    0.2796   -1.4786   -9.3614  -17.1274  -11.8603
    0.7479    0.8316    0.2866   -1.4893   -9.2797  -18.3307  -12.2182
    0.7453    0.8302    0.2854   -1.4855   -9.2900  -17.7091  -12.3769
    0.7386    0.8283    0.2850   -1.4806   -9.2861  -18.8451  -12.2903
    0.7299    0.8263    0.2872   -1.4777   -9.2475  -20.2567  -12.3023
    0.7186    0.8221    0.2901   -1.4693   -9.1894  -17.5458  -12.3243
    0.7368    0.8097    0.2903   -1.4785   -9.2045  -15.9597  -12.4477
    0.7225    0.8135    0.2883   -1.4760   -9.2281  -18.8061  -12.3614
    0.7028    0.8135    0.2872   -1.4755   -9.2433  -16.9722  -12.3628
    0.7063    0.8156    0.2819   -1.4771   -9.3241  -17.3116  -12.4125
    0.6720    0.8144    0.2882   -1.4753   -9.2285  -16.3265  -12.3728
    0.6340    0.8136    0.2860   -1.4721   -9.2536  -17.0042  -12.4240
    0.7147    0.8088    0.2926   -1.4704   -9.1554  -15.2775  -12.4444
    0.7109    0.8133    0.2881   -1.4764   -9.2317  -16.2325  -12.4295
    0.6966    0.8145    0.2876   -1.4751   -9.2369  -17.6640  -12.3654
    0.6810    0.8157    0.2865   -1.4772   -9.2563  -15.8838  -12.5596
    0.6691    0.8157    0.2868   -1.4750   -9.2479  -16.0709  -12.4750
    0.6700    0.8156    0.2877   -1.4746   -9.2343  -16.1153  -12.4163
    0.7101    0.8053    0.2899   -1.4724   -9.1971  -15.3692  -12.3028
    0.7019    0.8098    0.2852   -1.4758   -9.2726  -17.2108  -12.3678
    0.6884    0.8112    0.2867   -1.4761   -9.2520  -15.6704  -12.4595
    0.6789    0.8124    0.2896   -1.4746   -9.2074  -20.0872  -12.3951
    0.6693    0.8138    0.2873   -1.4754   -9.2423  -16.2141  -12.4364
    0.6294    0.8108    0.2932   -1.4718   -9.1499  -16.4772  -12.3913
    0.7136    0.8063    0.2896   -1.4756   -9.2091  -15.8396  -12.3343
    0.6945    0.8091    0.2873   -1.4756   -9.2418  -14.7385  -12.5274
    0.6820    0.8111    0.2868   -1.4759   -9.2502  -16.4425  -12.5025
    0.6753    0.8137    0.2851   -1.4768   -9.2767  -15.2726  -12.5126
    0.6610    0.8135    0.2875   -1.4767   -9.2416  -15.9818  -12.5735
    0.6370    0.8122    0.2934   -1.4725   -9.1486  -16.3854  -12.4370
    0.7175    0.8079    0.2862   -1.4702   -9.2475  -15.8775  -12.4885
    0.7088    0.8129    0.2860   -1.4744   -9.2580  -16.1552  -12.5130
    0.6915    0.8146    0.2877   -1.4753   -9.2357  -17.1445  -12.4393
    0.6843    0.8166    0.2875   -1.4773   -9.2425  -16.1541  -12.5272
    0.6733    0.8158    0.2881   -1.4749   -9.2293  -16.4971  -12.4828
    0.6478    0.8153    0.2902   -1.4743   -9.1972  -16.1972  -12.4274
    0.7433    0.8183    0.2850   -1.4820   -9.2877  -16.3954  -12.3600
    0.7148    0.8110    0.2854   -1.4728   -9.2644  -16.0888  -12.4128
    0.6930    0.8125    0.2892   -1.4743   -9.2118  -17.3041  -12.3396
    0.6954    0.8141    0.2914   -1.4750   -9.1817  -15.8086  -12.4953
    0.6590    0.8134    0.2891   -1.4748   -9.2148  -19.5166  -12.3048
    0.6211    0.8118    0.2882   -1.4708   -9.2189  -16.0236  -12.4042
    0.7411    0.8136    0.2874   -1.4766   -9.2414  -13.4455  -11.9021
    0.7077    0.7986    0.2929   -1.4640   -9.1380  -17.9153  -12.3598
    0.7278    0.8125    0.2874   -1.4706   -9.2304  -15.2013  -12.6230
    0.7357    0.8233    0.2823   -1.4885   -9.3414  -16.8716  -12.2028
    0.7207    0.8166    0.2874   -1.4799   -9.2495  -18.3875  -12.3715
    0.7054    0.8072    0.2918   -1.4657   -9.1576  -15.5041  -12.4778
    0.7338    0.8075    0.2857   -1.4729   -9.2588  -17.8695  -12.3751
    0.7185    0.8121    0.2888   -1.4761   -9.2214  -16.7813  -12.3880
    0.7056    0.8126    0.2899   -1.4739   -9.2016  -16.6549  -12.3807
    0.7048    0.8145    0.2829   -1.4769   -9.3083  -16.3382  -12.4000
    0.6703    0.8138    0.2886   -1.4754   -9.2226  -16.3580  -12.4082
    0.6324    0.8128    0.2861   -1.4718   -9.2523  -16.6224  -12.4400
    0.7180    0.8074    0.2899   -1.4714   -9.1962  -17.4377  -12.3672
    0.7007    0.8130    0.2880   -1.4756   -9.2322  -16.1427  -12.3820
    0.6938    0.8146    0.2874   -1.4763   -9.2419  -16.4923  -12.4109
    0.6815    0.8164    0.2867   -1.4777   -9.2550  -15.9112  -12.3783
    0.6565    0.8155    0.2881   -1.4755   -9.2299  -16.2185  -12.4365
    0.6556    0.8163    0.2894   -1.4752   -9.2116  -17.3539  -12.4113
    0.7057    0.8066    0.2880   -1.4766   -9.2335  -16.3811  -12.3117
    0.6875    0.8096    0.2864   -1.4747   -9.2525  -15.3930  -12.4836
    0.6894    0.8130    0.2866   -1.4765   -9.2534  -15.9763  -12.4284
    0.6765    0.8143    0.2863   -1.4764   -9.2575  -15.1680  -12.4133
    0.6619    0.8139    0.2868   -1.4762   -9.2508  -16.3668  -12.4202
    0.6321    0.8117    0.2929   -1.4723   -9.1551  -16.2911  -12.3860
    0.7030    0.8043    0.2889   -1.4752   -9.2174  -14.7445  -12.5700
    0.6973    0.8104    0.2859   -1.4764   -9.2640  -16.8244  -12.4064
    0.6858    0.8112    0.2868   -1.4757   -9.2493  -17.8267  -12.3701
    0.6766    0.8123    0.2876   -1.4745   -9.2354  -19.6157  -12.2584
    0.6545    0.8126    0.2876   -1.4760   -9.2388  -16.7837  -12.3734
    0.6344    0.8110    0.2929   -1.4711   -9.1524  -16.8264  -12.3755
    0.7215    0.8091    0.2914   -1.4725   -9.1765  -15.2931  -12.2638
    0.7015    0.8133    0.2887   -1.4759   -9.2216  -16.9821  -12.4063
    0.6962    0.8136    0.2887   -1.4748   -9.2202  -16.4136  -12.5167
    0.6834    0.8132    0.2873   -1.4741   -9.2396  -15.7657  -12.4308
    0.6703    0.8139    0.2886   -1.4750   -9.2223  -16.6422  -12.4566
    0.6579    0.8155    0.2903   -1.4740   -9.1956  -16.5245  -12.4104
    0.7435    0.8169    0.2855   -1.4787   -9.2747  -17.7244  -12.5795
    0.7268    0.8173    0.2856   -1.4734   -9.2618  -17.0041  -12.3391
    0.6969    0.8139    0.2890   -1.4731   -9.2129  -16.9421  -12.3722
    0.7087    0.8156    0.2916   -1.4723   -9.1733  -18.2918  -12.4071
    0.6715    0.8164    0.2889   -1.4736   -9.2147  -17.6663  -12.4456
    0.6252    0.8151    0.2878   -1.4700   -9.2233  -16.8630  -12.4479
    0.7334    0.8089    0.2851   -1.4753   -9.2733  -14.7469  -12.9226
    0.7292    0.8100    0.2881   -1.4739   -9.2263  -17.3785  -12.4211
    0.7150    0.8060    0.2897   -1.4679   -9.1921  -13.9821  -12.3864
    0.7332    0.8228    0.2830   -1.4878   -9.3295  -17.0885  -12.1635
    0.7117    0.8133    0.2870   -1.4755   -9.2466  -14.6501  -12.3445
    0.7039    0.8078    0.2908   -1.4667   -9.1732  -16.6402  -12.4292

];
s.wc(:,:,2) = [0.7395    0.8138    0.2766   -1.4457   -9.3401  -17.1259  -13.4237
    0.7329    0.8149    0.2725   -1.4483   -9.4068  -16.3933  -12.7690
    0.7302    0.8143    0.2750   -1.4472   -9.3669  -16.2619  -13.3012
    0.7279    0.8167    0.2696   -1.4505   -9.4566  -16.2961  -12.8483
    0.7204    0.8157    0.2734   -1.4501   -9.3976  -16.6267  -12.8581
    0.7115    0.8141    0.2735   -1.4466   -9.3887  -15.9792  -12.7232
    0.7272    0.7991    0.2740   -1.4496   -9.3871  -14.9078  -12.9792
    0.7132    0.8030    0.2726   -1.4480   -9.4051  -17.7483  -12.9328
    0.6948    0.8043    0.2712   -1.4502   -9.4316  -16.0421  -12.8797
    0.6979    0.8059    0.2660   -1.4507   -9.5121  -15.7531  -12.9496
    0.6628    0.8048    0.2724   -1.4490   -9.4101  -15.9852  -12.8819
    0.6255    0.8039    0.2703   -1.4457   -9.4354  -16.5883  -12.8772
    0.7079    0.8011    0.2744   -1.4478   -9.3766  -14.7296  -12.8037
    0.7003    0.8029    0.2718   -1.4487   -9.4186  -15.6271  -12.8549
    0.6888    0.8054    0.2708   -1.4494   -9.4351  -16.0388  -12.8263
    0.6729    0.8054    0.2698   -1.4493   -9.4514  -16.1435  -12.9323
    0.6614    0.8063    0.2704   -1.4491   -9.4412  -15.8809  -12.8758
    0.6628    0.8065    0.2718   -1.4492   -9.4197  -15.5325  -12.8863
    0.7031    0.7974    0.2746   -1.4496   -9.3777  -14.4446  -12.7331
    0.6938    0.8005    0.2701   -1.4503   -9.4481  -15.9292  -12.9007
    0.6813    0.8029    0.2696   -1.4522   -9.4602  -15.4888  -12.9194
    0.6703    0.8022    0.2731   -1.4466   -9.3958  -16.4034  -12.9404
    0.6622    0.8048    0.2703   -1.4503   -9.4453  -15.8412  -12.8877
    0.6208    0.8015    0.2776   -1.4461   -9.3261  -15.9571  -12.9244
    0.7061    0.7978    0.2725   -1.4513   -9.4126  -15.3288  -12.6663
    0.6867    0.8001    0.2716   -1.4506   -9.4260  -14.6905  -13.0076
    0.6776    0.8027    0.2703   -1.4520   -9.4481  -16.0911  -12.9896
    0.6700    0.8048    0.2697   -1.4523   -9.4589  -14.9882  -12.9280
    0.6551    0.8045    0.2707   -1.4517   -9.4419  -15.9790  -12.9819
    0.6298    0.8029    0.2779   -1.4470   -9.3237  -16.0897  -12.9322
    0.7107    0.8002    0.2698   -1.4481   -9.4487  -17.1467  -13.1658
    0.7011    0.8041    0.2697   -1.4498   -9.4535  -17.1452  -13.0637
    0.6837    0.8055    0.2716   -1.4498   -9.4243  -16.1125  -12.9545
    0.6767    0.8070    0.2717   -1.4512   -9.4252  -17.0474  -13.1063
    0.6657    0.8066    0.2719   -1.4494   -9.4187  -15.9656  -12.9745
    0.6419    0.8059    0.2738   -1.4481   -9.3878  -15.5842  -12.9252
    0.7326    0.8065    0.2692   -1.4508   -9.4623  -15.2399  -12.9187
    0.7081    0.8033    0.2669   -1.4506   -9.4987  -14.7726  -12.8105
    0.6847    0.8027    0.2744   -1.4480   -9.3785  -16.1707  -12.8979
    0.6875    0.8048    0.2762   -1.4498   -9.3538  -15.6765  -13.0900
    0.6510    0.8035    0.2737   -1.4482   -9.3890  -16.7401  -12.7728
    0.6138    0.8023    0.2724   -1.4448   -9.4022  -15.6217  -12.8935
    0.7347    0.8066    0.2658   -1.4552   -9.5238  -12.9598  -11.7943
    0.7004    0.7900    0.2766   -1.4391   -9.3275  -16.4005  -12.8339
    0.7225    0.8063    0.2704   -1.4515   -9.4466  -15.5655  -13.1158
    0.7244    0.8105    0.2668   -1.4551   -9.5097  -15.7712  -12.6833
    0.7113    0.8058    0.2727   -1.4509   -9.4091  -16.6302  -12.9009
    0.6987    0.7993    0.2749   -1.4429   -9.3596  -15.9571  -12.9556
    0.7253    0.7981    0.2683   -1.4469   -9.4689  -16.8457  -12.7291
    0.7097    0.8020    0.2733   -1.4489   -9.3959  -16.6231  -12.9936
    0.6982    0.8041    0.2730   -1.4500   -9.4032  -16.6449  -12.9450
    0.6929    0.8030    0.2665   -1.4467   -9.4963  -16.4810  -12.9473
    0.6605    0.8043    0.2727   -1.4494   -9.4060  -16.1945  -12.9776
    0.6239    0.8031    0.2699   -1.4452   -9.4408  -16.6183  -12.9180
    0.7109    0.7995    0.2732   -1.4483   -9.3971  -17.8684  -12.9958
    0.6955    0.8031    0.2735   -1.4486   -9.3923  -15.7800  -12.8807
    0.6903    0.8056    0.2711   -1.4510   -9.4347  -15.6034  -12.9431
    0.6761    0.8061    0.2697   -1.4495   -9.4530  -15.1192  -12.9548
    0.6510    0.8057    0.2723   -1.4489   -9.4114  -16.4768  -12.9211
    0.6504    0.8064    0.2744   -1.4484   -9.3793  -16.3404  -12.9369
    0.6982    0.7982    0.2738   -1.4534   -9.3982  -15.0534  -12.7042
    0.6801    0.8009    0.2712   -1.4504   -9.4318  -15.6512  -13.1177
    0.6830    0.8038    0.2702   -1.4508   -9.4480  -15.5270  -12.9570
    0.6703    0.8056    0.2711   -1.4522   -9.4375  -15.0807  -12.7953
    0.6549    0.8044    0.2717   -1.4504   -9.4245  -16.2669  -12.9709
    0.6254    0.8022    0.2773   -1.4463   -9.3311  -15.9508  -12.8706
    0.6946    0.7946    0.2722   -1.4486   -9.4125  -14.7964  -13.0374
    0.6891    0.8009    0.2719   -1.4506   -9.4220  -17.6808  -12.9686
    0.6776    0.8015    0.2717   -1.4494   -9.4225  -16.2229  -12.9444
    0.6712    0.8032    0.2716   -1.4493   -9.4234  -16.7312  -12.8354
    0.6507    0.8028    0.2720   -1.4491   -9.4164  -16.2608  -12.8700
    0.6283    0.8017    0.2777   -1.4456   -9.3241  -16.3565  -12.9210
    0.7117    0.7982    0.2785   -1.4437   -9.3074  -15.0806  -12.8106
    0.6961    0.8040    0.2718   -1.4502   -9.4217  -16.3852  -12.8828
    0.6901    0.8048    0.2718   -1.4498   -9.4219  -16.3269  -13.0702
    0.6755    0.8039    0.2705   -1.4484   -9.4385  -15.1723  -13.0715
    0.6615    0.8051    0.2717   -1.4502   -9.4237  -16.9561  -12.9949
    0.6507    0.8063    0.2734   -1.4483   -9.3934  -16.1226  -12.9169
    0.7339    0.8078    0.2710   -1.4529   -9.4397  -16.3015  -13.2376
    0.7190    0.8093    0.2684   -1.4505   -9.4748  -15.3386  -12.8136
    0.6888    0.8065    0.2722   -1.4516   -9.4185  -15.5224  -12.8724
    0.6989    0.8060    0.2765   -1.4463   -9.3427  -17.9617  -12.9082
    0.6632    0.8076    0.2734   -1.4494   -9.3964  -16.4880  -12.9847
    0.6179    0.8065    0.2714   -1.4462   -9.4194  -15.8277  -12.9580
    0.7243    0.7989    0.2696   -1.4486   -9.4528  -14.6532  -13.5421
    0.7195    0.8003    0.2724   -1.4477   -9.4084  -15.6385  -12.8111
    0.7087    0.7989    0.2727   -1.4469   -9.4015  -14.5759  -12.8806
    0.7221    0.8101    0.2676   -1.4547   -9.4956  -15.8054  -12.6634
    0.7021    0.8026    0.2731   -1.4472   -9.3958  -15.2440  -13.0690
    0.6959    0.7985    0.2755   -1.4410   -9.3482  -16.3981  -13.0787

];
s.wc(:,:,3) = [.1517    0.4234    0.9510   -0.3067   -0.9955   -2.0275   -3.4395
    0.1428    0.4174    0.9550   -0.2616   -0.8753   -1.9602   -3.2555
    0.1401    0.4117    0.9559   -0.2615   -0.8683   -1.9833   -3.2951
    0.1396    0.4129    0.9554   -0.2377   -0.8249   -1.8619   -3.0971
    0.1386    0.4183    0.9543   -0.2867   -0.9315   -1.9850   -3.2703
    0.1324    0.4079    0.9552   -0.2301   -0.8115   -1.8645   -3.1880
    0.1475    0.4179    0.9540   -0.2576   -0.8744   -1.8676   -3.0990
    0.1373    0.4084    0.9538   -0.2730   -0.9077   -1.9485   -3.2311
    0.1366    0.4131    0.9551   -0.2577   -0.8671   -1.9322   -3.2010
    0.1314    0.4116    0.9553   -0.2546   -0.8591   -1.8944   -3.1162
    0.1228    0.4110    0.9552   -0.2531   -0.8575   -1.9090   -3.1565
    0.1163    0.4090    0.9549   -0.2514   -0.8559   -1.9083   -3.1861
    0.1400    0.4157    0.9541   -0.2492   -0.8567   -1.9068   -3.2545
    0.1381    0.4154    0.9544   -0.2630   -0.8831   -1.9350   -3.2074
    0.1337    0.4246    0.9551   -0.2635   -0.8789   -1.9431   -3.2277
    0.1256    0.4116    0.9539   -0.2650   -0.8914   -1.8993   -3.1358
    0.1206    0.4114    0.9547   -0.2601   -0.8752   -1.9269   -3.2093
    0.1199    0.4101    0.9546   -0.2531   -0.8620   -1.8988   -3.1679
    0.1412    0.4133    0.9535   -0.2640   -0.8909   -1.9359   -3.2001
    0.1365    0.4166    0.9549   -0.2546   -0.8624   -1.9032   -3.1512
    0.1327    0.4135    0.9538   -0.2560   -0.8735   -1.8865   -3.1740
    0.1266    0.4181    0.9546   -0.2730   -0.9017   -1.9087   -3.1119
    0.1213    0.4131    0.9540   -0.2603   -0.8811   -1.9096   -3.1757
    0.1184    0.4084    0.9545   -0.2429   -0.8423   -1.8650   -3.1215
    0.1415    0.4123    0.9544   -0.2439   -0.8439   -1.8801   -3.1588
    0.1394    0.4169    0.9549   -0.2515   -0.8560   -1.9275   -3.2308
    0.1340    0.4182    0.9548   -0.2524   -0.8591   -1.9065   -3.2007
    0.1301    0.4124    0.9547   -0.2393   -0.8333   -1.8948   -3.1966
    0.1249    0.4121    0.9545   -0.2583   -0.8735   -1.8986   -3.1518
    0.1183    0.4095    0.9551   -0.2381   -0.8282   -1.8674   -3.1344
    0.1418    0.4158    0.9548   -0.2378   -0.8288   -1.8034   -3.0452
    0.1402    0.4172    0.9545   -0.2600   -0.8760   -1.9029   -3.1705
    0.1334    0.4098    0.9548   -0.2492   -0.8522   -1.8838   -3.1534
    0.1310    0.4083    0.9546   -0.2428   -0.8409   -1.8540   -3.1207
    0.1277    0.4087    0.9543   -0.2513   -0.8605   -1.8724   -3.1420
    0.1266    0.4111    0.9545   -0.2454   -0.8475   -1.8461   -3.0920
    0.1540    0.4345    0.9535   -0.3197   -1.0027   -2.1095   -3.4428
    0.1400    0.4196    0.9553   -0.2709   -0.8916   -1.9857   -3.2845
    0.1325    0.4138    0.9537   -0.2584   -0.8796   -1.8469   -3.0738
    0.1292    0.4080    0.9528   -0.2633   -0.8958   -1.8155   -3.0222
    0.1195    0.4099    0.9543   -0.2604   -0.8786   -1.9053   -3.1775
    0.1181    0.4092    0.9548   -0.2573   -0.8691   -1.9202   -3.2056
    0.1486    0.4135    0.9529   -0.2710   -0.9099   -1.8225   -2.9253
    0.1426    0.4126    0.9563   -0.2442   -0.8307   -1.8847   -3.1204
    0.1412    0.4160    0.9553   -0.2620   -0.8743   -1.9051   -3.1301
    0.1347    0.4016    0.9573   -0.1941   -0.7233   -1.7920   -3.0657
    0.1350    0.4106    0.9563   -0.2354   -0.8136   -1.8805   -3.1180
    0.1291    0.3986    0.9570   -0.1967   -0.7305   -1.7908   -3.0419
    0.1465    0.4224    0.9544   -0.2612   -0.8783   -1.9379   -3.2033
    0.1394    0.4133    0.9538   -0.2648   -0.8913   -1.9337   -3.2167
    0.1329    0.4081    0.9543   -0.2584   -0.8747   -1.9419   -3.2517
    0.1298    0.4139    0.9547   -0.2657   -0.8867   -1.9708   -3.2657
    0.1230    0.4081    0.9543   -0.2563   -0.8708   -1.9181   -3.2100
    0.1123    0.4079    0.9543   -0.2517   -0.8617   -1.9047   -3.1991
    0.1484    0.4268    0.9552   -0.2578   -0.8661   -1.9195   -3.1848
    0.1363    0.4103    0.9545   -0.2611   -0.8785   -1.9193   -3.2034
    0.1286    0.4129    0.9549   -0.2565   -0.8665   -1.9147   -3.1675
    0.1259    0.4139    0.9539   -0.2630   -0.8870   -1.8984   -3.1318
    0.1207    0.4145    0.9546   -0.2622   -0.8803   -1.9366   -3.2159
    0.1211    0.4123    0.9546   -0.2569   -0.8697   -1.9213   -3.2084
    0.1463    0.4211    0.9547   -0.2515   -0.8567   -1.8791   -3.1399
    0.1419    0.4236    0.9540   -0.2728   -0.9060   -1.9459   -3.2244
    0.1325    0.4168    0.9541   -0.2708   -0.9014   -1.9389   -3.2069
    0.1276    0.4178    0.9546   -0.2645   -0.8849   -1.9363   -3.1980
    0.1228    0.4214    0.9547   -0.2662   -0.8877   -1.9428   -3.2103
    0.1187    0.4086    0.9548   -0.2461   -0.8464   -1.8828   -3.1523
    0.1466    0.4236    0.9547   -0.2756   -0.9053   -1.9687   -3.2233
    0.1378    0.4123    0.9547   -0.2553   -0.8648   -1.9134   -3.2125
    0.1330    0.4169    0.9540   -0.2650   -0.8901   -1.9114   -3.1803
    0.1298    0.4107    0.9555   -0.2454   -0.8398   -1.9456   -3.2580
    0.1269    0.4115    0.9544   -0.2616   -0.8803   -1.9235   -3.2018
    0.1206    0.4085    0.9547   -0.2423   -0.8396   -1.8744   -3.1532
    0.1456    0.4123    0.9545   -0.2555   -0.8662   -1.8485   -3.0510
    0.1390    0.4152    0.9549   -0.2514   -0.8558   -1.8779   -3.1004
    0.1315    0.4111    0.9540   -0.2549   -0.8697   -1.8748   -3.1182
    0.1299    0.4092    0.9541   -0.2537   -0.8671   -1.8757   -3.1360
    0.1248    0.4059    0.9542   -0.2548   -0.8686   -1.8872   -3.1462
    0.1248    0.4097    0.9543   -0.2480   -0.8539   -1.8524   -3.0884
    0.1480    0.4103    0.9531   -0.2561   -0.8787   -1.9507   -3.3892
    0.1415    0.4192    0.9550   -0.2515   -0.8556   -2.0099   -3.4450
    0.1361    0.4101    0.9541   -0.2552   -0.8700   -1.8565   -3.1073
    0.1318    0.4089    0.9538   -0.2575   -0.8768   -1.8274   -3.0646
    0.1265    0.4114    0.9547   -0.2604   -0.8760   -1.9404   -3.2561
    0.1291    0.4102    0.9544   -0.2505   -0.8580   -1.8993   -3.2262
    0.1529    0.4309    0.9556   -0.2578   -0.8629   -1.8795   -3.0901
    0.1393    0.4114    0.9543   -0.2074   -0.7722   -1.7765   -3.0534
    0.1409    0.4295    0.9541   -0.2956   -0.9503   -1.9695   -3.1872
    0.1369    0.4157    0.9559   -0.2336   -0.8126   -1.8488   -3.0843
    0.1360    0.4206    0.9542   -0.2663   -0.8910   -1.9055   -3.1496
    0.1291    0.4061    0.9555   -0.2121   -0.7730   -1.8029   -3.1013

];

s.wifi(:,:,1) = [0.7466    0.9399    0.8300   -1.5686   -4.4209   -5.8931   -8.0444
    0.7104    0.9422    0.8265   -1.5730   -4.4572   -5.9002   -8.0817
    0.7130    0.9587    0.8251   -1.5735   -4.4696   -5.8922   -8.1356
    0.6943    0.9574    0.8256   -1.5713   -4.4611   -5.8686   -8.0795
    0.6731    0.9579    0.8251   -1.5710   -4.4648   -5.8634   -8.1075
    0.6641    0.9614    0.8258   -1.5676   -4.4530   -5.8317   -8.0754
    0.7447    0.9470    0.8239   -1.5742   -4.4768   -5.8824   -8.1376
    0.7125    0.9513    0.8255   -1.5700   -4.4583   -5.8563   -8.0935
    0.6545    0.9435    0.8268   -1.5651   -4.4401   -5.8311   -8.0898
    0.6114    0.9500    0.8267   -1.5647   -4.4402   -5.8194   -8.0678
    0.5702    0.9407    0.8262   -1.5697   -4.4538   -5.8678   -8.1100
    0.6154    0.9564    0.8255   -1.5693   -4.4589   -5.8476   -8.0880
    0.7612    0.9450    0.8249   -1.5701   -4.4613   -5.8502   -8.1030
    0.6995    0.9480    0.8232   -1.5741   -4.4841   -5.8825   -8.1710
    0.6723    0.9555    0.8248   -1.5706   -4.4657   -5.8536   -8.1010
    0.6234    0.9459    0.8257   -1.5668   -4.4517   -5.8287   -8.0697
    0.5854    0.9422    0.8239   -1.5716   -4.4748   -5.8593   -8.1224
    0.5799    0.9493    0.8238   -1.5705   -4.4733   -5.8400   -8.0913
    0.7646    0.9474    0.8219   -1.5752   -4.4936   -5.8654   -8.1149
    0.6893    0.9538    0.8254   -1.5702   -4.4601   -5.8617   -8.1187
    0.6470    0.9504    0.8265   -1.5658   -4.4432   -5.8298   -8.0724
    0.6291    0.9546    0.8246   -1.5706   -4.4677   -5.8508   -8.1082
    0.6275    0.9555    0.8244   -1.5697   -4.4671   -5.8396   -8.0874
    0.5748    0.9489    0.8256   -1.5671   -4.4536   -5.8340   -8.0952
    0.7778    0.9547    0.8248   -1.5724   -4.4664   -5.8626   -8.0745
    0.6954    0.9456    0.8256   -1.5662   -4.4507   -5.8207   -8.0621
    0.6402    0.9408    0.8250   -1.5682   -4.4591   -5.8344   -8.0822
    0.6387    0.9467    0.8233   -1.5681   -4.4721   -5.8156   -8.1057
    0.6260    0.9519    0.8245   -1.5692   -4.4659   -5.8400   -8.0980
    0.5718    0.9459    0.8234   -1.5712   -4.4781   -5.8449   -8.1074
    0.7643    0.9505    0.8243   -1.5688   -4.4630   -5.8285   -8.0761
    0.7088    0.9453    0.8235   -1.5701   -4.4733   -5.8347   -8.0908
    0.6519    0.9480    0.8222   -1.5696   -4.4830   -5.8155   -8.0953
    0.6082    0.9522    0.8219   -1.5725   -4.4911   -5.8375   -8.0987
    0.5813    0.9497    0.8230   -1.5716   -4.4815   -5.8439   -8.1091
    0.5736    0.9520    0.8225   -1.5720   -4.4858   -5.8422   -8.1087
    0.7446    0.9483    0.8252   -1.5701   -4.4594   -5.8704   -8.1462
    0.7001    0.9563    0.8240   -1.5724   -4.4744   -5.8643   -8.1274
    0.6569    0.9524    0.8254   -1.5691   -4.4583   -5.8503   -8.1062
    0.5960    0.9523    0.8258   -1.5646   -4.4466   -5.8101   -8.0738
    0.5826    0.9495    0.8259   -1.5684   -4.4538   -5.8468   -8.0927
    0.5834    0.9516    0.8252   -1.5690   -4.4601   -5.8428   -8.0924
    0.7372    0.9300    0.8217   -1.5689   -4.4826   -5.8043   -8.0674
    0.7209    0.9538    0.8242   -1.5716   -4.4712   -5.8626   -8.1320
    0.7014    0.9507    0.8244   -1.5689   -4.4649   -5.8355   -8.1030
    0.6814    0.9606    0.8236   -1.5691   -4.4715   -5.8247   -8.0928
    0.6714    0.9495    0.8238   -1.5718   -4.4756   -5.8626   -8.1338
    0.6473    0.9532    0.8260   -1.5661   -4.4484   -5.8230   -8.0679
    0.7559    0.9488    0.8323   -1.5605   -4.3879   -5.8326   -7.9718
    0.7281    0.9645    0.8234   -1.5695   -4.4731   -5.8242   -8.1006
    0.6550    0.9516    0.8271   -1.5691   -4.4457   -5.8716   -8.1068
    0.6391    0.9604    0.8217   -1.5759   -4.4996   -5.8778   -8.1827
    0.6071    0.9490    0.8242   -1.5708   -4.4711   -5.8500   -8.1072
    0.5872    0.9566    0.8254   -1.5686   -4.4577   -5.8407   -8.0804
    0.7675    0.9564    0.8273   -1.5680   -4.4396   -5.8558   -8.0822
    0.7106    0.9561    0.8210   -1.5747   -4.5011   -5.8647   -8.1854
    0.6400    0.9507    0.8246   -1.5693   -4.4646   -5.8382   -8.0917
    0.6055    0.9489    0.8229   -1.5705   -4.4795   -5.8299   -8.0905
    0.5932    0.9525    0.8251   -1.5686   -4.4603   -5.8344   -8.0773
    0.5934    0.9558    0.8234   -1.5703   -4.4756   -5.8342   -8.0961
    0.7354    0.9430    0.8217   -1.5728   -4.4899   -5.8461   -8.1326
    0.6837    0.9428    0.8236   -1.5730   -4.4784   -5.8702   -8.1441
    0.6448    0.9434    0.8250   -1.5677   -4.4580   -5.8259   -8.0730
    0.6304    0.9436    0.8273   -1.5647   -4.4358   -5.8270   -8.0627
    0.6077    0.9486    0.8254   -1.5687   -4.4580   -5.8364   -8.0603
    0.5824    0.9473    0.8259   -1.5700   -4.4567   -5.8663   -8.1146
    0.7549    0.9374    0.8248   -1.5746   -4.4710   -5.9101   -8.1941
    0.6856    0.9430    0.8247   -1.5681   -4.4607   -5.8399   -8.1245
    0.6551    0.9479    0.8246   -1.5692   -4.4644   -5.8350   -8.0776
    0.6461    0.9497    0.8261   -1.5722   -4.4599   -5.8941   -8.1397
    0.6106    0.9462    0.8250   -1.5713   -4.4664   -5.8656   -8.1091
    0.5774    0.9457    0.8258   -1.5705   -4.4589   -5.8703   -8.1182
    0.7452    0.9472    0.8277   -1.5656   -4.4317   -5.8441   -8.0829
    0.7333    0.9603    0.8245   -1.5736   -4.4731   -5.8833   -8.1242
    0.6578    0.9568    0.8247   -1.5723   -4.4700   -5.8724   -8.1188
    0.5951    0.9481    0.8251   -1.5691   -4.4609   -5.8469   -8.0950
    0.5883    0.9551    0.8255   -1.5709   -4.4618   -5.8673   -8.1074
    0.5803    0.9572    0.8257   -1.5703   -4.4594   -5.8622   -8.0981
    0.7463    0.9506    0.8263   -1.5655   -4.4415   -5.8299   -8.1008
    0.7243    0.9595    0.8254   -1.5666   -4.4526   -5.8153   -8.0558
    0.6712    0.9612    0.8240   -1.5732   -4.4770   -5.8771   -8.1557
    0.6412    0.9603    0.8219   -1.5762   -4.4982   -5.8895   -8.2095
    0.5877    0.9527    0.8236   -1.5729   -4.4796   -5.8679   -8.1384
    0.5733    0.9539    0.8241   -1.5720   -4.4743   -5.8606   -8.1190
    0.7449    0.9355    0.8267   -1.5673   -4.4425   -5.8356   -8.0274
    0.7093    0.9424    0.8249   -1.5720   -4.4671   -5.8755   -8.1264
    0.6916    0.9430    0.8269   -1.5699   -4.4488   -5.8730   -8.0708
    0.6823    0.9541    0.8251   -1.5704   -4.4632   -5.8617   -8.1110
    0.6484    0.9465    0.8252   -1.5695   -4.4611   -5.8509   -8.0955
    0.6373    0.9462    0.8270   -1.5667   -4.4420   -5.8398   -8.0611

];
s.wifi(:,:,2) = [0.3852    0.8183    0.1294   -1.5699  -12.6893   -9.0428   -8.8619
    0.3752    0.8151    0.1073   -1.5669  -13.4384   -9.0201   -8.8261
    0.3724    0.8167    0.1219   -1.5686  -12.9275   -9.0189   -8.8472
    0.3717    0.8185    0.1131   -1.5710  -13.2343   -9.0361   -8.8855
    0.3643    0.8189    0.1363   -1.5710  -12.4826   -9.0319   -8.8892
    0.3650    0.8171    0.1135   -1.5683  -13.2153   -9.0250   -8.8609
    0.3847    0.7987    0.1277   -1.5692  -12.7386   -9.0242   -8.8974
    0.3670    0.8005    0.1002   -1.5681  -13.7132   -9.0172   -8.8696
    0.3505    0.8026    0.1033   -1.5697  -13.5955   -9.0429   -8.8920
    0.3498    0.8042    0.1263   -1.5690  -12.7860   -9.0408   -8.8710
    0.3312    0.8030    0.1016   -1.5690  -13.6627   -9.0477   -8.8834
    0.3171    0.8043    0.1087   -1.5679  -13.3878   -9.0285   -8.8735
    0.3722    0.8008    0.1514   -1.5704  -12.0538   -9.0425   -8.9047
    0.3649    0.8020    0.1131   -1.5699  -13.2324   -9.0144   -8.8785
    0.3502    0.8048    0.1238   -1.5707  -12.8676   -9.0424   -8.8822
    0.3370    0.8060    0.1244   -1.5698  -12.8471   -9.0267   -8.8704
    0.3228    0.8048    0.1196   -1.5704  -13.0064   -9.0373   -8.8823
    0.3273    0.8072    0.1205   -1.5696  -12.9744   -9.0274   -8.8756
    0.3744    0.7924    0.0923   -1.5706  -14.0474   -9.0363   -8.8827
    0.3513    0.7948    0.0984   -1.5683  -13.7863   -9.0111   -8.8658
    0.3442    0.7998    0.0816   -1.5709  -14.5473   -9.0402   -8.8810
    0.3292    0.8008    0.0910   -1.5698  -14.1048   -9.0842   -8.8958
    0.3248    0.8012    0.0961   -1.5700  -13.8855   -9.0342   -8.8839
    0.3091    0.8001    0.1058   -1.5696  -13.5010   -9.0332   -8.8873
    0.3750    0.7930    0.1142   -1.5690  -13.1915   -9.0361   -8.8742
    0.3519    0.7973    0.1020   -1.5694  -13.6473   -9.0397   -8.8777
    0.3416    0.7988    0.0936   -1.5722  -13.9960   -9.0532   -8.8832
    0.3372    0.8017    0.0948   -1.5717  -13.9442   -8.9864   -8.8750
    0.3303    0.8024    0.0981   -1.5703  -13.8063   -9.0328   -8.8770
    0.3113    0.8027    0.1021   -1.5705  -13.6455   -9.0533   -8.8851
    0.3732    0.8003    0.0849   -1.5719  -14.3866   -9.1222   -8.9228
    0.3553    0.8012    0.0867   -1.5695  -14.2964   -9.0813   -8.8909
    0.3462    0.8052    0.0965   -1.5707  -13.8693   -9.0366   -8.8836
    0.3459    0.8061    0.0780   -1.5702  -14.7228   -9.0894   -8.8887
    0.3288    0.8041    0.0995   -1.5699  -13.7473   -9.0585   -8.8844
    0.3272    0.8063    0.0987   -1.5697  -13.7799   -9.0374   -8.8825
    0.3796    0.7990    0.1075   -1.5713  -13.4381   -9.0077   -8.8985
    0.3663    0.8046    0.0981   -1.5681  -13.7989   -9.0599   -8.8645
    0.3520    0.8044    0.1083   -1.5703  -13.4058   -9.0530   -8.8753
    0.3466    0.8028    0.0888   -1.5690  -14.2018   -9.0108   -8.8620
    0.3322    0.8033    0.1130   -1.5698  -13.2345   -9.0628   -8.8792
    0.3142    0.8034    0.1028   -1.5682  -13.6130   -9.0399   -8.8664
    0.3786    0.8018    0.1184   -1.5714  -13.0488   -9.1231   -8.9139
    0.3764    0.8039    0.0778   -1.5697  -14.7348   -8.9520   -8.8607
    0.3703    0.8031    0.0578   -1.5692  -15.9237   -8.9955   -8.8622
    0.3664    0.8062    0.0958   -1.5700  -13.8977   -9.0594   -8.8995
    0.3627    0.8054    0.0741   -1.5702  -14.9289   -9.0700   -8.8806
    0.3597    0.8029    0.0396   -1.5682  -17.4371   -8.9996   -8.8599
    0.3771    0.7953    0.0870   -1.5693  -14.2827   -9.0880   -8.8555
    0.3677    0.8012    0.1079   -1.5710  -13.4242   -9.0270   -8.8882
    0.3507    0.8022    0.0920   -1.5712  -14.0661   -9.0359   -8.8882
    0.3440    0.8013    0.1088   -1.5690  -13.3847   -9.0228   -8.8724
    0.3267    0.8030    0.0930   -1.5703  -14.0190   -9.0308   -8.8823
    0.3242    0.8043    0.0948   -1.5688  -13.9378   -9.0350   -8.8689
    0.3696    0.7982    0.1295   -1.5694  -12.6829   -9.0431   -8.8687
    0.3569    0.8009    0.1057   -1.5716  -13.5083   -9.0437   -8.8871
    0.3479    0.8043    0.1100   -1.5708  -13.3448   -9.0418   -8.8863
    0.3424    0.8066    0.1132   -1.5705  -13.2292   -9.0523   -8.8788
    0.3294    0.8033    0.0969   -1.5698  -13.8537   -9.0418   -8.8807
    0.3281    0.8035    0.0993   -1.5688  -13.7526   -9.0323   -8.8696
    0.3668    0.7941    0.0896   -1.5704  -14.1686   -8.9845   -8.8810
    0.3580    0.7986    0.1003   -1.5698  -13.7121   -9.0523   -8.8773
    0.3505    0.8014    0.1107   -1.5722  -13.3202   -9.0212   -8.8889
    0.3442    0.8037    0.1148   -1.5714  -13.1730   -8.9900   -8.8769
    0.3308    0.8020    0.1006   -1.5698  -13.7026   -9.0329   -8.8808
    0.3139    0.8019    0.1100   -1.5689  -13.3424   -9.0329   -8.8771
    0.3640    0.7864    0.1084   -1.5685  -13.3977   -9.0335   -8.8769
    0.3569    0.7953    0.1053   -1.5704  -13.5177   -9.0524   -8.8909
    0.3470    0.7994    0.1055   -1.5695  -13.5097   -9.0386   -8.8708
    0.3439    0.7994    0.1040   -1.5689  -13.5660   -9.0835   -8.8876
    0.3331    0.8006    0.1057   -1.5693  -13.5016   -9.0381   -8.8744
    0.3135    0.8020    0.1155   -1.5690  -13.1450   -9.0390   -8.8816
    0.3790    0.8005    0.1048   -1.5712  -13.5396   -9.0308   -8.8841
    0.3610    0.8013    0.0864   -1.5698  -14.3148   -9.0255   -8.8739
    0.3573    0.8034    0.1008   -1.5699  -13.6924   -9.0582   -8.8896
    0.3457    0.8048    0.1043   -1.5681  -13.5553   -9.0257   -8.8644
    0.3342    0.8048    0.0929   -1.5697  -14.0227   -9.0426   -8.8744
    0.3269    0.8050    0.0970   -1.5684  -13.8466   -9.0212   -8.8696
    0.3831    0.8046    0.0921   -1.5696  -14.0556   -9.0399   -8.8806
    0.3729    0.8080    0.1067   -1.5702  -13.4651   -9.0468   -8.8666
    0.3591    0.8048    0.1127   -1.5695  -13.2461   -9.0280   -8.8769
    0.3546    0.8079    0.0958   -1.5718  -13.9040   -9.0689   -8.9027
    0.3385    0.8089    0.1105   -1.5708  -13.3271   -9.0433   -8.8939
    0.3180    0.8078    0.1058   -1.5690  -13.4964   -9.0407   -8.8745
    0.3853    0.8007    0.1020   -1.5669  -13.6384   -8.8254   -8.8852
    0.3732    0.8008    0.1489   -1.5693  -12.1207   -8.9920   -8.8627
    0.3685    0.8047    0.0758   -1.5691  -14.8354   -9.0353   -8.8702
    0.3666    0.8061    0.1104   -1.5695  -13.3276   -9.0177   -8.8839
    0.3606    0.8055    0.0604   -1.5702  -15.7496   -9.0316   -8.8762
    0.3599    0.8025    0.0793   -1.5687  -14.6570   -9.0265   -8.8730

];
s.wifi(:,:,3) = [0.2578    0.6835    0.1291   -1.3854  -12.3294  -10.3543  -13.2860
    0.5955    0.6849    0.2310   -1.3872   -9.9681  -10.4183  -12.6829
    0.5899    0.6828    0.2272   -1.3858  -10.0335  -10.3411  -12.5394
    0.5834    0.6796    0.2063   -1.3816  -10.4205  -10.3083  -12.2179
    0.5812    0.6805    0.2395   -1.3826   -9.8097  -10.3822  -12.3224
    0.5742    0.6736    0.2107   -1.3709  -10.3117  -10.2749  -12.0932
    0.1376    0.6725    0.4709   -1.3916   -6.9458   -7.0471  -10.3159
    0.3494    0.6694    0.2124   -1.3859  -10.3097  -10.4267  -12.4218
    0.5639    0.6691    0.2054   -1.3860  -10.4476  -10.3872  -12.3328
    0.3288    0.6705    0.1929   -1.3873  -10.7063  -10.3752  -12.3246
    0.5269    0.6680    0.2010   -1.3839  -10.5314  -10.2845  -12.2899
    0.4965    0.6641    0.1818   -1.3749  -10.9227  -10.2623  -12.1686
    0.1791    0.6679    0.2171   -1.3894  -10.2262  -10.5492  -12.1906
    0.1710    0.6692    0.2127   -1.3850  -10.3014  -10.7017  -12.1621
    0.1235    0.6708    0.2130   -1.3862  -10.2977  -10.3105  -12.1061
    0.5452    0.6705    0.2100   -1.3867  -10.3575  -10.6676  -11.9026
    0.5289    0.6719    0.2136   -1.3861  -10.2866  -10.3738  -12.2396
    0.5259    0.6704    0.2021   -1.3807  -10.5021  -10.4173  -12.1357
    0.1384    0.6648    0.4477   -1.3895   -7.1682   -7.0786  -10.3830
    0.1702    0.6631    0.2154   -1.3803  -10.2401  -10.2524  -12.2552
    0.1619    0.6645    0.2092   -1.3811  -10.3611  -10.3141  -12.3192
    0.5393    0.6678    0.2009   -1.3812  -10.5272  -10.5281  -12.4868
    0.5319    0.6670    0.2017   -1.3815  -10.5119  -10.3936  -12.2592
    0.4992    0.6637    0.2058   -1.3778  -10.4230  -10.3870  -12.2399
    0.1781    0.6765    0.2735   -1.4134   -9.3217   -5.8753   -9.6913
    0.3362    0.6641    0.2065   -1.3828  -10.4184  -10.5395  -12.2384
    0.3289    0.6673    0.2079   -1.3869  -10.3988  -10.3113  -12.3883
    0.5450    0.6698    0.2070   -1.3854  -10.4129  -10.4195  -12.1662
    0.5259    0.6683    0.2078   -1.3855  -10.3983  -10.3889  -12.3773
    0.5040    0.6651    0.2083   -1.3791  -10.3762  -10.3328  -12.2820
    0.1808    0.6661    0.2164   -1.3844  -10.2287  -10.4861  -11.8441
    0.1690    0.6687    0.2161   -1.3828  -10.2332  -10.2813  -12.2169
    0.2359    0.6710    0.2109   -1.3878  -10.3413  -10.3294  -12.2672
    0.5463    0.6718    0.2097   -1.3841  -10.3589  -10.3405  -12.3722
    0.5295    0.6713    0.2127   -1.3857  -10.3036  -10.2051  -12.2302
    0.5200    0.6681    0.2066   -1.3788  -10.4085  -10.2854  -12.2486
    0.1342    0.6768    0.4648   -1.3925   -7.0063   -7.0405  -10.4045
    0.2429    0.6710    0.2310   -1.3829   -9.9594  -10.4325  -12.2751
    0.5654    0.6692    0.1990   -1.3847  -10.5728  -10.4501  -12.2905
    0.5607    0.6717    0.2066   -1.3897  -10.4308  -10.4540  -12.2813
    0.5213    0.6690    0.2052   -1.3842  -10.4466  -10.4270  -12.3743
    0.4952    0.6656    0.1901   -1.3753  -10.7406  -10.3694  -12.2685
    0.1094    0.6789    0.1869   -1.3849  -10.8293  -10.4364  -12.0857
    0.3593    0.6673    0.2314   -1.3788   -9.9436  -10.0832  -11.9528
    0.5671    0.6659    0.2070   -1.3835  -10.4093  -10.5960  -12.4478
    0.5774    0.6745    0.2059   -1.3845  -10.4348  -10.3277  -12.0398
    0.5649    0.6709    0.1941   -1.3830  -10.6709  -10.5677  -12.5255
    0.5503    0.6605    0.1914   -1.3706  -10.7043  -10.2714  -12.1313
    0.1301    0.6645    0.4541   -1.3921   -7.1094   -7.0515  -10.5757
    0.1040    0.6679    0.2132   -1.3868  -10.2962  -10.1683  -12.2222
    0.3363    0.6689    0.2111   -1.3859  -10.3343  -10.4278  -12.3170
    0.5505    0.6689    0.2038   -1.3854  -10.4777  -10.2988  -12.3685
    0.5220    0.6685    0.2112   -1.3837  -10.3277  -10.3880  -12.2780
    0.4956    0.6643    0.1916   -1.3764  -10.7120  -10.3333  -12.1953
    0.1757    0.6639    0.2156   -1.3793  -10.2351  -10.2976  -12.1733
    0.1687    0.6683    0.2109   -1.3818  -10.3297  -10.2804  -12.2443
    0.5586    0.6706    0.2063   -1.3875  -10.4325  -10.3268  -12.0585
    0.5469    0.6690    0.1975   -1.3786  -10.5926  -10.0975  -12.3913
    0.5334    0.6695    0.2085   -1.3826  -10.3777  -10.2623  -12.1820
    0.5254    0.6666    0.2035   -1.3750  -10.4632  -10.2901  -12.2289
    0.1719    0.6710    0.2632   -1.4106   -9.4749   -5.9124   -9.7054
    0.1696    0.6617    0.2007   -1.3799  -10.5293  -10.2343  -12.1902
    0.2305    0.6666    0.1974   -1.3871  -10.6111  -10.4341  -12.4114
    0.5416    0.6666    0.2087   -1.3821  -10.3738  -10.1905  -11.9970
    0.5219    0.6665    0.2017   -1.3821  -10.5123  -10.4126  -12.1692
    0.4963    0.6638    0.2047   -1.3780  -10.4455  -10.3504  -12.2243
    0.1749    0.6746    0.2594   -1.4075   -9.5293   -5.9108   -9.6206
    0.1014    0.6639    0.2106   -1.3842  -10.3406  -10.3553  -12.3134
    0.1626    0.6648    0.2077   -1.3807  -10.3903  -10.4508  -12.2975
    0.5420    0.6659    0.2096   -1.3773  -10.3465  -10.2829  -12.2372
    0.2179    0.6668    0.2017   -1.3817  -10.5113  -10.3979  -12.3842
    0.5046    0.6634    0.2009   -1.3781  -10.5211  -10.3530  -12.1868
    0.1784    0.6653    0.2177   -1.3855  -10.2074  -10.4391  -11.9646
    0.1726    0.6663    0.2106   -1.3832  -10.3393  -10.4528  -12.2001
    0.3274    0.6691    0.2043   -1.3841  -10.4641  -10.1898  -12.2039
    0.5479    0.6707    0.2002   -1.3830  -10.5455  -10.3749  -12.1629
    0.5277    0.6700    0.2079   -1.3846  -10.3953  -10.3742  -12.2982
    0.5155    0.6684    0.2075   -1.3788  -10.3908  -10.3176  -12.2511
    0.1808    0.6767    0.2158   -1.3961  -10.2644  -10.0784  -12.1413
    0.2470    0.6740    0.2023   -1.3867  -10.5095  -10.5650  -12.6952
    0.5710    0.6727    0.2078   -1.3853  -10.3975  -10.5032  -12.3242
    0.5642    0.6738    0.2158   -1.3866  -10.2459  -10.3167  -12.6491
    0.5382    0.6745    0.2082   -1.3870  -10.3925  -10.4318  -12.4029
    0.5022    0.6694    0.1961   -1.3777  -10.6184  -10.3971  -12.4220
    0.1356    0.6731    0.4494   -1.3926   -7.1576   -6.9758  -10.3655
    0.2428    0.6701    0.2058   -1.3859  -10.4382  -10.2155  -12.4173
    0.5697    0.6671    0.1978   -1.3836  -10.5957  -10.1553  -12.1616
    0.5760    0.6734    0.2056   -1.3840  -10.4395  -10.3590  -12.1306
    0.5678    0.6691    0.2096   -1.3826  -10.3581  -10.2905  -12.1439
    0.5538    0.6611    0.1845   -1.3714  -10.8552  -10.1833  -12.0752

];

s.lixo_extra(:,:,1) = [0.7367    0.7027    0.3175   -1.3203   -8.5119  -11.1493  -11.2958
    0.7153    0.6975    0.3271   -1.3063   -8.3585  -11.6423  -11.3690
    0.6997    0.6997    0.3291   -1.3120   -8.3450  -11.7458  -11.2136
    0.7008    0.7011    0.3387   -1.3117   -8.2214  -11.7812  -11.3541
    0.6836    0.6981    0.3422   -1.3053   -8.1659  -11.8020  -11.1055
    0.6438    0.6928    0.3483   -1.2975   -8.0746  -11.6270  -10.9878
    0.7345    0.7023    0.3330   -1.3241   -8.3181  -11.6283  -11.4435
    0.6977    0.6973    0.3405   -1.3111   -8.1977  -11.7393  -11.4321
    0.6539    0.6986    0.3416   -1.3142   -8.1904  -11.6876  -11.4062
    0.6196    0.6963    0.3400   -1.3111   -8.2050  -11.8090  -11.4908
    0.5899    0.6979    0.3384   -1.3149   -8.2318  -11.7521  -11.4974
    0.5691    0.6940    0.3427   -1.3078   -8.1643  -11.4828  -11.3785
    0.7260    0.6942    0.3474   -1.3074   -8.1048  -12.0042  -11.2964
    0.7072    0.7013    0.3459   -1.3164   -8.1412  -11.3766  -11.3918
    0.6546    0.6971    0.3481   -1.3110   -8.1047  -11.2011  -11.3267
    0.6056    0.6957    0.3373   -1.3075   -8.2313  -11.5930  -11.5851
    0.5942    0.6974    0.3445   -1.3138   -8.1547  -11.4511  -11.4025
    0.5987    0.6903    0.3530   -1.2990   -8.0210  -11.4045  -11.0021
    0.7359    0.7012    0.3137   -1.3162   -8.5549  -11.9724  -11.6627
    0.6953    0.6960    0.3405   -1.3087   -8.1938  -11.8723  -11.5694
    0.6457    0.6960    0.3420   -1.3118   -8.1816  -12.0967  -11.3055
    0.6339    0.6987    0.3388   -1.3142   -8.2253  -11.7861  -11.4760
    0.6115    0.6959    0.3411   -1.3120   -8.1921  -11.9028  -11.3685
    0.5871    0.6918    0.3361   -1.3037   -8.2392  -11.6694  -11.3715
    0.7196    0.6997    0.3315   -1.3212   -8.3320  -12.0784  -11.0047
    0.6980    0.6987    0.3401   -1.3123   -8.2058  -11.7619  -11.2657
    0.6555    0.6967    0.3391   -1.3119   -8.2168  -11.9573  -11.4111
    0.6387    0.6972    0.3387   -1.3119   -8.2226  -12.1111  -11.3843
    0.6160    0.6968    0.3390   -1.3134   -8.2212  -11.8231  -11.4214
    0.5966    0.6900    0.3497   -1.3006   -8.0639  -11.5010  -11.0730
    0.7367    0.7010    0.3287   -1.3168   -8.3584  -12.4753  -11.3796
    0.7050    0.6970    0.3318   -1.3091   -8.3044  -11.7597  -11.2207
    0.6497    0.6955    0.3373   -1.3062   -8.2291  -11.9121  -11.1567
    0.6075    0.6937    0.3379   -1.3025   -8.2135  -11.7961  -11.1943
    0.5836    0.6931    0.3401   -1.3054   -8.1920  -11.7590  -11.1909
    0.5804    0.6933    0.3404   -1.3013   -8.1803  -11.6723  -11.2358
    0.7298    0.6977    0.3526   -1.3165   -8.0598  -12.6371  -11.9445
    0.7121    0.7012    0.3372   -1.3133   -8.2443  -11.7921  -11.5898
    0.6448    0.7009    0.3343   -1.3178   -8.2891  -11.8328  -11.7370
    0.6060    0.6931    0.3367   -1.3019   -8.2275  -12.1233  -11.4781
    0.5849    0.6963    0.3343   -1.3114   -8.2769  -11.9092  -11.4158
    0.5472    0.6931    0.3397   -1.3053   -8.1971  -11.9147  -11.4335
    0.7296    0.6962    0.3457   -1.3115   -8.1345  -12.0209  -11.3439
    0.7200    0.7016    0.3387   -1.3143   -8.2277  -11.4665  -11.6332
    0.7133    0.7005    0.3373   -1.3114   -8.2390  -11.5923  -11.4539
    0.7015    0.7010    0.3371   -1.3105   -8.2393  -11.6763  -11.4371
    0.6801    0.7022    0.3380   -1.3161   -8.2392  -11.7053  -11.5336
    0.6456    0.6946    0.3494   -1.3018   -8.0699  -11.3616  -11.3550
    0.7267    0.6964    0.3307   -1.3078   -8.3153  -12.0828  -11.4443
    0.7058    0.6979    0.3360   -1.3074   -8.2470  -11.6074  -11.3632
    0.6576    0.6966    0.3391   -1.3096   -8.2132  -11.8259  -11.5453
    0.6217    0.6963    0.3339   -1.3076   -8.2743  -11.5296  -11.5029
    0.6175    0.6972    0.3374   -1.3094   -8.2332  -11.8388  -11.4169
    0.5851    0.6919    0.3442   -1.2999   -8.1301  -11.8228  -11.1423
    0.7327    0.6984    0.3364   -1.3121   -8.2506  -11.4820  -11.6031
    0.7040    0.6991    0.3398   -1.3134   -8.2115  -11.6031  -11.2535
    0.6612    0.6953    0.3499   -1.3071   -8.0750  -11.9090  -11.2670
    0.6221    0.6926    0.3488   -1.3019   -8.0773  -11.3385  -11.5164
    0.5973    0.6949    0.3454   -1.3105   -8.1362  -11.7028  -11.2739
    0.5897    0.6886    0.3464   -1.2951   -8.0934  -11.3912  -11.1359
    0.7256    0.6981    0.3385   -1.3198   -8.2399  -11.5426  -11.1855
    0.6915    0.6950    0.3463   -1.3080   -8.1203  -11.8501  -11.5200
    0.6619    0.6959    0.3415   -1.3103   -8.1842  -11.8876  -11.3592
    0.6481    0.6966    0.3381   -1.3094   -8.2247  -11.6737  -11.3647
    0.6291    0.6987    0.3390   -1.3133   -8.2212  -11.7497  -11.4643
    0.5952    0.6909    0.3410   -1.2986   -8.1672  -11.6990  -11.1489
    0.7102    0.6964    0.3332   -1.3125   -8.2925  -11.6563  -11.9223
    0.6980    0.6978    0.3361   -1.3107   -8.2523  -11.5654  -11.4356
    0.6698    0.6977    0.3380   -1.3124   -8.2315  -11.7352  -11.6392
    0.6469    0.6998    0.3366   -1.3163   -8.2577  -11.4548  -11.3743
    0.6208    0.6971    0.3377   -1.3109   -8.2329  -11.8221  -11.5182
    0.6015    0.6957    0.3375   -1.3090   -8.2316  -11.7135  -11.4606
    0.7313    0.6960    0.3387   -1.3092   -8.2165  -12.0914  -11.5167
    0.7064    0.6987    0.3277   -1.3135   -8.3657  -11.8154  -11.3519
    0.6334    0.6977    0.3410   -1.3130   -8.1953  -11.5788  -11.2241
    0.5930    0.6946    0.3434   -1.3074   -8.1547  -11.5687  -11.2227
    0.5764    0.6958    0.3416   -1.3101   -8.1820  -11.6552  -11.2501
    0.5797    0.6940    0.3428   -1.3053   -8.1586  -11.6963  -11.2673
    0.7338    0.6996    0.3285   -1.3178   -8.3631  -11.9758  -11.5548
    0.7037    0.6991    0.3373   -1.3134   -8.2422  -11.6908  -11.5492
    0.6503    0.6977    0.3326   -1.3130   -8.3016  -11.6204  -11.6610
    0.6101    0.6963    0.3461   -1.3127   -8.1320  -11.4986  -11.2590
    0.5922    0.6981    0.3374   -1.3144   -8.2442  -11.5587  -11.6983
    0.5794    0.6920    0.3519   -1.3033   -8.0426  -11.8409  -11.4100
    0.7277    0.6985    0.3361   -1.3162   -8.2629  -11.8071  -11.6324
    0.7138    0.6990    0.3334   -1.3116   -8.2884  -11.6914  -11.2799
    0.6946    0.6983    0.3433   -1.3102   -8.1615  -12.0978  -11.3269
    0.6815    0.7001    0.3376   -1.3111   -8.2344  -11.6276  -11.4579
    0.6726    0.7005    0.3371   -1.3135   -8.2453  -11.8590  -11.3875
    0.6376    0.6916    0.3509   -1.2948   -8.0376  -11.4721  -11.0841

];
s.lixo_extra(:,:,2) = [0.7566    0.7183    0.2927   -1.3549   -8.9217  -11.0104  -11.5340
    0.7337    0.7120    0.3018   -1.3393   -8.7634  -11.2456  -11.8844
    0.7269    0.7099    0.2988   -1.3282   -8.7829  -10.9784  -12.1894
    0.7041    0.7149    0.2992   -1.3438   -8.8086  -11.3528  -12.0683
    0.7043    0.7148    0.2971   -1.3424   -8.8355  -11.2364  -11.8111
    0.6459    0.6980    0.3406   -1.3161   -8.2066  -11.0790  -11.5352
    0.7595    0.7165    0.3028   -1.3483   -8.7660  -11.0391  -11.2074
    0.7169    0.7095    0.3020   -1.3393   -8.7601  -11.2737  -11.4458
    0.6562    0.7101    0.3048   -1.3418   -8.7273  -11.2555  -11.8742
    0.6178    0.7117    0.2943   -1.3444   -8.8783  -11.5105  -11.9447
    0.5807    0.7067    0.3023   -1.3366   -8.7506  -11.2845  -11.6677
    0.5677    0.7054    0.2987   -1.3337   -8.7957  -11.2545  -11.7991
    0.7498    0.7107    0.3044   -1.3368   -8.7209  -10.7498  -12.3183
    0.7078    0.7058    0.3074   -1.3257   -8.6588  -11.1848  -11.9017
    0.6576    0.7096    0.3087   -1.3375   -8.6641  -11.2321  -11.7411
    0.6404    0.7115    0.3006   -1.3463   -8.7939  -11.0811  -11.6485
    0.5900    0.7070    0.3028   -1.3354   -8.7425  -11.2103  -11.8971
    0.5965    0.7050    0.3125   -1.3326   -8.6036  -11.2836  -11.7722
    0.7546    0.7148    0.2558   -1.3430   -9.4571  -10.8941  -11.4043
    0.6949    0.7089    0.2945   -1.3426   -8.8723  -11.1045  -11.8535
    0.6729    0.7114    0.2976   -1.3429   -8.8294  -11.1163  -11.9646
    0.6547    0.7118    0.2907   -1.3433   -8.9283  -11.7715  -11.8386
    0.6220    0.7081    0.3009   -1.3363   -8.7694  -11.1849  -12.0878
    0.5941    0.7055    0.3063   -1.3345   -8.6923  -11.2153  -12.0057
    0.7540    0.7123    0.2735   -1.3407   -9.1765  -11.1545  -11.4484
    0.7118    0.7123    0.2935   -1.3485   -8.8979  -11.4541  -11.8010
    0.6762    0.7069    0.2941   -1.3373   -8.8668  -11.4112  -11.5859
    0.6443    0.7035    0.3205   -1.3340   -8.5002  -11.4979  -11.8903
    0.6339    0.7078    0.2998   -1.3407   -8.7935  -11.5438  -11.6846
    0.6018    0.6993    0.3271   -1.3255   -8.3976  -11.3100  -11.9348
    0.7288    0.7114    0.3010   -1.3473   -8.7900  -11.0211  -11.5278
    0.7080    0.7096    0.2987   -1.3424   -8.8127  -11.0519  -11.5554
    0.6725    0.7116    0.2865   -1.3414   -8.9850  -11.2914  -11.5154
    0.6275    0.7048    0.3022   -1.3301   -8.7396  -11.4162  -11.9258
    0.6102    0.7081    0.2991   -1.3368   -8.7965  -11.1291  -11.9912
    0.5993    0.7036    0.3011   -1.3275   -8.7502  -10.9404  -11.9792
    0.7605    0.7152    0.2630   -1.3455   -9.3479  -11.3022  -11.7030
    0.7143    0.7126    0.2931   -1.3422   -8.8908  -11.1885  -11.7721
    0.6809    0.7077    0.2945   -1.3329   -8.8531  -11.0883  -12.0002
    0.6442    0.7074    0.3091   -1.3303   -8.6446  -11.6615  -12.2802
    0.6233    0.7062    0.2971   -1.3314   -8.8130  -11.1980  -12.1780
    0.5861    0.6987    0.2965   -1.3124   -8.7840  -11.2019  -11.8415
    0.7457    0.7106    0.2790   -1.3429   -9.0975  -12.4174  -11.4108
    0.7305    0.7099    0.2983   -1.3367   -8.8068  -11.0990  -11.7231
    0.7093    0.7141    0.3008   -1.3446   -8.7878  -11.9274  -12.0547
    0.7076    0.7150    0.2901   -1.3445   -8.9400  -11.1555  -11.7759
    0.6861    0.7140    0.2997   -1.3436   -8.8018  -11.5217  -11.7421
    0.6298    0.6831    0.3548   -1.2836   -7.9680  -11.1987  -10.6103
    0.7511    0.7079    0.3145   -1.3336   -8.5776  -10.9703  -11.3725
    0.7183    0.7109    0.2943   -1.3420   -8.8737  -11.5510  -11.8002
    0.6623    0.7071    0.3046   -1.3386   -8.7231  -11.4541  -11.6845
    0.6353    0.7054    0.3012   -1.3336   -8.7601  -11.0419  -11.5124
    0.6252    0.7065    0.3042   -1.3341   -8.7204  -11.2857  -11.7766
    0.5889    0.7031    0.3050   -1.3276   -8.6964  -11.4130  -11.6582
    0.7509    0.7062    0.2881   -1.3313   -8.9410  -10.6107  -12.6688
    0.7111    0.7109    0.2950   -1.3408   -8.8623  -10.9560  -11.7989
    0.7034    0.7102    0.3042   -1.3360   -8.7236  -11.3581  -11.6581
    0.6553    0.7107    0.2970   -1.3391   -8.8301  -11.2331  -12.1973
    0.6202    0.7096    0.2943   -1.3412   -8.8721  -11.0719  -11.7647
    0.6244    0.7060    0.3020   -1.3310   -8.7443  -11.1973  -11.8501
    0.7503    0.7108    0.2862   -1.3437   -8.9925  -11.1130  -11.0762
    0.7108    0.7099    0.2708   -1.3361   -9.2091  -11.4211  -11.8244
    0.6808    0.7095    0.2946   -1.3380   -8.8624  -11.3516  -11.7844
    0.6446    0.7061    0.2922   -1.3280   -8.8761  -10.9648  -11.9662
    0.6324    0.7066    0.2967   -1.3311   -8.8179  -11.4100  -12.0648
    0.6130    0.7020    0.3072   -1.3215   -8.6539  -11.3339  -12.0246
    0.7544    0.7212    0.3054   -1.3586   -8.7513  -10.7426  -12.0171
    0.7141    0.7105    0.3106   -1.3400   -8.6444  -11.2404  -11.8412
    0.6927    0.7129    0.3023   -1.3441   -8.7664  -11.3805  -11.9275
    0.6720    0.7111    0.2983   -1.3416   -8.8163  -11.0871  -11.4920
    0.6370    0.7108    0.2958   -1.3426   -8.8548  -11.4581  -11.8194
    0.6183    0.7069    0.3086   -1.3352   -8.6616  -11.2699  -11.8487
    0.7315    0.7160    0.2974   -1.3527   -8.8506  -11.2189  -12.1776
    0.7186    0.7143    0.2940   -1.3453   -8.8848  -11.6301  -11.5936
    0.6659    0.7100    0.3000   -1.3366   -8.7835  -11.2140  -11.9976
    0.6254    0.7067    0.3058   -1.3346   -8.6983  -10.6448  -12.5585
    0.5900    0.7073    0.2990   -1.3351   -8.7941  -11.2226  -12.2532
    0.5861    0.7000    0.3064   -1.3200   -8.6621  -11.0893  -12.2754
    0.7597    0.7156    0.2881   -1.3471   -8.9727  -11.3783  -11.5106
    0.7017    0.7059    0.2939   -1.3276   -8.8502  -11.2950  -11.8682
    0.6661    0.7111    0.3032   -1.3437   -8.7526  -11.5302  -11.9787
    0.6424    0.7020    0.3026   -1.3183   -8.7100  -11.1507  -12.0120
    0.6081    0.7106    0.3051   -1.3427   -8.7242  -11.7028  -11.8354
    0.5991    0.7054    0.3004   -1.3318   -8.7686  -11.3989  -11.9834
    0.7513    0.7110    0.2755   -1.3378   -9.1394  -12.2541  -11.5819
    0.7470    0.7171    0.2996   -1.3409   -8.7966  -11.3891  -11.5118
    0.7318    0.7146    0.3014   -1.3385   -8.7670  -11.1776  -11.6441
    0.7113    0.7133    0.3010   -1.3412   -8.7789  -11.4630  -12.1442
    0.6878    0.7154    0.3014   -1.3441   -8.7783  -11.2780  -11.7301
    0.6508    0.7025    0.3113   -1.3232   -8.6012  -11.5561  -11.4605

];

s.wifi_extra = [0.371263102968285	0.792590246991767	0.0994307410705823	-1.57168005832098	-13.7522781845341	-9.10898770069201	-8.87653436078048
0.255999495409209	0.838462554484216	0.397665450179104	-1.58196588002341	-8.07380739850327	-12.8333314553972	-9.45144487665088
0.252464238095319	0.844421921378412	0.400186065435366	-1.58295757449236	-8.04844800692939	-12.6457485919392	-9.46973113165493
0.250903999861570	0.845033233178549	0.398396552145741	-1.58069363914170	-8.06345183270287	-12.8161573966604	-9.45908818195419
0.245273651046360	0.847393958365941	0.398931946437905	-1.58229625024834	-8.06084875954570	-12.7115803442809	-9.45698679367819
0.231680895025141	0.845665188976738	0.395355539986807	-1.58026014940617	-8.09589816208878	-12.6763362893984	-9.43731413563564
0.379417914564741	0.799041533546326	0.120045005778023	-1.57163444083834	-12.9940129290880	-9.02916013676424	-8.88120838820109
0.674575276621984	0.845723421262990	0.399393304135638	-1.58163269250882	-8.05430277824134	-12.6064286108736	-9.46561419623073
0.646358910583907	0.845233780355201	0.397732556827308	-1.58290496221687	-8.07506888136434	-12.9625584896445	-9.46412331778251
0.619326178479330	0.846178372961647	0.400565913960361	-1.58304955844990	-8.04455816140172	-12.7322479286167	-9.46114902733242
0.582045324526313	0.842895408957827	0.399629077456998	-1.58241340879819	-8.05349425643076	-12.7691883646146	-9.45536722484127
0.551939702949762	0.843996175037567	0.396486218788759	-1.58094163733878	-8.08486643465949	-12.7362030021833	-9.44641371400003
0.268276368522536	0.839784263959391	0.393344254818446	-1.58272565210591	-8.12238106454466	-12.5226785699360	-9.44436106149140
0.255182458427822	0.843629600106284	0.399378680981940	-1.58251710643820	-8.05623060991104	-12.8225639348142	-9.46513868467512
0.234066051449594	0.842976999855345	0.396834890578534	-1.58212412475026	-8.08331966360693	-12.8785798612122	-9.45664606895469
0.215647383197498	0.839954699886750	0.394993696706164	-1.58086656481149	-8.10104020099723	-12.6465567786511	-9.43163914572606
0.209302996659827	0.843453865336658	0.396479848464857	-1.58231286622123	-8.08766239265756	-12.8357272124541	-9.46116108601098
0.210980696986479	0.844888848396502	0.397417311427913	-1.58110099251555	-8.07500031954045	-12.8155371524596	-9.44256437897757
0.264602494252114	0.850128369704750	0.389526704890808	-1.58586582095943	-8.17090763416838	-12.0913797926699	-9.45799263734570
0.242143068789591	0.841393931363456	0.397901933822201	-1.58282917136857	-8.07295152636465	-12.7359920410627	-9.47024059931098
0.612413699115557	0.846360208816705	0.397188056991038	-1.58348822627583	-8.08218522969643	-12.3549298315592	-9.48585041104722
0.567539000664412	0.842846080262379	0.392176878270065	-1.58116370256281	-8.13266514228975	-13.4534161485313	-9.45030817774567
0.577378788718938	0.846069766715593	0.396418765601694	-1.58291986538936	-8.08954529248451	-12.6313369811248	-9.47188844987839
0.578081052615803	0.844353276353276	0.400048843908902	-1.57935049750140	-8.04282052527200	-12.8126713313384	-9.43466715366539
0.265955326151667	0.838463979688988	0.393231763913714	-1.58181473244275	-8.12179863087324	-12.8262706911168	-9.43027508706460
0.255314863950120	0.845785057930483	0.398081900138071	-1.58419289653137	-8.07371435345768	-12.7407468961350	-9.45991470545931
0.246629432246768	0.847995343083752	0.399494862163421	-1.58235009239452	-8.05474856523413	-12.7531465934955	-9.44997976029798
0.238340485713592	0.844537432911853	0.400738376411250	-1.58249715640711	-8.04158129140553	-12.7509925477911	-9.45725737893368
0.229542227561138	0.845658762145781	0.399765923315792	-1.58237905657339	-8.05193718005506	-12.7847214409486	-9.44922099877659
0.221190839595128	0.846018668553301	0.401265251320400	-1.58055780390285	-8.03203119207821	-13.0965227260404	-9.44371931950603
0.264101566596943	0.844458639412328	0.401752389528933	-1.58106556910295	-8.02719921874020	-12.6315231188646	-9.45708496966896
0.252534277296648	0.846112815042006	0.398301146013770	-1.58202341920102	-8.06698365762700	-12.8791174324802	-9.45811400808753
0.648133402119571	0.845211339085043	0.399211014618400	-1.58199006990747	-8.05711804820586	-12.8119659509338	-9.44466996218188
0.630997374009576	0.845210972386768	0.398550870277192	-1.58135228978686	-8.06308633392252	-12.5062785349353	-9.44104649368113
0.607284560502598	0.844558938329430	0.398456650541303	-1.58136879196826	-8.06417409115634	-12.6675911447925	-9.43929564682680
0.593343831664946	0.845878381460700	0.396400685519599	-1.58109856666883	-8.08611691227665	-12.6541286223114	-9.42767246357898
0.374020313686125	0.795238095238095	0.0790888840191067	-1.57260862743624	-14.6733373667217	-9.10788284736958	-8.86820817735966
0.250784802336686	0.842644320297952	0.393915909595232	-1.58233119583148	-8.11565217698487	-12.7945660666518	-9.44751361623183
0.231590304305774	0.844760110160893	0.399777000876963	-1.58347415081381	-8.05392751596638	-13.0923263833642	-9.47771100407378
0.218455999318927	0.841330425299891	0.398615320821258	-1.58111456910645	-8.06190814999494	-12.7559245202009	-9.44856963863230
0.209902613158236	0.844556030315317	0.399374693965107	-1.58356594812819	-8.05856719016702	-12.8892657794707	-9.46745594347837
0.285907233080418	0.799735835306978	0.112690143867144	-1.57065410623042	-13.2472706293003	-9.05872570324713	-8.87275687533143
0.264522459072021	0.830968148848944	0.392871093669673	-1.57879947785308	-8.11974606572870	-12.6321559887711	-9.43163262651208
0.254239907578211	0.838287752675387	0.398356105347057	-1.58037572587701	-8.06308923572313	-12.6227093221462	-9.43580332811730
0.252093653629006	0.839841498559078	0.398544607190700	-1.58077370405130	-8.06194587557857	-12.9611133044678	-9.45715739757318
0.248822088199957	0.842855845217549	0.396402079949432	-1.57976788790866	-8.08339678852937	-12.7358001346811	-9.42723523076738
0.242176324803028	0.840957860363863	0.397528677887374	-1.57949675114560	-8.07055882778975	-12.8542922450447	-9.44477595820944
0.230219251869075	0.838569291445668	0.394012899967641	-1.57753749827123	-8.10520956955971	-12.6982511980998	-9.42487591114143
0.376536966315956	0.799231508165226	0.119714547373921	-1.57142570116370	-13.0047004368669	-9.05277970022185	-8.86046241970462
0.683045153066456	0.849177697553149	0.397059022113259	-1.58473599866617	-8.08597483919201	-12.7576884730483	-9.46361727446964
0.633172145183339	0.844787588806728	0.400219445272961	-1.58233233682917	-8.04683481482571	-12.7463947781956	-9.46026135340293
0.229883560194630	0.845006839945281	0.397326419763776	-1.58285804215725	-8.07946387117567	-12.8731309237099	-9.45083044647573
0.578301584851950	0.846290968226170	0.399493740257589	-1.58307552939724	-8.05629084637054	-12.6898020723443	-9.45864588629217
0.555652140821468	0.845767624438004	0.397137624395483	-1.58195961705618	-8.07977479330133	-12.7221508235607	-9.43540012210087
0.270301421147407	0.843939878477774	0.401505859755817	-1.58219785026986	-8.03213262280174	-12.0474588103019	-9.45259122029180
0.364548974439577	0.800400000000000	0.0984521565208057	-1.57134226163935	-13.7917335278014	-9.06821009067811	-8.86721198492599
0.241516109976506	0.844062635928665	0.400410525353891	-1.58280921490477	-8.04571277741437	-12.7538595795781	-9.44156981790205
0.218469105741234	0.842389577375278	0.398116620115767	-1.58355886825349	-8.07223576682790	-12.8237149690653	-9.45679887432310
0.216794656084640	0.843822188924268	0.398353376991565	-1.58236521884248	-8.06729310280264	-12.6799538741530	-9.44981703282532
0.218351887873522	0.845317344519424	0.398978525270575	-1.58094678842949	-8.05765868182664	-12.7171368421286	-9.44074141614011
0.370521466725839	0.798338127197188	0.144432015405912	-1.56869657261021	-12.2418623391128	-9.01290323578959	-8.82668629789211
0.334212842234277	0.799203187250996	0.119847844857177	-1.57231252359093	-13.0023664416951	-9.02586202399358	-8.87707394961890
0.226592148308194	0.841084990958409	0.393630737250324	-1.58090252562226	-8.11605185729989	-13.1339037657190	-9.41021971672156
0.219952829002534	0.842981977061715	0.393826284127575	-1.58258114585635	-8.11730725661885	-12.0035378912786	-9.42528127738745
0.213007372328979	0.845053262940864	0.396056295623188	-1.58145071975122	-8.09057783584816	-12.6269740852197	-9.44717265806440
0.213135087668321	0.843239548923568	0.399365811266957	-1.58047008997620	-8.05248838430666	-12.8528410215561	-9.43027157699324
0.265217774457811	0.837460317460318	0.396853697008711	-1.58328628649691	-8.08496197847271	-12.5304161377695	-9.45828192533713
0.356587357715321	0.797243572753777	0.122784815458136	-1.57210976639734	-12.9044014972114	-9.02872254478674	-8.85748007730235
0.241239414235756	0.841728698417287	0.398018852290951	-1.58190474485760	-8.06994278768862	-12.7946334924416	-9.45309195940245
0.229395866497186	0.841699192157575	0.398382587293463	-1.58105034701181	-8.06431744107206	-12.8166128315481	-9.44458446616686
0.222311780698420	0.842706776283010	0.397841860129639	-1.58313683625835	-8.07441828784794	-12.7922722513593	-9.44817024443143
0.210419024354128	0.842790888009821	0.396711515573398	-1.58098282689757	-8.08248256549992	-12.8325287201861	-9.44474173202738
0.370674982020586	0.798724082934609	0.118186455008199	-1.57305585807461	-13.0597144139592	-9.09315976514019	-8.89255625242025
0.256680733508811	0.846790337648472	0.393823148341957	-1.58304126357980	-8.11809380117551	-12.5721615645178	-9.42680740448269
0.233569063615942	0.844365585168019	0.398269875389698	-1.58268989520274	-8.06877409830521	-12.8296631000576	-9.46089310865758
0.599346360170372	0.843859409812213	0.400041824762479	-1.58115895332233	-8.04647008223145	-12.8936222952395	-9.48495162090231
0.557807331192446	0.842188763765321	0.399959679777168	-1.58231249436822	-8.04969742741154	-12.7060372256585	-9.45821136545720
0.558317437826449	0.844578450335647	0.396239596871921	-1.58174953978879	-8.08918327683574	-12.7458851715723	-9.44541530945525
0.374083202861921	0.790741915028535	0.108693995212758	-1.57002134201744	-13.3907202016567	-9.03393245879233	-8.85379369141803
0.354562205536504	0.799256998805891	0.102737673704424	-1.57403599491150	-13.6258219278704	-9.05724858803530	-8.89455379811462
0.235195222756851	0.841808725801791	0.398465242415062	-1.58289990084211	-8.06706327323874	-12.7391788680660	-9.44141747696020
0.223815989722182	0.840768951759159	0.395985430301695	-1.58222941677828	-8.09288435638423	-12.7729611622497	-9.43791671277448
0.214037585883312	0.842154362833510	0.397938839610413	-1.58254249268381	-8.07217093753305	-12.7919990484591	-9.45655491775122
0.210251959477905	0.842799027294834	0.394612649283947	-1.58145673647376	-8.10645113454453	-12.6921919043177	-9.43900114272850
0.370142412102350	0.788510101010101	0.0789647545071164	-1.56861159436455	-14.6716466560794	-9.07098156130851	-8.85675366381613
0.251494261292986	0.837424122459752	0.394832754758625	-1.58166303602244	-8.10423309043918	-12.8200683218021	-9.44675357599432
0.252429067058336	0.840268262782145	0.398824370214031	-1.58075530023011	-8.05885981563509	-12.8722991851679	-9.44515766937182
0.249808673721041	0.842203251294160	0.397532308017604	-1.58003737669164	-8.07157297775685	-12.8120724683482	-9.45419671076733
0.241809388061941	0.840825716595163	0.398754507477949	-1.58067758371652	-8.05954477754650	-12.8906519069910	-9.45111438551083
0.228666373265477	0.840072611754028	0.393858973914655	-1.57967728455767	-8.11118351013024	-12.7122163619770	-9.43760137550538];

s.wc_extra_3_ch_2 = [0.428017774867514	0.733590733590734	0.867290157390592	-1.18381385555924	-3.37420789793350	-5.27764678435160	-7.45697565960600
0.466805327657115	0.693023255813954	0.865323408909751	-1.15775857077464	-3.33998373241199	-5.07605909583293	-7.28815436776801
0.316449940640112	0.695725062866723	0.862931142580269	-1.16640823235647	-3.37603765102874	-5.18097218426725	-7.45518832459579
0.383089066860249	0.683881064162754	0.861908298215904	-1.15430132215092	-3.35985930445008	-5.03759501695377	-7.22037309299923
0.407014959543154	0.703377386196770	0.863046739443923	-1.16299853742303	-3.36909693551404	-5.16249204337874	-7.40823919418379
0.372004149199461	0.684420772303595	0.865564195713533	-1.13488241875082	-3.29442083392388	-5.05054506560591	-7.26255559328904
0.402457589919205	0.690298507462687	0.855437598622272	-1.16713766211537	-3.42830236471590	-5.01593668926579	-7.37730745254201
0.352910636284389	0.717703349282297	0.869560931837035	-1.15708485538917	-3.30732752796974	-5.07962749801014	-7.28837486162608
0.283034335593258	0.683854606931530	0.865118657967850	-1.14658359824890	-3.32022225816023	-5.04044767648881	-7.30327496244257
0.280905736829000	0.693963254593176	0.867978390882540	-1.15969404461684	-3.32579938692039	-5.08473369985619	-7.33028712698523
0.266454475604537	0.688245931283906	0.864265711366690	-1.16234435576088	-3.35878163516681	-5.08537119969163	-7.31933995620069
0.267243748958744	0.678722280887012	0.865326963500065	-1.13951699310178	-3.30544119891345	-5.01677237257853	-7.29487404695161
0.528246138874174	0.701149425287356	0.861452546309031	-1.16591586522507	-3.38142311386092	-5.10064579713423	-7.48568433511066
0.475990166365527	0.698170731707317	0.864302369617809	-1.17270910257592	-3.37745036333834	-5.12512428892639	-7.38418943091343
0.394799648391350	0.678333333333333	0.861919565495675	-1.14568418254149	-3.34206083462980	-4.97627252593348	-7.20091341936988
0.359443198126031	0.669596690796277	0.865012052650788	-1.12623295263546	-3.28081136299485	-4.89328990330556	-7.21066181854705
0.348854224188462	0.678291814946619	0.865224930139708	-1.14446663934816	-3.31595034801331	-4.96383485556241	-7.21961053104657
0.344412188393113	0.677326343381389	0.865205274648305	-1.13422551669893	-3.29576154578547	-4.95831881707815	-7.16728642104834
0.443057598172598	0.707692307692308	0.860837292166305	-1.16573180681971	-3.38562965712323	-4.98320351141597	-7.17604509045845
0.349252816986159	0.697017268445840	0.867148610005391	-1.15854924084669	-3.32805811194713	-5.15876401875899	-7.43040850635995
0.444127904377913	0.713667820069204	0.865559281582273	-1.16584889569888	-3.35549753366662	-5.12562307220252	-7.39785377874529
0.393149685900733	0.689325544344132	0.867903715171450	-1.14590928373842	-3.29877831963551	-5.14978259702294	-7.44456095519559
0.378175938222630	0.692024990812201	0.867040301442451	-1.14849432208019	-3.31058067593252	-5.08408778804970	-7.31819366210318
0.255919177764138	0.687016795521194	0.864805549882570	-1.14625418298909	-3.32276623527489	-5.06897444289351	-7.28080765315820
0.428898033543067	0.698473282442748	0.867376935661877	-1.16816252592797	-3.34213728977417	-5.05360679010514	-7.20412419212854
0.366083318684916	0.697160883280757	0.867717029402029	-1.15183528909125	-3.31043340284124	-5.15113484970845	-7.31902891444670
0.341380776219356	0.704565030146426	0.865661246894912	-1.16048896367254	-3.34402010740284	-5.14620171212038	-7.39333454431147
0.300659627248338	0.700428724544480	0.870630650321122	-1.15256120001355	-3.29191444305837	-5.19563952758258	-7.35133568817762
0.291255531578248	0.698675496688742	0.865328948495568	-1.16300173966622	-3.35223964112132	-5.15970145270482	-7.33492826964217
0.285102639894064	0.691771642991155	0.866227647843788	-1.14465546093887	-3.30906187205709	-5.10078181016513	-7.31319248248240
0.431226983165733	0.701149425287356	0.865620381751999	-1.17027329863534	-3.35932900732680	-5.11361525117458	-7.23889298007632
0.360348943623548	0.708534621578100	0.869252957722513	-1.14478898219401	-3.28498950617575	-5.01562387415461	-7.18375014876057
0.325799313482401	0.695504664970314	0.866490939334656	-1.15441169552644	-3.32574502741202	-5.06336611262676	-7.25899653559830
0.295483787917516	0.701242571582928	0.864941259722198	-1.15691063177577	-3.34266693255110	-5.00375729178682	-7.16250040353390
0.271324742623300	0.701107011070111	0.868648429947095	-1.16033600384216	-3.32237314543462	-5.11967542884830	-7.31639037560426
0.272514797677979	0.697908177125781	0.866523782740240	-1.15107904037990	-3.31971587078130	-5.10279888200678	-7.30485288118377
0.550772713203913	0.713740458015267	0.864789165219367	-1.17045066068464	-3.36594952912005	-5.18417319387728	-7.15786612066859
0.471339152838194	0.703349282296651	0.862406483230981	-1.15607183546868	-3.35812791074560	-5.02415542167194	-7.08971975877020
0.338558299929361	0.694229112833764	0.864717231424221	-1.15015416374184	-3.33031973834687	-5.07206947599271	-7.14929157692575
0.405944023615972	0.701193058568330	0.866857764179165	-1.14661240565942	-3.30791299228900	-5.12863696579948	-7.36643546240737
0.387198890954090	0.692933777284499	0.866735586266377	-1.14658945612799	-3.30902135546584	-5.09237828514731	-7.25596267498466
0.363176240004774	0.678533190578158	0.865848972781381	-1.13370953347348	-3.28996560129704	-5.07026425391247	-7.30316359982434
0.477296256778875	0.689655172413793	0.867432321790368	-1.13592375563354	-3.27734135220883	-4.96447385066629	-7.17693777125330
0.364820883940181	0.697965571205008	0.862037743584646	-1.17341996929993	-3.39553228341681	-5.22119599215034	-7.63316592376775
0.431303716840476	0.682700421940928	0.864705874264303	-1.14779178183837	-3.32568678978935	-5.01607627041298	-7.32193787085716
0.422503788191720	0.690476190476191	0.862809536196194	-1.15333894780419	-3.35127682388866	-5.04438574553765	-7.33314499451265
0.399118807918015	0.682047584715213	0.862951923972158	-1.15248387895336	-3.34876667429507	-5.07094311256281	-7.45208565506555
0.386624608576031	0.679989412387507	0.864096247692214	-1.14160655229012	-3.31871152812373	-5.04456997266977	-7.43374599257904
0.413035123884905	0.693486590038314	0.853310589856313	-1.15744253227851	-3.42452314495405	-4.95233857753121	-7.02820804726055
0.518141329197926	0.717105263157895	0.865308482661899	-1.15452506945465	-3.33357505532962	-5.06497757952310	-7.15138346537995
0.425940508164189	0.691716481639624	0.865575159037867	-1.15944298904537	-3.34255283083932	-5.15266198268058	-7.43144283640878
0.432795082696499	0.699086512627620	0.862355163762462	-1.17445593373473	-3.39684626642209	-5.18548651280161	-7.53445011847720
0.417862766275418	0.702802359882006	0.864213665900968	-1.17108635664246	-3.37664607449204	-5.14717436637316	-7.27227866551110
0.395408812714693	0.692723830132540	0.867146807395169	-1.13874009554620	-3.29043667212654	-5.03369881813113	-7.25478324957537
0.452281479419832	0.712062256809339	0.872416555684153	-1.14360480245026	-3.25588546259305	-5.03306120907562	-7.19420799032172
0.362360011086976	0.700315457413249	0.869988583825107	-1.14044560507800	-3.27089335400046	-5.04764349654666	-7.27023695649584
0.331815651439007	0.705272255834054	0.868088946064050	-1.16444255467312	-3.33397446931149	-5.20850691519092	-7.43876445264368
0.290611585144118	0.689397975492808	0.867978788713882	-1.13733907265540	-3.28108726277145	-5.03445855688847	-7.36154774484705
0.274185806676558	0.692959823074088	0.866837209361533	-1.14176121900642	-3.29861851725773	-5.06163472943982	-7.32207367760082
0.276829633498579	0.687701396348013	0.867906151039940	-1.13312294141689	-3.27359105021835	-5.01718904204374	-7.22816403813326
0.439813031542744	0.689922480620155	0.861396137078963	-1.13227858009968	-3.31459947567357	-5.02641337963728	-7.26865307006777
0.350240875055771	0.691705790297340	0.864773361785566	-1.15443061118188	-3.33737294829272	-5.22866866824923	-7.57959410607959
0.328797916858146	0.697334479793637	0.865990787340864	-1.15018556882736	-3.32098031757138	-5.12203877611157	-7.30875629972647
0.309082052513186	0.697711548696115	0.864680880865087	-1.15997267309350	-3.35072032020604	-5.18826131402407	-7.55537655072347
0.290123772068745	0.697665802148944	0.865356907784982	-1.15873646145267	-3.34350013477150	-5.20127954195682	-7.44179111871650
0.281490544346794	0.690329493704795	0.865797536893600	-1.14811390588525	-3.31915527145238	-5.13171667970157	-7.36693325460978
0.445335651544778	0.706766917293233	0.868725440477384	-1.17468466359437	-3.34532938007568	-5.07572742612654	-7.47730197257326
0.492412830057924	0.700000000000000	0.866729431494823	-1.15670107344584	-3.32748589851202	-5.17769245480371	-7.47177158010636
0.408954865963589	0.687290969899666	0.866240923728323	-1.15602665454688	-3.33082378517904	-5.10372928241459	-7.32116393568030
0.361951671289474	0.677316293929713	0.867046428393851	-1.12618029234871	-3.26565750437901	-5.02229882696530	-7.25405475308524
0.369723242690967	0.685714285714286	0.865833189255879	-1.15296226979153	-3.32843942949553	-5.11988355574418	-7.42642711010759
0.346807772792446	0.681722134178553	0.864695482890624	-1.14629957750448	-3.32367080448395	-5.06020687366002	-7.32602620424842
0.445774890633168	0.716535433070866	0.869527884856587	-1.16361139005201	-3.31712474440831	-5.14940157789100	-7.31305319147444
0.358338945425446	0.704472843450479	0.863151108484572	-1.16656949328199	-3.37359905266173	-5.11825506161675	-7.40994606136948
0.315879101911423	0.697872340425532	0.867613860010397	-1.15432296137100	-3.31726720412418	-5.08441475669238	-7.35704119920272
0.272624290910009	0.694117647058824	0.866597392897758	-1.15613938623048	-3.32888682341310	-5.09196708894111	-7.34251244155365
0.274259948380388	0.690423976608187	0.864938002922764	-1.15093954617398	-3.33100758028273	-5.05933681370349	-7.33063147600036
0.261308944523887	0.684854937450093	0.865637782714139	-1.14580141128224	-3.31571110467203	-5.03178158129057	-7.33554480103216
0.426327221700802	0.712643678160920	0.869739334099315	-1.16793799021631	-3.32431489131657	-5.12030136899867	-7.48343388981118
0.504895115201391	0.708860759493671	0.867716305521216	-1.15920707732776	-3.32519685215487	-5.15608857305754	-7.44173900417177
0.446831223862289	0.699130434782609	0.868924254326845	-1.13703887571554	-3.27300684517022	-5.07230393479439	-7.29640608397907
0.405810220919683	0.697990222705052	0.864845235253243	-1.16910782545539	-3.36775238764013	-5.24621809241846	-7.52959560591975
0.372002456030681	0.691449814126394	0.870222494218565	-1.13479088782342	-3.25964333095668	-5.07515991420440	-7.32331758295935
0.360889992194312	0.681927710843373	0.867990481786074	-1.13373640898501	-3.27419202113566	-5.04561946446933	-7.30195121390770
0.440777800634108	0.722891566265060	0.867748291167412	-1.15385641392381	-3.31076672737180	-5.14537718439384	-7.35932902086766
0.489645397707371	0.712918660287081	0.862505743294067	-1.15658136073017	-3.35844499569532	-5.13024924729408	-7.48541816315488
0.340372292650429	0.706806282722513	0.866903229154580	-1.14855774575071	-3.31098150076989	-5.07523843472062	-7.35503634044261
0.428070713361307	0.705755782678860	0.861985589243024	-1.16547789451471	-3.38163333778990	-5.09212581891033	-7.44422140388445
0.404397418078923	0.700295639320030	0.863154765670287	-1.15767734123836	-3.35765356203856	-5.14865816086669	-7.48616339874964
0.383354141277506	0.692286947141316	0.865479557134306	-1.14597532603423	-3.31722708587477	-5.05924238223955	-7.33333292036918];

s.corrosive_3_extra(:,:,1) = [0.4206    0.6466    0.8763   -1.1776   -3.2881   -6.5521   -7.4341
    0.3071    0.6398    0.8974   -1.0965   -2.9766   -6.4836   -7.0609
    0.2957    0.6357    0.8914   -1.0886   -3.0077   -6.0479   -6.5308
    0.3883    0.6553    0.8896   -1.1233   -3.0916   -6.3440   -6.8777
    0.3584    0.6326    0.8950   -1.0864   -2.9782   -6.0804   -6.5055
    0.3233    0.6294    0.8931   -1.0900   -2.9997   -6.1707   -6.5809
    0.4943    0.7054    0.8766   -1.1507   -3.2318   -6.5404   -6.9854
    0.3558    0.6633    0.8930   -1.1023   -3.0207   -5.6407   -6.1028
    0.3340    0.6301    0.8847   -1.1196   -3.1198   -6.5170   -7.2641
    0.3362    0.6323    0.8919   -1.0987   -3.0254   -6.2924   -6.8561
    0.3300    0.6260    0.8907   -1.0992   -3.0358   -6.3898   -6.9073
    0.2931    0.6194    0.8937   -1.0766   -2.9686   -6.3368   -6.7908
    0.4079    0.6429    0.9010   -1.0435   -2.8374   -5.6162   -5.7214
    0.3004    0.6171    0.8893   -1.0845   -3.0132   -6.4734   -6.8925
    0.2584    0.6248    0.8994   -1.0686   -2.9082   -6.3392   -6.6203
    0.2389    0.6346    0.8970   -1.0817   -2.9531   -6.3264   -6.8304
    0.2191    0.6338    0.8994   -1.0813   -2.9355   -6.4124   -6.7187
    0.2114    0.6238    0.8974   -1.0657   -2.9191   -6.2392   -6.5977
    0.4606    0.6640    0.8884   -1.1356   -3.1149   -6.1105   -6.4174
    0.3051    0.6535    0.8929   -1.0970   -3.0110   -6.4831   -6.8240
    0.2653    0.6340    0.9007   -1.0471   -2.8559   -6.0810   -6.5200
    0.2448    0.6458    0.8868   -1.1063   -3.0782   -6.5436   -6.9049
    0.2355    0.6370    0.8927   -1.0876   -2.9980   -6.2617   -6.6741
    0.2421    0.6393    0.8905   -1.0971   -3.0338   -6.3401   -6.7566
    0.6104    0.7193    0.8859   -1.1776   -3.2164   -5.8282   -6.5447
    0.4265    0.6585    0.8779   -1.1219   -3.1719   -5.6040   -6.2274
    0.3815    0.6533    0.8734   -1.1618   -3.2878   -5.9853   -6.6944
    0.2444    0.6049    0.8899   -1.0690   -2.9809   -6.7183   -7.2095
    0.3377    0.6152    0.8962   -1.0652   -2.9268   -6.1634   -6.6787
    0.3322    0.6252    0.8973   -1.0683   -2.9249   -6.2603   -6.6672
    0.4421    0.6557    0.8835   -1.1061   -3.0923   -6.1520   -6.8323
    0.3334    0.6623    0.8988   -1.1300   -3.0332   -6.7379   -7.0412
    0.3008    0.6437    0.8894   -1.1049   -3.0551   -6.0763   -6.5609
    0.2522    0.6200    0.8971   -1.0777   -2.9446   -6.3980   -6.8976
    0.2410    0.6217    0.8915   -1.0936   -3.0185   -6.3148   -6.7905
    0.2903    0.6381    0.8883   -1.0911   -3.0377   -5.6975   -6.2674
    0.5662    0.7241    0.8746   -1.2054   -3.3557   -6.4391   -6.9317
    0.3371    0.6567    0.8884   -1.1415   -3.1337   -6.4176   -6.8725
    0.3554    0.6330    0.8918   -1.0878   -3.0033   -6.1143   -6.5900
    0.2569    0.6355    0.8865   -1.1195   -3.1071   -6.2011   -6.6854
    0.2301    0.6187    0.8955   -1.0776   -2.9567   -6.1079   -6.6081
    0.3304    0.6416    0.8745   -1.1289   -3.2156   -5.5296   -6.2397
    0.4399    0.6800    0.8781   -1.1873   -3.2943   -6.5698   -7.2626
    0.4251    0.6476    0.8844   -1.1155   -3.1114   -6.4285   -7.0629
    0.3957    0.6322    0.8916   -1.0847   -2.9986   -6.4210   -6.9733
    0.3777    0.6449    0.8903   -1.1050   -3.0497   -6.5283   -6.9648
    0.3689    0.6393    0.8871   -1.1036   -3.0714   -6.3342   -6.8624
    0.3382    0.6340    0.8922   -1.0888   -3.0046   -6.3746   -6.7578
    0.4125    0.6694    0.8705   -1.1506   -3.2771   -7.0443   -7.0508
    0.3823    0.6478    0.9039   -1.0347   -2.8049   -6.1064   -6.4391
    0.3895    0.6561    0.8965   -1.0984   -2.9897   -6.3612   -6.7970
    0.2422    0.6171    0.8991   -1.0555   -2.8853   -6.3508   -6.7032
    0.2992    0.6246    0.8967   -1.0816   -2.9558   -6.4730   -6.9021
    0.2720    0.6192    0.8983   -1.0664   -2.9141   -6.3269   -6.6871
    0.4623    0.6875    0.8975   -1.1527   -3.0818   -6.7085   -7.0452
    0.3921    0.6355    0.8959   -1.0851   -2.9652   -6.6110   -6.9288
    0.3358    0.6354    0.8961   -1.0872   -2.9700   -6.1471   -6.5772
    0.3148    0.6411    0.9024   -1.0715   -2.8932   -6.2109   -6.6816
    0.2992    0.6276    0.8910   -1.0978   -3.0307   -6.0987   -6.5219
    0.2862    0.6180    0.8952   -1.0689   -2.9422   -6.2361   -6.5821
    0.4463    0.6721    0.8939   -1.0756   -2.9543   -5.5378   -5.9507
    0.2933    0.6118    0.8962   -1.0084   -2.8099   -5.7237   -6.1198
    0.3689    0.6317    0.8881   -1.0812   -3.0176   -6.1645   -6.7344
    0.3400    0.6312    0.8962   -1.0687   -2.9335   -6.0520   -6.5773
    0.3293    0.6266    0.8949   -1.0647   -2.9352   -6.2012   -6.6879
    0.3097    0.6216    0.8971   -1.0451   -2.8802   -5.9560   -6.4386
    0.5124    0.6612    0.8695   -1.1657   -3.3143   -5.8059   -6.4211
    0.3275    0.6340    0.8864   -1.0932   -3.0521   -6.1593   -6.6712
    0.2884    0.6386    0.8945   -1.1119   -3.0315   -6.6160   -6.8829
    0.2603    0.6334    0.8950   -1.1127   -3.0303   -6.3649   -6.8388
    0.2419    0.6157    0.8956   -1.0770   -2.9547   -6.1902   -6.6152
    0.2333    0.6118    0.8974   -1.0722   -2.9326   -6.2771   -6.7416
    0.4553    0.6562    0.8843   -1.1382   -3.1506   -5.9685   -6.4591
    0.3579    0.6545    0.8967   -1.0935   -2.9758   -5.8418   -6.2037
    0.2962    0.6327    0.8910   -1.1038   -3.0414   -6.1814   -6.6202
    0.2461    0.6176    0.8928   -1.0929   -3.0073   -6.4351   -6.7713
    0.2419    0.6200    0.8939   -1.0948   -3.0030   -6.2255   -6.6567
    0.2337    0.6156    0.8916   -1.0831   -2.9974   -6.0762   -6.5043
    0.4319    0.6538    0.8845   -1.1301   -3.1332   -6.3062   -6.5643
    0.3575    0.6623    0.8885   -1.1279   -3.1056   -6.4233   -6.9115
    0.2830    0.6304    0.8941   -1.0938   -2.9982   -6.2093   -6.6492
    0.2583    0.6268    0.8939   -1.0840   -2.9811   -6.2772   -6.6296
    0.2488    0.6332    0.8978   -1.0882   -2.9609   -6.2086   -6.6446
    0.2524    0.6281    0.8976   -1.0740   -2.9341   -6.2247   -6.5709
    0.4482    0.6967    0.8729   -1.1741   -3.3063   -6.3229   -7.0217
    0.4170    0.6467    0.8864   -1.1198   -3.1055   -6.5097   -6.9609
    0.2901    0.6246    0.8887   -1.0746   -3.0005   -6.1334   -6.6616
    0.4030    0.6539    0.8825   -1.1407   -3.1790   -6.3937   -6.9576
    0.3513    0.6148    0.8886   -1.0758   -3.0051   -6.0747   -6.5843
    0.3073    0.6288    0.8907   -1.0867   -3.0111   -6.3682   -6.8446

];
s.corrosive_3_extra(:,:,2) = [0.3394    0.6350    0.8945   -1.0793   -2.9614   -5.9725   -6.5296
    0.2874    0.6278    0.8961   -1.0868   -2.9689   -6.6298   -7.0762
    0.2776    0.6304    0.8959   -1.1003   -2.9986   -6.3028   -6.7269
    0.2686    0.6398    0.8906   -1.1184   -3.0756   -6.3074   -6.6831
    0.2481    0.6325    0.8966   -1.0955   -2.9851   -6.3244   -6.7535
    0.2429    0.6219    0.8949   -1.0743   -2.9558   -6.1700   -6.6034
    0.3455    0.6533    0.8996   -1.1078   -2.9804   -6.3257   -6.7829
    0.3099    0.6411    0.8912   -1.1078   -3.0477   -6.0052   -6.5441
    0.2501    0.6253    0.8965   -1.0719   -2.9372   -6.1962   -6.7117
    0.2424    0.6257    0.8988   -1.0627   -2.9026   -5.9555   -6.3324
    0.2261    0.6231    0.8964   -1.0709   -2.9374   -6.1091   -6.6381
    0.2239    0.6164    0.8962   -1.0573   -2.9118   -5.9410   -6.4178
    0.3789    0.6738    0.8916   -1.1139   -3.0516   -6.3709   -6.7800
    0.3304    0.6381    0.8836   -1.1109   -3.1098   -5.9298   -6.5227
    0.2620    0.6390    0.8905   -1.1002   -3.0391   -6.1451   -6.7034
    0.2387    0.6356    0.8876   -1.1041   -3.0689   -6.2449   -6.8367
    0.2331    0.6332    0.8920   -1.1028   -3.0342   -6.4271   -6.9461
    0.2272    0.6301    0.8879   -1.1056   -3.0704   -6.2907   -6.8306
    0.4080    0.7143    0.8944   -1.1851   -3.1730   -6.8625   -7.2872
    0.3097    0.6583    0.8945   -1.1067   -3.0208   -6.5618   -6.9824
    0.2675    0.6472    0.8978   -1.0781   -2.9398   -6.1561   -6.5188
    0.2447    0.6282    0.8948   -1.0843   -2.9756   -6.3525   -6.8256
    0.2412    0.6339    0.8928   -1.0956   -3.0133   -6.2817   -6.7491
    0.2247    0.6160    0.8956   -1.0625   -2.9264   -6.1238   -6.5221
    0.5594    0.6686    0.8780   -1.1443   -3.2124   -6.1348   -6.7027
    0.4266    0.6443    0.8759   -1.1286   -3.2023   -5.7929   -6.4716
    0.2844    0.6379    0.8973   -1.0895   -2.9664   -6.3078   -6.6974
    0.2533    0.6157    0.8906   -1.0724   -2.9829   -5.9554   -6.4896
    0.2354    0.6255    0.8940   -1.0812   -2.9756   -6.3096   -6.7857
    0.2309    0.6212    0.8940   -1.0710   -2.9555   -6.2176   -6.6455
    0.4562    0.6596    0.8731   -1.1228   -3.2067   -5.4762   -6.2409
    0.3041    0.6287    0.8930   -1.0647   -2.9478   -5.9801   -6.4433
    0.2790    0.6253    0.8890   -1.0907   -3.0308   -6.1685   -6.5584
    0.2322    0.6313    0.8920   -1.0754   -2.9786   -6.0877   -6.5579
    0.2312    0.6215    0.8900   -1.0942   -3.0313   -6.3336   -6.7116
    0.2267    0.6190    0.8908   -1.0813   -2.9996   -6.1528   -6.5827
    0.3821    0.6404    0.8806   -1.1126   -3.1315   -6.0178   -6.4929
    0.2861    0.6382    0.8933   -1.0940   -3.0044   -6.2073   -6.7483
    0.2578    0.6308    0.8945   -1.0822   -2.9731   -6.1762   -6.6393
    0.3128    0.6321    0.8949   -1.0795   -2.9655   -6.0682   -6.6765
    0.2965    0.6242    0.8935   -1.0875   -2.9919   -6.1953   -6.7204
    0.2974    0.6191    0.8960   -1.0617   -2.9224   -6.0679   -6.5749
    0.3761    0.6650    0.8922   -1.1507   -3.1209   -6.2838   -6.8587
    0.2998    0.6479    0.8974   -1.0904   -2.9665   -6.3679   -6.7810
    0.2757    0.6310    0.8924   -1.0759   -2.9759   -6.1299   -6.5538
    0.2553    0.6272    0.8876   -1.0831   -3.0270   -6.2372   -6.6455
    0.2463    0.6290    0.8960   -1.0863   -2.9711   -6.0583   -6.5358
    0.2438    0.6263    0.8976   -1.0629   -2.9124   -6.1520   -6.5317
    0.3680    0.6480    0.9000   -1.0948   -2.9512   -6.1296   -6.5740
    0.3076    0.6373    0.8913   -1.0948   -3.0203   -6.2331   -6.7330
    0.2596    0.6369    0.8950   -1.1081   -3.0209   -6.1062   -6.7041
    0.2262    0.6215    0.8889   -1.0956   -3.0421   -6.2128   -6.7495
    0.2127    0.6212    0.8956   -1.0857   -2.9728   -6.1441   -6.6536
    0.2089    0.6162    0.8926   -1.0806   -2.9854   -6.1786   -6.6703
    0.3767    0.6429    0.8951   -1.1182   -3.0339   -6.2188   -6.7060
    0.3795    0.6460    0.8984   -1.0872   -2.9526   -6.1667   -6.5981
    0.2580    0.6284    0.8971   -1.0871   -2.9638   -6.0046   -6.4562
    0.2302    0.6291    0.8949   -1.0848   -2.9756   -6.2365   -6.7001
    0.2276    0.6338    0.8967   -1.0926   -2.9787   -6.2481   -6.7157
    0.2005    0.6237    0.8941   -1.0773   -2.9673   -6.1979   -6.6750
    0.3751    0.6755    0.9047   -1.0922   -2.9106   -6.4543   -7.3206
    0.3059    0.6460    0.8905   -1.1053   -3.0473   -6.2873   -6.8166
    0.2863    0.6267    0.8885   -1.0881   -3.0295   -5.9679   -6.4408
    0.2704    0.6397    0.8961   -1.0850   -2.9677   -6.2228   -6.6628
    0.2232    0.6265    0.8889   -1.0970   -3.0454   -6.1673   -6.6366
    0.2067    0.6259    0.8932   -1.0853   -2.9902   -6.1966   -6.6135
    0.3739    0.6531    0.8857   -1.1435   -3.1547   -6.0695   -6.4460
    0.3049    0.6422    0.8913   -1.0904   -3.0117   -6.7328   -6.9979
    0.2630    0.6305    0.8947   -1.0877   -2.9823   -6.5593   -7.0194
    0.2418    0.6246    0.8936   -1.0773   -2.9706   -6.4318   -6.8206
    0.2328    0.6169    0.8952   -1.0628   -2.9297   -6.2196   -6.6946
    0.2191    0.6140    0.8955   -1.0629   -2.9283   -6.3064   -6.7569
    0.3794    0.6477    0.8864   -1.1192   -3.1011   -6.2789   -6.8568
    0.3181    0.6456    0.8926   -1.0977   -3.0164   -6.3039   -6.7925
    0.2321    0.6239    0.8886   -1.0875   -3.0273   -5.9041   -6.3954
    0.2157    0.6241    0.8962   -1.0707   -2.9380   -6.2562   -6.6442
    0.2119    0.6234    0.8916   -1.0794   -2.9901   -6.0600   -6.5378
    0.2043    0.6218    0.8921   -1.0796   -2.9868   -6.1187   -6.6330
    0.3914    0.6350    0.8981   -1.0798   -2.9354   -6.7151   -7.2218
    0.2784    0.6117    0.8965   -0.9946   -2.7815   -5.9656   -6.2307
    0.2583    0.6327    0.8919   -1.0872   -3.0025   -6.2472   -6.7364
    0.2310    0.6228    0.8976   -1.0676   -2.9214   -6.0026   -6.4605
    0.3061    0.6294    0.8929   -1.0866   -2.9950   -6.2838   -6.7845
    0.3068    0.6285    0.8956   -1.0746   -2.9510   -6.1773   -6.6581
    0.3383    0.6232    0.8976   -1.0795   -2.9384   -6.9617   -7.0707
    0.3308    0.6519    0.8983   -1.1019   -2.9828   -6.5468   -6.9788
    0.2672    0.6373    0.8992   -1.0784   -2.9301   -6.3815   -6.7083
    0.2680    0.6464    0.8926   -1.1162   -3.0560   -6.3446   -6.6848
    0.2426    0.6273    0.8989   -1.0775   -2.9317   -6.3063   -6.6805
    0.2438    0.6254    0.8976   -1.0629   -2.9124   -6.1520   -6.5317

];

s.info_3_extra(:,:,1) = [0.763340082546716	0.743362831858407	0.514252580180626	-1.45527958675207	-6.67053181192543	-8.93195612916529	-8.91652790337609
0.741460747884858	0.751694393099199	0.508449230250368	-1.47328046598915	-6.76039865065191	-9.71176681725313	-9.22174314875952
0.727543638825438	0.753706199460916	0.503362412596820	-1.46232797909669	-6.78516457991992	-9.15227804937165	-9.12399718153967
0.720481844505775	0.755305602716469	0.510425945376158	-1.47352312583861	-6.74383196562966	-9.27425812033482	-9.15429893400165
0.700885537555173	0.758590564938847	0.503564113747234	-1.47022963295619	-6.79949751672104	-9.08290305575612	-9.13446700106469
0.664626888412911	0.751250665247472	0.518963382586759	-1.46344310704093	-6.64739530232750	-9.36021221111948	-9.08998232612759
0.326912387137100	0.747761194029851	0.517866164198055	-1.47243209521697	-6.67245886710600	-9.32887581168065	-9.25427791680669
0.719896790551193	0.741697416974170	0.505428787585006	-1.46300712140973	-6.76719718483766	-9.14428723402873	-9.12014702167590
0.668636281426211	0.746068919371027	0.501813530001739	-1.47216832926067	-6.81894610630864	-9.25847899531785	-9.16340532833468
0.667111221374846	0.748888418378150	0.497228477120153	-1.47087790959625	-6.85855596566464	-9.39453648083638	-9.16355134815236
0.641324534062037	0.746417716022579	0.500274315341658	-1.46643742234487	-6.82190750557105	-9.13468306508703	-9.09304899440602
0.618193190304605	0.744446795007404	0.497955158575776	-1.46483652597427	-6.84001153130475	-9.19522512243277	-9.09585722951076
0.771742982393276	0.750737463126844	0.511276794974709	-1.47579292387479	-6.73825560385506	-9.22612036782038	-9.15655607912978
0.724185254623970	0.750769230769231	0.510847086487538	-1.47670959040458	-6.74562414756358	-9.05774791004677	-9.19908799583589
0.691148489158867	0.750839489590329	0.502435267347318	-1.47292774136214	-6.81479888385489	-9.20496409323488	-9.10397316269877
0.666944927277592	0.752583843071082	0.516002408586972	-1.47619044783086	-6.69911107496591	-9.32606722845197	-9.16478146978940
0.656673317493540	0.752069716775599	0.514816849535934	-1.46959419152028	-6.69665770859575	-9.10914626406710	-9.14120290394394
0.608673708493867	0.747387311305817	0.510444802095805	-1.46919961891166	-6.73521919851920	-9.16009947350539	-9.12203136929362
0.375337622156027	0.734693877551020	0.509294530871320	-1.47227204653274	-6.74905996956196	-9.51432459293905	-9.10542582821514
0.717878089166608	0.749537322640345	0.504975232317046	-1.47445353205616	-6.79420160297262	-9.29213866070781	-9.11609011763549
0.692448648201608	0.748904617458713	0.502210236254882	-1.46910803193461	-6.80920850482885	-9.21693401589658	-9.10754083733663
0.661631468904509	0.747787610619469	0.507155206218486	-1.47424365150999	-6.77480857857970	-9.35885394144474	-9.15895368770125
0.630278957194307	0.746423927178153	0.506932059690869	-1.47051196959483	-6.76949469091730	-9.23097827257853	-9.09560338737120
0.605181066288550	0.741741424802111	0.513567040885790	-1.46385383868863	-6.69645543286384	-9.30441873574740	-9.06655241916379
0.337829127069677	0.749631811487482	0.516829855585335	-1.46556301537029	-6.66803615133153	-8.70201615567341	-9.06635106755273
0.700901718526375	0.748002458512600	0.501521131683841	-1.46878600132457	-6.81431043583795	-9.06074450940122	-9.04678768768351
0.682432324828239	0.748730104977989	0.517376821924588	-1.46654114108040	-6.66728870060326	-9.28804773183116	-9.10108140233107
0.648375301955986	0.744869896340173	0.508153409013936	-1.46513194739765	-6.74755704415455	-9.07119875934085	-9.05820880627068
0.612427882416497	0.743608262314026	0.513570775320679	-1.46712854120455	-6.70289555406690	-9.21788311625094	-9.06627519523658
0.610371966119693	0.742860153862367	0.517280287120998	-1.46419831570159	-6.66391362972878	-9.29912909472655	-9.06417885413072
0.775584634527342	0.752592592592593	0.508949005362119	-1.46926741863793	-6.74620084667984	-8.98370768100629	-9.12022931180369
0.750147543577494	0.756823821339950	0.499363894683504	-1.47452407809330	-6.84549357202976	-9.16716224159082	-9.17838062510118
0.670588810694428	0.752695417789758	0.496805790073439	-1.46829233329196	-6.85702844226974	-8.89314796806231	-9.13109753843998
0.630544697579689	0.752060875079264	0.496515674535400	-1.47678718887628	-6.87692094372780	-9.33686607761139	-9.18673970750000
0.599031618438935	0.746683967704729	0.508673403614819	-1.46893961866921	-6.75060413624944	-9.21489325286773	-9.11604388235035
0.610077909031120	0.744988394175987	0.506997635867421	-1.46688676083724	-6.76172666480305	-9.20874624429949	-9.08882034176668
0.348459759465287	0.744807121661721	0.513685743841980	-1.46996748658698	-6.70493707732743	-8.80643068684382	-9.05825764899924
0.723628084798831	0.743034055727554	0.513440698505501	-1.45987912828887	-6.68864904003794	-9.06900713335443	-9.00322279896217
0.681067415809355	0.750337381916329	0.503629012102113	-1.46791727088619	-6.79391036096384	-9.04651442281279	-9.09025901155400
0.676353942471791	0.750582257040017	0.502203556008084	-1.47120417384906	-6.81370403063389	-9.17999583113018	-9.09003227998071
0.629835059793608	0.745848375451264	0.505648467210719	-1.46712276336955	-6.77434914906409	-9.18934705195922	-9.08391817042782
0.617386335685655	0.744139387539599	0.509539374719744	-1.46368025831437	-6.73234372915184	-9.21062224760681	-9.06699711121856
0.765738098135088	0.747800586510264	0.488080376527293	-1.45528900955911	-6.90952611365115	-8.60087253037090	-9.02250546567205
0.733516683949845	0.750463248919086	0.487364610146117	-1.46597719812542	-6.93917631184053	-8.78511857920332	-9.04831358934278
0.725244297364368	0.750757830919502	0.512914608154952	-1.46616542451318	-6.70649190863120	-9.13175000304535	-9.01264716927877
0.708084430971416	0.755362072626885	0.507182042589471	-1.46855754237099	-6.76319670790935	-9.21704763353839	-9.05324644134056
0.678607491257442	0.751012145748988	0.503804457112402	-1.46809598575623	-6.79304398928380	-9.12229441286944	-9.04113026735193
0.664624338965802	0.746815286624204	0.518996651413568	-1.46444965743146	-6.64911120701715	-9.48673627602742	-9.07943739030607
0.330053818208905	0.743323442136499	0.521535837773734	-1.46836731620340	-6.63168664917801	-9.09476241591529	-9.12037018892613
0.718941725272947	0.746609124537608	0.509078887123019	-1.47011844090792	-6.74838311277159	-9.19863160042837	-9.17948594910851
0.684112070322718	0.749579266240323	0.508560601404781	-1.46991888982940	-6.75321038501096	-9.00707399405024	-9.10883729954236
0.626340987488616	0.748631578947368	0.505111151619107	-1.47279121710469	-6.79043250276847	-9.26621779801296	-9.13464408619579
0.599026455119882	0.742520138089758	0.505128319309210	-1.46818688798241	-6.78119648198341	-9.13677051223575	-9.11840238675766
0.578665398227295	0.741305033098666	0.512582309442051	-1.46622801950848	-6.71004484011168	-9.20735616102854	-9.12520974605036
0.762019407598067	0.750737463126844	0.516112302765535	-1.47834819596436	-6.69998268755368	-8.92156283705523	-9.16401070512148
0.732059987979124	0.748771498771499	0.505914055175103	-1.46946228575227	-6.77571127623913	-9.13551418425710	-9.15847445935150
0.672806030092887	0.750335570469799	0.513776111180402	-1.47468329164911	-6.71579256600946	-9.05696408002862	-9.17726881964685
0.661221809552974	0.748253228879949	0.512563004991598	-1.47229823809234	-6.72215228583437	-9.28864826613760	-9.13386267292706
0.629566607944628	0.747435341713625	0.505762157516892	-1.46843723768359	-6.77594705494996	-9.08605206524459	-9.10130373782557
0.606665460739130	0.745669623996620	0.505230229998914	-1.46661402288654	-6.77720147728800	-9.19688886065024	-9.09800398210035
0.373404356211182	0.744047619047619	0.512272924588573	-1.46847495243899	-6.71463048295358	-9.57071108611810	-9.14495044215207
0.726512154503317	0.743857493857494	0.511580721919935	-1.46566265808979	-6.71693145287878	-9.13280050910061	-9.06391789975041
0.675223256375577	0.746068919371027	0.513753366114634	-1.46791369985493	-6.70246004571513	-9.14172673883513	-9.13199420363318
0.652006970405380	0.744868035190616	0.510715902418738	-1.46417563260776	-6.72252834637883	-9.16452850844305	-9.10009573047604
0.622931884603181	0.744300144300144	0.514092083161885	-1.46774526671879	-6.69945396784047	-9.35454541846131	-9.10230233715397
0.585336654402946	0.740385628490149	0.510356934501531	-1.46265833169653	-6.72292816964947	-9.30766348258462	-9.05965368205190
0.338241556771676	0.752238805970149	0.504010547805124	-1.47893739675990	-6.81023133534250	-9.33964328340456	-9.24026887484265
0.700896234516566	0.743400859422959	0.504878305861077	-1.46474156988520	-6.77566536488848	-9.11710768475137	-9.10105447083037
0.671991831121504	0.747732616728250	0.520595707339244	-1.47438116672984	-6.65429783610055	-9.26222340810749	-9.12861735515437
0.646561370077442	0.746309574019401	0.515712243817894	-1.46629053271103	-6.68190722897367	-8.97655762458102	-9.09271081544243
0.616760441731942	0.743251005169443	0.511348794590609	-1.46970604807562	-6.72801663031016	-9.10198397003747	-9.07672670974018
0.607490647017865	0.739322069472138	0.514503311951649	-1.46270921286347	-6.68577217029205	-9.21385510306962	-9.05140095771540
0.769440928328408	0.749629629629630	0.525486235316445	-1.45992111569312	-6.57986981085654	-9.30729671415092	-9.10648179449519
0.716698936455238	0.754962779156328	0.503138217443600	-1.47184603521375	-6.80569096860299	-9.40703789445078	-9.14252911405727
0.685479663941687	0.749578699022582	0.502790994046984	-1.47084676308039	-6.80739447809854	-9.35779650334282	-9.11863307396868
0.645829930879603	0.752701843610935	0.510477856370119	-1.47696301407294	-6.75024180623425	-9.31800957060818	-9.14043980958977
0.630806136200573	0.747613537749494	0.509044350953078	-1.47010457386873	-6.74958359800548	-9.28944750957921	-9.09537817041127
0.615918075927758	0.747042670046472	0.514513209565102	-1.46849134592897	-6.69724737857608	-9.26962005420210	-9.10164665708878
0.758640636569634	0.749627421758569	0.505310538781712	-1.45505358168671	-6.75071901225170	-9.40046530238833	-9.06215369394745
0.732486432168599	0.749845583693638	0.513669622877060	-1.46558059061959	-6.69800509027988	-9.15416514027132	-9.13841638874611
0.698653176211054	0.750505050505051	0.510095809732277	-1.46829260998921	-6.73610579990199	-9.08169657019105	-9.11026095205424
0.642984343395639	0.752171150180047	0.496108969705199	-1.46896761543033	-6.86502241398961	-9.11485243575006	-9.11032130179939
0.629263880924278	0.746728031065727	0.502578604882803	-1.46752000011052	-6.80304919439569	-9.12974703102294	-9.11512236141976
0.596366413239431	0.746252902681022	0.510219677028402	-1.46170494552660	-6.72225980773935	-9.10215475876105	-9.07603763880309
0.780125817051798	0.758518518518519	0.475926158610168	-1.48136335895422	-7.07573868986807	-8.65182454009061	-9.18676462471613
0.752527634621302	0.757500000000000	0.505871556126338	-1.47321942303722	-6.78359953411802	-9.26063094017921	-9.19109348709354
0.744088016805483	0.760027192386132	0.515517137519767	-1.47725056918420	-6.70532804191231	-9.11442797906171	-9.22202837841506
0.731272908131345	0.756993380311766	0.510171373906234	-1.47330295806679	-6.74568428419206	-9.10938408370799	-9.18680708879203
0.716845022514659	0.758439281017098	0.513448671655353	-1.47624910722049	-6.72223198457371	-9.04685786317031	-9.19302218593490
0.671576082674863	0.746727679046504	0.523032709676221	-1.46244575069433	-6.60925080479080	-9.35979291186313	-9.12297443883552];

s.info_3_extra(:,:,2) = [0.4177    0.5639    0.5772   -1.2600   -5.7390   -6.1310   -6.7367
    0.4183    0.5688    0.5730   -1.2677   -5.7895   -6.1717   -6.8274
    0.4114    0.5668    0.5750   -1.2622   -5.7627   -6.1405   -6.7494
    0.4054    0.5680    0.5856   -1.2662   -5.6827   -6.2403   -6.8144
    0.3932    0.5661    0.5749   -1.2628   -5.7648   -6.1671   -6.7678
    0.3732    0.5612    0.5887   -1.2603   -5.6457   -6.2573   -6.7782
    0.4259    0.5732    0.5621   -1.2655   -5.8765   -6.1391   -6.7860
    0.4034    0.5699    0.5756   -1.2633   -5.7594   -6.1611   -6.7986
    0.3658    0.5646    0.5796   -1.2624   -5.7246   -6.2193   -6.8019
    0.3506    0.5657    0.5888   -1.2702   -5.6644   -6.2910   -6.8435
    0.3224    0.5630    0.5800   -1.2640   -5.7242   -6.2171   -6.8113
    0.3281    0.5621    0.5803   -1.2610   -5.7164   -6.1881   -6.7819
    0.4233    0.5640    0.5814   -1.2628   -5.7097   -6.2297   -6.7735
    0.4104    0.5626    0.5725   -1.2662   -5.7914   -6.2061   -6.7978
    0.3770    0.5627    0.5872   -1.2635   -5.6641   -6.2236   -6.8046
    0.3553    0.5645    0.5651   -1.2626   -5.8467   -6.1187   -6.7932
    0.3523    0.5631    0.5761   -1.2639   -5.7568   -6.1907   -6.7996
    0.3479    0.5626    0.5780   -1.2600   -5.7331   -6.1790   -6.7789
    0.4161    0.5576    0.5875   -1.2586   -5.6508   -6.2921   -6.7862
    0.4044    0.5618    0.5708   -1.2621   -5.7971   -6.1800   -6.7797
    0.3815    0.5612    0.5758   -1.2662   -5.7636   -6.1949   -6.7852
    0.3688    0.5626    0.5859   -1.2635   -5.6748   -6.2419   -6.8201
    0.3570    0.5615    0.5776   -1.2650   -5.7464   -6.1896   -6.7909
    0.3401    0.5578    0.5828   -1.2613   -5.6958   -6.2209   -6.7941
    0.4172    0.5578    0.5892   -1.2488   -5.6168   -6.2328   -6.7338
    0.3962    0.5595    0.5825   -1.2703   -5.7159   -6.2996   -6.8599
    0.3759    0.5616    0.5765   -1.2597   -5.7453   -6.1123   -6.7722
    0.3555    0.5588    0.5816   -1.2638   -5.7111   -6.2277   -6.8048
    0.3478    0.5593    0.5942   -1.2657   -5.6112   -6.3132   -6.8330
    0.3414    0.5559    0.5946   -1.2636   -5.6040   -6.3162   -6.8388
    0.4228    0.5679    0.5587   -1.2667   -5.9074   -6.1469   -6.8156
    0.4083    0.5645    0.5798   -1.2699   -5.7377   -6.2334   -6.8235
    0.3769    0.5648    0.5789   -1.2678   -5.7413   -6.2197   -6.8213
    0.3581    0.5607    0.5816   -1.2659   -5.7153   -6.2201   -6.8269
    0.3391    0.5649    0.5812   -1.2687   -5.7240   -6.2544   -6.8414
    0.3349    0.5623    0.5775   -1.2672   -5.7523   -6.2343   -6.8084
    0.4216    0.5634    0.5719   -1.2595   -5.7814   -6.1338   -6.7261
    0.4012    0.5638    0.5735   -1.2638   -5.7777   -6.1572   -6.7831
    0.3717    0.5608    0.5785   -1.2621   -5.7330   -6.1884   -6.7946
    0.3666    0.5595    0.5837   -1.2568   -5.6792   -6.1658   -6.7681
    0.3417    0.5595    0.5813   -1.2620   -5.7102   -6.1922   -6.7895
    0.3421    0.5591    0.5836   -1.2599   -5.6868   -6.1860   -6.7707
    0.4257    0.5636    0.5647   -1.2731   -5.8696   -6.2569   -6.8090
    0.4121    0.5588    0.5798   -1.2658   -5.7295   -6.3198   -6.8060
    0.4055    0.5652    0.5793   -1.2688   -5.7400   -6.3030   -6.8113
    0.4003    0.5625    0.5663   -1.2580   -5.8274   -6.1244   -6.7292
    0.3841    0.5618    0.5758   -1.2621   -5.7562   -6.2038   -6.7621
    0.3734    0.5581    0.5857   -1.2605   -5.6707   -6.2884   -6.7544
    0.4220    0.5642    0.5672   -1.2618   -5.8254   -6.1220   -6.7891
    0.3999    0.5616    0.5760   -1.2630   -5.7556   -6.1745   -6.7937
    0.3514    0.5590    0.5820   -1.2639   -5.7080   -6.2388   -6.8158
    0.3533    0.5625    0.5765   -1.2594   -5.7444   -6.1223   -6.7848
    0.3366    0.5621    0.5826   -1.2659   -5.7074   -6.2427   -6.8226
    0.3303    0.5595    0.5857   -1.2606   -5.6708   -6.2422   -6.7809
    0.4176    0.5651    0.5685   -1.2679   -5.8271   -6.1583   -6.8016
    0.4052    0.5645    0.5697   -1.2663   -5.8147   -6.2174   -6.8002
    0.3992    0.5639    0.5868   -1.2651   -5.6705   -6.1858   -6.7991
    0.3703    0.5657    0.5723   -1.2671   -5.7949   -6.2503   -6.8219
    0.3553    0.5618    0.5795   -1.2682   -5.7369   -6.2251   -6.8102
    0.3563    0.5626    0.5801   -1.2633   -5.7225   -6.2367   -6.7961
    0.4145    0.5652    0.5641   -1.2592   -5.8463   -6.1406   -6.7563
    0.4059    0.5631    0.5853   -1.2649   -5.6822   -6.2564   -6.8169
    0.3694    0.5595    0.5775   -1.2632   -5.7438   -6.2166   -6.7858
    0.3686    0.5595    0.5787   -1.2593   -5.7260   -6.1936   -6.7747
    0.3551    0.5589    0.5800   -1.2633   -5.7233   -6.2207   -6.7971
    0.3336    0.5543    0.5882   -1.2594   -5.6479   -6.2583   -6.7944
    0.4122    0.5742    0.5495   -1.2743   -6.0008   -6.0812   -6.8251
    0.3870    0.5641    0.5784   -1.2695   -5.7486   -6.2509   -6.8438
    0.3755    0.5588    0.5942   -1.2639   -5.6072   -6.3657   -6.8130
    0.3698    0.5639    0.5933   -1.2742   -5.6357   -6.2745   -6.8652
    0.3554    0.5621    0.5879   -1.2658   -5.6634   -6.2699   -6.8172
    0.3405    0.5556    0.5899   -1.2580   -5.6309   -6.2331   -6.7608
    0.4208    0.5640    0.5606   -1.2640   -5.8855   -6.0200   -6.7276
    0.4034    0.5622    0.5876   -1.2640   -5.6613   -6.2653   -6.8152
    0.3762    0.5628    0.5798   -1.2698   -5.7382   -6.2307   -6.7979
    0.3628    0.5628    0.5798   -1.2660   -5.7303   -6.2067   -6.8275
    0.3447    0.5610    0.5825   -1.2659   -5.7078   -6.2136   -6.8205
    0.3392    0.5628    0.5772   -1.2653   -5.7509   -6.2140   -6.7986
    0.4249    0.5624    0.5759   -1.2640   -5.7575   -6.1858   -6.7861
    0.4081    0.5654    0.5737   -1.2672   -5.7829   -6.2386   -6.7820
    0.3854    0.5622    0.5826   -1.2613   -5.6977   -6.1906   -6.8001
    0.3742    0.5657    0.5871   -1.2728   -5.6834   -6.2792   -6.8578
    0.3450    0.5615    0.5804   -1.2631   -5.7194   -6.1911   -6.8051
    0.3403    0.5611    0.5826   -1.2627   -5.7009   -6.2364   -6.7918
    0.4180    0.5642    0.5791   -1.2553   -5.7133   -6.1776   -6.8021
    0.4141    0.5651    0.5838   -1.2655   -5.6957   -6.1260   -6.8232
    0.4072    0.5641    0.5764   -1.2613   -5.7487   -6.1224   -6.7911
    0.3996    0.5672    0.5862   -1.2675   -5.6804   -6.2290   -6.8318
    0.3878    0.5656    0.5807   -1.2634   -5.7174   -6.2134   -6.8307
    0.3720    0.5594    0.5896   -1.2600   -5.6376   -6.1958   -6.7995
];

s.phone_3_extra(:,:,1) = [0.5261    0.4991    0.2878   -1.0490   -8.3803  -10.0996   -8.8643
    0.5073    0.4933    0.2981   -1.0297   -8.1963  -10.5400   -8.6065
    0.4959    0.4932    0.2795   -1.0246   -8.4547  -10.2348   -8.5718
    0.4977    0.5045    0.2957   -1.0542   -8.2788  -10.3324   -8.8090
    0.4808    0.4946    0.2920   -1.0315   -8.2868  -10.5589   -8.6523
    0.4281    0.4616    0.2329   -0.9326   -9.0247   -9.3085   -7.4932
    0.5145    0.4893    0.3116   -1.0251   -8.0004  -10.7350   -8.5104
    0.4823    0.4893    0.2878   -1.0330   -8.3489  -11.2171   -8.4089
    0.4545    0.4929    0.2810   -1.0361   -8.4550  -10.4411   -8.6836
    0.4426    0.4956    0.2955   -1.0409   -8.2551  -11.4471   -8.6139
    0.4074    0.4897    0.2866   -1.0322   -8.3654  -10.8462   -8.6704
    0.4054    0.4861    0.2969   -1.0240   -8.2019  -10.7567   -8.6180
    0.5189    0.4933    0.2893   -1.0310   -8.3229  -11.9950   -8.5904
    0.4981    0.4921    0.3031   -1.0311   -8.1284  -10.6244   -8.6436
    0.4559    0.4932    0.3038   -1.0348   -8.1263  -10.5646   -8.6011
    0.4234    0.4885    0.2845   -1.0306   -8.3921  -10.1718   -8.5817
    0.4025    0.4907    0.2918   -1.0322   -8.2902  -10.5914   -8.6145
    0.4068    0.4872    0.3055   -1.0245   -8.0828  -10.2937   -8.6290
    0.5185    0.4976    0.2866   -1.0428   -8.3860  -11.0102   -8.7284
    0.4886    0.4927    0.2795   -1.0395   -8.4848  -10.5809   -8.6422
    0.4579    0.4915    0.2917   -1.0344   -8.2966  -10.5499   -8.6944
    0.4351    0.4908    0.2853   -1.0344   -8.3884  -10.6217   -8.6078
    0.4310    0.4919    0.2879   -1.0339   -8.3502  -10.4514   -8.6861
    0.4103    0.4860    0.2965   -1.0222   -8.2038  -10.5435   -8.6241
    0.5197    0.4970    0.2958   -1.0454   -8.2592  -10.1897   -8.5949
    0.4841    0.4919    0.2837   -1.0299   -8.4033  -10.4917   -8.7356
    0.4616    0.4917    0.2854   -1.0345   -8.3869  -10.8211   -8.6720
    0.4405    0.4882    0.2850   -1.0258   -8.3759  -10.6484   -8.6455
    0.4402    0.4938    0.2886   -1.0361   -8.3433  -10.9869   -8.7079
    0.4250    0.4877    0.2868   -1.0227   -8.3440  -10.6413   -8.6095
    0.5093    0.4906    0.2783   -1.0297   -8.4811  -10.6199   -8.7233
    0.4995    0.4928    0.2931   -1.0358   -8.2785  -10.6050   -8.6545
    0.4403    0.4922    0.2948   -1.0381   -8.2588  -10.8603   -8.7112
    0.4270    0.4868    0.2806   -1.0242   -8.4379  -10.8669   -8.7319
    0.4221    0.4906    0.2906   -1.0380   -8.3191  -10.6449   -8.6905
    0.4049    0.4876    0.2789   -1.0287   -8.4715  -10.8899   -8.6291
    0.5218    0.4958    0.2812   -1.0473   -8.4743  -10.7224   -8.6316
    0.4973    0.4919    0.2839   -1.0356   -8.4110  -10.7328   -8.6259
    0.4481    0.4934    0.2910   -1.0384   -8.3135  -11.1064   -8.6370
    0.4237    0.4967    0.2841   -1.0427   -8.4225  -10.9447   -8.7241
    0.4023    0.4919    0.2945   -1.0366   -8.2611  -10.9382   -8.6192
    0.3909    0.4887    0.2838   -1.0290   -8.3999  -10.7764   -8.6095
    0.5140    0.4946    0.2886   -1.0414   -8.3530  -11.2934   -8.7148
    0.4945    0.4894    0.3004   -1.0282   -8.1608  -10.6800   -8.6436
    0.4635    0.4670    0.2406   -0.9502   -8.9255   -9.7164   -7.4646
    0.4954    0.5046    0.3024   -1.0558   -8.1879  -10.9146   -8.8310
    0.4688    0.4933    0.2949   -1.0326   -8.2467  -10.9018   -8.6931
    0.4298    0.4605    0.2398   -0.9324   -8.9043   -9.3497   -7.5264
    0.5070    0.4924    0.2886   -1.0331   -8.3366  -11.1989   -8.6976
    0.4953    0.4937    0.2880   -1.0353   -8.3508   -9.9606   -8.6644
    0.4356    0.4928    0.2828   -1.0372   -8.4302  -10.9361   -8.6925
    0.4281    0.4982    0.2933   -1.0469   -8.2983  -10.1213   -8.7407
    0.4110    0.4919    0.2818   -1.0351   -8.4421  -10.4693   -8.7103
    0.3997    0.4859    0.2931   -1.0242   -8.2564  -10.3935   -8.6239
    0.4935    0.4825    0.2822   -1.0200   -8.4044  -11.1104   -8.6929
    0.4921    0.4914    0.2730   -1.0289   -8.5616  -11.0025   -8.6503
    0.4450    0.4934    0.2869   -1.0381   -8.3731  -11.1947   -8.7426
    0.4183    0.4894    0.2920   -1.0277   -8.2786  -11.3657   -8.6635
    0.4230    0.4922    0.2793   -1.0333   -8.4747  -11.1187   -8.7091
    0.4044    0.4884    0.3099   -1.0243   -8.0220  -11.0769   -8.6851
    0.5286    0.5009    0.2827   -1.0544   -8.4652  -10.4619   -8.7028
    0.4806    0.4913    0.2858   -1.0331   -8.3789  -10.1466   -8.7418
    0.4593    0.4928    0.2932   -1.0359   -8.2775  -10.7583   -8.6570
    0.4406    0.4899    0.2849   -1.0285   -8.3831  -10.4257   -8.6626
    0.4281    0.4925    0.2869   -1.0331   -8.3627  -11.0075   -8.6371
    0.4289    0.4876    0.2986   -1.0230   -8.1754  -10.6614   -8.6163
    0.5206    0.4985    0.2743   -1.0577   -8.5979   -9.9646   -8.8551
    0.4768    0.4895    0.2971   -1.0289   -8.2080   -9.7746   -8.5771
    0.4612    0.4933    0.2914   -1.0370   -8.3049  -10.6480   -8.6977
    0.4365    0.4905    0.2929   -1.0312   -8.2729  -10.2628   -8.7687
    0.4314    0.4925    0.2909   -1.0353   -8.3098  -10.7509   -8.7172
    0.3873    0.4576    0.2633   -0.9313   -8.5161   -9.0098   -7.4634
    0.4923    0.4903    0.2956   -1.0421   -8.2551  -10.3565   -8.8865
    0.4765    0.4903    0.2966   -1.0329   -8.2235  -10.0454   -8.6162
    0.4360    0.4916    0.2887   -1.0336   -8.3375   -9.9678   -8.6925
    0.4108    0.4914    0.2914   -1.0365   -8.3046  -10.3082   -8.5828
    0.3932    0.4913    0.2884   -1.0358   -8.3466  -10.3984   -8.7185
    0.3875    0.4869    0.2798   -1.0279   -8.4570  -10.4714   -8.6733
    0.5228    0.4967    0.2978   -1.0467   -8.2338  -10.7828   -8.8131
    0.4861    0.4880    0.2988   -1.0275   -8.1809  -10.4247   -8.7779
    0.4651    0.4932    0.2883   -1.0354   -8.3470  -10.6864   -8.7359
    0.4466    0.4937    0.2854   -1.0377   -8.3945  -10.2795   -8.6199
    0.4199    0.4922    0.2865   -1.0339   -8.3707  -10.7620   -8.7186
    0.4075    0.4874    0.2794   -1.0263   -8.4597  -10.5863   -8.6272
    0.5252    0.5009    0.3139   -1.0529   -8.0244  -10.7556   -8.6738
    0.5099    0.4964    0.2849   -1.0388   -8.4027  -11.6531   -8.5515
    0.4970    0.4927    0.2651   -1.0247   -8.6749  -10.1627   -8.7570
    0.4911    0.5058    0.2889   -1.0546   -8.3766  -10.7694   -8.7984
    0.4582    0.4947    0.2852   -1.0334   -8.3876  -10.7946   -8.6749
    0.4246    0.4597    0.2329   -0.9290   -9.0182   -9.2769   -7.4924

];
s.phone_3_extra(:,:,2) = [.5314    0.5014    0.3069   -1.0594   -8.1321   -9.7792   -8.8185
    0.5084    0.4936    0.3021   -1.0360   -8.1529  -10.8984   -8.4732
    0.4966    0.4953    0.2954   -1.0417   -8.2576  -10.3028   -8.6092
    0.4804    0.4893    0.2756   -1.0234   -8.5102  -11.2432   -8.4189
    0.4779    0.4945    0.2904   -1.0359   -8.3179  -10.9850   -8.7187
    0.4468    0.4860    0.2879   -1.0129   -8.3085  -10.9749   -8.6296
    0.5094    0.4913    0.2729   -1.0382   -8.5796  -10.5241   -8.6784
    0.4919    0.4920    0.2742   -1.0363   -8.5565  -10.2980   -8.6802
    0.4498    0.4913    0.2891   -1.0362   -8.3364  -11.1570   -8.6882
    0.4332    0.4890    0.2887   -1.0310   -8.3320  -11.8233   -8.6772
    0.4128    0.4899    0.2839   -1.0337   -8.4074  -10.8641   -8.7203
    0.4047    0.4856    0.2944   -1.0234   -8.2361  -11.0365   -8.6857
    0.5201    0.4927    0.2896   -1.0326   -8.3217  -12.9403   -8.6977
    0.4919    0.4926    0.2630   -1.0330   -8.7235  -10.9640   -8.6407
    0.4625    0.4888    0.2896   -1.0243   -8.3065  -11.0028   -8.5749
    0.4318    0.4944    0.2908   -1.0383   -8.3165  -10.9636   -8.6760
    0.4149    0.4908    0.2901   -1.0314   -8.3126  -11.0874   -8.6012
    0.4109    0.4876    0.3072   -1.0245   -8.0592  -10.9015   -8.6014
    0.4945    0.4927    0.2673   -1.0467   -8.6829  -10.7451   -8.6479
    0.4911    0.4907    0.2964   -1.0344   -8.2292  -11.2138   -8.5913
    0.4568    0.4886    0.2977   -1.0304   -8.2026  -10.4716   -8.6467
    0.4337    0.4847    0.2971   -1.0185   -8.1870  -11.9625   -8.4365
    0.4167    0.4877    0.2855   -1.0309   -8.3790  -10.6685   -8.7120
    0.4025    0.4827    0.2976   -1.0205   -8.1844  -10.5773   -8.5954
    0.5116    0.4931    0.2633   -1.0440   -8.7408  -11.0811   -8.5774
    0.4695    0.4890    0.2905   -1.0324   -8.3083  -10.4947   -8.7089
    0.4737    0.4939    0.3004   -1.0388   -8.1820  -11.4734   -8.6499
    0.4531    0.4912    0.2995   -1.0374   -8.1918  -10.6458   -8.6855
    0.4341    0.4893    0.2856   -1.0297   -8.3744  -11.3021   -8.6966
    0.4016    0.4831    0.2838   -1.0168   -8.3749  -11.2543   -8.5880
    0.5290    0.4993    0.3001   -1.0490   -8.2059  -12.7796   -8.7427
    0.5030    0.4962    0.2589   -1.0444   -8.8108  -11.2977   -8.6935
    0.4692    0.4932    0.2833   -1.0398   -8.4292  -11.1128   -8.6285
    0.4486    0.4951    0.2856   -1.0398   -8.3957  -11.1397   -8.5457
    0.4221    0.4918    0.2884   -1.0346   -8.3445  -11.1700   -8.6339
    0.4135    0.4875    0.2787   -1.0268   -8.4717  -11.0158   -8.6014
    0.5175    0.4913    0.2820   -1.0395   -8.4469   -9.5199   -8.7983
    0.4840    0.4858    0.3021   -1.0217   -8.1236  -10.8984   -8.6329
    0.4527    0.4905    0.2775   -1.0327   -8.5004  -10.9230   -8.7031
    0.4481    0.4931    0.2599   -1.0329   -8.7732  -10.9941   -8.8574
    0.4222    0.4908    0.2723   -1.0306   -8.5749  -11.1597   -8.7078
    0.4110    0.4868    0.2618   -1.0244   -8.7253  -11.1027   -8.6543
    0.5064    0.4896    0.2813   -1.0417   -8.4608  -11.4387   -8.8689
    0.5086    0.4973    0.2962   -1.0386   -8.2394  -11.7521   -8.4650
    0.4900    0.4917    0.2778   -1.0300   -8.4909  -11.2212   -8.4417
    0.4868    0.4960    0.2708   -1.0408   -8.6184  -10.2841   -8.8653
    0.4706    0.4925    0.2851   -1.0357   -8.3948  -10.9133   -8.7437
    0.3933    0.4310    0.3222   -0.8345   -7.4788   -8.9076   -7.2193
    0.5117    0.4872    0.2448   -1.0278   -9.0094  -11.0815   -8.7026
    0.4868    0.4919    0.2946   -1.0317   -8.2492  -10.6917   -8.7315
    0.4446    0.4900    0.2879   -1.0300   -8.3419  -11.2621   -8.7192
    0.4138    0.4923    0.2618   -1.0350   -8.7471  -10.5867   -8.6924
    0.3868    0.4897    0.2887   -1.0306   -8.3322  -11.2771   -8.6984
    0.3840    0.4861    0.2917   -1.0217   -8.2710  -11.0541   -8.6783
    0.5253    0.4986    0.2922   -1.0402   -8.2997  -11.6220   -8.6611
    0.4897    0.4938    0.2683   -1.0390   -8.6528  -11.1952   -8.6015
    0.4511    0.4879    0.2969   -1.0266   -8.2067  -10.8678   -8.6024
    0.4348    0.4944    0.2942   -1.0406   -8.2724  -10.6688   -8.6348
    0.4142    0.4904    0.2877   -1.0334   -8.3518  -10.7884   -8.5670
    0.4015    0.4872    0.3017   -1.0281   -8.1421  -11.0614   -8.5831
    0.5272    0.4986    0.2788   -1.0523   -8.5188  -10.9228   -8.6416
    0.4964    0.4914    0.2780   -1.0280   -8.4841  -11.2605   -8.7958
    0.4607    0.4905    0.2893   -1.0345   -8.3303  -10.8354   -8.6223
    0.4345    0.4891    0.2989   -1.0314   -8.1880  -10.0246   -8.7221
    0.4223    0.4894    0.2688   -1.0313   -8.6299  -11.1374   -8.6398
    0.4027    0.4826    0.2850   -1.0170   -8.3582  -10.7851   -8.6093
    0.5087    0.4903    0.2928   -1.0361   -8.2832   -9.9540   -8.6543
    0.4723    0.4855    0.2841   -1.0220   -8.3814  -11.2358   -8.6568
    0.4618    0.4853    0.2993   -1.0208   -8.1617  -10.5911   -8.5331
    0.4470    0.4851    0.3038   -1.0167   -8.0910  -10.4183   -8.6353
    0.4259    0.4875    0.2851   -1.0265   -8.3763  -10.7380   -8.6126
    0.3888    0.4561    0.3333   -0.9311   -7.5289   -9.7459   -7.5303
    0.5275    0.4990    0.2995   -1.0501   -8.2153  -11.3577   -8.8322
    0.5043    0.4921    0.2615   -1.0348   -8.7508  -11.6733   -8.6993
    0.4746    0.4916    0.2920   -1.0325   -8.2875  -10.6382   -8.7103
    0.4197    0.4917    0.2868   -1.0341   -8.3663  -10.8002   -8.6066
    0.4150    0.4896    0.2991   -1.0301   -8.1821  -10.9657   -8.6611
    0.3959    0.4858    0.2819   -1.0242   -8.4181  -10.7571   -8.7057
    0.5125    0.4903    0.3191   -1.0419   -7.9327  -10.0090   -8.7145
    0.4939    0.4899    0.2967   -1.0324   -8.2205  -10.7254   -8.4256
    0.4452    0.4894    0.2905   -1.0317   -8.3082  -10.6943   -8.5699
    0.4316    0.4917    0.2670   -1.0336   -8.6625  -10.4390   -8.4382
    0.4068    0.4895    0.2880   -1.0307   -8.3416  -10.6848   -8.6292
    0.3976    0.4854    0.2758   -1.0230   -8.5076  -10.6382   -8.5864
    0.5242    0.4958    0.3265   -1.0471   -7.8462  -10.0247   -8.6299
    0.5124    0.4961    0.2840   -1.0382   -8.4154  -12.4846   -8.5510
    0.4949    0.4965    0.2751   -1.0430   -8.5581  -10.8860   -8.7957
    0.4902    0.4941    0.2572   -1.0327   -8.8154  -12.4042   -8.3894
    0.4677    0.4951    0.2859   -1.0361   -8.3835  -11.1444   -8.7312
    0.4396    0.4843    0.2815   -1.0143   -8.4050  -10.6202   -8.6021

];
end