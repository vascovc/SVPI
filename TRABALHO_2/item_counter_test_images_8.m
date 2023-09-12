clear all
close all
clc

nn = 97746;

dificuldade = 1; % 1-hardest; 2-mid; 3 - easier

% lista = dir("Imagens de Referência/frame/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("Imagens de Referência/noframe/svpi2023_TP2_img_*"+dificuldade+"_*.png");
lista = dir("Seq39x/imagens/svpi2023_TP2_img_*"+dificuldade+"_*.png");
% lista = dir("../svpi2023_TP2_img_*"+dificuldade+"_*.png");

num_files = size(lista,1);
matrix = zeros(num_files,18);

i = 8;
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
% figure
% imshow(Z)
Z_hsv_2 = autobin(imadjust(Z_hsv(:,:,2)));
Z_hsv_1 = autobin(imadjust(Z_hsv(:,:,1)));
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
Z_show = bsxfun(@times,image,cast(N, 'like', image));
asfas = rgb2hsv(Z_show);
Z_hsv_2 = autobin(imadjust(asfas(:,:,2)));
Z_hsv_1 = autobin(imadjust(asfas(:,:,1)));
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

    Z = triangles | squares |circles;
    Z_hsv_2 = and((TRI | SQU |CIR),Z_hsv_2);
    Z_hsv_3 = and((TRI | SQU |CIR),Z_hsv_1);

    figure
    imshow(Z)
    title("Canal 3")
    figure
    imshow(Z_hsv_2)
    title("Canal 2")
    figure
    imshow(Z_hsv_3)
    title("Canal 1")
%   
%     imshow(Z)
%     figure
%     
%     %Z = bwmorph(Z,'close');
%     %Z = bwmorph(Z,'open');
%     imshow(Z)
    figure
    mask = bwareaopen(Z, 200);
    [L,num] = bwlabel(mask);
    features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
    for counter=1:num
        imshow(L==counter)
        x = features(counter).Centroid(1);
        y = features(counter).Centroid(2);
        text(x,y, {['Obj ' num2str(counter)], num2str(features(counter).Solidity)}, 'Color','r')
        %pause
    end
    title("Canal 3")

    figure
    mask = bwareaopen(Z_hsv_2, 200);
    [L,num] = bwlabel(mask);
    features_2 = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
    for counter=1:num
        imshow(L==counter)
        x = features_2(counter).Centroid(1);
        y = features_2(counter).Centroid(2);
        text(x,y, {['Obj ' num2str(counter)], num2str(features_2(counter).Solidity)}, 'Color','r')
        %pause
    end
    title("Canal 2")

    figure
    mask = bwareaopen(Z_hsv_3, 200);
    [L,num] = bwlabel(mask);
    features_3 = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
    for counter=1:num
        imshow(L==counter)
        x = features_3(counter).Centroid(1);
        y = features_3(counter).Centroid(2);
        text(x,y, {['Obj ' num2str(counter)], num2str(features_3(counter).Solidity)}, 'Color','r')
        %pause
    end
    title("Canal 1")
    %close all

    % corrosive canal 3
    info_2 = 1;


    load parameters_advanced.mat
    %s.lixo_extra(:,:,2) = get_parameters(features(lixo_1).Image);
    s.info_2_extra(:,:,2) = get_parameters(features_2(info_2).Image);

    %s.corrosive_3_extra(:,:,1) = get_parameters(features(corr_1).Image);


    save('parameters_advanced.mat',"s")


close all
function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
  end
end
function parameters = get_parameters(obj)
% figure
% imshow(obj)

rotate_angles = linspace(0,359,15);
scale_factors = linspace(0.4,1.5,6);
parameters = nan(length(rotate_angles)*length(scale_factors),7);
%parameters = zeros(length(rotate_angles)*length(scale_factors),3);
count = 1;
for r = rotate_angles
    for s = scale_factors
        rotated_obj = imrotate(obj, r);
        scaled_obj = imresize(rotated_obj, s);
        all = regionprops(scaled_obj,'All');
        [~, idx] = max([all.Area]);
        invariant_moments = feature_vec(all(idx).Image);
        parameters(count,:) = [all(idx).Circularity all(idx).Solidity all(idx).Eccentricity log(invariant_moments(1)) log(invariant_moments(2)) log(invariant_moments(3)) log(invariant_moments(4))];
        if any(isinf(parameters(count,:)))
            parameters(count,:) = nan(1,7);
        end
        %parameters(count,:) = [all(idx).Circularity all(idx).Solidity all(idx).Eccentricity];
        count = count+1;
    end
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