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
Z_hsv_2 = autobin(imadjust(Z_hsv(:,:,2)));

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
L_2 = and(L,Z_hsv_2);
imshow(L)
features = regionprops(logical(L),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
features_2 = regionprops(logical(L_2),'Circularity','Centroid','Solidity','Area','BoundingBox','Eccentricity','Image');
% n = size([features.Circularity]',1);
% entropy_array = zeros(n,1);
% for ent = 1:n
%     entropy_array(ent) = entropy(features(ent).Image);
% end
Patts = [[features.Circularity]' [features.Solidity]' [features.Eccentricity]' [features.Eccentricity]'];
Patts_2 = [[features_2.Circularity]' [features_2.Solidity]' [features_2.Eccentricity]' [features_2.Eccentricity]'];
for inter = 1:size(Patts,1)
invariant_moments = feature_vec(features(inter).image);
Patts(inter,4) = invariant_moments(1);
end
for inter = 1:size(Patts_2,1)
invariant_moments = feature_vec(features_2(inter).image);
Patts_2(inter,4) = invariant_moments(1);
end
%Patts = [[features.Circularity]' [features.Solidity]' [features.Eccentricity]' entropy_array];
noInfRows = ~any(isinf(Patts), 2);
Patts = Patts(noInfRows,:);

noInfRows_2 = ~any(isinf(Patts_2), 2);
Patts_2 = Patts_2(noInfRows_2,:);

%
boxes = zeros(size([features.Circularity],2),4);
for b=1:size([features.Circularity],2)
    boxes(b,:) = features(b).BoundingBox;
end
boxes = boxes(noInfRows,:);
%

d_bio = nan(size(Patts,1),size(Patts,2));
d_corrosive = d_bio;
d_elet = d_bio;
d_explosive = d_bio;
d_info = d_bio; % info_2
d_laser = d_bio;
d_lixo = d_bio; %lixo_2
d_phone = d_bio; %phone_3
d_cigar = d_bio;
d_toxic = d_bio;
d_wc = d_bio; %wc_3
d_wifi = d_bio;%wifi_1

%bio
load parameters.mat

for iter=1:3
  d_bio(:,iter) = mahal(Patts,s.bio(:,:,iter));
  d_bio(:,iter) = d_bio(:,iter)./max(d_bio(:,iter));
  
  d_corrosive(:,iter) = mahal(Patts,s.corrosive(:,:,iter));
  d_corrosive(:,iter) = d_corrosive(:,iter)./max(d_corrosive(:,iter));

  d_elet(:,iter) = mahal(Patts,s.eletric(:,:,iter));
  d_elet(:,iter) = d_elet(:,iter)./max(d_elet(:,iter));
   
  d_explosive(:,iter) = mahal(Patts,s.explosive(:,:,iter));
  d_explosive(:,iter) = d_explosive(:,iter)./max(d_explosive(:,iter));

  if iter ~=1
      d_wifi(:,iter) = mahal(Patts,s.wifi(:,:,iter));
      d_wifi(:,iter) = d_wifi(:,iter)./max(d_wifi(:,iter));
  end
  if iter~=2
      d_info(:,iter) = mahal(Patts,s.info(:,:,iter));
      d_info(:,iter) = d_info(:,iter)./max(d_info(:,iter));
      d_lixo(:,iter) = mahal(Patts,s.lixo(:,:,iter));
      d_lixo(:,iter) = d_lixo(:,iter)./max(d_lixo(:,iter));
  end
  if iter~=3
      d_phone(:,iter) = mahal(Patts,s.phone(:,:,iter));
      d_phone(:,iter) = d_phone(:,iter)./max(d_phone(:,iter));
      d_wc(:,iter) = mahal(Patts,s.wc(:,:,iter));
      d_wc(:,iter) = d_wc(:,iter)./max(d_wc(:,iter));
  end
%   d_info(:,iter) = mahal(Patts,s.info(:,:,iter));
%   d_info(:,iter) = d_info(:,iter)./max(d_info(:,iter));

  d_laser(:,iter) = mahal(Patts,s.laser(:,:,iter));
  d_laser(:,iter) = d_laser(:,iter)./max(d_laser(:,iter));
    
%   d_lixo(:,iter) = mahal([Patts(:,1) Patts_2(:,2) Patts(:,3)],s.lixo(:,:,iter));
%   d_lixo(:,iter) = d_lixo(:,iter)./max(d_lixo(:,iter));
%    
%   d_phone(:,iter) = mahal([Patts(:,1) Patts(:,2) Patts_2(:,3)],s.phone(:,:,iter));
%   d_phone(:,iter) = d_phone(:,iter)./max(d_phone(:,iter));

  d_cigar(:,iter) = mahal(Patts,s.cigar(:,:,iter));
  d_cigar(:,iter) = d_cigar(:,iter)./max(d_cigar(:,iter));

  d_toxic(:,iter) = mahal(Patts,s.toxic(:,:,iter));
  d_toxic(:,iter) = d_toxic(:,iter)./max(d_toxic(:,iter));

%   d_wc(:,iter) = mahal([Patts(:,1) Patts(:,2) Patts_2(:,3)],s.wc(:,:,iter));
%   d_wc(:,iter) = d_wc(:,iter)./max(d_wc(:,iter));
% 
%   d_wifi(:,iter) = mahal([Patts_2(:,1) Patts(:,2) Patts(:,3)],s.wifi(:,:,iter));
%   d_wifi(:,iter) = d_wifi(:,iter)./max(d_wifi(:,iter));
end
dist = 50;
for iter=1:size(Patts,1)
    for counter=1:size(Patts_2,1)
        norm([ [features(iter).Centroid] ; [features_2(counter).Centroid] ]);
        if dist > norm([ [features(iter).Centroid] ; [features_2(counter).Centroid] ])
            d_info(iter,2) = mahal(Patts_2,s.info(:,:,2));
        end
    end
end
d_info(:,2) = d_info(:,2)./max(d_info(:,2))

% d_info_2 = mahal(Patts_2,s.info(:,:,2));
% d_info_2 = d_info_2./max(d_info_2);
% d_info(:,2) = mahal(Patts_2,s.info(:,:,2));
% d_info(:,2) = d_info(:,iter)./max(d_info(:,2));
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
cor = sum(any(d_corrosive   <0.0001,2));
elec = sum(any(d_elet       <0.001,2));
explo = sum(any(d_explosive <0.001,2));
info = sum(any(d_info       <0.0001,2));
laser = sum(any(d_laser     <0.0001,2));
lit = sum(any(d_lixo        <0.001,2));
phone = sum(any(d_phone     <0.001,2));
smoking = sum(any(d_cigar   <0.0001,2));
tox = sum(any(d_toxic       <0.001,2));
wc = sum(any(d_wc           <0.001,2));
wifi = sum(any(d_wifi       <0.001,2));
end
function Z = autobin(A)
  mask = graythresh(A);
  Z = imbinarize(A,mask);
  if mask < mean(Z(:))
      Z = 1-Z;
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