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

i = 9;
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
    
    Z_hsv_all = rgb2hsv(image);
    Z_hsv = Z_hsv_all(:,:,3);
    Z_hsv = imadjust(Z_hsv);
    Z_hsv = autobin(Z_hsv);
    Z_hsv_2 = autobin(imadjust(Z_hsv_all(:,:,2)));
    Z_hsv_3 = autobin(imadjust(Z_hsv_all(:,:,1)));
    Z = Z_hsv;
    %Z = imbinarize(Z);
    
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

    %lixo 
    lixo_1 = 8;
    % wifi
    wifi_1 = 29;

    load parameters.mat

    s.lixo_extra(:,:,1) = get_parameters(features(lixo_1).Image);
    s.wifi_extra(:,:,1) = get_parameters(features(wifi_1).Image);

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