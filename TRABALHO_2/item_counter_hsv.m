clear all
close all
clc
tic
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

    % com frame
    lixo_1 = 6;lixo_3 = 30;
    bio_1 = 8; bio_2 = 18; bio_3 = 29;
    explosive_1 = 9; explosive_2 = 19; explosive_3 = 31;
    toxic_1 = 10; toxic_2 = 20; toxic_3 = 27;
    phone_1 = 37; phone_2 = 49;
    wc_1 = 39; wc_2 = 45;
    info_1 = 40;
    corrosive_1 = 41; corrosive_2 = 50; corrosive_3 = 58;
    cigar_1 = 66; cigar_2 = 78; cigar_3 = 86;
    eletric_1 = 70;eletric_2 = 81; eletric_3 = 91;
    laser_1 = 71; laser_2 = 80; laser_3 = 92;
    wifi_2 = 79; wifi_3 = 87;


    % canal 2
    lixo_2 = 11;
    info_2 = 21;
    phone_3 = 25;
    wc_3 = 28;
    wifi_1 = 36;

    %canal 1
    info_3 = 9;

    s = struct;
    %bio  
    s.bio(:,:,1) = get_parameters(features(bio_1).Image);
    s.bio(:,:,2) = get_parameters(features(bio_2).Image);
    s.bio(:,:,3) = get_parameters(features(bio_3).Image);

    %corrosive
    s.corrosive(:,:,1) = get_parameters(features(corrosive_1).Image);
    s.corrosive(:,:,2) = get_parameters(features(corrosive_2).Image);
    s.corrosive(:,:,3) = get_parameters(imresize(features(corrosive_3).Image,1.5));

    %electric
    s.eletric(:,:,1) = get_parameters(features(eletric_1).Image);
    s.eletric(:,:,2) = get_parameters(features(eletric_2).Image);
    s.eletric(:,:,3) = get_parameters(features(eletric_3).Image);

    %explosive
    s.explosive(:,:,1) = get_parameters(features(explosive_1).Image);
    s.explosive(:,:,2) = get_parameters(features(explosive_2).Image);
    s.explosive(:,:,3) = get_parameters(features(explosive_3).Image);

    %info
    s.info(:,:,1) = get_parameters(features(info_1).Image);
    s.info(:,:,2) = get_parameters(features_2(info_2).Image);
    s.info(:,:,3) = get_parameters(features_3(info_3).Image);

    %laser
    s.laser(:,:,1) = get_parameters(features(laser_1).Image);
    s.laser(:,:,2) = get_parameters(features(laser_2).Image);
    s.laser(:,:,3) = get_parameters(features(laser_3).Image);

    %lixo
    s.lixo(:,:,1) = get_parameters(features(lixo_1).Image);
    s.lixo(:,:,2) = get_parameters(features_2(lixo_2).Image);
    s.lixo(:,:,3) = get_parameters(features(lixo_3).Image);

    %phone
    s.phone(:,:,1) = get_parameters(features(phone_1).Image);
    s.phone(:,:,2) = get_parameters(features(phone_2).Image);
    s.phone(:,:,3) = get_parameters(features_2(phone_3).Image);

    %cigar
    s.cigar(:,:,1) = get_parameters(features(cigar_1).Image);
    s.cigar(:,:,2) = get_parameters(features(cigar_2).Image);
    s.cigar(:,:,3) = get_parameters(features(cigar_3).Image);
    
    %toxic
    s.toxic(:,:,1) = get_parameters(features(toxic_1).Image);
    s.toxic(:,:,2) = get_parameters(features(toxic_2).Image);
    s.toxic(:,:,3) = get_parameters(features(toxic_3).Image);

    %wc
    s.wc(:,:,1) = get_parameters(features(wc_1).Image);
    s.wc(:,:,2) = get_parameters(features(wc_2).Image);
    s.wc(:,:,3) = get_parameters(features_2(wc_3).Image);
    
    %wifi
    s.wifi(:,:,1) = get_parameters(features_2(wifi_1).Image);
    s.wifi(:,:,2) = get_parameters(features(wifi_2).Image);
    s.wifi(:,:,3) = get_parameters(features(wifi_3).Image);
    save('parameters.mat',"s")
end
toc
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