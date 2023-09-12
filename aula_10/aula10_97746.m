%Aula 10
%% TemplateCodeSVPI -------------------------------------------------------
% Name:       Vasco Costa
% Num. Mec:   97746
% Date:       2022/2023
%% Initial configurations
clc % Clear all text from command window
close all % Close all figures previously opened
clear % Clear previous environment variables
%addpath('../lib') % Update yout matlab path (the folder must exist)

list_of_exercises = { %Add, comment or uncomment at will
   'Ex1a'
   'Ex1b'
   'Ex2'
   'Ex3'
   'Ex4'
   'Ex5'
   'Ex6'
   'Ex7'
   'Ex8'
   'Ex9'
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1a -------------------------------------------------------------------

exercise = 'Ex1a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("artemoderna2.png"));

  subplot(2,4,1)
  imshow(Z)
  
  subplot(2,4,2)
  points_r = and(1-Z(:,:,2),1-Z(:,:,3));
  imshow(points_r)

  subplot(2,4,3)
  points_g = and(1-Z(:,:,1),1-Z(:,:,3));
  imshow(points_g)

  subplot(2,4,4)
  points_b = and(1-Z(:,:,1),1-Z(:,:,2));
  imshow(points_b)

  subplot(2,4,6)
  img = zeros(size(Z));
  img(:,:,1) = points_r;
  imshow(img)
  subplot(2,4,7)
  img = zeros(size(Z));
  img(:,:,2) = points_g;
  imshow(img)
  subplot(2,4,8)
  img = zeros(size(Z));
  img(:,:,3) = points_b;
  imshow(img)
end
%% Ex1b -------------------------------------------------------------------

exercise = 'Ex1b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("artemoderna2.png"));

  subplot(1,3,1)
  imshow(Z)

  subplot(1,3,2) 
  points = (Z(:,:,1) > 0.8) & (Z(:,:,2)>0.6) & (Z(:,:,3) < 0.2);
  imshow(points)

  subplot(1,3,3)
  yellow_only = Z;
  yellow_only(repmat(~points,[1 1 3])) = 1;
  imshow(yellow_only);

end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("mongolia.jpg"));
  figure
  imshow(Z)

  figure
  [cR,cG,cB,x] = rgbhist(Z);
  plot(x(:,1),cR,'r')
  hold on
  plot(x(:,2),cG,'g')
  plot(x(:,3),cB,'b')
end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("morangos.jpg"));
  figure
  imshow(Z)

  figure
  [cR,cG,cB,x] = rgbhist(Z);
  plot(x(:,1),cR,'r')
  hold on
  plot(x(:,2),cG,'g')
  plot(x(:,3),cB,'b')

  figure
  red_points = x(cG>cR);
  green_points = x(cR>cG);
  points = (Z(:,:,1) > 0.19) & (Z(:,:,2) < 0.1) & (Z(:,:,3) < 0.2);
  red_only = Z;
  red_only(repmat(~points,[1 1 3])) = 0;
  imshow(red_only)

  figure
  points = (Z(:,:,1) < 0.25) & (Z(:,:,2) > 0.1) & (Z(:,:,3) < 0.2);
  green_only = Z;
  green_only(repmat(~points,[1 1 3])) = 0;
  imshow(green_only)

end
%% Ex4 -------------------------------------------------------------------

exercise = 'Ex4'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('ArteModerna1.jpg'));
  subplot(1,3,1)
  imshow(Z)

  Z_hsv = rgb2hsv(Z);
  subplot(1,3,2)
  yellow_mask = (Z_hsv(:,:,1) >= 0.12) & (Z_hsv(:,:,1) <= 0.20) & (Z_hsv(:,:,2) > 0.5);
  bin = and(yellow_mask, Z_hsv(:,:,3)>0.01);
  imshow(bin)

  subplot(1,3,3)
  yellow_only = Z;
  yellow_only(repmat(~bin,[1 1 3])) = 0;
  imshow(yellow_only);

  % o formato jpeg efetua a remocao das frequencias mais altas da imagem
  % original o que resulta da perca de informacao mais detalhada. Esta
  % perda de informacao pode levar a que as cores sejam menos saturadas.
  % Adicionalmente, e possivel que sejam adicionados aterfactos em blocos
  % que resultam em decontinuidades nos valores do brilho de pixeis
  % adjacentes

  %Isto leva a que se utilizando o valor da saturacao, maior que 1% se
  %obtenha a imagem correta.
end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('ArteModerna1.jpg'));
  Z_hsv = rgb2hsv(Z);
  subplot(1,4,1)
  red_mask = ((Z_hsv(:,:,1) >= 0) & (Z_hsv(:,:,1) <= 0.1) | (Z_hsv(:,:,1) >= 0.9) & (Z_hsv(:,:,1) <= 1)) & (Z_hsv(:,:,2) > 0.5);
  bin = and(red_mask, Z_hsv(:,:,3)>0.01);
  red_only = Z;
  red_only(repmat(~bin,[1 1 3])) = 0;
  imshow(red_only);
  
  subplot(1,4,2)
  yellow_mask = (Z_hsv(:,:,1) >= 0.15) & (Z_hsv(:,:,1) <= 0.2) & (Z_hsv(:,:,2) > 0.5);
  bin = and(yellow_mask, Z_hsv(:,:,3)>0.01);
  yellow_only = Z;
  yellow_only(repmat(~bin,[1 1 3])) = 0;
  imshow(yellow_only);

  subplot(1,4,3)
  green_mask = (Z_hsv(:,:,1) >= 0.25) & (Z_hsv(:,:,1) <= 0.4) & (Z_hsv(:,:,2) > 0.5);
  bin = and(green_mask, Z_hsv(:,:,3)>0.01);
  green_only = Z;
  green_only(repmat(~bin,[1 1 3])) = 0;
  imshow(green_only);

  subplot(1,4,4)
  blue_mask = (Z_hsv(:,:,1) >= 0.55) & (Z_hsv(:,:,1) <= 0.75) & (Z_hsv(:,:,2) > 0.5);
  bin = and(blue_mask, Z_hsv(:,:,3)>0.01);
  blue_only = Z;
  blue_only(repmat(~bin,[1 1 3])) = 0;
  imshow(blue_only);
end
%% Ex6 -------------------------------------------------------------------
%opcional
exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

end
%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('feet2.jpg'));
  Z_hsv = rgb2hsv(Z);
  subplot(2,2,1)
  red_mask = ((Z_hsv(:,:,1) >= 0) & (Z_hsv(:,:,1) <= 0.1) | (Z_hsv(:,:,1) >= 0.9) & (Z_hsv(:,:,1) <= 1)) & (Z_hsv(:,:,2) > 0.4);
  bin = and(red_mask, Z_hsv(:,:,3)>0.01);
  bin = bwareaopen(bin,50);
  bin = bwmorph(bin,"erode");
  bin = bwmorph(bin,'dilate');
  red_only = Z;
  red_only(repmat(~bin,[1 1 3])) = 0;
  imshow(red_only);
  
  subplot(2,2,2)
  yellow_mask = (Z_hsv(:,:,1) >= 0.15) & (Z_hsv(:,:,1) <= 0.2) & (Z_hsv(:,:,2) > 0.4);
  bin = and(yellow_mask, Z_hsv(:,:,3)>0.01);
  bin = bwareaopen(bin,50);
  bin = bwmorph(bin,"erode");
  bin = bwmorph(bin,'dilate');
  yellow_only = Z;
  yellow_only(repmat(~bin,[1 1 3])) = 0;
  imshow(yellow_only);

  subplot(2,2,3)
  green_mask = (Z_hsv(:,:,1) >= 0.25) & (Z_hsv(:,:,1) <= 0.4) & (Z_hsv(:,:,2) > 0.4);
  bin = and(green_mask, Z_hsv(:,:,3)>0.01);
  bin = bwareaopen(bin,50);
  bin = bwmorph(bin,"erode");
  bin = bwmorph(bin,'dilate');
  green_only = Z;
  green_only(repmat(~bin,[1 1 3])) = 0;
  imshow(green_only);

  subplot(2,2,4)
  blue_mask = (Z_hsv(:,:,1) >= 0.55) & (Z_hsv(:,:,1) <= 0.75) & (Z_hsv(:,:,2) > 0.4);
  bin = and(blue_mask, Z_hsv(:,:,3)>0.01);
  bin = bwareaopen(bin,50);
  bin = bwmorph(bin,"erode");
  bin = bwmorph(bin,'dilate');
  blue_only = Z;
  blue_only(repmat(~bin,[1 1 3])) = 0;
  imshow(blue_only);
end
%% Ex8 -------------------------------------------------------------------

exercise = 'Ex8'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('morangos.jpg'));
  Z_hsv = rgb2hsv(Z);
  red_mask = ((Z_hsv(:,:,1) >= 0) & (Z_hsv(:,:,1) <= 0.1) | (Z_hsv(:,:,1) >= 0.9) & (Z_hsv(:,:,1) <= 1)) & (Z_hsv(:,:,2) > 0.5);
  bin = and(red_mask, Z_hsv(:,:,3)>0.01);
  bin = bwareaopen(bin,100);
  bin = bwmorph(bin,'close');
  bin = imfill(bin,"holes");
  %bin = bwmorph(bin,"erode");
  %bin = bwmorph(bin,'dilate');
  imshow(bin)
  %red_only = Z;
  %red_only(repmat(~bin,[1 1 3])) = 0;
  %imshow(red_only);
end
%% Ex9 -------------------------------------------------------------------

exercise = 'Ex9'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  option = 3;
  for i=1:7
      if i==1
          Z = im2double(imread('morangos.jpg'));
      else
          Z = im2double(imread(['morangos',num2str(i),'.jpg']));
      end
      Z_hsv = rgb2hsv(Z);
      red_mask = ((Z_hsv(:,:,1) >= 0) & (Z_hsv(:,:,1) <= 0.1) | (Z_hsv(:,:,1) >= 0.9) & (Z_hsv(:,:,1) <= 1)) & (Z_hsv(:,:,2) > 0.65) & (Z_hsv(:,:,2) < 0.81);
      bin = and(red_mask, Z_hsv(:,:,3)>0.1);
      bin = bwareaopen(bin,100);
      bin = bwmorph(bin,'close');
      bin = imfill(bin,"holes");
      subplot(3,3,i+2)
      if option==1
          imshow(bin)
      elseif option==2
          red_only = Z;
          red_only(repmat(~bin,[1 1 3])) = 0;
          imshow(red_only);
      else
          stats = regionprops(bin, 'Centroid','EquivDiameter');
          centroids = cat(1, stats.Centroid);
          imshow(Z)
          hold on
          viscircles(centroids, [stats.EquivDiameter], 'Color', 'b');
      end
  end

end