%Aula 9
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
   'Ex1c'
   'Ex2'
   'Ex3a'
   'Ex3b'
   'Ex4a'
   'Ex4b'
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1a -------------------------------------------------------------------

exercise = 'Ex1a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  tesoura = im2double(imread("tesoura_org_template.png"));
  pa = im2double(imread("pa_org_template.png"));

  figure
  subplot(1,2,1)
  a = normxcorr2(tesoura,Z);
  [x,y] = find(a>0.9);
  imshow(a)
  hold on
  plot(y,x,'r.','MarkerSize',16)
  
  subplot(1,2,2)
  b = normxcorr2(pa,Z);
  [x,y] = find(b>0.9);
  imshow(b)
  hold on
  plot(y,x,'b.','MarkerSize',16)
end
%% Ex1b -------------------------------------------------------------------

exercise = 'Ex1b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  tesoura = im2double(imread("tesoura_org_template.png"));
  pa = im2double(imread("pa_org_template.png"));

  figure
  a = normxcorr2(tesoura,Z);
  [max_a,index] = max(a(:));
  imshow(Z)
  hold on
  [y, x] = ind2sub(size(a), index);
  coord_x1 = x - size(tesoura,2)/2;
  coord_y1 = y - size(tesoura,1)/2;

  rotulo_img = logical(bwlabel(Z));

  stats = regionprops(rotulo_img, 'centroid', 'BoundingBox');
  centroide = cat(1, stats.Centroid);

  distancias = sqrt((centroide(:,1) - coord_x1).^2 + (centroide(:,2) - coord_y1).^2);
  [menor_distancia, num_objeto] = min(distancias);
  centroide_menor_distancia = centroide(num_objeto,:);

  plot(centroide_menor_distancia(1), centroide_menor_distancia(2), 'b*', 'MarkerSize', 10);
  rectangle('Position', stats(num_objeto).BoundingBox, 'EdgeColor', 'magenta', 'LineWidth', 2);

end
%% Ex1c -------------------------------------------------------------------

exercise = 'Ex1c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  tesoura = im2double(imread("tesoura_org_template.png"));
  pa = im2double(imread("pa_org_template.png"));

  figure
  a = normxcorr2(tesoura,Z);
  [max_a,index] = max(a(:));
  imshow(Z)
  hold on
  [y, x] = ind2sub(size(a), index);
  coord_x1 = x - size(tesoura,2)/2;
  coord_y1 = y - size(tesoura,1)/2;

  rotulo_img = logical(bwlabel(Z));

  stats = regionprops(rotulo_img, 'centroid', 'BoundingBox');
  centroide = cat(1, stats.Centroid);

  distancias = sqrt((centroide(:,1) - coord_x1).^2 + (centroide(:,2) - coord_y1).^2);

  [menor_distancia, num_objeto] = min(distancias);
  centroide_menor_distancia = centroide(num_objeto,:);

  plot(centroide_menor_distancia(1), centroide_menor_distancia(2), 'b*', 'MarkerSize', 10);
  rectangle('Position', stats(num_objeto).BoundingBox, 'EdgeColor', 'magenta', 'LineWidth', 2);

  disp(num_objeto)
end
% %% Ex2 -------------------------------------------------------------------
% 
% exercise = 'Ex2'; % Define the name of the current exercise
% if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
%   disp(['Executing ' exercise ':'])
%   clearvars -except list_of_exercises % Delete all previously declared vars
%   close all
% 
%   Z = im2double(imread("TP2_img_01_01b.png"));
%   pa = im2double(imread("pa_org_template.png"));
%   pa_rotated = imrotate(pa, 45);
%   threshold = 0.8;
% 
%   figure
%   a = normxcorr2(pa,Z);
%   [y,x] = find(a>threshold);
%   coord_x1 = x - size(pa,2)/2;
%   coord_y1 = y - size(pa,1)/2;
% 
%   rotulo_img = logical(bwlabel(Z));
% 
%   stats = regionprops(rotulo_img, 'centroid', 'BoundingBox','Orientation','Image');
%   centroide = cat(1, stats.Centroid);
% 
%   distancias = sqrt((centroide(:,1) - coord_x1).^2 + (centroide(:,2) - coord_y1).^2);
% 
%   [menor_distancia, num_objeto] = min(distancias);
%   centroide_menor_distancia = centroide(num_objeto,:);
% 
%   plot(centroide_menor_distancia(1), centroide_menor_distancia(2), 'b*', 'MarkerSize', 10);
%   rectangle('Position', stats(num_objeto).BoundingBox, 'EdgeColor', 'magenta', 'LineWidth', 2);
% end
%% Ex3a -------------------------------------------------------------------

exercise = 'Ex3a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  %Z = imbinarize(Z,'adaptive');
  Z = bwareaopen(Z,200);
  imshow(Z)

  stats = regionprops(Z, 'Area', 'Centroid', 'Eccentricity', 'Solidity', 'Perimeter', 'Circularity', 'BoundingBox');

  Patts = [[stats.Circularity]' [stats.Solidity]' [stats.Eccentricity]'];

  for n=1:size(Patts,1)
      mstr = num2str(Patts(n,:)',3);
      text(stats(n).Centroid(1)+10,stats(n).Centroid(2),mstr,'Color',[1 1 0],'BackgroundColor',[0 0 0])
      text(stats(n).Centroid(1)-10,stats(n).Centroid(2),num2str(n),'Color',[0 1 0],'BackgroundColor',[0 0 0])
  end
  
end
%% Ex3b -------------------------------------------------------------------

exercise = 'Ex3b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  %Z = imbinarize(Z,'adaptive');
  Z = bwareaopen(Z,200);
  imshow(Z)

  stats = regionprops(Z, 'Area', 'Centroid', 'Eccentricity', 'Solidity', 'Perimeter', 'Circularity', 'BoundingBox');

  Patts = [[stats.Circularity]' [stats.Solidity]' [stats.Eccentricity]'];

  for n=1:size(Patts,1)
      mstr = num2str(Patts(n,:)',3);
      text(stats(n).Centroid(1)+10,stats(n).Centroid(2),mstr,'Color',[1 1 0],'BackgroundColor',[0 0 0])
      text(stats(n).Centroid(1)-10,stats(n).Centroid(2),num2str(n),'Color',[0 1 0],'BackgroundColor',[0 0 0])
  end

  pA = [0.2163 0.5354 0.9871];
  pB = [0.4121 0.8289 0.9712];
  dA = zeros(size(Patts,1),1); dB=dA;
  for n=1:size(Patts,1)
      dA(n) = norm(Patts(n,:)-pA);
      dB(n) = norm(Patts(n,:)-pB);
  end
  disp("dA")
  disp(dA(1))
  disp(dA(12))
  disp(dA(18))
  disp("dB")
  disp(dB(4))
  disp(dB(6))
end
%% Ex4a -------------------------------------------------------------------

exercise = 'Ex4a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  %Z = imbinarize(Z,'adaptive');
  Z = bwareaopen(Z,200);
  imshow(Z)

  stats = regionprops(Z, 'Area', 'Centroid', 'Eccentricity', 'Solidity', 'Perimeter', 'Circularity', 'BoundingBox');

  Patts = [[stats.Circularity]' [stats.Solidity]' [stats.Eccentricity]'];

  for n=1:size(Patts,1)
      mstr = num2str(Patts(n,:)',3);
      text(stats(n).Centroid(1)+10,stats(n).Centroid(2),mstr,'Color',[1 1 0],'BackgroundColor',[0 0 0])
      text(stats(n).Centroid(1)-10,stats(n).Centroid(2),num2str(n),'Color',[0 1 0],'BackgroundColor',[0 0 0])
  end
  
  Patts_A = Patts([1 12 14 16 17 18],:);
  Patts_B = Patts([4 6 19],:);

  dA = zeros(size(Patts,1),1); dB=dA;
  for n=1:size(Patts,1)
      dA(n) = norm(Patts(n,:)-Patts_A);
      dB(n) = norm(Patts(n,:)-Patts_B);
  end
  disp("dA")
  disp(dA(1))
  disp(dA(12))
  disp(dA(18))
  disp("dB")
  disp(dB(4))
  disp(dB(6))
  
end
%% Ex4b -------------------------------------------------------------------

exercise = 'Ex4b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("TP2_img_01_01b.png"));
  %Z = imbinarize(Z,'adaptive');
  Z = bwareaopen(Z,200);
  imshow(Z)

  stats = regionprops(Z, 'Area', 'Centroid', 'Eccentricity', 'Solidity', 'Perimeter', 'Circularity', 'BoundingBox');

  Patts = [[stats.Circularity]' [stats.Solidity]' [stats.Eccentricity]'];

  for n=1:size(Patts,1)
      mstr = num2str(Patts(n,:)',3);
      text(stats(n).Centroid(1)+10,stats(n).Centroid(2),mstr,'Color',[1 1 0],'BackgroundColor',[0 0 0])
      text(stats(n).Centroid(1)-10,stats(n).Centroid(2),num2str(n),'Color',[0 1 0],'BackgroundColor',[0 0 0])
  end

  Patts_A = Patts([1 12 14 16 17 18],:);
  Patts_B = Patts([4 6 19],:);

  dA = mahal(Patts,Patts_A);
  dB = mahal(Patts,Patts_B);
  dA = dA./max(dA);
  dB = dB./max(dB);
end