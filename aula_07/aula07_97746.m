%Aula 7
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
   'Ex1'
   'Ex2'
   'Ex3'
   'Ex4'
   'Ex5'
   'Ex6'
   'Ex7'
   'Ex8'
   'Ex9'
   'Ex10'
   'Ex11'
   'Ex12'
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1 -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("Manycoins.png"));

  A = imbinarize(Z,0.18);
  subplot(2,3,1)
  imshow(A)
  title('A')

  X = bwmorph(A,'erode',3);
  subplot(2,3,2)
  imshow(X)
  title('X')

  Y = bwmorph(A,'dilate',3);
  subplot(2,3,3)
  imshow(Y)
  title('Y')

  X_i = bwmorph(X,'dilate',3);
  subplot(2,3,4)
  imshow(X_i)
  title('Dilate after erode')

  Y_i = bwmorph(Y,'erode',3);
  subplot(2,3,5)
  imshow(Y_i)
  title('Erode after dilate')
end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("Manycoins.png"));
  A = imbinarize(Z,0.18);
 
  BW = A;
  tot = sum(sum(A));
  part = tot;
  n=0;
  frac=3;
  while(part>tot/frac) %contam-se o numero de pixels nesta iteracao, se for maior que a fracao que se deseja manter entao repete-se
      BW = bwmorph(BW,'erode');
      part = sum(sum(BW));
      n = n+1;
  end
  subplot(1,3,1)
  imshow(A)
  title(sprintf('Original image\n %d pixels',tot));

  subplot(1,3,2)
  imshow(BW)
  title(sprintf('Eroded %d times\n %d pixels',n,nnz(BW)));

  BW = bwmorph(A,'erode',n-1);
  subplot(1,3,3)
  imshow(BW)
  title(sprintf('Eroded %d times\n %d pixels',n-1,nnz(BW)));
end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("Manycoins.png"));
  A = imbinarize(Z,0.18);

  tot = sum(sum(A));
  subplot(1,3,1)
  imshow(A)
  title('original a 18%')
  str=sprintf('Total pixels: %d',tot);
  xlabel(str)

  subplot(1,3,2)
  BW = bwmorph(A,'close');
  imshow(BW)
  title('close')
  str=sprintf('Total pixels: %d',sum(sum(and(BW,A))));
  xlabel(str)
 
  subplot(1,3,3)
  BW = bwmorph(A,'open');
  imshow(BW)
  title('open')
  str=sprintf('Total pixels: %d',sum(sum(and(BW,A))));
  xlabel(str)
end
%% Ex4 -------------------------------------------------------------------

exercise = 'Ex4'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A = false(300);
  A(10:20:end,10:20:end) = 1;
  subplot(1,2,1)
  imshow(A)

  SE = false(10);
  SE(end,:) = 1;

  B = imdilate(A,SE);
  subplot(1,2,2)
  imshow(B)
end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A = rand(300,300)>0.9995;
  subplot(1,2,1)
  imshow(A)

  SE = strel('diamond',8);
  B = imdilate(A,SE);
  subplot(1,2,2)
  imshow(B)
end
%% Ex6 -------------------------------------------------------------------

exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("pcb2.png"));
  subplot(1,2,1)
  imshow(Z)

  Z = rgb2gray(Z);
  bin = bwmorph(Z,'skel',Inf);
  subplot(1,2,2)
  imshow(bin)

  SE1 = ones(3);
  SE1([1 end],[1 end]) = -1;
  BW2 = bwhitmiss(bin,SE1);
  BW3 = zeros(4,size(Z,1),size(Z,2));
  for i=1:4
    SE2 = ones(3);
    SE2(2:end,[1 end]) = -1;
    SE2 = rot90(SE2,i);
    BW3(i,:,:) = bwhitmiss(bin,SE2);
  end
  hold on
  for i=1:size(Z,1)
      for j=1:size(Z,2)
          if any(BW3(:,i,j)==1)
              plot(j,i,'r*')
          end
          if BW2(i,j)
              plot(j,i,'g*')
          end
      end
  end
% nao deteta o que esta sublinhado entre outros porque o esqueleto nao tem
% o T com aquela forma
end
%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all
  Z = im2double(imread("pcb.png"));
  subplot(1,2,1)
  imshow(Z)

  Z = rgb2gray(Z);

  bin = bwmorph(Z,'shrink',inf);
  ppi = filter2([1 1 1; 1 -8 1; 1 1 1], bin);
  mask = (ppi==-8);
  SE = strel("disk",9);
  dilated = imdilate(mask,SE);
  subplot(1,2,2)
  imshow(dilated)
 
end
%% Ex8 -------------------------------------------------------------------

exercise = 'Ex8'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all


end
%% Ex9 -------------------------------------------------------------------

exercise = 'Ex9'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("pcb_holes.png"));
  Z = rgb2gray(Z);
  subplot(1,2,1)
  imshow(Z)

  bin = bwmorph(1-Z,'shrink',inf);
  ppi = filter2([1 1 1; 1 -8 1; 1 1 1], bin);

  furos = sum(sum(ppi==-8));
  disp(['furos: ',num2str(furos)])

  subplot(1,2,2)
  imshow(Z)
  hold on
  for i=1:size(Z,1)
      for j=1:size(Z,2)
          if ppi(i,j) == -8
              plot(j,i,'r*')
          end
      end
  end
end
%% Ex10 -------------------------------------------------------------------

exercise = 'Ex10'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("anilhas.png"));
  subplot(1,3,1)
  imshow(Z)
  subplot(1,3,2)
  BW = imbinarize(Z);
  imshow(BW)
  BW2 = imfill(1-BW,'holes');
  subplot(1,3,3)
  imshow(BW2)
  title('Filled Image')
end
%% Ex11 -------------------------------------------------------------------

exercise = 'Ex11'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("anilhas.png"));
  subplot(1,3,1)
  imshow(Z)
  BW = ~imbinarize(Z);
  %imshow(BW)

  bin = bwmorph(BW,'shrink',inf);
  ppi = filter2([1 1 1; 1 -8 1; 1 1 1], bin);

  subplot(1,3,3)
  disp(['roscas: ',num2str(sum(sum(ppi==-8)))])
  bin_roscas = bin;
  bin_roscas(ppi==-8)=0;
  roscas = imreconstruct( bin_roscas,BW );
  imshow(roscas)

  subplot(1,3,2)
  bin_parafusos = bin;
  bin_parafusos(ppi~=-8)=0;
  parafusos = imreconstruct( bin_parafusos,BW );
  imshow(parafusos)
end
%% Ex12 -------------------------------------------------------------------

exercise = 'Ex12'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

 
end
