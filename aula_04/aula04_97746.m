%Aula 4
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
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1 -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  cols = 200;
  lins = 200;
  dx = 50;
  dy = 80;
  Z = zeros(cols,lins);
  
  Z(cols/2 - dx/2:cols/2+dx/2 , lins/2-dy/2:lins/2+dy/2) = 1;

  figure(1)
  subplot(1,4,1)
  imshow(Z)
  title('A');
  B = imnoise(Z,'salt & pepper',0.05);
  subplot(1,4,2)
  imshow(B)
  title('B');

  C = filter2(ones(3,3)/9,B);
  subplot(1,4,3)
  imshow(C)
  title('C');
  D = medfilt2(B,[3,3]);
  subplot(1,4,4)
  imshow(D)
  title('D');
end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  cols = 200;
  lins = 200;
  dx = 50;
  dy = 80;
  Z = zeros(cols,lins);
  rng(0)
  Z(cols/2 - dx/2:cols/2+dx/2 , lins/2-dy/2:lins/2+dy/2) = 1;
  figure(1)
  subplot(1,3,1)
  imshow(Z)
  title('A');
  B = imnoise(Z,'salt & pepper',0.01);
  subplot(1,3,2)
  imshow(B)
  title('B');
  F = [-1 -1 -1; -1 8 -1; -1 -1 -1];
  C = filter2(F,B);
  subplot(1,3,3)
  idx_pos = (C==8);
  idx_neg = (C==-8);
  idx = idx_neg+idx_pos;
  imshow(idx)
  
  number_of_ones = sum(idx(:));
  disp(['Number of points: ', num2str(number_of_ones)])
end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  cols = 200;
  lins = 200;
  dx = 50;
  dy = 80;
  A = zeros(cols,lins);
  rng(0)
  A(cols/2 - dx/2:cols/2+dx/2 , lins/2-dy/2:lins/2+dy/2) = 1;
  figure(1)
  subplot(1,3,1)
  imshow(A)
  title('A');

  T=imnoise(A,'salt & pepper',0.005);
  F = zeros(3,3,4);
  F(:,:,1) = [1 1 1; 1 -8 1; 1 1 1];
  F(:,:,2) = [1 2 1; 2 -12 2; 1 2 1];
  F(:,:,3) = [-1 1 -1; 1 4 1; -1 1 -1];
  F(:,:,4) = [1 2 3; 4 -100 5; 6 7 8];

  %W = [-8 -12 4 -100];
  %NW = [8 12 0 36];
  W = [F(2,2,1) F(2,2,2) F(2,2,3) F(2,2,4)];
  NW = [sum(sum(F(:,:,1)))-W(1) sum(sum(F(:,:,2)))-W(2) sum(sum(F(:,:,3)))-W(3) sum(sum(F(:,:,4)))-W(4)];
  for n=1:4
      X = filter2(F(:,:,n),T);
      ZW = (X == W(n));
      ZB = (X == NW(n));

      subplot(2,4,n), imshow(ZW) %nnz -> number of nonzero matrix elements
      str = sprintf('Total %d',nnz(ZW));xlabel(str)

      subplot(2,4,4+n), imshow(ZB)
      str = sprintf('Total %d',nnz(ZB));xlabel(str)
      
  end
end
%% Ex4 -------------------------------------------------------------------

exercise = 'Ex4'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];

subplot(1,3,1)
imshow(A)
title('A')

F = [0 1 0
     1 4 1
     0 1 0];
A2 = filter2(F,A);
subplot(1,3,2)
B = (A2==7);
imshow(B)

subplot(1,3,3)
imshow(A)
hold on
for i=1:length(B)
    for j=1:length(B)
        if(B(j,i))
            plot(i,j,'b*')
        end
    end
end
end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all
A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];


end
%% Ex6 -------------------------------------------------------------------

% e possivel usar outros filtros e caso se usasse
% F = [0 1 0; 1 1 1; 0 1 0] dava para ambos os casos em que se teria 
% que alterar o valor a visualizar
exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];

  subplot(1,3,1)
  imshow(A)
  title('A')

  F =[0 -1 0
      -1 4 -1
      0 -1 0];
  A2 = filter2(F,A);
  subplot(1,3,2)
  B = ( A2 == 3 );
  B(1,:)=0;B(end,:)=0;
  B(:,1)=0;B(:,end)=0;
  imshow(B)

  subplot(1,3,3)
  imshow(A)
  hold on
    for i=1:length(B)
        for j=1:length(B)
            if(B(j,i))
                plot(i,j,'r*')
            end
        end
    end
end

%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  for i=1:4
      figure(i)
    A=[
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0
0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0
0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0
0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0
0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0
0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0
0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0
0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0
0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0
0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0
0 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0
0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 0
0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];
A =rot90(A,i-1);
  subplot(1,2,1)
  imshow(A)
  title('A')

  F =[0 -1 0
      -1 4 -1
      0 -1 0];
  num_becos = 100;
  while num_becos ~= 0
    A2 = filter2(F,A);
    B = ( A2 == 3 );
    B(1,:)=0;B(end,:)=0;
    B(:,1)=0;B(:,end)=0;
    A(B) = 0;
    num_becos = sum(B(:));
  end

  subplot(1,2,2)
  imshow(A)
  title('S')
  end
end
