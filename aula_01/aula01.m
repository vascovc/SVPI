%Aula 1
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
   'Ex2a'
   'Ex2b'
   'Ex3a'   
   'Ex3b'
   'Ex3c'
   'Ex3d'
   'Ex4a'
   'Ex4b'
   'Ex4c'
   'Ex4d'
   'Ex4e'
   'Ex5'
   'desafio_b'
   'desafio_c'
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1a -------------------------------------------------------------------

exercise = 'Ex1a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  imshow(Z)
end

%% Ex1b -------------------------------------------------------------------
exercise = 'Ex1b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  clearvars -except list_of_exercises exercise
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

 Z = zeros(100,200);
 Z(30:70,:)=255;
 figure(1)
 imshow(Z)

 Z = zeros(100,200);
 Z(30:70,50:90)=255;
 figure(2)
 imshow(Z)
end
%% Ex2a -------------------------------------------------------------------
exercise = 'Ex2a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all
    Z = zeros(100,200);
    Z(30:70,50:90)=255;
    Z(30:70,120:160)=128;
    imshow(Z)
    % nao funcionou por causa dos tipos de dados
  
end

%% Ex2b -------------------------------------------------------------------

exercise = 'Ex2b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

    Z=zeros(100,200); % gera em double
    Z(30:70,50:90)=1;
    Z(30:70,120:160)=0.5;
    figure(1)
    imshow(Z)
    whos Z
    %usando em uint8
    Z=zeros(100,200,'uint8');
    %alternativa
    %Z=uint8(zeros(100,200));
    Z(30:70,50:90)=255;
    Z(30:70,120:160)=128;
    figure(2)
    imshow(Z)
    whos Z

end
%% Ex3a -------------------------------------------------------------------

exercise = 'Ex3a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  Z = AddSquare(Z,20,30);
  imshow(Z)

end

%% Ex3b -------------------------------------------------------------------
exercise = 'Ex3b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  Z = AddSquare(Z,20,30);
  imshow(Z)

end
%% Ex3c -------------------------------------------------------------------
exercise = 'Ex3c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  for cc=10:20:180
    Z = AddSquare(Z,20,cc);
  end
  imshow(Z)

end
%% Ex3d -------------------------------------------------------------------
exercise = 'Ex3d'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  for ll=10:20:100
      for cc=10:20:200
        Z = AddSquare(Z,ll,cc);
      end
  end
  imshow(Z)
end
%% Ex4a -------------------------------------------------------------------
exercise = 'Ex4a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  x_0=50;y_0=60;r=20;

  for x=1:200
      for y=1:100
          if (x-x_0)^2+(y-y_0)^2 <=r^2
              Z(y,x)=1;
          end
      end
  end
  imshow(Z)
end

%% Ex4b -------------------------------------------------------------------
exercise = 'Ex4b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  x_0=50;y_0=60;r=20;
  x=1:size(Z,2);
  y=1:size(Z,1);
  [X,Y]=meshgrid(x,y);
  C = (((X-x_0).^2+(Y-y_0).^2)<=r*r);
  Z(C)=1;
  imshow(Z)
  
end
%% Ex4c -------------------------------------------------------------------
exercise = 'Ex4c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  x_0=50;y_0=60;r=20;
  Z=AddCircle(Z,x_0,y_0,r);
  imshow(Z)
  
end
%% Ex4d -------------------------------------------------------------------
exercise = 'Ex4d'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  for ll=13:25:100
      for cc=13:25:200
        Z=AddCircle(Z,cc,ll,11);
      end
  end
  imshow(Z)
  
end
%% Ex4e -------------------------------------------------------------------
exercise = 'Ex4e'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  for ll=13:25:100
      for cc=13:25:200
        Z=AddCircle(Z,cc,ll,11*rand(1));
      end
  end
  imshow(Z)
  
end
%% Ex5 -------------------------------------------------------------------
exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=zeros(100,200);
  x_0=100;y_0=50;
  Z=AddHeart(Z,x_0/30,y_0/30);
  imshow(Z)
  
end
%% desafio a) e o exercicio 3d
%% desafio b -------------------------------------------------------------------
exercise = 'desafio_b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  color = 1;
  for ll=10:20:100
      for cc=10:20:200
          size = randi([5 20]);
          Z = AddSquare_random(Z,ll,cc,size,color);
      end
  end
  imshow(Z)
  
end
%% desafio c
exercise = 'desafio_c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = zeros(100,200);
  for ll=10:20:100
      for cc=10:20:200
          color = rand();
          size = randi([5 20]);
          Z = AddSquare_random(Z,ll,cc,size,color);
      end
  end
  imshow(Z)
  
end