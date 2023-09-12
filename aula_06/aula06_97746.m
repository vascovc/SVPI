%Aula 6
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
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1 -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("coins.png"));
  Sx = [-1 0 1; -2 0 2; -1 0 1];
  Sy = [-1 -2 -1; 0 0 0; 1 2 1];

  figure(1)
  subplot(1,3,1)
  imshow(abs(filter2(Sx,Z)))
  title('|Gx|')

  subplot(1,3,2)
  imshow(abs(filter2(Sy,Z)))
  title('|Gy|')

  subplot(1,3,3)
  imshow(abs(filter2(Sx,Z))+abs(filter2(Sy,Z)))
  title('|Gx|+|Gy|')
end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("coins.png"));
  [Gx,Gy]=imgradientxy(Z,'sobel');
  figure(1)
  subplot(1,5,1)
  imshow(abs(Gx),[])
  title('Sobel |Gx| norm')

  subplot(1,5,2)
  imshow(abs(Gy),[])
  title('Sobel |Gy| norm')

  subplot(1,5,3)
  imshow(abs(Gx)+abs(Gy),[])
  title('Sobel |Gx|+|Gy| norm')

  subplot(1,5,4)
  [~,Gdir]=imgradient(Gx,Gy);
  imshow(Gdir,[])
  title('Sobel \theta norm (gx,gy)')

  subplot(1,5,5)
  [~,Gdir]=imgradient(Z,'sobel');
  imshow(Gdir,[])
  title('Sobel \theta norm (sobel)')
end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("coins.png"));

  B = edge(Z,'sobel');
  imshow(B)

end
%% Ex4 -------------------------------------------------------------------

exercise = 'Ex4'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("coins.png"));

  figure(1)
  subplot(2,2,1)
  B = edge(Z,'sobel');
  imshow(B)
  
  subplot(2,2,2)
  B = edge(Z,'canny');
  imshow(B)
  
  subplot(2,2,3)
  B = edge(Z,'prewitt');
  imshow(B)
  
  subplot(2,2,4)
  B = edge(Z,'log');
  imshow(B)
end
%% Ex5 -------------------------------------------------------------------
% a dar diferente
exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("Tcomluz.jpg"));

  figure(1)
  subplot(2,2,1)
  B = edge(Z,'sobel');
  imshow(B)
  
  subplot(2,2,2)
  B = edge(Z,'canny');
  imshow(B)
  
  subplot(2,2,3)
  B = edge(Z,'prewitt');
  imshow(B)
  
  subplot(2,2,4)
  B = edge(Z,'log');
  imshow(B)
end
%% Ex6 -------------------------------------------------------------------

exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  cam = webcam();
  h = figure(1);

  while isvalid(h)
      A = snapshot(cam);
      subplot(1,3,1)
      imshow(A)
      Z = rgb2gray(A);
      title('original')

      subplot(1,3,2)
      %[L, N] = bwlabel(Z);
      B = edge(Z,'sobel');
      imshow(B)
      title('Sobel')

      subplot(1,3,3)
      %B = bwboundaries(Z,'noholes');
      B = edge(Z,'canny');
      imshow(B)
      title('Canny')

      pause(0.05)
  end
  clear cam

end
%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A=im2double(imread("coins.png"));

  Z = edge(A,'sobel');
  X = false((size(Z))); % criar uma mascara de 0 para adicionar as bordas maiores
  subplot(1,2,1)
  imshow(Z)
  title('All edges')
  subplot(1,2,2)
  imshow(X)
  hold on
  title('Selected edges')
  minSize = 100;

  [L, N] = bwlabel(Z); % em L fica uma matriz como Z mas cada componente ligado fica com um numero diferente
  %em N fica o numero de connected objects

  for k=1:N
      C = (L==k); % fica a mascara binaria em C de cada componente ligado
      if(sum(sum(C))<minSize) % ignorar se o numero de pixels ligados for menor que minSize
          continue
      end
      X = X|C;
      subplot(1,2,2)
      imshow(X)
      pause(0.2) % para se ter uma pausa e ver frame a frame
  end
end
%% Ex8 -------------------------------------------------------------------

exercise = 'Ex8'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

    A=im2double(imread("coins.png"));

  Z = edge(A,'canny');
  X = false(size(Z)); % criar uma mascara de 0
  Y = false(size(Z));

  subplot(1,3,1)
  imshow(Z)
  title('All edges')
  subplot(1,3,2)
  imshow(X)

  minSize = 160;

  [L, N] = bwlabel(Z);

  for k=1:N
      C = (L==k);
      if(sum(sum(C))<minSize)
        Y = Y|C;
      else
        X = X|C;
      end

      subplot(1,3,2)
      imshow(X)
      title("large edges")
      subplot(1,3,3)
      imshow(Y)
      title("Small edges")
  end
end

%% Ex9 -------------------------------------------------------------------

exercise = 'Ex9'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A = im2double(imread("coins.png"));
  Z = edge(A,'sobel');
  X = false(size(Z));
  subplot(1,2,1)
  imshow(Z)
  hold on
  axis on
  title({'Edges overlayed with','larger outer countours'})

  my_axis = axis;
  subplot(1,2,2)
  hold on
  axis ij
  axis equal
  axis(my_axis)
  grid on
  title({'Separate plot of the','larger outer countours'})
  
  minSize = 100;
  [L,N] = bwlabel(Z);
  for k=1:N
      C = (L==k);
      if (nnz(C) < minSize), continue;end

      BB = bwboundaries(C, 'noholes');
      boundary = BB{1};

      subplot(1,2,1)
      plot(boundary(:,2),boundary(:,1),'r','LineWidth',4)
      subplot(1,2,2)
      plot(boundary(:,2),boundary(:,1),'b')
      pause(0.5)
  end


end
%% Ex10 -------------------------------------------------------------------

exercise = 'Ex10'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

end