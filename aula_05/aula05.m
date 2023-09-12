%Aula 5
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
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1 -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  matches_1 = im2double(imread('matches1.png'));
  matches_2 = im2double(imread('matches2.png'));
  
  figure(1)
  subplot(2,2,1)
  imshow(matches_1)
  subplot(2,2,2)
  imhist(matches_1)
  subplot(2,2,3)
  imshow(matches_2)
  subplot(2,2,4)
  imhist(matches_2)

  figure(2)
  subplot(2,1,1)
  matches_1(matches_1<0.2)=1;
  imshow(matches_1)
  subplot(2,1,2)
  matches_2(matches_2<0.2)=1;
  imshow(matches_2)

  % nao funciona tao bem no matches_2 porque a separacao no 0.2 nao e tao
  % acentuada e ao fazer-se apenas a ser menor tudo o que e mais preto
  % tambem vai ser transformado em branco, com maior ajuste e definindo o
  % minimo que e transformado em branco, por exemplo acima de 0.1
end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  figure(1)
  matches_1 = im2double(imread('matches1.png'));
  matches = matches_1;
  subplot(3,2,1)
  imshow(matches_1)
  subplot(3,2,2)
  imhist(matches_1)

  matches_1(matches_1<0.2)=1;
  subplot(3,2,3)
  imshow(matches_1)
  subplot(3,2,4)
  imhist(matches_1)

  matches_1 = imadjust(matches_1);
  subplot(3,2,5)
  imshow(matches_1)
  subplot(3,2,6)
  imhist(matches_1)

  mask = graythresh(matches_1);
  Bw = imbinarize(matches_1,mask);
  figure(2)
  subplot(2,2,1)
  imshow(matches_1)
  subplot(2,2,3)
  imshow(Bw)

  neg = 1-Bw;
  matches(neg==1)=0.5;
  subplot(2,2,4)
  imshow(matches)

end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  chess = im2double(imread("trimodalchess.png"));
  figure(1)
  subplot(1,2,1)
  imshow(chess)
  subplot(1,2,2)
  imhist(chess)

  thresh = multithresh(chess,2);
  line([thresh(1) thresh(1)],[0 1000],'color','r')
  line([thresh(2) thresh(2)],[0 1000],'color','r')

  figure(2)
  subplot(2,3,1)
  mask_1 = chess<thresh(1);
  mask_2 = chess>thresh(2);
  mask = or(mask_1,mask_2);
  imshow(mask)
  title('Pretos e brancos')

  subplot(2,3,2)      
  mask_1 = chess<thresh(1);
  mask_2 = chess>thresh(2);
  mask = not(or(mask_1,mask_2));
  imshow(mask)
  title('cinzentos')

  subplot(2,3,3)
  y = imbinarize(1-chess,mean(thresh));
  imshow(y)
  title('pretos')

  subplot(2,3,4)
  y = imbinarize(chess,thresh(2));
  imshow(y)
  title('brancos')

  subplot(2,3,5)
  y = imbinarize(chess,thresh(2));
  imshow(1-y)
  title('pretos e cinzentos')

  subplot(2,3,6)
  y = imbinarize(chess,thresh(1));
  imshow(y)
  title('brancos e cinzentos')

end
%% Ex4 -------------------------------------------------------------------

exercise = 'Ex4'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  s_1 = im2double(imread("seeds.png"));
  s_2 = im2double(imread("seeds_inv.png"));
  figure(1)
  subplot(2,2,1)
  imshow(s_1)
  subplot(2,2,2)
  imshow(autobin(s_1))
  subplot(2,2,3)
  imshow(s_2)
  subplot(2,2,4)
  imshow(autobin(s_2))
  
  %nao funcionara bem no sentido de aplicação de casos em que o método de
  %Otsu nao seja adequado a aplicar
end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("rice.png"));

  [M,N] = size(Z); %M -> rows; N-> columns
  figure(1)
  subplot(1,3,1)
  imshow(Z)

  img_1 = Z(1:M/2,1:M/2);
  img_2 = Z(1:M/2,M/2+1:end);
  img_3 = Z(M/2+1:end,1:M/2);
  img_4 = Z(M/2+1:end,M/2+1:end);

  subplot(1,3,2)
  img_1_b = autobin(img_1);
  img_2_b = autobin(img_2);
  img_3_b = autobin(img_3);
  img_4_b = autobin(img_4);
  img_b_all = [img_1_b img_2_b; img_3_b img_4_b];
  imshow(img_b_all)

  subplot(1,3,3)
  img_b = autobin(Z);
  imshow(img_b)

  figure(2)
  subplot(2,2,1)
  imshow(img_b)
  title("Limiar Global")

  subplot(2,2,2)
  imshow(img_b_all)
  title("Multi-histograma")

  subplot(2,2,3)
  imshow(xor(img_b_all,img_b))
  title("Diferenças")

  subplot(2,2,4)
  imshow(medfilt2(img_b_all,[3,3]))
  title("Multi-histograma + mediana")
end
%% Ex6 -------------------------------------------------------------------

exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("rice.png"));
  figure(1)
  subplot(1,2,1)
  imshow(Z)
  subplot(1,2,2)
  imshow(MultiRegionBin(Z,5,3))

  figure(2)
  subplot(1,2,1)
  imshow(Z)
  subplot(1,2,2)
  imshow(MultiRegionBin(Z,25,25))

end
%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  
end
%% Ex8 -------------------------------------------------------------------
% verificar a limpleza se e com a media
exercise = 'Ex8'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("rice.png"));
  figure(1)
  subplot(1,3,1)
  imshow(Z)
  title('Original')
  
  subplot(1,3,2)
  value = adaptthresh(Z);
  y = imbinarize(Z,value);
  imshow(y)
  title('Adaptive binarization')
  
  subplot(1,3,3)
  imshow(medfilt2(y,[3,3])) % ou entao como vem no proximo exercicio da questao do grupo de pixeis
  title('Cleaned image')
end
%% Ex9 -------------------------------------------------------------------

exercise = 'Ex9'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("samples2.png"));
  figure(1)
  subplot(1,2,1)
  imshow(Z)

  mask = adaptthresh(Z,0.645); % a sensibilidade muda tudo
  y = imbinarize(Z,mask);
  y = bwareaopen(y,100);
  subplot(1,2,2)
  imshow(1-y)
end

