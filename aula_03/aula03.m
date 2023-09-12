%Aula 3
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
   'Ex2a'
   'Ex2b'
   'Ex2c'
   'Ex2d'
   'Ex3a'
   'Ex3b'
   'Ex4a'
   'Ex4b'
   'Ex4c'
   'Ex4d'
   'Ex4e'
   'Ex4f'
   'Ex4g'
   'Ex5'
  }; %Defines the exercise to be executed (one or more at a time).
addpath 'C:\Users\Vasco Costa\Documents\MEGAsync\4-2\SVPI\lib'
addpath 'C:\Users\Vasco\Documents\MEGAsync\4-2\SVPI\lib'
%% Ex1a -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('rice.png'));
  a=pi/4;
  
  % rotacao especifica
  newZ2 = imrotate(Z,-a*180/pi);
  figure(1)
  imshow(newZ2)

  %abordagem geral
  T = rot(a);
  tf = affine2d(T');
  newZ1 = imwarp(Z, tf);
  figure(2)
  imshow(newZ1)
end
%% Ex2a -------------------------------------------------------------------

exercise = 'Ex2a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('bolt1.png'));
  cols=600;
  lins = 400;
  x = cols+(-cols)*rand();y = lins+(-lins)*rand(); a = 2*pi+(-2*pi)*rand();
  T = trans(x,y)*rot(a);
  tf = affine2d(T');
  Ro = imref2d([lins cols]);
  tempA = imwarp(Z,tf,'OutputView',Ro,'SmoothEdges',true);
  
  figure(1)
  imshow(tempA)

  Z(1,:)=1;Z(end,:)=1;
  Z(:,1)=1;Z(:,end)=1;
  x = rand();y = rand(); a = rand();
  T = trans(x,y)*rot(a);
  tf = affine2d(T');
  Ro = imref2d([lins cols]);
  tempA = imwarp(Z,tf,'OutputView',Ro,'SmoothEdges',true);
  
  figure(2)
  imshow(tempA)
end
%% Ex2b -------------------------------------------------------------------
%
exercise = 'Ex2b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread('bolt1.png'));
  cols=600;
  lins = 400;
  x = cols+(-cols)*rand();y = lins+(-lins)*rand(); a = 2*pi+(-2*pi)*rand();
  T = trans(x,y)*rot(a);
  tf = affine2d(T');
  Ro = imref2d([lins cols]);
  wObj=size(Z,2);hObj=size(Z,1);
  imxlim=[-wObj/2 wObj/2];
  imylim = [-hObj/2 hObj/2];
  Ri=imref2d(size(Z),imxlim,imylim);
  tempA = imwarp(Z,Ri,tf,'OutputView',Ro,'SmoothEdges',true);
  
  figure(1)
  imshow(tempA)

  Z(1,:)=1;Z(end,:)=1;
  Z(:,1)=1;Z(:,end)=1;
  %x = rand();y = rand(); a = rand();
  T = trans(x,y)*rot(a);
  tf = affine2d(T');
  Ro = imref2d([lins cols]);
  tempA = imwarp(Z,tf,'OutputView',Ro,'SmoothEdges',true);
  
  figure(2)
  imshow(tempA)
end
%% Ex2c -------------------------------------------------------------------

exercise = 'Ex2c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  bordo = true;
  Z = im2double(imread('bolt1.png'));
  if bordo
    Z(1,:)=1;Z(end,:)=1;
    Z(:,1)=1;Z(:,end)=1;
  end
  cols=600;
  lins = 400;
  tempA=zeros(lins,cols);
  tempB = tempA;
  tempC = tempA;
  Ro = imref2d([lins cols]);
  wObj=size(Z,2);hObj=size(Z,1);
  imxlim=[-wObj/2 wObj/2];
  imylim = [-hObj/2 hObj/2];
  Ri=imref2d(size(Z),imxlim,imylim);
  for i=1:5
      x = cols+(-cols)*rand();y = lins+(-lins)*rand(); a = 2*pi+(-2*pi)*rand();
      T = trans(x,y)*rot(a);
      tf = affine2d(T');
      single = imwarp(Z,Ri,tf,'OutputView',Ro,'SmoothEdges',true);
      tempA = tempA + single;% da valores maiores que 1
      tempB = max(tempB,single); % da o maximo que so pode ser 1 por isso nao origina valores maiores que um, nao faz a soma
      mask = (single>0);
      tempC(mask) = single(mask); % faz a substituição dos valores que eram diferentes de 0
  end
  figure(1)
  imshow(tempA)
  title('Soma') % fica mais branco
  figure(2)
  imshow(tempB)
  title('Max') % ficam merged
  figure(3)
  imshow(tempC)
  title('mask') % da um recorte
end
%% Ex2d -------------------------------------------------------------------

exercise = 'Ex2d'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  bordo = true;
  Z = im2double(imread('bolt1.png'));
  if bordo
    Z(1,:)=1;Z(end,:)=1;
    Z(:,1)=1;Z(:,end)=1;
  end
  cols = 600;
  lins = 400;
  tempA = zeros(lins,cols);
  tempA(1,:)=1;tempA(end,:)=1;
  tempA(:,1)=1;tempA(:,end)=1;
  Ro = imref2d([lins cols]);
  wObj=size(Z,2);hObj=size(Z,1);
  imxlim=[-wObj/2 wObj/2];
  imylim = [-hObj/2 hObj/2];
  Ri=imref2d(size(Z),imxlim,imylim);
  i=0;
  while i<5
      x = cols+(-cols)*rand();y = lins+(-lins)*rand(); a = 2*pi+(-2*pi)*rand();
      T = trans(x,y)*rot(a);
      tf = affine2d(T');
      single = imwarp(Z,Ri,tf,'OutputView',Ro,'SmoothEdges',true);
      tempA = tempA + single;% da valores maiores que 1
      if any(tempA(:)>1)
          tempA = tempA - single;
          disp("houve sobreposicao")
          i = i-1;
      end
      i=i+1;
  end
  figure(1)
  imshow(tempA)
end
%% Ex3a -------------------------------------------------------------------

exercise = 'Ex3a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

end
%% Ex3b -------------------------------------------------------------------

exercise = 'Ex3b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  cols = 800;
  lins = 600;
  tempA = zeros(lins,cols);
  tempA(1,:)=1;tempA(end,:)=1;
  tempA(:,1)=1;tempA(:,end)=1;
  Ro = imref2d([lins cols]);
  i = 0;
  while i<20
      dots = round(1 + (6-1).*rand(1,2));
      Z = create_domino( dots(1),dots(2) );
      wObj=size(Z,2);hObj=size(Z,1);
      imxlim=[-wObj/2 wObj/2];
      imylim = [-hObj/2 hObj/2];
      Ri=imref2d(size(Z),imxlim,imylim);

      x = cols+(-cols)*rand();y = lins+(-lins)*rand(); a = 2*pi+(-2*pi)*rand();
      T = trans(x,y)*rot(a);
      tf = affine2d(T');
      single = imwarp(Z,Ri,tf,'OutputView',Ro,'SmoothEdges',true);
      tempA = tempA + single;% da valores maiores que 1
      if any(tempA(:)>1)
          tempA = tempA - single;
          disp("houve sobreposicao")
          i = i-1;
      end
      i=i+1;
  end
  figure(1)
  imshow(tempA)
  % para verificar se nao existem dominos repetidos poder-se-ia guardar os
  % pares aleatorios criados e se essa combinação, sem ordem, já exista
  % então não se gera
end
%% Ex4a -------------------------------------------------------------------

exercise = 'Ex4a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

end
%% Ex4b -------------------------------------------------------------------

exercise = 'Ex4b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;
end
%% Ex4c -------------------------------------------------------------------

exercise = 'Ex4c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')
end
%% Ex4d -------------------------------------------------------------------

exercise = 'Ex4d'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')

  alpha = [500 500];
  center = [size(A,2) size(A,1)]/2;
  K = PerspectiveTransform(alpha, center);
end
%% Ex4e -------------------------------------------------------------------

exercise = 'Ex4e'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')

  alpha = [500 500];
  center = [size(A,2) size(A,1)]/2;
  K = PerspectiveTransform(alpha, center);

  Ch = K*P;

  C = round(Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
  C(2,:) = size(A,1) - C(2,:); % para transformar as coordenadas do yy
end
%% Ex4f -------------------------------------------------------------------

exercise = 'Ex4f'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(0)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')

  alpha = [500 500];
  center = [size(A,2) size(A,1)]/2;
  K = PerspectiveTransform(alpha, center);

  Ch = K*P;

  C = round(Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
  C(2,:) = size(A,1) - C(2,:); %

  Oks = (C(2,:)>0 & C(2,:)<=imLins) & (C(1,:)>0 & C(1,:)<=imCols);
  C2 = C(2,Oks);
  C1 = C(1,Oks);
  C = [C1;C2];
end
%% Ex4g -------------------------------------------------------------------

exercise = 'Ex4g'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  imLins=240;
  imCols=320;
  A=zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  P=[5*cos(t);
     5*sin(t);
     30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(pi/4)*roty(0)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')

  alpha = [500 500];
  center = [size(A,2) size(A,1)]/2;
  K = PerspectiveTransform(alpha, center);

  Ch = K*P;

  C = round(Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
  C(2,:) = size(A,1) - C(2,:); %

  Oks = (C(2,:)>0 & C(2,:)<=imLins) & (C(1,:)>0 & C(1,:)<=imCols);
  C2 = C(2,Oks);
  C1 = C(1,Oks);
  C = [C1;C2];

  idx = sub2ind(size(A),C(2,:),C(1,:));
  A(idx) = 1;
  plot3(A(1,:),A(2,:),zeros(1,size(A,2)),'w')
  figure(2)
  imshow(A)

end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  A_1 = [0 0 0]';B_1 = [3 6 12]';

  imLins=240;
  imCols=320;
  A = zeros(imLins,imCols);
  t = linspace(0,2*pi,50);
  %P=[5*cos(t);
  %   5*sin(t);
  %   30*ones(size(t))];
  %define-se a circunferencia atraves das coordenadas polares
  %assim fica-se com uma circunferencia de raio 5 no plano z=30

  %P = genpts(A_1,B_1,4);
  load P.mat
  P = [P; ones(1,size(P,2))];
  P = trans3(0,0,30)*rotx(0)*roty(pi/3)*rotz(pi/2)*trans3(0,0,-30)*P;

  plot3(P(1,:),P(2,:),P(3,:),'.')
  hold on; axis equal; grid on;
  axis([-5 5 -5 5 0 40])
  zlabel('Z');xlabel('X');ylabel('Y');
  line([0 0], [0 0], [0 50])
  fill3([4 -4 -4 4],[-3 -3 3 3], [0 0 0 0],'k')

  alpha = [500 500];
  center = [size(A,2) size(A,1)]/2;
  K = PerspectiveTransform(alpha, center);

  Ch = K*P;

  C = round(Ch(1:2,:) ./ repmat(Ch(3,:),2,1));
  C(2,:) = size(A,1) - C(2,:); %

  Oks = (C(2,:)>0 & C(2,:)<=imLins) & (C(1,:)>0 & C(1,:)<=imCols);
  C2 = C(2,Oks);
  C1 = C(1,Oks);
  C = [C1;C2];

  idx = sub2ind(size(A),C(2,:),C(1,:));
  A(idx) = 1;
  figure(2)
  imshow(A)
end
