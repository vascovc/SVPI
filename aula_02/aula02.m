%Aula 2
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
   'Ex2c'
   'Ex3a'
   'Ex3b'
   'Ex3c'
   'Ex4a'
   'Ex4b'
   'Ex4c'
   'Ex5'
   'Ex6'
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1a -------------------------------------------------------------------

exercise = 'Ex1a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P=[3 0]';
  plot(P(1),P(2),'*');
  a = pi/3;
  Rot = [cos(a) -sin(a)
         sin(a)  cos(a)];
  axis([-1 4 -1 4]);
  hold on
  grid on
  axis square
  Pc = Rot*P;
  plot(Pc(1),Pc(2),'*r')
end
%% Ex1b -------------------------------------------------------------------

exercise = 'Ex1b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P=[3 0]';
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N = 20;
  angs = linspace(0,2*pi,N);
  for a=angs
      Q=rota(a)*P;
      plot(Q(1),Q(2),'*r');
      pause(0.1)
  end
end
%% Ex2a -------------------------------------------------------------------

exercise = 'Ex2a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P=[3 0]';
  h=plot(P(1),P(2),'dr');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N = 100;
  angs = linspace(0,2*pi,N);
  for a=angs
      Q=rota(a)*P;
      set(h,'Xdata',Q(1),'Ydata',Q(2));
      pause(0.1)
  end
end
%% Ex2b -------------------------------------------------------------------

exercise = 'Ex2b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P=[3 0]';
  P2 = [2 0]';
  h=plot(P(1),P(2),'dr');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N = 500;
  h2 = plot(P2(1),P2(2),'ob');
  angs = linspace(0,10*pi,N);
  for a=angs
      Q=rota(a)*P;
      Q2 = rota(a)*P2;
      set(h,'Xdata',Q(1),'Ydata',Q(2));
      set(h2,'Xdata',Q2(1),'Ydata',Q2(2));
      pause(0.1)
  end
end
%% Ex2c -------------------------------------------------------------------

exercise = 'Ex2c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P=[3 0]';
  P2 = [2 0]';
  h=plot(P(1),P(2),'dr');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N = 500;
  h2 = plot(P2(1),P2(2),'ob');
  angs = linspace(0,10*pi,N);
  for a=angs
      Q=rota(a)*P;
      Q2 = rota(2*a)*P2; %fazendo o dobro do angulo ele vai ao dobro da velocidade
      set(h,'Xdata',Q(1),'Ydata',Q(2));
      set(h2,'Xdata',Q2(1),'Ydata',Q2(2));
      pause(0.1)
  end
end
%% Ex3a -------------------------------------------------------------------

exercise = 'Ex3a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P = [-0.5 0.5 0
      0 0 2];
  h = fill(P(1,:),P(2,:),'y');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N=200;
  angs = linspace(0,20*pi,N);
  for a=angs
      Q = rota(a)*P;
      set(h,'XData',Q(1,:),'YData',Q(2,:))
      pause(0.05)
  end
end
%% Ex3b -------------------------------------------------------------------

exercise = 'Ex3b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P = [-0.5 0.5 0
      0 0 2];
  P = P+[3 0]'; % faz a todos os elementos
  %P = P+repmat([3 0]',1,size(P,2));
  h = fill(P(1,:),P(2,:),'y');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N=500;
  angs = linspace(0,20*pi,N);
  for a=angs
      Q = rota(a)*P;
      set(h,'XData',Q(1,:),'YData',Q(2,:))
      set(h,'FaceColor',rand(1,3));
      pause(0.05)
  end
end
%% Ex3c -------------------------------------------------------------------

exercise = 'Ex3c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  a = -2;b=2;
  P=a + (b-a).*rand([2,10]);
  h = fill(P(1,:),P(2,:),'y');
  axis([-4 4 -4 4]);
  hold on
  grid on
  axis square
  N=500;
  angs = linspace(0,20*pi,N);
  for a=angs
      Q = rota(a)*P;
      set(h,'XData',Q(1,:),'YData',Q(2,:))
      set(h,'FaceColor',rand(1,3));
      pause(0.05)
  end
end
%% Ex4a -------------------------------------------------------------------

exercise = 'Ex4a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P = [-0.5 0.5 0
      0 0 2
      1 1 1];
  h = fill(P(1,:),P(2,:),'y');
  axis([-1 4 -1 4])
  hold on
  grid on
  axis square

  T1 = trans(3,0);
  T2 = rot(pi/4);
  Q1=T1*T2*P; %primeiro rotacao, depois rotacao
  Q2 = T2*T1*P;% primeiro translacao, depois rotacao
  h1=fill(Q1(1,:),Q1(2,:),'r');
  h2=fill(Q2(1,:),Q2(2,:),'g');

  text(mean(P(1,:)), mean(P(2,:)),'P')
  text(mean(Q1(1,:)), mean(Q1(2,:)),'Q_1')
  text(mean(Q2(1,:)), mean(Q2(2,:)),'Q_2')
  % a ordem importa 
end
%% Ex4b -------------------------------------------------------------------

exercise = 'Ex4b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P = [-0.5 0.5 0
      0 0 2
      1 1 1];
  h1 = fill(P(1,:),P(2,:),'r');
  axis([-1 4 -1 7])
  hold on
  grid on
  axis square
  for t=linspace(0,3,20)
      Q = trans(0,t)*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
end
%% Ex4c -------------------------------------------------------------------

exercise = 'Ex4c'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  P = [-0.5 0.5 0
      0 0 2
      1 1 1];
  h1 = fill(P(1,:),P(2,:),'r');
  axis([-12 12 -12 12])
  hold on
  grid on
  axis square
  
  for t=linspace(0,3,20)%trans(0,3)
      Q = trans(0,t)*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
  for a=linspace(0,pi/2,20)%rot(+90)
      Q = rot(a)*trans(0,3)*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
  for t=linspace(0,-6,20)%trans(-6,0)
      Q = trans(t,0)*rot(pi/2)*trans(0,3)*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
  for a=linspace(0,-pi/2,20)%rot(-90)
      Q = trans(-6,0)*rot(pi/2)*trans(0,3)*rot(a)*P;
      Q2 = rot(a)*Q;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
end
%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  V = [1 1 0
      -1 1 0
      -1 -1 0
      1 -1 0
      1 1 2
      -1 1 2
      -1 -1 2
      1 -1 2];
  F = [1 2 3 4
      5 6 7 8
      1 2 6 5
      1 5 8 4
      3 7 8 4
      2 6 7 3];
  h = patch('Vertices', V, 'Faces',F,'Facecolor','c');
  grid on
  axis([-2 2 -2 2])
  axis equal
  view(3)
  
  for a=linspace(0,20*pi,20)
      rotacao = [cos(a) -sin(a) 0 
                 sin(a) cos(a) 0 
                 0 0 1];
      Q = V*rotacao;
      cla
      patch('Vertices', Q, 'Faces',F,'Facecolor','c');
      pause(0.05)
  end

end
%% Ex6 -------------------------------------------------------------------

exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  h_plot = plot(0,0,'o-b');
  xs = [];
  ys = [];
  while true
      [x,y] = ginput(1);
      if isempty(x)
          break
      end
      xs = [xs x];
      ys = [ys y];
      set(h_plot,'XData',xs,'YData',ys)
  end

  P = [xs;
      ys;
      ones([1 length(xs)])]';

  h1 = fill(P(1,:),P(2,:),'r');
  hold on
  grid on
  axis square
  axis([-10 10 -10 10])
  
  for t=linspace(0,3,20)%trans(0,3)
      Q = trans(0,t)*P; 
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
  for a=linspace(0,pi/2,20)%rot(+90)
      Q = rot(a)*trans(0,3)*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
  ang = linspace(0,pi,20);
  t = linspace(0,-6,20);
  for i=1:20%trans(-6,0)
      Q = trans(t(i),0)*rot(pi/2)*trans(0,3)*rot(ang(i))*P;
      set(h1,'XData',Q(1,:),'YData',Q(2,:));
      pause(0.05)
  end
end