%Aula 8
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
   'Ex4a'
   'Ex4b'
   'Ex5' % da mal para baixo por causa de fechar as tesouras
   'Ex6' % a dar mal
   'Ex7'
   'Ex8'
   'Ex9'% dar mal
  }; %Defines the exercise to be executed (one or more at a time).
%% Ex1 -------------------------------------------------------------------

exercise = 'Ex1'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("nuts2a.jpg"));
  figure
  subplot(1,3,1)
  imshow(Z)

  Z = 1-rgb2gray(Z);

  bin=imbinarize(Z);
  fill = imfill(bin,'holes');
  subplot(1,3,2)
  imshow(fill)
 
  S = false(size(fill));
  S(1,:)=1;S(end,:)=1;
  S(:,1)=1;S(:,end)=1;
  S=and(S,fill);
  M = imreconstruct(S,fill);
  N = and(fill,not(M));
  subplot(1,3,3)
  imshow(N)

  figure
  [L,num] = bwlabel(N);
  for i=1:num
      subplot(num/4,num/3,i)
      imshow(L==i)
  end
end
%% Ex2 -------------------------------------------------------------------

exercise = 'Ex2'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("traffic_signs.jpg"));
  Z = im2gray(Z);
  Z = imbinarize(Z);
  Z = imfill(1-Z,"holes");
  
  Z = bwmorph(Z,'close');

  imshow(Z)

  [L,num] = bwlabel(Z);

  sts = regionprops(Z,'Circularity','Centroid');

  ff = [sts.Circularity];

  tr_lim = 0.69;
  cir_lim = 0.9;
  squ_lim = 0.79;

  tri_idx = find(ff<tr_lim);
  cir_idx = find(ff>cir_lim);
  squ_idx = find(ff>tr_lim & ff < cir_lim);

  TRI = ismember(L,tri_idx);
  SQU = ismember(L,squ_idx);
  CIR = ismember(L,cir_idx);

  for i=1:size(sts,1)
        x = sts(i).Centroid(1);
        y = sts(i).Centroid(2);
        text(x,y, {['Obj ' num2str(i)], num2str(sts(i).Circularity)}, 'Color','r')
  end

  figure
  subplot(1,3,1)
  imshow(TRI)
  title('Triangles')
    subplot(1,3,2)
  imshow(CIR)
  title('Circles')
    subplot(1,3,3)
  imshow(SQU)
  title('Squares')
end
%% Ex3 -------------------------------------------------------------------

exercise = 'Ex3'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("traffic_signs_jam1.jpg"));
  Z = im2gray(Z);
  Z = imbinarize(Z);
  Z = imfill(1-Z,"holes");
  
  Z = bwmorph(Z,'close');

  imshow(Z)

  [L,num] = bwlabel(Z);

  sts = regionprops(Z,'Circularity');

  ff = [sts.Circularity];

  tr_lim = 0.69;
  cir_lim = 0.9;
  squ_lim = 0.8;

  tri_idx = find(ff<tr_lim);
  cir_idx = find(ff>cir_lim);
  squ_idx = find(ff>tr_lim & ff < cir_lim);

  TRI = ismember(L,tri_idx);
  SQU = ismember(L,squ_idx);
  CIR = ismember(L,cir_idx);

  figure
  subplot(1,3,1)
  imshow(TRI)
  title('Triangles')
  subplot(1,3,2)
  imshow(CIR)
  title('Circles')
  subplot(1,3,3)
  imshow(SQU)
  title('Squares')

end
%% Ex4a -------------------------------------------------------------------

exercise = 'Ex4a'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("talheres_individuais.jpg"));
  Z = im2gray(Z);
  Z = 1-imbinarize(Z);
  Z = imfill(Z,"holes");
  
  Z = bwmorph(Z,'close');
  figure
  imshow(Z)

  [L,num] = bwlabel(Z);
  sts = regionprops(Z,'Solidity','Area','Perimeter','Circularity');
  for i=1:num
      subplot(1,num,i)
      imshow(L==i)
      %xlabel(sprintf('Solidity=%.4f\nForm factor=%.4f',sts(i).Solidity,4*pi*sts(i).Area / sts(i).Perimeter^2) )
      xlabel(sprintf('Solidity=%.4f\nForm factor=%.4f',sts(i).Solidity,sts(i).Circularity) )
  end
end
%% Ex4b -------------------------------------------------------------------

exercise = 'Ex4b'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z = im2double(imread("talheres.jpg"));
  Z = im2gray(Z);
  Z = 1-imbinarize(Z);
  Z = logical(imfill(Z,"holes"));

  S = false(size(Z));
  S(1,:)=1;S(end,:)=1;
  S(:,1)=1;S(:,end)=1;
  S=and(S,Z);
  M = imreconstruct(S,Z);
  N = and(Z,not(M));

  N = bwmorph(N,'close');

  figure
  imshow(N)

  [L,num] = bwlabel(N);
  sts = regionprops(N,'Solidity','Area','Perimeter');

  solidity_fork = 0.6936;
  solidity_knife = 0.7561;
  solidity_spoon = 0.7221;
  form_fork = 0.1100;
  form_knife = 0.2468;
  form_spoon = 0.2447;
  error = 0.03;
  upper_error = 1+error;
  lower_error = 1-error;

  ff = zeros(2,num);
  ff(1,:) = [sts.Solidity];
  for i=1:size(ff,2)
      ff(2,i) = 4*pi*sts(i).Area / sts(i).Perimeter^2; % mais facilmente se pode usar o 'circularity'
  end

  fork_idx  = find(ff(1,:) <= solidity_fork*upper_error  & ff(1,:) >= solidity_fork*lower_error & ff(2,:) <= form_fork*upper_error  & ff(2,:) >= form_fork*lower_error);
  knife_idx = find(ff(1,:) <= solidity_knife*upper_error & ff(1,:) >= solidity_knife*lower_error & ff(2,:) <= form_knife*upper_error  & ff(2,:) >= form_knife*lower_error);
  spoon_idx = find(ff(1,:) <= solidity_spoon*upper_error & ff(1,:) >= solidity_spoon*lower_error & ff(2,:) <= form_spoon*upper_error  & ff(2,:) >= form_spoon*lower_error);

  fork = ismember(L,fork_idx);
  knife = ismember(L,knife_idx);
  spoon = ismember(L,spoon_idx);

  figure
  subplot(1,3,1)
  imshow(fork)
  title('Fork')
  subplot(1,3,2)
  imshow(spoon)
  title('Spoon')
  subplot(1,3,3)
  imshow(knife)
  title('Knife')

end

%% Ex5 -------------------------------------------------------------------

exercise = 'Ex5'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("Seq1\TP2_img_01_01.png"));
  Z=rgb2gray(Z);
  Z=imbinarize(Z,0.06);
  %Z=imfill(Z,'holes');
  imshow(Z)

  S = false(size(Z));
  S(1,:)=1;S(end,:)=1;
  S(:,1)=1;S(:,end)=1;
  S=and(S,Z);
  M = imreconstruct(S,Z);
  N = and(Z,not(M));
  N = bwmorph(N,"close");

  [L,num] = bwlabel(N);
  sts = regionprops(N,'Area','EulerNumber');
  idx = find([sts.Area]>100);
  tools = ismember(L,idx);
  figure
  imshow(tools)
end
%% Ex6 -------------------------------------------------------------------

exercise = 'Ex6'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("Seq1\TP2_img_01_01.png"));
  Z=rgb2gray(Z);
  Z=imbinarize(Z,0.06);
  %Z=imfill(Z,'holes');
  imshow(Z)

  S = false(size(Z));
  S(1,:)=1;S(end,:)=1;
  S(:,1)=1;S(:,end)=1;
  S=and(S,Z);
  M = imreconstruct(S,Z);
  N = and(Z,not(M));
  N = bwmorph(N,"close");

  [L,num] = bwlabel(N);
  sts = regionprops(N,'Area','EulerNumber');
  idx = find([sts.Area]>100);
  tools = ismember(L,idx);
  figure
  imshow(tools)

  figure
  idx_h2 = find([sts.EulerNumber]==-1);
  tools_h2 = ismember(L,idx_h2);
  subplot(1,3,1)
  imshow(tools_h2)
  title("2 holes => euler number -1")
  
  idx_h1 = find([sts.EulerNumber]==0);
  tools_h1 = ismember(L,idx_h1);
  subplot(1,3,2)
  imshow(tools_h1)
  title("1 hole => euler number 0")

  idx_h0 = find([sts.EulerNumber]>0);
  tools_h0 = ismember(L,idx_h0);
  subplot(1,3,3)
  imshow(tools_h0)
  title("0 holes => euler number 1")
end
%% Ex7 -------------------------------------------------------------------

exercise = 'Ex7'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("Seq1\TP2_img_01_01.png"));
  Z=rgb2gray(Z);
  Z=imbinarize(Z,0.06);
  imshow(Z)

  B = imclearborder(Z);
  B = bwmorph(B,"close");

  L=bwlabel(B); %obter matriz de 'labels'
  s=regionprops(B,'Solidity'); %obter lista das propriedades todas
  soli=[0 0.5 0.6 0.7 1]; %limites dos intervalos
  lins=2; cols=2; %medidas para o subplot
  for i=2:numel(soli)
    idx=find([s.Solidity]>soli(i-1)&[s.Solidity]<=soli(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('Sol>%0.2f&Sol<=%0.2f',soli(i-1),soli(i));
    title(str);
  end

end
%% Ex8 -------------------------------------------------------------------

exercise = 'Ex8'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("Seq1\TP2_img_01_01.png"));
  Z=rgb2gray(Z);
  Z=imbinarize(Z,0.06);
  imshow(Z)

  B = imclearborder(Z);
  B = bwmorph(B,"close"); 

  L=bwlabel(B); %obter matriz de 'labels'
  s=regionprops(B,'Eccentricity'); %obter lista das propriedades todas
  excen=[0 0.94 0.96 0.98 1]; %limites dos intervalos
  lins=2; cols=2; %medidas para o subplot
  for i=2:numel(excen)
    idx=find([s.Eccentricity]>excen(i-1)&[s.Eccentricity]<=excen(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('Eccen>%0.2f&Eccen<=%0.2f',excen(i-1),excen(i));
    title(str);
  end
end
%% Ex9 -------------------------------------------------------------------

exercise = 'Ex9'; % Define the name of the current exercise
if ismember(exercise, list_of_exercises) %... if exer. in list_of_exercises
  disp(['Executing ' exercise ':'])
  clearvars -except list_of_exercises % Delete all previously declared vars
  close all

  Z=im2double(imread("Seq1\TP2_img_01_01.png"));
  Z=rgb2gray(Z);
  Z=imbinarize(Z,0.06);
  imshow(Z)

  B = imclearborder(Z);
  %B = bwmorph(B,"close");

  L=bwlabel(B); %obter matriz de 'labels'
  s=regionprops(B,'Circularity'); %obter lista das propriedades todas
  circ=[0 0.15 0.2 0.3 1]; %limites dos intervalos
  lins=2; cols=2; %medidas para o subplot
  for i=2:numel(circ)
    idx=find([s.Circularity]>circ(i-1)&[s.Circularity]<=circ(i));
    m=ismember(L,idx); %imagem binaria dos objetos detetados
    subplot(lins,cols,i-1); imshow(m);
    str=sprintf('Circ>%0.2f&Circ<=%0.2f',circ(i-1),circ(i));
    title(str);
  end
end