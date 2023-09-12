clear all
close all
clc

%enunciado - 
%EX1 - 106,   178,    50,    30 -> 1657
%EX2 - 107,   243,    77,    25 -> 1332
%EX3 - 109,   255,    60,    15 -> 1356
%EX4 - 109,   164,   104,    -5 -> 1319
%EX5 - 101,   212,    88,     5 -> 1165
%EX6 - 109,   164,   104,    -5 -> 1390
%EX7 -  85,   291,    96,    35 -> 796
%EX8 -  98,   112,   139,   -25 -> 1141
%EX9 - 107,    69,    96,    35 -> 1498
%EX10 - 94,   215,    90,    20 -> 1089
cd Exemplos\
cd Ex1\
result_01 = q4(106,   178,    50,    30);
cd ..\
cd Ex2\
result_02 = q4(107,   243,    77,    25);
cd ..\
cd Ex3\
result_03 = q4(109,   255,    60,    15);
cd ..\
cd Ex4\
result_04 = q4(109,   164,   104,    -5);
cd ..\
cd Ex5\
result_05 = q4(101,   212,    88,     5);
cd ..\
cd Ex6\
result_06 = q4(109,   164,   104,    -5);
cd ..\
cd Ex7\
result_07 = q4(85,   291,    96,    35);
cd ..\
cd Ex8\
result_08 = q4(98,   112,   139,   -25);
cd ..\
cd Ex9\
result_09 = q4(107,    69,    96,    35);
cd ..\
cd Ex10\
result_10 = q4(94,   215,    90,    20);
cd ..\..\

function result = q4(p1,p2,p3,p4)
%p1=106;p2=200;p3=100;p4=45;
Z = im2double(imread("image_TP1_2023_2.png"));
square = true(p1);

[lins,cols]=size(Z);
T = [1 0 p2; 0 1 p3; 0 0 1]*[cosd(p4) -sind(p4) 0;sind(p4) cosd(p4) 0;0 0 1];
%T = [cosd(p4) -sind(p4) 0;sind(p4) cosd(p4) 0;0 0 1]*[1 0 p2; 0 1 p3; 0 0 1];
T = T*[1 0 -1; 0 1 -1; 0 0 1];
tf = affine2d(T');
Ro = imref2d([lins cols]);
%tempA = imwarp(square,tf,'OutputView',Ro);
%tempA = imwarp(square,tf,'OutputView',Ro,'SmoothEdges',true); % falha 1 no
%4
tempA = imwarp(square,tf,'OutputView',Ro,'SmoothEdges',true,'interp','nearest');

filter = [1 1 1; 1 8 1; 1 1 1];
outcome = filter2(filter,Z);

% figure
% imshow(Z.*.4+tempA.*.6)
% hold on
% for line=1:lins
%     for col=1:cols
%         if line==p3 && col==p2
%             plot(col,line,'r*')
%         end
%         if tempA(line,col) && outcome(line,col)>8
%             plot(col,line,'g.')
%         end
%     end
% end
%disp(tempA(p3,p2))

result = sum(sum( outcome(tempA)>8 ));
end