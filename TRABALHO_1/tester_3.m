clear all
close all
clc

%enunciado - 
%EX1 - [0     0     1; 0     1     1; 1     1  1] -> 110
%EX2 - [0     0     1; 1     1     1; 0     0  1] -> 814
%EX3 - [0     1     0; 1     1     1; 0     1  0] -> 846
%EX4 - [0     0     1; 0     1     1; 0     0  1] -> 844
%EX5 - [1     1     1; 1     1     0; 1     0  0] -> 52
%EX6 - [0     0     1; 0     1     1; 0     0  1] -> 885
%EX7 - [1     1     1; 0     1     0; 0     0  0] -> 883
%EX8 - [0     0     1; 0     1     1; 0     0  1] -> 827
%EX9 - [0     0     1; 1     1     0; 0     0  1] -> 608
%EX10 -[1     1     1; 1     1     0; 1     0  0] -> 66

result_1  = q3([0     0     1; 0     1     1; 1     1     1]);
result_2  = q3([0     0     1; 1     1     1; 0     0     1]);
result_3  = q3([0     1     0; 1     1     1; 0     1     0]);
result_4  = q3([0     0     1; 0     1     1; 0     0     1]);
result_5  = q3([1     1     1; 1     1     0; 1     0     0]);
result_6  = q3([0     0     1; 0     1     1; 0     0     1]);
result_7  = q3([1     1     1; 0     1     0; 0     0     0]);
result_8  = q3([0     0     1; 0     1     1; 0     0     1]);
result_9  = q3([0     0     1; 1     1     0; 0     0     1]);
result_10 = q3([1     1     1; 1     1     0; 1     0     0]);

function result = q3(p1)

img = im2double(imread("Exemplos\Ex10\image_TP1_2023_2.png"));
filter = -1*ones(size(p1));
filter = filter + 2*p1;
filtered_image = filter2(filter,img);
soma = sum(sum(p1));
result = sum(sum(filtered_image==soma));

% primeiro faz-se o filtro a ser [-1 -1 -1...] e depois soma-se duas vezes o 
% filtro para se ter assim os 0 substituidos por -1 no filtro dado como
% parametro. Isto é relevante uma vez que de facto nao se pretende ignorar 
% o pixel nessa posicao que seria o que aconteceria com o valor 0 original.
% assim, ao ser um valor negativo, caso este seja branco, leva a que haja 
% uma penalização e que este nao seja considerado.
end