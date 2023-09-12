clear all
close all
clc

%enunciado - 5,7,1
%EX1 - 15,    21,     1 -> 94
%EX2 - 9,     11,     2 -> 306809
%EX3 - 13,    17,     3 -> 9229
%EX4 - 19,    13,     2 -> 54046
%EX5 - 11,     9,     3 -> 448964
%EX6 - 19,    13,     2 -> 7275
%EX7 -  9,    11,     2 -> 426539
%EX8 - 17,    13,     2 -> 227339
%EX9 -  9,    15,     1 -> 31808
%EX10 - 9,    13,     1 -> 259399

result_1 = q2(15,    21,     1);
result_2 = q2(9,     11,     2);
result_3 = q2(13,    17,     3);
result_4 = q2(19,    13,     2);
result_5 = q2(11,     9,     3);
result_6 = q2(19,    13,     2);
result_7 = q2(9,    11,     2);
result_8 = q2(17,    13,     2);
result_9 = q2(9,    15,     1);
result_10 = q2(9,    13,     1);

function result = q2(p1,p2,p3)

img = im2double(imread("Exemplos\Ex9\image_TP1_2023_1.png"));
filter = ones(p1,p2);
filter(round(p1/2),round(p2/2)) = p1*p2;
img_filtered = filter2(filter,img);
result = sum(sum(img_filtered<=p3));

%um filtro de valor 1 para todos os vizinhos que se querem
%considerar para contar se eles são brancos.
%Contudo, existe a possibilidade do pixel central ser um branco e apenas se
%querem os que sao pretos, por isso, atribui-se um valor grande o 
%suficiente, p1*p2, para o caso deste ser um branco e os seus vizinhos
%todos pretos. No fim so se querem aqueles cuja soma, que se tira
%diretamente do filtro, vai ser menor ou igual ao enunciado, p3, se for o 1
%ou 0.
end