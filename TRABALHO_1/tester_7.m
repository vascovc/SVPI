clear all
close all
clc

%enunciado - 
%EX1 -
ex1 = [1 1 1 1 1; 1 0 1 0 1; 1 1 1 1 1];
% -> 1184
%EX2 -
ex2 = [1   0   0   0   0   0   0   0   0   0   1;
       0   1   0   0   0   0   0   0   0   1   0;
       0   0   1   0   0   0   0   0   1   0   0;
       0   0   0   1   0   0   0   1   0   0   0;
       0   0   0   0   1   0   1   0   0   0   0;
       0   0   0   0   0   1   0   0   0   0   0;
       0   0   0   0   1   0   1   0   0   0   0;
       0   0   0   1   0   0   0   1   0   0   0;
       0   0   1   0   0   0   0   0   1   0   0;
       0   1   0   0   0   0   0   0   0   1   0;
       1   0   0   0   0   0   0   0   0   0   1];
%-> 80
%EX3 -
ex3 = [1     0     0     0     1     0     1     1     1     0     1     1     1     0     1     1     1;
     1     1     0     1     1     0     1     0     1     0     1     0     0     0     1     0     0;
     1     0     1     0     1     0     1     0     1     0     1     1     1     0     1     1     1;
     1     0     0     0     1     0     1     0     1     0     0     0     1     0     0     0     1;
     1     0     0     0     1     0     1     1     1     0     1     1     1     0     1     1     1];
% -> 258
%EX4 -
ex4=[1     1     1     0     1;
     1     0     1     0     1;
     1     0     1     1     1];
% -> 1381
%EX5 -
ex5=[1     1     1;
     1     0     1;
     1     1     1;
     1     0     1;
     1     1     1 ];
%-> 1220
%EX6 -
ex6=[1     1     1     0     1;     
     1     0     1     0     1;     
     1     0     1     1     1]; 
% 1304

%EX7 -
ex7=[1     0     0     0     0     1     0     0     0     1     0     0     0     1     1     1     1;
     1     0     0     0     0     1     0     0     1     0     1     0     0     1     0     0     0;
     1     0     1     1     1     1     0     0     1     0     1     0     0     1     1     1     1;
     1     0     1     0     0     1     0     1     0     0     0     1     0     0     0     0     1;
     1     0     1     1     1     1     0     1     0     0     0     1     0     1     1     1     1];
%-> 220
%EX8 - 
ex8=[1     0     0     0     0     1     0     0     0     1     0     0     0     1     1     1     1;
     1     0     0     0     0     1     0     0     1     0     1     0     0     1     0     0     0;
     1     0     1     1     1     1     0     0     1     0     1     0     0     1     1     1     1;
     1     0     1     0     0     1     0     1     0     0     0     1     0     0     0     0     1;
     1     0     1     1     1     1     0     1     0     0     0     1     0     1     1     1     1];
% -> 90
%EX9 - 
ex9=[1     1     1;
     0     0     1;
     1     1     1;
     1     0     0;
     1     1     1];
% -> 1472
%EX10 -
ex10=[1     0     0     0     0     1     0     0     0     1     0     0     0     1     1     1     1;
      1     0     0     0     0     1     0     0     1     0     1     0     0     1     0     0     0;
      1     0     1     1     1     1     0     0     1     0     1     0     0     1     1     1     1;
      1     0     1     0     0     1     0     1     0     0     0     1     0     0     0     0     1;
      1     0     1     1     1     1     0     1     0     0     0     1     0     1     1     1     1];
%-> 109

result_1 = q7(ex1);   % 1184
result_2 = q7(ex2);   % 80
result_3 = q7(ex3);   % 258
result_4 = q7(ex4);   % 1381
result_5 = q7(ex5);   % 1220
result_6 = q7(ex6);   % 1304
result_7 = q7(ex7);   % 220 
result_8 = q7(ex8);   % 90
result_9 = q7(ex9);   % 1472
result_10 = q7(ex10); % 109

function result = q7(p1)
img = im2double(imread("Exemplos\Ex10\image_TP1_2023_4.png"));
[lins,cols] = size(img);
result = 0;

[lins_filter,cols_filter] = size(p1);

% matriz_final = false(lins,cols);
% Ro       = imref2d([lins cols]);
% imxlim   = [-cols_filter/2 cols_filter/2];
% imylim   = [-lins_filter/2 lins_filter/2];
% Ri   = imref2d([lins_filter,cols_filter],imxlim,imylim);

for line=1:lins
    for col=1:cols
        if img(line,col) == 1
            if line-lins_filter < 1
                limit_1 = 1;
            else
                limit_1 = line-lins_filter;
            end
            if line+lins_filter > lins
                limit_2 = lins;
            else
                limit_2 = line+lins_filter;
            end
            if col-cols_filter < 1
                limit_3 = 1;
            else
                limit_3 = col-cols_filter;
            end
            if col+cols_filter > cols
                limit_4 = cols;
            else
                limit_4 = col+cols_filter;
            end

            neighboors = img(limit_1:limit_2,limit_3:limit_4);
            if sum(sum(neighboors(:))) < 2
                result = result+1;

%                 T = [1 0 col; 0 1 line; 0 0 1];
%                 tf = affine2d(T');
%                 tempA = imwarp(p1,Ri,tf,'OutputView',Ro);
%                 matriz_final = tempA + matriz_final;
            end
        end
    end
end
% figure
% imshow(matriz_final.*0.7+img)

% para a resolucao deste exercicio o mais simples acaba por ser verificar
% os vizinhos de cada um dos pontos da imagem. atentando a situacao de este
% estar numa das bordas e dai as clausulas if de limitacao. a distancia 
% minima para que se possa colocar um filtro vai ser a dimensao do filtro 
% em si. como se pode ver na imagem do enunciado dos dois mais juntos que 
% sao aceites e dos dois verticais que se recusam. Existe a vantagem de ao
% se contar a interacao A->B e B->A se removem os dois casos.
end