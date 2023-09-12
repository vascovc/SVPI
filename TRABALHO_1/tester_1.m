clear all
close all
clc

%enunciado - 108,46,192,89,227
%EX1 - 96,    30,   163,     39,   173 -> 1678
%EX2 - 117,   74,   221,     75,   221 -> 1113
%EX3 -  88,   84,   211,     81,   198 -> 1294
%EX4 - 116,  115,   257,     24,   172 -> 1081
%EX5 - 114,   42,   185,     44,   189 -> 518
%EX6 - 116,   115,   257,    24,   172 -> 1715
%EX7 - 113,    29,   176,   150,   302 -> 483
%EX8 - 104,    24,   157,    27,   160 -> 569
%EX9 - 88,     85,   210,   164,   282 -> 1004
%EX10 - 84,    47,   161,    74,   190 -> 380

%enunciado
%result = q1(108,46,192,89,227);
result_1 = q1(96,30,163,39,173);
result_2 = q1(117,   74,   221,     75,   221);
result_3 = q1(88,   84,   211,     81,   198);
result_4 = q1(116,  115,   257,     24,   172);
result_5 = q1(114,   42,   185,     44,   189);
result_6 = q1(116,   115,   257,    24,   172);
result_7 = q1(113,    29,   176,   150,   302);
result_8 = q1(104,    24,   157,    27,   160);
result_9 = q1(88,     85,   210,   164,   282);
result_10 = q1(84,    47,   161,    74,   190);

function result = q1(p1,p2,p3,p4,p5)

img = im2double(imread("Exemplos\Ex10\image_TP1_2023_1.png")); %591,787

[M_A,N_A] = size(img); % m linhas, n colunas
horizontais = false([M_A, N_A]);
verticais = false([M_A, N_A]);
% criar mascaras tanto para as linhas veriticais como para as horizontais

%nao sao precisos mais if porque so as duas margens tanto da direita como a
%mais inferior podem ultrapassar os limites. como nao existe sobreposicao
%nao e necessario fazer mais

if (p3+p1) > M_A
    %disp('here')
    line_2 = M_A;
else
    line_2 = p3+p1-1;
end

if (p5+p1) > N_A
    %disp('or here')
    col_2 = N_A;
else
    col_2 = p5+p1-1;
end

horizontais(p2:p2+p1-1,:) = 1;
horizontais(p3:line_2,:) = 1;

verticais(:,p4:p4+p1-1) = 1;
verticais(:,p5:col_2) = 1;
% figure(1)
% imshow(verticais+horizontais) % mostrar ambas as mascaras

mask_join = and(verticais,horizontais);
% figure(2)
% imshow(mask)
result = sum(sum(img(mask_join)));

%criam-se as mascaras tanto das linhas horizontais como verticais, tendo
%atencao a possibilidade destas sairem fora da imagem dada, e depois faz-se
%o "and" delas e contam-se os pixeis brancos, que vao ter o valor 1, nessa
%mascara
end