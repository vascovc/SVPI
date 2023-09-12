clear all
close all
clc

%enunciado - 
%EX1 - 4,   111,   261,   210,   120,     3,    81,   123,    77,   160 -> 310
%EX2 - 4,    85,   205,   265,   212,     3,   121,   175,   157,   179 -> 52
%EX3 - 2,    77,   107,    74,   128,     2,    85,   111,    14,   103 -> 16
%EX4 - 2,   101,   141,   435,    57,     4,    63,   179,   362,     9 -> 39
%EX5 - 4,    67,   167,   298,   168,     3,    99,   133,   389,   198 -> 97
%EX6 - 2,   101,   141,   435,    57,     4,    63,   179,   362,     9 -> 22
%EX7 - 2,   109,   135,   261,   325,     3,   119,   165,   337,  278 -> 166
%EX8 - 2,   117,   147,   512,   145,     2,   113,   137,   431,   108 -> 6
%EX9 - 4,    87,   209,   135,   158,     2,    75,   127,   240,   212 -> 74
%EX10 -2,    67,   121,   184,    67,     2,    91,   147,   251,    27 -> 93

%result = q5(1,   65,   105,   98,   173,     2,    89,   191,    174,   119 );
result_1 = q5(4,   111,   261,   210,   120,     3,    81,   123,    77,   160 );
result_2 = q5(4,    85,   205,   265,   212,     3,   121,   175,   157,   179);
result_3 = q5(2,    77,   107,    74,   128,     2,    85,   111,    14,   103);
result_4 = q5(2,   101,   141,   435,    57,     4,    63,   179,   362,     9);
result_5 = q5(4,    67,   167,   298,   168,     3,    99,   133,   389,   198);
result_6 = q5(2,   101,   141,   435,    57,     4,    63,   179,   362,     9);
result_7 = q5(2,   109,   135,   261,   325,     3,   119,   165,   337,  278);
result_8 = q5(2,   117,   147,   512,   145,     2,   113,   137,   431,   108);
result_9 = q5(4,    87,   209,   135,   158,     2,    75,   127,   240,   212);
result_10 = q5(2,    67,   121,   184,    67,     2,    91,   147,   251,    27);

function result = q5(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)

tipo_1 = p1; tipo_2 = p6;
r1_min = p2;r1_max = p3; x1=p4;y1=p5;
r2_min = p7;r2_max = p8; x2=p9;y2=p10;

img = im2double(imread("Exemplos\Ex10\image_TP1_2023_3.png"));
[lins,cols] = size(img);

if tipo_1==1 %circular
    mask_1 = draw_type_1(lins,cols,r1_min,r1_max,x1,y1);
elseif tipo_1==2 %quadrada
    mask_1 = draw_type_2(lins,cols,r1_min,r1_max,x1,y1);
elseif tipo_1==3%circular fora; quadrangular dentro
    mask_1 = draw_type_3(lins,cols,r1_min,r1_max,x1,y1);
else% quadrangular fora; circular dentro
    mask_1 = draw_type_4(lins,cols,r1_min,r1_max,x1,y1);
end

if tipo_2==1 %circular
    mask_2 = draw_type_1(lins,cols,r2_min,r2_max,x2,y2);
elseif tipo_2==2 %quadrada
    mask_2 = draw_type_2(lins,cols,r2_min,r2_max,x2,y2);
elseif tipo_2==3%circular fora; quadrangular dentro
    mask_2 = draw_type_3(lins,cols,r2_min,r2_max,x2,y2);
else% quadrangular fora; circular dentro
    mask_2 = draw_type_4(lins,cols,r2_min,r2_max,x2,y2);
end

mask_join = and(mask_1,mask_2); % fazem-se as mascaras individuais e depois 
% ve-se o que ha na mascara em comum. para depois somar os pixels, valor 1 
% nessa regiao da imagem
result = sum(sum(img(mask_join)));

% figure
% subplot(1,4,1)
% imshow(mask_1)
% subplot(1,4,2)
% imshow(mask_2)
% subplot(1,4,3)
% imshow(mask_1+mask_2)
% subplot(1,4,4)
% imshow(mask_join)
end

%decidi resolver tudo com ciclos for de forma a se conseguir evitar a
%questao das margens da mascara ficarem de fora da imagem
function mask = draw_type_1(lins,cols,r_min,r_max,x,y) % circular
    r_max = r_max^2;
    r_min = r_min^2; % nao e necessario calcular sempre, mais vale armazenar o valor ao quadrado
    mask = false(lins,cols);
    for line=1:lins
        for col=1:cols
            if ((x-col)^2 + (y-line)^2) <= r_max && ((x-col)^2 + (y-line)^2) >= r_min
                % o ponto comprir a condicao de estar dentro das duas
                % condicoes circulares
                mask(line,col) = 1;
            end
        end
    end
end

%aqui a ideia mais simples acaba por ser considerar a mais exterior com o
%valor de 1 e depois retirar o interior, dai nao se fazer com o igual nos
%mais interiores
function mask = draw_type_2(lins,cols,r_min,r_max,x,y) % quadrada
    mask = false(lins,cols);
    for line=1:lins
        if line >= (y-floor(r_max/2)) && line <= (y+floor(r_max/2)) % passa-se logo para aqui para poupar umas iteracoes
            for col=1:cols
               if col >= (x-floor(r_max/2)) && col <= (x+floor(r_max/2))
                    %quadrado exterior
                    mask(line,col) = 1; % faz-se o quadrado mais de fora e considera-se com 1
                    % contudo se for dos mais interiores muda-se para 0
                    if col > (x-floor(r_min/2)) && col < (x+floor(r_min/2))
                    if line > (y-floor(r_min/2)) && line < (y+floor(r_min/2))
                        %quadrado interior
                        mask(line,col) = 0; % se fizer parte do mais interior muda-se o ponto para 0
                    end
                    end
               end
           end
        end
    end
end

function mask = draw_type_3(lins,cols,r_min,r_max,x,y)%circular fora; quadrangular dentro
    mask = false(lins,cols);
    r_max = r_max^2; % so se calcula uma vez
    for line=1:lins
        for col=1:cols
            if ((x-col)^2 + (y-line)^2) <= r_max
                mask(line,col) = 1;
                if col > (x-floor(r_min/2)) && col < (x+floor(r_min/2))
                if line > (y-floor(r_min/2)) && line < (y+floor(r_min/2))
                    %quadrado interior
                    mask(line,col) = 0;
                end
                end
            end
        end
    end
end

function mask = draw_type_4(lins,cols,r_min,r_max,x,y)% quadrangular fora; circular dentro
    mask = false(lins,cols);
    r_min = r_min^2;
    for line=1:lins
        if line >= (y-floor(r_max/2)) && line <= (y+floor(r_max/2)) % mais uma vez podem-se poupar iteracoes logo aqui
            for col=1:cols
               if col >= (x-floor(r_max/2)) && col <= (x+floor(r_max/2)) 
                    %quadrado exterior
                    mask(line,col) = 1;
                    if ((x-col)^2 + (y-line)^2) < r_min % circulo de dentro, aberto
                        mask(line,col) = 0;
                    end
               end
           end
        end
    end
end

