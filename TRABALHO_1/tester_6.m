clear all
close all
clc

%enunciado - 
%EX1 - 4,     9,     3,     5 -> 42
%EX2 - 6,     7,     1,     9 -> 3
%EX3 - 6,     7,     5,     9 -> 44
%EX4 - 6,     3,     5,     9 -> 20
%EX5 - 7,     5,     1,    11 -> 5
%EX6 - 6,     3,     5,     9 -> 11
%EX7 - 5,     7,     3,     7 -> 227
%EX8 - 7,     9,     3,    11 -> 236
%EX9 - 5,     3,     3,     7 -> 29
%EX10 - 6,     5,     5,     9 -> 184

result_1 = q6(4,     9,     3,     5);
result_2 = q6(6,     7,     1,     9);
result_3 = q6(6,     7,     5,     9);
result_4 = q6(6,     3,     5,     9);
result_5 = q6(7,     5,     1,    11);
result_6 = q6(6,     3,     5,     9);
result_7 = q6(5,     7,     3,     7);
result_8 = q6(7,     9,     3,    11);
result_9 = q6(5,     3,     3,     7);
result_10 = q6(6,     5,     5,     9);

function result = q6(p1,p2,p3,p4)
%p1=5; p2 =3; p3 = 5;p4=7;
img = im2double(imread("Exemplos\Ex10\image_TP1_2023_3.png"));
[lins,cols] = size(img);

img(1:p1-1,:)=0;img(end-p1+2:end,:)=0;
img(:,1:p1-1)=0;img(:,end-p1+2:end)=0;
%para por os limites como e pedido no enunciado para serem sempre postas a
%0

filter = [-1 -1 -1;-1 8 -1;-1 -1 -1];
isolated_points = filter2(filter,img); % os que forem 8 sao brancos isolados;
% e os que forem -8 sao isolados pretos

for line=p1:(lins-p1+2) %so vale a pena comecar a analisar a partir dos pontos que podem ser isolados
    for col=p1:(cols-p1+2)
        if isolated_points(line,col)==8 || isolated_points(line,col)==-8
            limit = floor(p2/2); % para se ter a distancia de onde vai comecar
            %para que nao se excedam os limites do array pode-se fazer esta
            %conta, seria talvez mais eficiente fazer um padding a imagem e
            %assim analisava-se sempre dentro e seriam escusados os 4 if a
            %cada iteração
            if line-limit < 1
               limit_1 = 1;
            else
               limit_1 = line-limit;
            end
            if line+limit > lins
               limit_2 = lins;
            else
               limit_2 = line+limit;
            end
            if col-limit < 1
               limit_3 = 1;
            else
               limit_3 = col-limit;
            end
            if col+limit > cols
               limit_4 = cols;
            else
               limit_4 = col+limit;
            end

            img(limit_1:limit_2,limit_3:limit_4) = 1; % fazer a regiao a 1s
        end
    end
end

result = 1; %faz-se sempre pelo menos uma vez
%figure
while true 
    img_new=medfilt2(img,[p3,p4]);
    if img_new == img % se a imagem agora for igual a anterior e porque se pode parar
        break
    else
        result = result+1; % e preciso fazer mais uma vez e a imagem obtida agora passa a ser a da proxima
        img = img_new;
    end
    %imshow(img)
    %pause(0.01)
end

end