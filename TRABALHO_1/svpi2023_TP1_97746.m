function [nn,a1,a2,a3,a4,a5,a6,a7,a8] = svpi2023_TP1_97746()
nn = 97746;
a1 = q1(115  ,  57 ,  198  ,  56 ,  206);
a2 = q2(13  ,  19  ,   2);
a3 = q3([0     1     0;
         1     1     1;
         0     1     0]);
a4 = q4(94,    32,    82,    -35);
a5 = q5(2,    87  , 147  , 480  , 140  ,   2  ,  69 ,  119 ,  404  , 176);
a6 = q6(7  ,   5  ,   1  ,  11);
a7 = q7([1     1     1;
         0     0     1;
         1     1     1;
         1     0     0;
         1     1     1]);
a8 = q8(460,    13);
end

function result = q1(p1,p2,p3,p4,p5)
img = im2double(imread("image_TP1_2023_1.png")); %591,787
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

function result = q2(p1,p2,p3)

img = im2double(imread("image_TP1_2023_1.png"));
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

function result = q3(p1)

img = im2double(imread("image_TP1_2023_2.png"));
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

function result = q5(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)

tipo_1 = p1; tipo_2 = p6;
r1_min = p2;r1_max = p3; x1=p4;y1=p5;
r2_min = p7;r2_max = p8; x2=p9;y2=p10;

img = im2double(imread("image_TP1_2023_3.png"));
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

function result = q6(p1,p2,p3,p4)
%p1=5; p2 =3; p3 = 5;p4=7;
img = im2double(imread("image_TP1_2023_3.png"));
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

function result = q7(p1)
img = im2double(imread("image_TP1_2023_4.png"));
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
% imshow(matriz_final.*0.5+img)

% para a resolucao deste exercicio o mais simples acaba por ser verificar
% os vizinhos de cada um dos pontos da imagem. atentando a situacao de este
% estar numa das bordas e dai as clausulas if de limitacao. a distancia 
% minima para que se possa colocar um filtro vai ser a dimensao do filtro 
% em si. como se pode ver na imagem do enunciado dos dois mais juntos que 
% sao aceites e dos dois verticais que se recusam. Existe a vantagem de ao
% se contar a interacao A->B e B->A se removem os dois casos.
end

function result = q8(p1,p2)
img = im2double(imread("image_TP1_2023_5.png"));
img_edges = edge(img,'log'); % obter so os contornos, este foi o melhor em 
% cofamparacao aos restantes, como e de segunda ordem pelo brilho das imagens
% acabou por resultar bem. noutras situacoes poderia ser necessario aplicar
% mais tratamento as imagens por causa disso

result = 0; %pode nao haver nenhuma individual

[label,num] = bwlabel(img_edges); % saber quantas regioes fechadas existem e dar um label
val_menor = p1-p2;
val_maior = p1+p2;
% figure
% subplot(1,2,1)
% imshow(img_edges)

% final_forms = false(size(img_edges)); %matriz para ficar so com as bordas
% finais das pecas individuais
for i=1:num
      mask = (label==i);
      boundary_array = bwboundaries(mask,'noholes'); %procurar so pelo objeto pai e filho
      boundary = boundary_array{1}; % so a mais exterior

      tamanho = length(boundary); % obtem-se assim o comprimento da aresta exterior da peca
      if tamanho >= val_menor && tamanho <= val_maior % se estiver dentro do range dado
        %final_forms = final_forms|mask; % juntar a peca obtida individual
        result = result+1;
      end
end
% subplot(1,2,2)
% imshow(final_forms)
end