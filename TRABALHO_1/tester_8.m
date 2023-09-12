clear all
close all
clc

%enunciado - 
%EX1 - 403,    18 -> 1
%EX2 - 460,    12 -> 2
%EX3 - 403,    15 -> 6
%EX4 - 460,    14 -> 7
%EX5 - 460,    15 -> 2
%EX6 - 460,    14 -> 2
%EX7 - 460,    16 -> 1
%EX8 - 460,    13 -> 4
%EX9 - 403,    17 -> 5
%EX10 - 403,    14 -> 5

result_1 = q8(403,    18);
result_2 = q8(460,   12);
result_3 = q8(403,    15);
result_4 = q8(460,    14);
result_5 = q8(460,    15);
result_6 = q8(460,    14);
result_7 = q8(460,    16);
result_8 = q8(460,    13);
result_9 = q8(403,    17);
result_10 = q8(403,    14);

function result = q8(p1,p2)
img = im2double(imread("image_TP1_2023_5_trial_1.png"));
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
%         final_forms = final_forms|mask; % juntar a peca obtida individual
%         result = result+1;
      end
end
subplot(1,2,2)
imshow(final_forms)
end