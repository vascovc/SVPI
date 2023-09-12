% Vasco Costa 97746
function [cR,cG,cB,x] = rgbhist(A)
    [cR,xR] = imhist(A(:,:,1));
    [cG,xG] = imhist(A(:,:,2));
    [cB,xB] = imhist(A(:,:,3));
    x = [xR xG xB];
end