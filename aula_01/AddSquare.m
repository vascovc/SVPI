function X=AddSquare(I,line,col)
% function to add a white square with dimensions (line,col) to image 1
I(line:line+10-1,col:col+10-1)=1;
X=I;
end