function I=AddHeart(Z,x_0,y_0)
x = 1:size(Z,2);
y = 1:size(Z,1);
[X,Y]=meshgrid(x,y);
X=X/30;Y=Y/30;
C = ( ( (X-x_0).^2+(Y-y_0).^2-1 ).^3-(X-x_0).^2.*(Y-y_0).^3 <= 0);
I=Z;
I(C)=1;
I = flipud(I);
end