function I=AddCircle(Z,x_0,y_0,r)
x=1:size(Z,2);
y=1:size(Z,1);
[X,Y]=meshgrid(x,y);
C=(((X-x_0).^2+(Y-y_0).^2)<=r*r);
I=Z;
I(C)=1;
end