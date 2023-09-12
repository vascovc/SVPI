function Z = MultiRegionBin(A,N,M)
[M_A,N_A] = size(A);

lines = floor(linspace(0,M_A,N));
columns = floor(linspace(0,N_A,M));

Z = zeros([M_A, N_A]);
for l=2:N
    for c=2:M
        img = A(lines(l-1)+1:lines(l),columns(c-1)+1:columns(c));
        mask = graythresh(img);
        Z(lines(l-1)+1:lines(l),columns(c-1)+1:columns(c)) = imbinarize(img,mask);
        if mask < mean(img)
            Z(lines(l-1)+1:lines(l),columns(c-1)+1:columns(c)) = 1-Z(lines(l-1)+1:lines(l),columns(c-1)+1:columns(c));
        end
    end
end
end