function K = PerspectiveTransform(alpha, center)
K = [alpha(1) 0 center(1) 0
        0    alpha(2) center(2) 0
        0 0 1 0    ];
end