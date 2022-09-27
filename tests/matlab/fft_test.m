pd = makedist('Normal', 0, 1);

x = cast(1 + random(pd, [64, 64, 10]) + 1i * random(pd, [64, 64, 10]), 'single');

y = fft2(x);

yi = ifft2(x);

save('fft.mat', 'x', 'y', 'yi', '-v7.3');
