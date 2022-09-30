pd = makedist('Uniform', -1, 1);

x0 = gpuArray(cast(1 + random(pd, [64, 64, 10]) + 1i * random(pd, [64, 64, 10]), 'double'));
x = x0;
error = zeros(100);

for i = 1:100
    x = fft2(x);
    x = ifft2(x);
    if i == 1
        y_double = gather(x);
    end
    error(i) = mean(reshape(abs((x-x0)./x0), 1, []));
end

figure();
plot(1:100, error);
title({['Mean Absolute Relative Error'], ['for Double Precision on GPU']});
xlabel('Calls to FFT/iFFT');

x0_double = gather(x0);

x0 = cast(x0, 'single');
x = x0;
error = zeros(100);

for i = 1:100
    if i == 1
        x0_single = gather(x);
    end
    x = fft2(x);
    if i == 1
        y_single = gather(x);
    end
    x = ifft2(x);
    if i == 1
        x_single = gather(x);
    end
    error(i) = mean(reshape(abs((x-x0)./x0), 1, []));
end

figure();
plot(1:100, error);
title({['Mean Absolute Relative Error'], ['for Single Precision on GPU']});
xlabel('Calls to FFT/iFFT');

save('fft.mat', 'x0_double', 'y_double', 'x_single', 'x0_single', 'y_single', '-v7.3');
