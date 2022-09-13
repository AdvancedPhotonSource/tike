import core.*

pd = makedist('Normal');

x0 = cast(random(pd, [32, 32, 5]) + 1i * random(pd, [32, 32, 5]), 'single');

[x1, I_n, eval] = probe_modes_ortho(x0);

save('ortho-in.mat', 'x0', '-v7.3');

save('ortho-out.mat', 'I_n', 'eval', 'x1', '-v7.3');
