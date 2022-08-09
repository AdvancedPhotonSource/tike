import engines.GPU.GPU_wrapper.*

pd = makedist('Normal');

probe = cast(random(pd, [236, 236]) + 1i * random(pd, [236, 236]), 'single');
obj_proj_tmp = cast(random(pd, [236, 236, 120]) + 1i * random(pd, [236, 236, 120]), 'single');
chi_tmp = cast(random(pd, [236, 236, 120]) + 1i * random(pd, [236, 236, 120]), 'single');
g_ind_tmp = cast(1:120, 'int32');
probe_evolution = cast(random(pd, [120, 1]), 'single');
kk = 1;

save('variable_intensity_input0.mat', 'probe', 'obj_proj_tmp', '-v7.3');
save('variable_intensity_input1.mat', 'chi_tmp', 'g_ind_tmp', 'probe_evolution', '-v7.3');

% correction to account for variable intensity
mean_probe = probe(:,:,kk,1); 
% compare P*0 and chi to estimate best update of the intensity 
[nom, denom] = get_coefs_intensity(chi_tmp, mean_probe, obj_proj_tmp);

probe_evolution(g_ind_tmp,1)  = probe_evolution(g_ind_tmp,1) + 0.1 * squeeze(Ggather(sum2(nom)./ sum2(denom)));

save('variable_intensity_output.mat', 'probe_evolution', 'nom', 'denom', '-v7.3');

function [nom1, denom1] = get_coefs_intensity(xi, P, O)
    OP = O.*P;
    nom1 = real(conj(OP) .* xi);
    denom1 = abs(OP).^2; 
end

function x = sum2(x)
    x = sum(sum(x,1),2);
end