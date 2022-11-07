load('forward-in.mat');
pd = makedist('Normal', 0, 1);
peps = cast(padarray(imread('peppers.png'), [64, 0, 0]), 'single');
corn = cast(imread('ngc6543a.jpg'), 'single');
coins = cast(imread('coins_2048.png'), 'single');
satyre = cast(imread('satyre_2048.png'), 'single');

Np_p = self.Np_p;
self.probe{1} = 0 .* self.probe{1} + peps(:, :, 1) + 1i .* corn(1:512, 1:512, 2);
probe0 = gather(self.probe{1});
self.probe{2} = 0 .* self.probe{2} + peps(:, :, 2) + 1i .* corn(1:512, 1:512, 3);
probe1 = gather(self.probe{2});
probe_evolution = self.probe_evolution(g_ind, :);
z_distance = self.z_distance;
modes = self.modes;
self.object{1} = 0 .* self.object{1} + satyre(1:1568, 1:1568, 2) + 1i .* coins(1:1568, 1:1568);
object = gather(self.object{1});
object_modes = par.object_modes;
probe_modes = par.probe_modes;
variable_intensity = par.variable_intensity;
Nlayers = par.Nlayers;
apply_subpix_shift = par.apply_subpix_shift;
apodwin = cache.apodwin;
cache.oROI_s{1}{1}(g_ind(1), :) = 0;
cache.oROI_s{1}{1}(g_ind(2), :) = size(object, 1) - 512;
cache.oROI_s{1}{1}(g_ind(3), :) = 0;
cache.oROI_s{1}{1}(g_ind(4), :) = size(object, 1) - 512;
cache.oROI_s{1}{1}(g_ind(5), :) = 512;
cache.oROI_s{1}{2}(g_ind(1), :) = 0;
cache.oROI_s{1}{2}(g_ind(2), :) = 0;
cache.oROI_s{1}{2}(g_ind(3), :) = size(object, 2) - 512;
cache.oROI_s{1}{2}(g_ind(4), :) = size(object, 2) - 512;
cache.oROI_s{1}{2}(g_ind(5), :) = 512;
positions0 = cache.oROI_s{1}{1}(g_ind, 1);
positions1 = cache.oROI_s{1}{2}(g_ind, 1);

save('forward-in1.mat', 'Np_p',...
'probe0',...
'probe1',...    
'probe_evolution',...
'z_distance',...
'modes',...
'object',...
'object_modes',...
'probe_modes',...
'variable_intensity',...
'Nlayers',...
'apply_subpix_shift',...
'positions0', 'positions1', ...
'apodwin', '-v7.3');

import engines.GPU.shared.*
import engines.GPU.GPU_wrapper.*
import engines.GPU.LSQML.*
import math.*
import utils.*
import plotting.*

if isempty(obj_proj{1})
    for ll = 1:par.object_modes
        obj_proj{ll} = Gzeros([self.Np_p, 0], true);
    end
end
probe = self.probe; 

% get illumination probe 
for ll = 1:par.probe_modes
    if (ll == 1 && (par.variable_probe || par.variable_intensity))
        % add variable probe (OPRP) part into the constant illumination 
        probe{ll,1} =  get_variable_probe(self.probe{ll}, self.probe_evolution(g_ind,:),p_ind{ll});
    else
        % store the normal (constant) probe(s)
        probe{ll,1} = self.probe{min(ll,end)}(:,:,min(end,p_ind{ll}),1);
    end

%     if (ll == 1 && par.apply_subpix_shift && isinf(self.z_distance(end)))  || is_used(par,'fly_scan')
%         % only in farfield mode 
%         probe{ll} = apply_subpx_shift(probe{ll}, self.modes{min(end,ll)}.sub_px_shift(g_ind,:) );
%     end
%     if (ll == 1)
%         probe{ll} = apply_subpx_shift_fft(probe{ll}, self.modes{1}.probe_fourier_shift(g_ind,:)); 
%     end
end

% get projection of the object and probe 
for layer = 1:par.Nlayers
   for ll = 1:max(par.object_modes, par.probe_modes)
       llo = min(ll, par.object_modes); 
       llp = min(ll, par.probe_modes); 
        % get objects projections 
        obj_proj{llo} = get_views(self.object, obj_proj{llo},layer_ids(layer),llo, g_ind, cache, scan_ids,[]);
        if (ll == 1 && par.apply_subpix_shift && ~isinf(self.z_distance(end)))
            % only in nearfield mode , apply shift in the opposite direction 
            obj_proj{ll} = apply_subpx_shift(obj_proj{ll} .* cache.apodwin, -self.modes{min(end,ll)}.sub_px_shift(g_ind,:) ) ./ cache.apodwin;
        end

        % get exitwave after each layer
        psi{ll} =probe{llp,layer} .* obj_proj{llo};
        % fourier propagation  
%         [psi{ll}] = fwd_fourier_proj(psi{ll} , self.modes{layer}, g_ind);  
        if par.Nlayers > 1
             probe{llp,layer+1} = psi{llp};
        end
   end
end

% figure();
% imshow(real(object), [0, 255]);
% savefig('forward-00.png');

% figure();
% for i = 1:5
%     subplot(5, 1, i);
%     imshow(real(obj_proj{1}(:, :, i)), [0, 255]);
% end
% savefig('forward-01.png');

psi1 = gather(psi{1});
psi2 = gather(psi{2});
obj_proj_ = gather(obj_proj{1});
probe0 = gather(probe{1});
probe1 = gather(probe{2});

save('forward-out.mat', 'probe0', 'probe1', 'obj_proj_', '-v7.3');
save('forward-out2.mat', 'psi2', '-v7.3');
save('forward-out1.mat', 'psi1', '-v7.3');

