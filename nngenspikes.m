function nngenspikes(nn, test_x, test_y, opts)
dt = opts.dt;
sim_len = round(opts.duration/opts.dt);
num_examples = 10; %size(test_x,1);

% Precache answers
[~,   ans_idx] = max(test_y');

% Time-stepped simulation
for n=1:num_examples
        % Create poisson distributed spikes from the input images
        %   (for all images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(sim_len, nn.size(1)) * rescale_fac;
        inp_image = spike_snapshot <= test_x(1:sim_len,:);

        ff_name = strcat("./spikes/matlab_spikes_", int2str(n-1), ".txt");
        writematrix(inp_image, ff_name);
        
        ff_name = strcat("./spikes/matlab_label_", int2str(n-1), ".txt");
        writematrix(ans_idx(n)-1, ff_name);
end

end