function nn=nndisclifsim(nn, test_x, test_y, opts)
dt = opts.dt;
nn.performance = [];
num_examples = size(test_x,1);

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    nn.layers{l}.disc_mem = int32(blank_neurons);     
    nn.layers{l}.disc_sum_spikes = int32(blank_neurons);
end

% Precache answers
[~,   ans_idx] = max(test_y');

% Time-stepped simulation
for t=dt:dt:opts.duration
        % Create poisson distributed spikes from the input images
        %   (for all images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(size(test_x)) * rescale_fac;
        inp_image = spike_snapshot <= test_x;

        nn.layers{1}.spikes = inp_image;
        nn.layers{1}.disc_sum_spikes = nn.layers{1}.disc_sum_spikes + int32(inp_image);
        
        nn.disc_input_spikes{int8(t/dt)} = inp_image;
        
        for l = 2 : numel(nn.size)
            % Get input impulse from incoming spikes
            impulse = int32(nn.layers{l-1}.spikes*nn.scaled_W{l-1}');
            % Add input to membrane potential
            nn.layers{l}.disc_mem = nn.layers{l}.disc_mem + impulse;
            % Check for spiking
            nn.layers{l}.spikes = nn.layers{l}.disc_mem >= int32(opts.threshold);
            % Reset
            nn.layers{l}.disc_mem(nn.layers{l}.spikes) = 0;
            % Store result for analysis later
            nn.layers{l}.disc_sum_spikes = nn.layers{l}.disc_sum_spikes + int32(nn.layers{l}.spikes);            
        end
        if(mod(round(t/dt),round(opts.report_every/dt)) == round(opts.report_every/dt)-1)
            [~, guess_idx] = max(nn.layers{end}.disc_sum_spikes');
            acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
            fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
            nn.performance(end+1) = acc;
        else
            fprintf('.');            
        end
end
    
    
% Get answer
[~, guess_idx] = max(nn.layers{end}.disc_sum_spikes');
acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
fprintf('\nFinal spiking accuracy: %2.2f%%\n', acc);

end
