%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Generates files with poisson distributed spike vectors for each       %
%   image of MNIST test set.                                              %
%                                                                         %
%   Authors: Martin Haar                                                  %
%            Max Geiselbrechtinger                                        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nngenspikes(nn, test_x, test_y, opts, path)
dt = opts.dt;
sim_len = round(opts.duration/opts.dt);
num_examples = size(test_x,1);

% Transform labels to decimals
[~,   ans_idx] = max(test_y');

% Generate poisson spikes for each image
for n=1:num_examples
        % Create poisson distributed spikes from the input images
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(sim_len, nn.size(1)) * rescale_fac;
        inp_image = spike_snapshot <= (ones(sim_len,1)*test_x(n,:));

        % Write nn.size(1) x sim_len matrix of spikes to file
        ff_name = strcat(path, "spikes_", int2str(n-1), ".txt");
        writematrix(inp_image, ff_name);
        
        % Write decimal label value to file
        ff_name = strcat(path, "label_", int2str(n-1), ".txt");
        writematrix(ans_idx(n)-1, ff_name);
end

end