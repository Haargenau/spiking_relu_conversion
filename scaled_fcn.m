%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);
load mnist_uint8;
train_x = 1 - double(train_x) / 255;
test_x  = 1 - double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
% Initialize net
%nn = nnsetup([784 1200 1200 10]);
nn = nnsetup([784 200 10]);
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between as proposed in Dre05
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 2 * sqrt(3 / (nn.size(i - 1)));
end
%% ReLU Train
% Set up learning constants
nn.activation_function = 'relu';
nn.output ='relu';
nn.learningRate = 0.002;
nn.momentum = 0.0005;
nn.learn_bias = 0;
% Train 70 epochs for optimal performance with scaled weights
opts.numepochs =  70;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
%% ReLU Test
% Test - should be 93.65% after 70 epochs
[er, train_bad] = nntest(nn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
%% Spike-based Testing of Fully-Connected NN
% Sim - should be 93.24%
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    = 1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.035;
t_opts.report_every = 0.001;
t_opts.max_rate     =   800;

nn = nnlifsim(nn, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Determine scale factor (manually)
max_pos = 7;
max_neg = 8;
max_vals = [0 0];
min_vals = [0 0];
for i = 2 : nn.n
    max_vals(i-1) = abs(max(nn.W{i-1}(:)));
    min_vals(i-1) = abs(min(nn.W{i-1}(:)));
end
max_val = max(max_vals);
min_val = max(min_vals);
fprintf('Maximum elem = %f\n', max_val);
fprintf('Minimum elem = %f\n', min_val);
scales = [0 0];
scales(1) = max_neg/min_val;
scales(2) = max_pos/max_val;
scale = min(scales);
fprintf('Scale value = %d\n', round(scale));
% maybe scale layers independently
%% Write scaled weights to file
for i = 2 : nn.n
    % scale, cast and write to file
    f_name = strcat("weights_200_neurons_4_bit_layer_", int2str(i-1), ".txt");
    writematrix(int8(nn.W{i-1}.*double(scale)), f_name);
end
%% Import scaled weights
nn_scaled = nnsetup([784 200 10]);
nn_scaled.activation_function = 'relu';
nn_scaled.output ='relu';
for i = 2 : nn_scaled.n
    f_name = strcat("weights_200_neurons_4_bit_layer_", int2str(i-1), ".txt");
    nn_scaled.W{i-1} = readmatrix(f_name);
end
%% Test scaled spiking network
t_scaled_opts = struct;
t_scaled_opts.t_ref        = 0.000;
t_scaled_opts.threshold    = 1.0*round(scale);
t_scaled_opts.dt           = 0.001;
t_scaled_opts.duration     = 0.035;
t_scaled_opts.report_every = 0.001;
t_scaled_opts.max_rate     = 900;

nn_scaled = nnlifsim(nn_scaled, test_x, test_y, t_scaled_opts);
fprintf('Done.\n');
%% Test scaled spiking network on HW-simulation
t_scaled_opts = struct;
t_scaled_opts.t_ref        = 0.000;
t_scaled_opts.threshold    = 1.0*round(scale);
t_scaled_opts.dt           = 0.001;
t_scaled_opts.duration     = 0.035;
t_scaled_opts.report_every = 0.001;
t_scaled_opts.max_rate     = 900;

nn_scaled = nndisclifsim(nn_scaled, test_x, test_y, t_scaled_opts);
fprintf('Done.\n');

%% Generate input spikes
nngenspikes(nn_scaled, test_x, test_y, t_scaled_opts)
