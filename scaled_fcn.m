%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%       FF-ReLU Network Training for FPGA implementation                %%
%%                                                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   In this file a feedforwad network of rectified linear units is        %
%   is trained using backpropagation. The weights are then scaled         %
%   to be used with discrete LIF-neurons implemented on an FPGA.          %
%                                                                         %
%   This is an extension of, P. U. Diehl, D. Neil, J. Binas, M. Cook,     %
%   S.-C. Liu, and M. Pfeiffer, “Fast-classifying, high-accuracy spiking  %
%   deep networks through weight and threshold balancing,” in 2015        %
%   International Joint Conference on Neural Networks (IJCNN), pp. 1–8,   %
%   IEEE.                                                                 %
%                                                                         %
%   Further informations can be found at...                               %
%                                                                         %
%   Authors: Martin Haar                                                  %
%            Max Geiselbrechtinger                                        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Load data
% Load DeepLearningToolbox
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));

% Load MNIST dataset
rand('state', 0);
load mnist_uint8;
train_x = 1 - double(train_x) / 255;
test_x  = 1 - double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% 2. Initialize training network
% Feedforward network - hiddenlayer constrained by FPGA size
nn = nnsetup([784 200 10]);
nn.activation_function = 'relu';
nn.output ='relu';

% Rescale weights
for i = 2 : nn.n   
    % Weights - choose as proposed in Dre05
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 2 * sqrt(3 / (nn.size(i - 1)));
end

%% 3. Train training network
% Set up learning constants
nn.learningRate = 0.002;
nn.momentum = 0.0005;
nn.learn_bias = 0;
% Train 70 epochs for optimal performance with scaled weights
opts.numepochs =  65;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);

%% 4. Test training network
% Test - should be 93.65% after 70 epochs
[er, train_bad] = nntest(nn, train_x, train_y);
fprintf('Training Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

%% 5. Simulate deployment network with float weights
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

%% 6. Scale deployment network weights
% Determine scale factor
% Set values to maximum negative and positive range of weights in HW
max_pos = 7;
max_neg = 8;
max_vals = zeros(1,nn.n-1);
min_vals = zeros(1,nn.n-1);
for i = 2 : nn.n
    max_vals(i-1) = abs(max(nn.W{i-1}(:)));
    min_vals(i-1) = abs(min(nn.W{i-1}(:)));
end
max_val = max(max_vals);
min_val = max(min_vals);
fprintf('Maximum elem before scaling: %f\n', max_val);
fprintf('Minimum elem before scaling: -%f\n', min_val);
scales = [0 0];
scales(1) = max_neg/min_val;
scales(2) = max_pos/max_val;
scale = min(scales);
fprintf('Scale value = %d\n', scale);

% Scale weights
for i = 2 : nn.n
    nn.scaled_W{i-1} = round(nn.W{i-1}.*scale);
end

fprintf('Maximum elem after scaling: %d\n', int16(max_val*scale));
fprintf('Minimum elem after scaling: -%d\n', int16(min_val*scale));

%% 7. Simulate deployment network with scaled weights
t_scaled_opts = struct;
t_scaled_opts.t_ref        = 0.000;
t_scaled_opts.threshold    = 1.0*round(scale); 
t_scaled_opts.dt           = 0.001;
t_scaled_opts.duration     = 0.035;
t_scaled_opts.report_every = 0.001;
t_scaled_opts.max_rate     =   900;

nn = nndisclifsim(nn, test_x, test_y, t_scaled_opts);
fprintf('Done.\n');
