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

%% Simulate different configs and write results to csv files
% Load DeepLearningToolbox
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));

% Load MNIST dataset
rand('state', 0);
load mnist_uint8;
train_x = 1 - double(train_x) / 255;
test_x  = 1 - double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

opts.batchsize = 100;

t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    = 1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.035;
t_opts.report_every = 0.001;
t_opts.max_rate     =   800;

t_scaled_opts = struct;
t_scaled_opts.t_ref        = 0.000; 
t_scaled_opts.dt           = 0.001;
t_scaled_opts.duration     = 0.035;
t_scaled_opts.report_every = 0.001;
t_scaled_opts.max_rate     =   900;

num_avg = 20;
num_epochs = 15;
configs = {[784 200 10], [784 400 10], [784 800 10]};

% produce file for all configs
for c=1:size(configs)
    fprintf('Sim for config %d started.\n', c);
    % acc vector [ann_acc, snn_acc, snn_4_acc, snn_6_acc, snn_8_acc]
    tmp_acc = zeros(5,num_epochs);
    % iterate 10 times over each config
    for a=1:num_avg
        % init network
        clearvars nn;
        nn = nnsetup(configs{c});
        nn.activation_function = 'relu';
        nn.output ='relu';
        nn.learningRate = 0.002;
        nn.momentum = 0.0005;
        nn.learn_bias = 0;
        for i = 2 : nn.n   
            nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 2 * sqrt(3 / (nn.size(i - 1)));
        end
        % train and test network
        for e=1:num_epochs
            % train
            opts.numepochs =  10;
            nn = nntrain(nn, train_x, train_y, opts);
            % test ann
            [er, bad] = nntest(nn, test_x, test_y);
            tmp_acc(1,e) = tmp_acc(1,e) + (1-er)*100;
            % test snn
            nn = nnlifsim(nn, test_x, test_y, t_opts);
            tmp_acc(2,e) = tmp_acc(2,e) + nn.sim_acc;
            % test snn 4bit
            max_pos = 2^3-1;
            max_neg = 2^3;
            max_vals = zeros(1,nn.n-1);
            min_vals = zeros(1,nn.n-1);
            for i = 2 : nn.n
                max_vals(i-1) = abs(max(nn.W{i-1}(:)));
                min_vals(i-1) = abs(min(nn.W{i-1}(:)));
            end
            max_val = max(max_vals);
            min_val = max(min_vals);
            scales = [0 0];
            scales(1) = max_neg/min_val;
            scales(2) = max_pos/max_val;
            scale = min(scales);
            for i = 2 : nn.n
                nn.scaled_W{i-1} = round(nn.W{i-1}.*scale);
            end
            t_scaled_opts.threshold    = 1.0*round(scale);
            nn = nndisclifsim(nn, test_x, test_y, t_scaled_opts);
            tmp_acc(3,e) = tmp_acc(3,e) + nn.sim_acc;
            % test snn 6bit
            max_pos = 2^5-1;
            max_neg = 2^5;
            max_vals = zeros(1,nn.n-1);
            min_vals = zeros(1,nn.n-1);
            for i = 2 : nn.n
                max_vals(i-1) = abs(max(nn.W{i-1}(:)));
                min_vals(i-1) = abs(min(nn.W{i-1}(:)));
            end
            max_val = max(max_vals);
            min_val = max(min_vals);
            scales = [0 0];
            scales(1) = max_neg/min_val;
            scales(2) = max_pos/max_val;
            scale = min(scales);
            for i = 2 : nn.n
                nn.scaled_W{i-1} = round(nn.W{i-1}.*scale);
            end
            t_scaled_opts.threshold    = 1.0*round(scale);
            nn = nndisclifsim(nn, test_x, test_y, t_scaled_opts);
            tmp_acc(4,e) = tmp_acc(4,e) + nn.sim_acc;
            % test snn 8bit
            max_pos = 2^7-1;
            max_neg = 2^7;
            max_vals = zeros(1,nn.n-1);
            min_vals = zeros(1,nn.n-1);
            for i = 2 : nn.n
                max_vals(i-1) = abs(max(nn.W{i-1}(:)));
                min_vals(i-1) = abs(min(nn.W{i-1}(:)));
            end
            max_val = max(max_vals);
            min_val = max(min_vals);
            scales = [0 0];
            scales(1) = max_neg/min_val;
            scales(2) = max_pos/max_val;
            scale = min(scales);
            for i = 2 : nn.n
                nn.scaled_W{i-1} = round(nn.W{i-1}.*scale);
            end
            t_scaled_opts.threshold    = 1.0*round(scale);
            nn = nndisclifsim(nn, test_x, test_y, t_scaled_opts);
            tmp_acc(5,e) = tmp_acc(5,e) + nn.sim_acc;
        end
    end
    tmp_acc = tmp_acc .* 1/num_avg;
    % write result
    f_name = strcat('csv/results_', int2str(c), '.csv');
    writematrix(tmp_acc, f_name);
    fprintf('Sim for config %d finished.\n', c);
end

