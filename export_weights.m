%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Generates files with the scaled weight matrizes for each layer        %
%   of the network.                                                       %
%                                                                         %
%   Authors: Martin Haar                                                  %
%            Max Geiselbrechtinger                                        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_weights(nn, path)

for i = 2 : nn.n
    f_name = strcat(path, "weights_layer_", int2str(i-1), ".txt");
    writematrix(int16(nn.scaled_W{i-1}), f_name);
end

end

