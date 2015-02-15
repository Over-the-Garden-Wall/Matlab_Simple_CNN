classdef conv_neural_network < handle
    properties
        num_layers = 0;
        
        input_block = [];
        label_block = [];
        
        label_patch = [];
        
        test_block = [];
        test_labels = [];
        
        
        pF = cell(0,1);
        F = cell(0,1);
        W = cell(0,1);
        dW = cell(0,1);
        lW = cell(0,1);
        
        B = cell(0,1);
        dB = cell(0,1);
        lB = cell(0,1);
        
        f = cell(0,1);
        df = cell(0,1);
        
        block_size = 0;
        
        train_order = [];
        figure_handle = [];
        axis_handle = [];
        
        Ef = @(x,t) (0);
        dEf = @(x,t) (0);
        E = 0;
        
        old_net = [];
        in_params = [];
        
        shrinkage = [];
        
    end
    methods
        
        function cnn = conv_neural_network(feature_maps, varargin)
%             nn_size, lambda, sigma, non_lin, error_fun
            p = inputParser;    
            p.addRequired('feature_maps', @isnumeric);
            p.addParameter('lambda', .01, @(x) (isnumeric(x) || iscell(x)));
            p.addParameter('Blambda', .01, @(x) (isnumeric(x) || iscell(x)));
            p.addParameter('sigma', .01, @(x) (isnumeric(x) || iscell(x)));
            p.addParameter('non_lin', 'logsig', @(x) (ischar(x) || iscell(x)));
            p.addParameter('error_fun', 'square', @ischar);
            p.addParameter('filter_size', [5 5], @(x) (isnumeric(x) || iscell(x)));
    
            p.parse(feature_maps, varargin{:});
    
            s = p.Results;
                        
            cnn.in_params = {feature_maps, varargin{:}};
            
            cnn.num_layers = length(feature_maps);
            
            if ischar(s.non_lin)
                nlin = cell(cnn.num_layers,1);
                nlin{1} = 'linear';
                for l = 2:cnn.num_layers
                    nlin{l} = s.non_lin;
                end
                s.non_lin = nlin;
            end
            
            if size(s.filter_size,1) == 1
                s.filter_size = ones(cnn.num_layers-1,1)*s.filter_size;
            end
            
            if ~iscell(s.filter_size)
                fs = cell(1, cnn.num_layers-1);
                for l = 1:cnn.num_layers-1
                    fs{l} = s.filter_size(l,:);
                end
                s.filter_size = fs;
            end
            
            weight_size = cell(1,cnn.num_layers-1);
            for l = 1:cnn.num_layers-1
                weight_size{l} = [s.filter_size{l} feature_maps(l) feature_maps(l+1)];
            end
            
            
            if length(s.lambda) == 1
                s.lambda = s.lambda*ones(1, cnn.num_layers);
            end
            
            if ~iscell(s.lambda)
                lambda = cell(1, cnn.num_layers);
                for l = 1:cnn.num_layers-1
                    lambda{l} = s.lambda(l)*ones(weight_size{l});
                end
                s.lambda = lambda;
            end
            
            
            if length(s.Blambda) == 1
                s.Blambda = s.Blambda*ones(1, cnn.num_layers);
            end
            
            if ~iscell(s.Blambda)
                Blambda = cell(1, cnn.num_layers);
                for l = 1:cnn.num_layers
                    Blambda{l} = s.Blambda(l)*ones(1,feature_maps(l));
                end
                s.Blambda = Blambda;
            end
            
            
            if length(s.sigma) == 1
                s.sigma = s.sigma*ones(1, cnn.num_layers);
            end
            
            nl = cnn.num_layers;
            cnn.pF = cell(nl,1);
            cnn.F = cell(nl,1);
            
            cnn.W = cell(nl-1,1);
            cnn.dW = cell(nl-1,1);
            cnn.lW = cell(nl-1,1);

            cnn.B = cell(nl,1);
            cnn.dB = cell(nl,1);
            cnn.lB = cell(nl,1);

            cnn.f = cell(nl,1);
            cnn.df = cell(nl,1);
            
            cnn.shrinkage = [0 0];
            
            for l = 2:cnn.num_layers
                cnn.W{l-1} = randn(weight_size{l-1}) * s.sigma(l-1);
                cnn.lW{l-1} = s.lambda{l-1};
                cnn.shrinkage = cnn.shrinkage + [weight_size{l-1}(1) weight_size{l-1}(2)] - 1;
                
                cnn.B{l} = randn(1,weight_size{l-1}(end))*s.sigma(l);
                cnn.lB{l} = s.Blambda{l};
                
                if strcmp(s.non_lin{l}, 'linear')
                    cnn.f{l} = @(x) (x);
                    cnn.df{l} = @(x,y) (1);
                elseif strcmp(s.non_lin{l}, 'logsig')
                    cnn.f{l} = @(x) (1./(1 + exp(-x)));
                    cnn.df{l} = @(x,y) ( y .* (1-y) );
                elseif strcmp(s.non_lin{l}, 'tanh')
                    cnn.f{l} = @(x) (tanh(x));
                    cnn.df{l} = @(x,y) ((1 - y.^2));
                end
                
            end
            
            if strcmp(s.error_fun, 'square')
                cnn.Ef = @(x,t) ( (x - t).^2 );
                cnn.dEf = @(x,t) ( 2*(x - t) );
            end
        end
        
        
        
        
        
        function set_input_block(cnn, input_block, label_block, test_block, test_labels, mirror_data)
            if ~exist('mirror_data', 'var') || isempty(mirror_data)
                mirror_data = false;
            end
            
            if ~exist('label_block', 'var')
                label_block = [];
            end
                        
            if exist('test_block','var')
                cnn.test_block = test_block;
                cnn.test_labels = test_labels;
            end
            
            if mirror_data
                offset = cnn.shrinkage/2;
                input_block = cat(1, input_block(offset(1):-1:1,:,:,:), input_block, input_block(end:-1:end-offset(1)+1,:,:,:));
                input_block = cat(2, input_block(:, offset(2):-1:1,:,:), input_block, input_block(:, end:-1:end-offset(2)+1,:,:));
            end
            cnn.input_block = input_block;
            
            cnn.block_size = size(input_block,3);
            
            if mirror_data && ~isempty(label_block)
                cnn.label_block = zeros(size(label_block,1) + cnn.shrinkage(1), ...
                    size(label_block,2) + cnn.shrinkage(2), ...
                    size(label_block,3), size(label_block,4));
                cnn.label_block(offset(1)+1:end-offset(1), offset(2)+1:end-offset(2), :, :) = label_block;
            else
                cnn.label_block = label_block;
            end
            
        end
        
        
        function forward(cnn)
            for l = 1:cnn.num_layers-1
                cnn.pF{l+1} = zeros(size(cnn.F{l},1)-size(cnn.W{l},1)+1, ...
                    size(cnn.F{l},2)-size(cnn.W{l},2)+1, ...
                    size(cnn.F{l},3), ...
                    size(cnn.W{l},4));
                for postmap = 1:size(cnn.W{l},4)
                    for premap = 1:size(cnn.W{l},3)
                        for sample = 1:size(cnn.F{l},3)
                            cnn.pF{l+1}(:,:,sample, postmap) = cnn.pF{l+1}(:,:,sample, postmap) + ...
                                conv2(cnn.F{l}(:,:,sample, premap ), cnn.W{l}(:,:,premap,postmap), 'valid');
                        end
                    end
                    cnn.pF{l+1}(:,:,:,postmap) = cnn.pF{l+1}(:,:,:,postmap) + ...
                        cnn.B{l+1}(postmap);                        
                end
                cnn.F{l+1} = cnn.f{l+1}(cnn.pF{l+1});
            end            
        end
        
        function backward(cnn)
            
            cnn.E = cnn.Ef(cnn.F{end}, cnn.label_patch);
            dEdF = cnn.dEf(cnn.F{end}, cnn.label_patch);
            
            for l = cnn.num_layers:-1:2
                cnn.dB{l} = dEdF.*cnn.df{l}(cnn.pF{l},cnn.F{l});                                
                
                if l > 2
                
                    dEdF = zeros(size(cnn.F{l-1})); %dEdF of layer l-1
                    for premap = 1:size(cnn.W{l-1},3)
                        for postmap = 1:size(cnn.W{l-1},4)
                            for sample = 1:size(cnn.F{l},3)
                                dEdF(:,:,sample, premap ) = dEdF(:,:,sample, premap) + ...
                                    xcorr2(cnn.dB{l}(:,:,sample, postmap), cnn.W{l-1}(:,:,premap,postmap));
                            end
                        end                                          
                    end
                
                end
                
            end
                            
        end
        
        function gradient(cnn)
            for l = 1:cnn.num_layers-1
                cnn.dW{l} = zeros(size(cnn.W{l}));
                for premap = 1:size(cnn.W{l},3)
                    for postmap = 1:size(cnn.W{l},4)
                        for sample = 1:size(cnn.F{l},3)
                            cnn.dW{l}(:,:,premap, postmap) = cnn.dW{l}(:,:,premap, postmap) + ...
                                rot90(conv2(cnn.F{l}(:,:,sample,premap), cnn.dB{l+1}(end:-1:1,end:-1:1,sample, postmap), 'valid'),2);                            
                        end
                    end                                          
                end                
            end
        end
        
        function update(cnn)
            for l = 2:cnn.num_layers;
                cnn.W{l-1} = cnn.W{l-1} - cnn.dW{l-1}.*cnn.lW{l-1};
                
                for k = 1:length(cnn.B{l})
                    cnn.B{l}(k) = cnn.B{l}(k) - cnn.lB{l}(k) * sum(sum(sum(cnn.dB{l}(:,:,:,k),1),2),3);
                end
            end
        end
        
        function prep_run(cnn, out_size, num_samples)
            if ~exist('out_size','var') || isempty(out_size)
                in_size = [size(cnn.input_block,1) size(cnn.input_block,2)];
                
                out_size = in_size - cnn.shrinkage;
                
                num_samples = size(cnn.input_block,3);
                
                sample_x = ones(1,num_samples);
                sample_y = ones(1,num_samples);
                sample_z = 1:num_samples;
                
            else
                if ~exist('num_samples','var') || isempty(num_samples)
                    num_samples = 1;
                end
                
                in_size = out_size + cnn.shrinkage;
                
                sample_x = randi(size(cnn.input_block,1)-in_size(1)+1, num_samples, 1);
                sample_y = randi(size(cnn.input_block,2)-in_size(2)+1, num_samples, 1);
                sample_z = randi(cnn.block_size, num_samples, 1);
            end
                
            cnn.F{1} = zeros([in_size, num_samples, size(cnn.W{1},3)]);
            for k = 1:num_samples
                cnn.F{1}(:,:,k,:) = cnn.input_block( ...
                    sample_x(k) + (0:in_size(1)-1), ...
                    sample_y(k) + (0:in_size(2)-1), ...
                    sample_z(k), :);
            end
            
            cnn.label_patch = zeros([out_size, num_samples, size(cnn.W{end},4)]);
            if ~isempty(cnn.label_block)
                for k = 1:num_samples
                    cnn.label_patch(:,:,k,:) = cnn.label_block( ...
                        sample_x(k) + (0:out_size(1)-1) + floor(cnn.shrinkage(1)/2), ...
                        sample_y(k) + (0:out_size(2)-1) + floor(cnn.shrinkage(2)/2), ...
                        sample_z(k), :);
                end
            end
        end
        
        function test_run(cnn)
            cnn.F{1} = cnn.test_block;
            cnn.label_patch = cnn.test_labels(floor(cnn.shrinkage(1)/2)+1:end-floor(cnn.shrinkage(1)/2), ...
                floor(cnn.shrinkage(2)/2)+1:end-floor(cnn.shrinkage(2)/2), :);
            cnn.forward;
            cnn.backward;
            
        end
        
        function [E test_E] = train(cnn, varargin)
            if ~isempty(varargin)
                num_runs = varargin{1};
            else
                num_runs = 1;
            end
            
            if length(varargin)>1
                batch_size = varargin{2};  
            else
                batch_size = [1 1 1];
            end
            
            if length(varargin)>2
                test_interval = varargin{3};
            else
                test_interval = 500;
            end
            
            c = colormap('Lines');
            
            E = zeros(num_runs,1);
            test_E = zeros(floor(num_runs/test_interval),1);
            
            if isempty(cnn.figure_handle)
                cnn.figure_handle = figure;
                cnn.axis_handle = gca;
                hold on
            end
            
            
            for t = 1:num_runs
                
                cnn.prep_run(batch_size(1:2), batch_size(3));

                cnn.forward();
                cnn.backward();
                cnn.gradient();
                cnn.update();
                E(t) = mean(cnn.E(:));
                
                
                if mod(t,test_interval) == 0 && ~isempty(cnn.test_block)
                    cnn.test_run;
                    test_E(t/test_interval) = mean(cnn.E(:));
                    
                    scatter(cnn.axis_handle, t, mean(E(t+(-test_interval+1:0))), 'MarkerEdgeColor', c(1,:));                    
                    scatter(cnn.axis_handle, t, test_E(t/test_interval), 'MarkerEdgeColor', c(2,:));                    
                end
                
                
                
                drawnow
                
                
                
            end
        end
        
        function [E test_E] = train_constrain(cnn, varargin)
            if ~isempty(varargin)
                num_runs = varargin{1};
            else
                num_runs = 1;
            end
            
            if length(varargin)>1
                batch_size = varargin{2};  
            else
                batch_size = [1 1 1];
            end
            
            if length(varargin)>2
                test_interval = varargin{3};
            else
                test_interval = 500;
            end
            
            c = colormap('Lines');
            
            E = zeros(num_runs,1);
            test_E = zeros(floor(num_runs/test_interval),1);
            
            if isempty(cnn.figure_handle)
                cnn.figure_handle = figure;
                cnn.axis_handle = gca;
                hold on
            end
            
            
            for t = 1:num_runs
                
                cnn.prep_run(batch_size(1:2), batch_size(3));

                cnn.forward();
                cnn.backward();
                cnn.gradient();
                cnn.update();
                
                cnn = constrain_net(cnn, varargin{4}, varargin{5});
                
                E(t) = mean(cnn.E(:));
                
                
                if mod(t,test_interval) == 0 && ~isempty(cnn.test_block)
                    cnn.test_run;
                    test_E(t/test_interval) = mean(cnn.E(:));
                    
                    scatter(cnn.axis_handle, t, mean(E(t+(-test_interval+1:0))), 'MarkerEdgeColor', c(1,:));                    
                    scatter(cnn.axis_handle, t, test_E(t/test_interval), 'MarkerEdgeColor', c(2,:));                    
                end
                
                
                
                drawnow
                
                
                
            end
        end
        
        function new_cnn = add_layer(cnn, new_layer_size)
            if ~exist('new_layer_size', 'var') || isempty(new_layer_size)
                new_layer_size = size(cnn.W{end},3);
            end
            
            p = cnn.in_params;
            
            p{1} = [p{1}(1:end-1) new_layer_size p{1}(end)];
            
            for k = 2:length(p)
                if length(p{k}) == cnn.num_layers && isnumeric(p{k})
                    p{k} = p{k}([1:end-1, end-1, end]);
                end
            end
            
            
            
            new_cnn = conv_neural_network(p{:});
            
            
            
            new_cnn.old_net = cnn;

            new_cnn.set_input_block(new_cnn.old_net.input_block, new_cnn.old_net.label_block);
            
            new_cnn.old_net.clear_net;
            
            
            
            for l = 1:cnn.num_layers-2
                new_cnn.W{l} = cnn.W{l};
                new_cnn.B{l+1} = cnn.B{l+1};
            end
            
            
        end
        
        function clear_net(cnn)
            cnn.F = cell(cnn.num_layers,1);
            cnn.pF = cell(cnn.num_layers,1);
            cnn.dB = cell(cnn.num_layers,1);
            cnn.input_block = [];
            cnn.label_block = [];
            cnn.test_block = [];
            cnn.test_labels = [];
            cnn.label_patch = [];
            
            
        end
    end
end
