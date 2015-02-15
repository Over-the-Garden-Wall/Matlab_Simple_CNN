function cnn = create_cnn_2d(feature_maps, varargin)



    %Lots of input handling
    p = inputParser;    
    p.addRequired('feature_maps', @isnumeric);
    p.addParameter('lambda', .01, @(x) (isnumeric(x) || iscell(x)));
    p.addParameter('Blambda', .01, @(x) (isnumeric(x) || iscell(x)));
    p.addParameter('sigma', .01, @(x) (isnumeric(x) || iscell(x)));
    p.addParameter('nonlinearity', 'logsig', @(x) (ischar(x) || iscell(x)));
    p.addParameter('error_fxn', 'square', @ischar);
    p.addParameter('filter_size', [5 5], @(x) (isnumeric(x) || iscell(x)));
    p.addParameter('max_pooling', [1 1], @(x) (isnumeric(x) || iscell(x)));

    p.parse(feature_maps, varargin{:});

    s = p.Results;

    cnn.in_params = s;

    cnn.num_layers = length(feature_maps);    
    
    if ischar(s.nonlinearity)
        nlin = cell(cnn.num_layers,1);
        nlin{1} = 'linear';
        for l = 2:cnn.num_layers
            nlin{l} = s.nonlinearity;
        end
        s.nonlinearity = nlin;
    end

    if size(s.filter_size,1) == 1
        s.filter_size = ones(cnn.num_layers-1,1)*s.filter_size;
    end

%     if ~iscell(s.filter_size)
%         fs = cell(1, cnn.num_layers-1);
%         for l = 1:cnn.num_layers-1
%             fs{l} = s.filter_size(l,:);
%         end
%         s.filter_size = fs;
%     end

    
    if size(s.max_pooling,1) == 1
        s.max_pooling = ones(cnn.num_layers-1,1)*s.max_pooling;
    end
    cnn.max_pooling = s.max_pooling;
    
%     if ~iscell(s.max_pooling)
%         mp = cell(1, cnn.num_layers-1);
%         for l = 1:cnn.num_layers-1
%             mp{l} = s.max_pooling(l,:);
%         end
%         s.max_pooling = mp;
%     end
    
%     weight_size = cell(1,cnn.num_layers-1);
%     for l = 1:cnn.num_layers-1
%         weight_size{l} = [s.filter_size(l,:) feature_maps(l) feature_maps(l+1)];
%     end


    if length(s.lambda) == 1
        s.lambda = s.lambda*ones(1, cnn.num_layers);
    end
    cnn.lambda = s.lambda;
%     if ~iscell(s.lambda)
%         lambda = cell(1, cnn.num_layers);
%         for l = 1:cnn.num_layers-1
%             lambda{l} = s.lambda(l)*ones(weight_size{l});
%         end
%         s.lambda = lambda;
%     end


    if length(s.Blambda) == 1
        s.Blambda = s.Blambda*ones(1, cnn.num_layers-1);
    end
    cnn.Blambda = s.Blambda;

%     if ~iscell(s.Blambda)
%         Blambda = cell(1, cnn.num_layers);
%         for l = 1:cnn.num_layers
%             Blambda{l} = s.Blambda(l)*ones(1,feature_maps(l));
%         end
%         s.Blambda = Blambda;
%     end


    if length(s.sigma) == 1
        s.sigma = s.sigma*ones(1, cnn.num_layers);
    end

    nl = cnn.num_layers;
    cnn.pF = cell(nl,1);
    cnn.F = cell(nl,1);

    cnn.W = cell(nl-1,1);
    cnn.dEdW = cell(nl-1,1);
    cnn.lW = cell(nl-1,1);

    cnn.B = cell(nl-1,1);
    cnn.dEdB = cell(nl-1,1);
    cnn.lB = cell(nl-1,1);

    cnn.f = cell(nl-1,1);
    cnn.df = cell(nl-1,1);

    cnn.num_dims = size(s.filter_size,2);
            
    
    %initialize weights
        
    for l = 1:nl-1
        cnn.W{l} = randn([s.filter_size(l,:), s.feature_maps(l), s.feature_maps(l+1)]) * s.sigma(l);
    end
    
    %initialize biases
    for l = 1:nl-1
        cnn.B{l} = randn(1, s.feature_maps(l+1)) * s.sigma(l);
    end

    %initialize nonlinearities
    for l = 1:nl-1
        
        if strcmp(s.nonlinearity{l}, 'linear')
            cnn.f{l} = @(x) (x);
            cnn.df{l} = @(x,y) (1);
        elseif strcmp(s.nonlinearity{l}, 'logsig')
            cnn.f{l} = @(x) (1./(1 + exp(-x)));
            cnn.df{l} = @(x,y) ( y .* (1-y) );
        elseif strcmp(s.nonlinearity{l}, 'tanh')
            cnn.f{l} = @(x) (tanh(x));
            cnn.df{l} = @(x,y) ((1 - y.^2));
        elseif strcmp(s.nonlinearity{l}, 'rectify')
            cnn.f{l} = @(x) (max(x,0));
            cnn.df{l} = @(x,y) (y > 0);       
        else
            error('unknown nonlinearity');
        end
        
    end
    
    if strcmp(s.error_fxn, 'square')
        cnn.Ef = @(x, t) ((x-t).^2);
        cnn.dEf = @(x, t) (2*(x-t));
    elseif strcmp(s.error_fxn, 'hinge')
        if strcmp(s.non_lin{end}, 'logsig')
            cnn.Ef = @(x, t) (max( abs(x-t) - .5, 0));
            cnn.dEf = @(x, t) (abs(x-t) > .5);
        else
            cnn.Ef = @(x, t) (max( abs(x-t), 0));
            cnn.dEf = @(x, t) (abs(x-t) > 0);
        end                
    else
        error('unknown error function');

    end

    
    %figure out input size
    out_size_1pix = ones(1,cnn.num_dims);
    out_size_2pix = 2*ones(1,cnn.num_dims);
    
    for l = nl-1:-1:1
        out_size_1pix = out_size_1pix .* s.max_pooling(l,:);
        out_size_1pix = out_size_1pix + s.filter_size(l,:) - 1;

        out_size_2pix = out_size_2pix .* s.max_pooling(l,:);
        out_size_2pix = out_size_2pix + s.filter_size(l,:) - 1;
    end
    
    cnn.min_input_size = out_size_1pix;
    cnn.inc_input_size = out_size_2pix - out_size_1pix;
        
    
end    