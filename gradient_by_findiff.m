function cnn = gradient_by_findiff(cnn, input_image, label, delta)
    %this is a simple function that finds the gradient by finite
    %difference. This is O(n^2) and thus much less efficient than backprop,
    %it should only be used for debugging or demonstration.
    if ~exist('delta','var') || isempty(delta)
        delta = .00001;
    end
    
    cnn = run_cnn_2d(cnn, input_image);
    base_E = cnn.Ef(cnn.F{end}, label);
    base_E = sum(base_E(:));
    
    for l = 1:cnn.num_layers-1;
        cnn.dEdW{l} = zeros(size(cnn.W{l}));
        
        for k = 1:numel(cnn.dEdW{l});
            cnn.W{l}(k) = cnn.W{l}(k) + delta;
            cnn = run_cnn_2d(cnn, input_image);
            new_E = cnn.Ef(cnn.F{end}, label);
            cnn.W{l}(k) = cnn.W{l}(k) - delta;
            cnn.dEdW{l}(k) = (new_E - base_E) / delta;
        end
    end
    
    for l = 1:cnn.num_layers-1;
        cnn.dEdB{l} = zeros(size(cnn.B{l}));
        
        for k = 1:numel(cnn.dEdB{l});
            cnn.B{l}(k) = cnn.B{l}(k) + delta;
            cnn = run_cnn_2d(cnn, input_image);
            new_E = cnn.Ef(cnn.F{end}, label);
            cnn.B{l}(k) = cnn.B{l}(k) - delta;
            cnn.dEdB{l}(k) = (new_E - base_E) / delta;
        end
    end
    
end
            
            