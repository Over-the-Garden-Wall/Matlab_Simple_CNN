function cnn = backprop_cnn_2d(cnn, label)

    if ~all(size(cnn.F{end}) == size(label))
        error('label is the wrong size');
    end

    cnn.E = cnn.Ef(cnn.F{end}, label);    
    
    nl = cnn.num_layers;
    
    cnn.dEdB{end} = cnn.dEf(cnn.F{end}, label) .* cnn.df{end}(cnn.pF{end}, cnn.F{end});
    
    for l = nl-2:-1:1
        cnn.dEdB{l} = zeros(size(cnn.F{l+1}));
        
        for pm = 1:size(cnn.F{l+1}, 3)
            for nm = 1:size(cnn.F{l+2}, 3)
                cnn.dEdB{l}(:,:,pm) = cnn.dEdB{l}(:,:,pm) + ...
                    convn(cnn.dEdB{l+1}(:,:,nm), cnn.W{l}(end:-1:1,end:-1:1,pm, nm), 'full');
            end
        end
        
    end
    
    for l = 1:nl-1
        cnn.dEdW{l} = zeros(size(cnn.W{l}));
        
        for pm = 1:size(cnn.F{l}, 3)
            for nm = 1:size(cnn.F{l+1}, 3)
                cnn.dEdW{l}(:,:,pm, nm) = ...
                    convn(cnn.F{l}(:,:,pm), cnn.dEdB{l}(end:-1:1,end:-1:1, nm), 'valid');
            end
        end
        
    end
    
    for l = 1:nl - 1
        for m = 1:length(cnn.B{l})
            cnn.B{l}(m) = cnn.B{l}(m) - sum(sum(cnn.dEdB{l})) * cnn.Blambda(l);
        end
        
        cnn.W{l} = cnn.W{l} - cnn.dEdW{l} * cnn.lamba(l);
    end

end
        