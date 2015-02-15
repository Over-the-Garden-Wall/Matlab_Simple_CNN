function cnn = backprop_cnn_2d(cnn, label)

    if ~all(size(cnn.F{end}) == size(label))
        error('label is the wrong size');
    end

    cnn.E = cnn.Ef(cnn.F{end}, label);    
    
    nl = cnn.num_layers;
    
    cnn.dEdB{end} = cnn.dEf(cnn.F{end}, label);
    
    
    for l = nl-2:-1:1
        
        if any(cnn.max_pooling(l+1,:) > 1)
            cnn.dEdB{l+1} = zeros([prod(cnn.max_pooling(l+1,:)) size(cnn.dEdB{l+1})]);
            for k = 1:prod(cnn.max_pooling(l+1,:))
                is_me = cnn.max_pick{l+1} == k;
                cnn.dEdB{l+1}(k,is_me) = cnn.dEdB{l+1}(is_me);
            end
            cnn.dEdB{l+1} = permute(dEdprepool, [2 3 4 1]);
            cnn.dEdB{l+1} = reshape(dEdprepool, ...
                [size(cnn.dEdB{l+1},1) size(cnn.dEdB{l+1},2) size(cnn.dEdB{l+1},3) cnn.max_pooling(l+1,:)]);
            cnn.dEdB{l+1} = permute(dEdprepool, [1 4 2 5 3]);

        end
        
        cnn.dEdB{l} = zeros(size(cnn.F{l+1}));
        
        for pm = 1:size(cnn.F{l+1}, 3)
            for nm = 1:size(cnn.F{l+2}, 3)
                cnn.dEdB{l}(:,:,pm) = cnn.dEdB{l}(:,:,pm) + ...
                    convn(cnn.dEdB{l+1}(:,:,nm), cnn.W{l+1}(end:-1:1,end:-1:1,pm, nm), 'full');
            end
        end
        
    end
    
    cnn.dEdB{l+1} = cnn.dEdB{l+1} .* cnn.df{l+1}(cnn.pF{l+2}, cnn.F{l+2});
    
    
    for l = 1:nl-1
        cnn.dEdW{l} = zeros(size(cnn.W{l}));
        
        for pm = 1:size(cnn.F{l}, 3)
            for nm = 1:size(cnn.F{l+1}, 3)
                cnn.dEdW{l}(:,:,pm, nm) = ...
                    convn(cnn.F{l}(end:-1:1,end:-1:1,pm), cnn.dEdB{l}(:, :, nm), 'valid');
            end
        end
        
    end
    
    for l = 1:nl - 1
        cnn.B{l} = cnn.B{l} - squeeze(sum(sum(cnn.dEdB{l})))' * cnn.Blambda(l);
        
        cnn.W{l} = cnn.W{l} - cnn.dEdW{l} * cnn.lambda(l);
    end

end
        