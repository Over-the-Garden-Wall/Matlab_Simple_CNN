function cnn = backprop_cnn_2d(cnn, label)

    if ~all(size(cnn.F{end}) == size(label))
        error('label is the wrong size');
    end

    cnn.E = cnn.Ef(cnn.F{end}, label);    
    
    nl = cnn.num_layers;
    
    
    
    for l = nl-1:-1:1
        
        if l == nl-1
            cnn.dEdB{l} = cnn.dEf(cnn.F{l+1}, label);            
        else

            cnn.dEdB{l} = zeros(size(cnn.F{l+1}));

            for pm = 1:size(cnn.F{l+1}, 3)
                for nm = 1:size(cnn.F{l+2}, 3)
                    cnn.dEdB{l}(:,:,pm) = cnn.dEdB{l}(:,:,pm) + ...
                        convn(cnn.dEdB{l+1}(:,:,nm), cnn.W{l+1}(end:-1:1,end:-1:1,pm, nm), 'full');
                end
            end

        end
        
        if any(cnn.max_pooling(l,:) > 1)
            pp = zeros([prod(cnn.max_pooling(l,:)) size(cnn.dEdB{l})]);
            for k = 1:prod(cnn.max_pooling(l,:))
                is_me = cnn.max_pick{l} == k;
                pp(k,is_me) = cnn.dEdB{l}(is_me);
            end
            pp = permute(pp, [2 3 4 1]);
            pp = reshape(pp, ...
                [size(cnn.dEdB{l},1) size(cnn.dEdB{l},2) size(cnn.dEdB{l},3) cnn.max_pooling(l,:)]);
            pp = permute(pp, [4 1 5 2 3]);
            pp = reshape(pp, size(cnn.pF{l+1}));
            
            cnn.dEdB{l} = pp .* cnn.df{l}(cnn.pF{l+1}, cnn.fpF{l+1});

        else
            cnn.dEdB{l} = cnn.dEdB{l} .* cnn.df{l}(cnn.pF{l+1}, cnn.F{l+1});            
        end
        
    end
    
    
    for l = 1:nl-1
        cnn.dEdW{l} = zeros(size(cnn.W{l}));
        
        for pm = 1:size(cnn.F{l}, 3)
            for nm = 1:size(cnn.F{l+1}, 3)

                %this is the backprop bottleneck
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
        