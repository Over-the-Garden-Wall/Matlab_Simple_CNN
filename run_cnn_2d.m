function cnn = run_cnn_2d(cnn, input_image)

    %check if the input is inefficiently sized and crop it if so.
    out_size = size(input_image);
    out_size = floor((out_size(1:2) - cnn.min_input_size) ./ cnn.inc_input_size);
    %out_size is actually the output size - 1
    
    if any(out_size ~= floor(out_size))
        in_size = out_size.*cnn.inc_input_size + cnn.min_input_size;
        
        input_image = input_image(1:in_size(1), 1:in_size(2), :);
    end

    %set the input!
    cnn.F{1} = input_image;
    
    %actual run
    for l = 1:cnn.num_layers-1
%         disp(l)
        
        inSz = size(cnn.F{l});
        filter_size = size(cnn.W{l});
        
        pFsize = [(inSz(1:2) - filter_size(1:2) + 1), size(cnn.W{l}, 3)];
        cnn.pF{l+1} = zeros(pFsize);

        %convolution
        for pm = 1:size(cnn.F{l}, 3)
            for nm = 1:size(cnn.pF{l+1}, 3)
                %this line should be far and away the most expensive
                cnn.pF{l+1}(:,:, nm) = cnn.pF{l+1}(:,:, nm) + convn(cnn.F{l}(:,:,pm), cnn.W{l}(:,:,pm,nm), 'valid');
            end
            cnn.pF{l+1}(:,:, nm) = cnn.pF{l+1}(:,:, nm) + cnn.B{l}(nm);
        end
        
        cnn.F{l+1} = cnn.f{l}(cnn.pF{l+1});
        
        if any(cnn.max_pooling(l,:) > 1);
            cnn.F{l+1} = reshape(cnn.F{l+1}, ...
                [pFsize(1) / cnn.max_pooling(l, 1), cnn.max_pooling(l, 1), ...
                pFsize(2) / cnn.max_pooling(l, 2), cnn.max_pooling(l, 2), ...
                pFsize(3)]);
            cnn.F{l+1} = permute(cnn.F{l+1}, [1 3 5 2 4]);
            cnn.F{l+1} = cnn.F{l+1}(:,:,:,:);
            cnn.F{l+1} = max(cnn.F{l+1}, [], 4);
        end
    end
end