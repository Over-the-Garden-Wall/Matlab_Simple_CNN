function dEdBp = back_conv_layer(dEdBn, W)
                
    dEdBp = zeros(size(dEdBn,1) + size(W,1) - 1, size(dEdBn,2) + size(W,2) - 1, size(W,3));

                
    for pm = 1:size(W, 3)
        for nm = 1:size(W, 4)
            dEdBp(:,:,pm) = dEdBp(:,:,pm) + ...
                conv2(dEdBn(:,:,nm), W(end:-1:1,end:-1:1,pm, nm), 'full');
        end
    end
    
end