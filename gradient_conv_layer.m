function dEdW = gradient_conv_layer(F, dEdB)
            
    wSz = [size(F,1) - size(dEdB,1) + 1, size(F,2) - size(dEdB,2) + 1, size(F,3), size(dEdB,3)];
    dEdW = zeros(wSz);

    for pm = 1:wSz(3)
        for nm = 1:wSz(4)

            %this is the backprop bottleneck
            dEdW(:,:,pm, nm) = ...
                conv2(F(end:-1:1,end:-1:1,pm), dEdB(:, :, nm), 'valid');
        end
    end
    
end