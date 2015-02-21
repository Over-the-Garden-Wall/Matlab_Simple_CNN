function dEdBp = back_fftconv_layer(dEdBn, W)
                

    inSz = [size(dEdBn,1) + size(W,1) - 1, size(dEdBn,2) + size(W,2) - 1, size(W,3)];
%     WSz = size(W);
    dEdBp = zeros(inSz);
                
    
    %precompute ffts
    fftdBn = zeros([inSz(1:2), size(dEdBn,3)]);
    fftdBn(1:size(dEdBn,1), 1:size(dEdBn,2), :) = dEdBn;        
    for nm = 1:size(W, 4)
        fftdBn(:,:,nm) = fft2(fftdBn(:,:,nm));        
    end
    
    for pm = 1:size(W, 3)
        for nm = 1:size(W, 4)

            padW = zeros(inSz(1), inSz(2));
            padW(1:size(W,1), 1:size(W,2)) = W(end:-1:1,end:-1:1,pm, nm);
            padW = fft2(padW);
            
            dEdBptemp = padW.*fftdBn(:,:,nm);
            dEdBptemp = real(ifft2(dEdBptemp));
            
            dEdBp(:,:,pm) = dEdBp(:,:, pm) + dEdBptemp; 
            
        end
    end
            
    
    
end