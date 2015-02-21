function dEdW = gradient_fftconv_layer(F, dEdB)

    fftSz = [size(F,1) size(F,2)];
    wSz = [fftSz(1) - size(dEdB,1) + 1, fftSz(2) - size(dEdB,2) + 1, size(F,3), size(dEdB,3)];
 
   %compute ffts for F
    Ffft = zeros([fftSz, wSz(3)]);
    for pm = 1:wSz(3)
        Ffft(:,:,pm) = fft2(F(end:-1:1,end:-1:1,pm));
    end
    
    %compute ffts for dEdB
    dEdBfft = zeros([fftSz, wSz(4)]);
    dEdBfft(1:size(dEdB,1),1:size(dEdB,2),:) = dEdB;
    for nm = 1:wSz(4)
        dEdBfft(:,:,nm) = fft2(dEdBfft(:,:,nm));
    end
    
    
    dEdW = zeros([fftSz wSz(3) wSz(4)]);
    
    for pm = 1:wSz(3)
        for nm = 1:wSz(4)
            dEdW(:,:,pm, nm) = real(ifft2(dEdBfft(:,:,nm).*Ffft(:,:,pm)));
        end
    end
       
    dEdW = dEdW(end - wSz(1) + 1 : end, end - wSz(2) + 1 : end, :, :);
end