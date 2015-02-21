function dEdBp = back_MLP_layer(dEdBn, W)
                
    dEdBp = squeeze(W) * squeeze(dEdBn);
    dEdBp = shiftdim(dEdBp, -2);

end
