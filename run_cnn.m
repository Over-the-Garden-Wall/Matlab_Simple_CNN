function cnn = run_cnn(cnn, input_image)
    %runs a forward pass on the cnn using input_image as input

    nd = cnn.num_dims; %brevity

    if nd == 2
        cnn = run_cnn_2d(cnn, input_image);
    elseif nd == 3
        cnn = run_cnn_3d(cnn, input_image);
    else
        cnn = run_cnn_nd(cnn, input_image);
    end
end

    

    