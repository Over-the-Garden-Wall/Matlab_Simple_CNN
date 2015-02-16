%train an architecture like lenet-5 on the mnist dataset


    %training_parameters
    testing_frequency = 500;
    training_iterations = 10000;
    
    

    %make network
%     cnn = create_cnn([1 6 16 12 84 10], ...
%         'max_pooling', [2 2; 2 2; 1 1; 1 1], ...
%         'filter_size', [5 5; 5 5; 5 5; 1 1]);
    
    
    %read data
    test_img_raw = read_IDX('t10k-images.idx3-ubyte');
    test_lbl_raw = read_IDX('t10k-labels.idx1-ubyte');
    train_img_raw = read_IDX('train-images.idx3-ubyte');
    train_lbl_raw = read_IDX('train-labels.idx1-ubyte');
    
    %pad data (natively 28) and reorder dimensions for convenience, rescale
    %to [0 1] from [0 255]
    test_img_raw = permute(test_img_raw, [2 3 1]);
    test_img = zeros(size(test_img_raw) + [4 4 0]);
    test_img(3:end-2, 3:end-2, :) = test_img_raw;
    
    train_img_raw = permute(train_img_raw, [2 3 1]);
    train_img = zeros(size(train_img_raw) + [4 4 0]);
    train_img(3:end-2, 3:end-2, :) = train_img_raw;

    
    train_img = train_img/255;
    test_img = test_img/255;    
    
    

    
    
    train_error = zeros(training_iterations,1);
    test_error = zeros(floor(training_iterations / testing_frequency),1);

    tic    
    for t = 1:training_iterations

        image_pick = ceil(rand*size(train_img,3));
        
        cnn = run_cnn(cnn, train_img(:,:,image_pick));        

        lbl = zeros(10,1);
        lbl(train_lbl(image_pick)) = 1;
        cnn = backprop_cnn(cnn, lbl);
        
        train_error(t) = sum(cnn.E(:));
        
        if mod(t, testing_frequency) == 0
            for n = 1:size(test_img,3)
                cnn = run_cnn(cnn, test_img(:,:,n));        

                lbl = zeros(10,1);
                lbl(test_lbl(n)) = 1;
                
                E = cnn.Ef(cnn.F{end}, lbl);
                test_error(t/testing_frequency) = test_error(t/testing_frequency) + sum(E(:));
            end
            test_error(t/testing_frequency) = test_error(t/testing_frequency) / size(test_img,3);
           
            time_spent = toc;
            disp(['testing! ' num2str(t) ' iterations completed in ' toc ' seconds']);
        end
    end
      
    figure; hold all
    plot(train_error, 'lineWidth', 2);
    plot(testing_frequency:testing_frequency:training_iterations, test_error, 'lineWidth', 2);
    
    
        
        
    
    
    
    