function data = read_IDX(fn)

    fid = fopen(fn, 'r');
    fdata = fread(fid, 4, 'uint8');
    
    switch fdata(3)
        case 8
            readtype = 'uint8';
        case 9
            readtype = 'int8';
        case 11
            readtype = 'int16';
        case 12
            readtype = 'int32';
        case 13
            readtype = 'float32';
        case 14
            readtype = 'double';
    end
    numdims = fdata(4);
    
    dim_data = double(fread(fid, double(numdims)*4, 'uint8'));
    dim_data = reshape(dim_data, [4 numdims]);
    dim_data = sum(dim_data .* ((256.^(3:-1:0)')*ones(1,numdims)));
    
    data = fread(fid, prod(dim_data), readtype);
    data = double(data);
    if numdims > 1        
        data = reshape(data, dim_data(end:-1:1));
        data = permute(data, numdims:-1:1);
    end
    
    
end

    