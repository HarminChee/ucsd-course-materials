function [compressedSignal,D,compressionMatrix]=loading_data(address)
    load([address,'compressedSignal.mat']);
    load([address,'compressionMatrix.mat']);
    compressedSignal=double(compressedSignal);
    D_compressed=imread([address,'CompressedBasis.tiff']);
    D=double(D_compressed)./255.*0.1284-0.0525;
end