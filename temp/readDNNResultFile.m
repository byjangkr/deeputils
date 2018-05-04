function outList = readDNNResultFile(predFile,infoFile)

% Read the result of DNN and save to list
% pronList is structure value
% uttinx : utterance index
% prob : probability matrix

pronList = [];
pronList_cnt = 1;
fid = fopen(predFile);
fprintf('read file : %s\n',predFile);
probMat = [];
tline = fgetl(fid);
while ischar(tline)
    buf = str2double(regexp(deblank(tline),'\s+','split'));
    probMat = [probMat; buf];
    tline = fgetl(fid); 
end
fclose(fid);
clear fid tline buf


fid = fopen(infoFile);
tline = fgetl(fid);
while ischar(tline)
    buf = regexp(deblank(tline),'\s+','split');
    uttinx = buf{1};
    beginx = str2double(buf{2});
    endinx = str2double(buf{3});
    
    pronList(pronList_cnt).uttinx = uttinx;
    pronList(pronList_cnt).prob = probMat(beginx:endinx,:);   
    
    tline = fgetl(fid);
    pronList_cnt = pronList_cnt + 1;
end
fclose(fid);

clear fid buf tline uttinx beginx endinx

outList = pronList;

end


