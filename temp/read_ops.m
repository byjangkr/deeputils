function [out_data,out_featid] = read_ops(filename)


    [fid, message]= fopen(filename); % file open
    if(fid == -1)
        disp(message);
        disp(filename);
    end

    fprintf('read file : %s \n',filename);

    data=[];
    firstline = fgetl(fid); % The first line is feature index
    indexStr = regexp(firstline, ';', 'split');
    featid = indexStr(3:end);
    
    tline = fgets(fid);
    while ischar(tline)
        splitdata = regexp(tline, ';', 'split');
        framedata = [];
        for i=1:length(splitdata)
            framedata = [framedata str2double(splitdata{i})];
        end
        data = [data; framedata(3:end)];
        
        tline = fgets(fid);
    end
    
    out_data = data;
    out_featid = featid;
    fclose(fid);




end