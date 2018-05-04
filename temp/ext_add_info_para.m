    %% Extract the additive information
    function output = ext_add_info_para(input)
    inputD = input;
    sizePara = size(inputD,2);
    for i=1:sizePara
        % name information
        segStr = regexp(inputD(i).uttinx, '_', 'split');
        inputD(i).gender = deblank(segStr{1});
        inputD(i).set = str2double(segStr{2});
        inputD(i).spkname = deblank(segStr{3});
        inputD(i).task = str2double(segStr{4});
    
        % phone index - 0 : consonants, 1 : vowels, 2 : silence
%         tmpcel = deblank(mat2cell(inputD(i).phnary,ones(1,size(inputD(i).phnary,1))));

%         phnid=zeros(size(inputD(i).phnary,1),1); % 
%         for j=1:size(vowels,2)
%             phnid = phnid + strcmpi(tmpcel,vowels{1,j});    
%         end
%         phnid = phnid + 2*strcmpi(tmpcel,silence);
%     
%         inputD(i).phnid = phnid;
    
    end
    output = inputD;
    % clear phnid sizePara inputD segStr tmpcel i j
    end
