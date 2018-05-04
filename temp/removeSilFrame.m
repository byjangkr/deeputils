    %% Remove silence frame
    function output = removeSilFrame(input)
    inputD = input;
    sizePara = size(inputD,2);
    for i=1:sizePara
        data = inputD(i).prob;
        [maxVal,maxId] = max(data,[],2);
        inputD(i).nonSilProb = data(maxId~=1,:);        

    end
    output = inputD;

    end
