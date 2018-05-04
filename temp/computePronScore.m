    %% Compute pronunciation score
    % Regression method : score = sum(prob*score)
    function output = computePronScore(input)
    inputD = input;
    sizePara = size(inputD,2);
    for i=1:sizePara
        data = inputD(i).prob;
        %data = inputD(i).nonSilProb;
        clsData = sum(data);
        totalVal = sum(clsData);
        score = 0;
        for j=1:size(clsData,2)
            score = score + j*clsData(j)/totalVal;
        end
        inputD(i).clsData = clsData;
        inputD(i).predScore= round(score);
        

    end
    output = inputD;

    end
