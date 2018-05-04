    %% Compute utterance level feature
    % Regression method : roulette wheel and counting with silence frame
    function output = computeUtterFeat(input)
    inputD = input;
    sizePara = size(inputD,2);
    for i=1:sizePara
        data = inputD(i).prob;
        
        cdf = zeros(size(data));
        cdf(:,1) = data(:,1);
        for j=2:size(data,2)
            cdf(:,j) = (cdf(:,j-1) + data(:,j))./sum(data,2);
             inputD(i).cdf = cdf;
        end
                               
        labCnt = zeros(1,size(cdf,2));
        for k=1:size(cdf,1)
            randN = rand(1);
            upper = (cdf(k,:)>=randN);
            lower = (cdf(k,:)<randN);
            label = diff(upper-lower)>0;
             if sum(label)==0, % for silProb
                label = [1 label];
             else
                label = [0 label];
             end
            labCnt = labCnt + label;            
        end

        clsData = labCnt(2:end);
        inputD(i).labCnt = clsData;        
        totalVal = sum(clsData);
        inputD(i).uttFeat = labCnt;
        score = 0;
        for j=1:size(clsData,2)
            score = score + j*clsData(j)/totalVal;
        end
        inputD(i).predScore= round(score);
        
    end
    output = inputD;

    end
