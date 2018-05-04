    %% Compute pronunciation score
    % Regression method : roulette wheel and regression
    % modified 2016.03
    function output = computePronScore2(input)
    inputD = input;
    sizePara = size(inputD,2);
    for i=1:sizePara
        %data = inputD(i).nonSilProb;
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
            upper = find(cdf(k,:)>=randN);
            label = upper(1);
            labCnt(1,label) = labCnt(1,label) + 1;            
        end

        inputD(i).labCnt = labCnt;
        [C,I] = max(labCnt);
        inputD(i).predScore = I;
        
    end
    output = inputD;

    end
