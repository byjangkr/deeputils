function outList = storeRefScore(inList,scorematfile,rubric) 

inputList = inList;
load(scorematfile);
offset = 3; 


if length(rubric)>1,
    scoreAll = [score(:,offset+rubric(1)) score(:,offset+rubric(2)) score(:,offset+rubric(3))];
    meanscore= round(mean(scoreAll,2));
    target = [score(:,1:3) meanscore];
else
    target = [score(:,1:3) sum(score(:,offset+rubric),2)];
end


for i = 1:length(inputList)
    infocel = regexp(deblank(inputList(i).uttinx),'.*(?<setInfo>[0-9]+)_(?<spkInfo>\w+)_(?<taskInfo>\w+)','tokens'); % extract speaker information

    setnum = str2double(infocel{1}{1});
    spkname = deblank(infocel{1}{2});
    spknum = Map_SpkNum(spkname);
    tasknum = str2double(infocel{1}{3});

%     [spknum setnum tasknum]
    tarinx = (target(:,1)==spknum & target(:,2)==setnum & target(:,3)==tasknum);
%     find(tarinx==1)

    inputList(i).spkNum = spknum;
    inputList(i).refScore = target(tarinx,4);
end

outList = inputList;

end