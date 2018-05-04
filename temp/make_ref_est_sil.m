function outref = make_ref_est_sil(scorematfile,uttinx,feat,rubric,engInx) 

load(scorematfile);
offset = 3; 

if length(rubric)>1,
    scoreAll = [score(:,offset+rubric(1)) score(:,offset+rubric(2)) score(:,offset+rubric(3))];
    meanscore= round(mean(scoreAll,2));
    target = [score(:,1:3) meanscore];
else
    target = [score(:,1:3) score(:,offset+rubric)];
end

infocel = regexp(deblank(uttinx),'.*(?<setInfo>[0-9]+)_(?<spkInfo>\w+)_(?<taskInfo>\w+)','tokens'); % extract speaker information

setnum = str2double(infocel{1}{1});
spkname = deblank(infocel{1}{2});
spknum = Map_SpkNum(spkname);
tasknum = str2double(infocel{1}{3});

tarinx = (target(:,1)==spknum & target(:,2)==setnum & target(:,3)==tasknum);

outref = [];
for i=1:size(feat,1)
%outref = repmat(target(tarinx,4),[1 data_size]);

    if feat(i,engInx) > 0.005,
        outref = [outref target(tarinx,4)];
    else
        outref = [outref 0];
    end

end

end