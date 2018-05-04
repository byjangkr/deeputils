if(isdir(DNNDir))
    rmdir(DNNDir,'s');
end
mkdir(DNNDir);

if(~isdir(segFeatDir))
    error('error!!! not exist directory... check inDir!!');
end

% read group list file
gOrder = 0;
gfid = fopen(gListFile);
gline = fgetl(gfid);
while ischar(gline)
    gOrder = gOrder + 1;
    gList(gOrder,:) = regexp(gline,'\s+','split');
    gline = fgetl(gfid);
end
fclose(gfid);
clear gfid gline

% create output directory for train/test using DNN
if gOrder < 1,
    error('error!!! group is not divided for train/test ');
end

for nDir=1:gOrder
    mkdir(sprintf('%s%d',DNNDir,nDir));    
end
clear nDir

numDirFile = [DNNDir 'numDir'];
dfid = fopen(numDirFile,'w');
fprintf(dfid,'%d',gOrder);
fclose(dfid);
clear dfid numDirFile 
