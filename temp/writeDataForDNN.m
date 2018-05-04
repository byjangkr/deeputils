function writeDataForDNN(inList,fileName,type)

if nargin < 3,
    type = 'nooptimum';
end

dataFile = fileName.dataFile;
labFile = fileName.labFile;
infoFile = fileName.infoFile;

data = [];
dCnt = 1;
lab = [];

ifid = fopen(infoFile,'w');
for i=1:size(inList,1)
    idata = inList(i).segFeat;
    data = [data; idata];    
    
    switch(type)
        case 'nooptimum' 
            lab = [lab; repmat(inList(i).uttScore,size(idata,1),1)];
        case 'optimum' 
            lab = [lab; (inList(i).segLab)'];                       
    end
       
    fprintf(ifid,'%s %d %d\n',inList(i).uttInx,dCnt,(dCnt+size(idata,1)-1));
    dCnt = dCnt + size(idata,1);
end
fclose(ifid);

dfid = fopen(dataFile,'w');
for nLine = 1:size(data,1)
    fprintf(dfid,'%f ',data(nLine,:));
    fprintf(dfid,'\n');
end
fclose(dfid);

lfid = fopen(labFile,'w');

for j=1:size(lab,1)
   fprintf(lfid,'%d\n',lab(j)); 
end

fclose('all');

end