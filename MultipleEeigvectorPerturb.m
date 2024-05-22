function [estimated,keynodes]=MultipleEeigvectorPerturb(mat,nodeNumber,vectorNumber)

    maxDegree=max(sum(mat));

    [row,vol]=size(mat);
    [V,D]=eigs(mat,vectorNumber,'LA');
    for i=1:vectorNumber
        s=sum(V(:,i));
        V(:,i)=s*V(:,i);
        V(:,i)=V(:,i)/norm(V(:,i));
    end
    V=V.^2;
    V=V*maxDegree/2;    
    D=diag(D)';
    
    V2=V;
    keynodes=[];
    estimated=[];
    i=1;
    while i<=nodeNumber
        DD=repmat(D,row,1);
        DD=DD-V2;
        [C1,I1]=max(DD');
        [C2,I2]=min(C1);
        chosenNode=I2;
        contributionNew=max(D)-C2;
        V2(chosenNode,:)=0;
        D=D-V2(chosenNode,:);
        keynodes(end+1)=chosenNode;
        
        if length(keynodes)<=1
            continue
        end
        V2temp=V(keynodes,:);
        DD=repmat(D,size(V2temp,1),1);
        DD=DD+V2temp;
        [C3,I3]=max(DD');
        [C4,I4]=min(C3);
        chosenExistingNodeIndex=I4;
        chosenExistingNode=keynodes(chosenExistingNodeIndex);
        contributionOld=C4-max(D);
        if contributionOld<contributionNew && chosenNode~=chosenExistingNode
            keynodes(chosenExistingNodeIndex)=[];
            V2(chosenExistingNode,:)=V(chosenExistingNode,:);
            D=D+V2(chosenExistingNode,:);
            continue;
        end
        estimated(end+1)=max(D);
        i=i+1;

    end

end