cd Dataset/
folderlist = dir();
branch = 4;
folder_num = 84 ;
per_fold = 60;
q_im = 72-per_fold;
N = per_fold * folder_num;
level = 4;
image_id = 1;
%extracting features
for i=3:(folder_num + 2)
    cd (folderlist(i).name);    
    filelist = dir('*.jpg');
    disp(i);
    for j=1:per_fold
        I = imread(filelist(j).name);
        I = rgb2gray(I);
        regions = detectMSERFeatures(I);
        [features] = extractFeatures(I, regions);
        A = zeros(size(features,1),1);
        A(1:size(features,1),1) = image_id;
        features = [features, A];
        if((i==3)&&(j==1))
            res = features;
        else
            res = [res; features];
        end
        image_id = image_id + 1;
    end
    cd (folderlist(2).name);
end
%making the first node of vocabulary tree
features = res;
temp_features = features(1:size(features,1), 1:64);
[idx, CC] = kmeans(temp_features,1);
database_centroid = CC;
A = features(1:size(features,1), 65);
temp_features = [temp_features, A];
res = temp_features(idx == 1,:);
graph = res;
layer_size = size(res,1);
count = 0;
last = (branch*((branch^level) - 1))/(branch - 1) + 1;
i = 2;
W = 1.000;
%build rest of the nodes in vocabulary tree
while (i<=last)
    disp(i);
    parent_index = int32((i-2)/branch) + 1;
    if(parent_index == 1)
        start = 1;
    else
        start = layer_size(parent_index-1) + 1;
    end
    end_ = layer_size(parent_index);
    features = graph(start:end_, 1:65);  
    A = features(1:size(features,1), 65);
    temp_features = features(1:size(features,1), 1:64);
    [idx, CC] = kmeans(temp_features, branch);
    temp_features = [temp_features,A];
    for k=1:branch
        database_centroid = [database_centroid; CC(k,:)];
          res = temp_features(idx == k,:);
          images = zeros(N);
          for m=1:size(res, 1)
              if(images(res(m,65)) == 0)
                  images(res(m,65)) = 1;
                  count = count + 1;
               end
          end
          graph = [graph; res];
          layer_size = [layer_size, layer_size(i-1) + size(res,1)];
          W = [W, log10((N/count))];
          count = 0;
          i = i+1;
    end
end
    
%database image index
database = zeros(N,last + 1);
first = 1;
for i=1:(last)
    if(first == 1)
        start = 1;
        first = 2;
    else
        start = layer_size(i-1) + 1;
    end
    last_ = layer_size(i);
    temp_features = graph(start:last_, 65);
    for j=1:size(temp_features,1)
        database(temp_features(j),i) = database(temp_features(j),i)+ W(i);
        database(temp_features(j),last+1) = temp_features(j);
    end
end
%query vector
cd Dataset/
folderlist = dir();
N =q_im*folder_num;
image_id = 1;
query_vector = zeros(N,last);
N = per_fold * folder_num;
match = 0;
m_= level+1;
for i=3:(folder_num+2)
    cd (folderlist(i).name);
    filelist = dir('*.jpg');
    for j=(per_fold+1):72
        I = imread(filelist(j).name);
        I = rgb2gray(I);
        regions = detectMSERFeatures(I);
        [features] = extractFeatures(I, regions);
        A = zeros(size(features,1),1);
        A(1:size(features,1),1) = image_id;
        features = [features, A];
        for k=1:size(features,1)
            level = 1;
            index = 1;
            query_vector(image_id,index) = query_vector(image_id,index) + W(index);
            ind = [branch*index-2, branch*index-1, branch*index, branch*index+1];
            level = 2;
            while (level<=m_)
                D1  = sqrt(sum((features(k,1:64) - database_centroid(branch*index-2, 1:64)) .^ 2));
                D2  = sqrt(sum((features(k,1:64) - database_centroid(branch*index-1, 1:64)) .^ 2));
                D3  = sqrt(sum((features(k,1:64) - database_centroid(branch*index, 1:64)) .^ 2));
                D4  = sqrt(sum((features(k,1:64) - database_centroid(branch*index+1, 1:64)) .^ 2));
               
                [m,i_] = min([D1, D2, D3, D4]); 
                query_vector(image_id, ind(i_)) = query_vector(image_id, ind(i_)) + W(ind(i_));
                index = ind(i_);
                ind = [branch*index-2, branch*index-1, branch*index, branch*index+1];
                level = level + 1;
            end
        end
        %check similarity
        for l=1:N
            if(l == 1)
                D  = sqrt(sum((query_vector(image_id,:) - database(l, 1:last)) .^ 2));
            else
                D = [D, sqrt(sum((query_vector(image_id,:) - database(l, 1:last)) .^ 2))];
            end
        end
        [B, I__] = sort(D(1,:));
        %print result
        fprintf('test image folder: %d :---match folder id:- %d, %d, %d, %d, %d\n',i,int32((I__(1)-1)/60)+3,int32((I__(2)-1)/60)+3,int32((I__(3)-1)/60)+3,int32((I__(4)-1)/60)+3, int32((I__(5)-1)/60)+3);
        if((i == int32(((I__(1)-1)/60)+3)) || (i == int32(((I__(2)-1)/60)+3)) || (i == int32(((I__(3)-1)/60)+3)) || (i == int32(((I__(4)-1)/60)+3)) || (i == int32(((I__(5)-1)/60)+3)))
            match = match + 1;
        end
        image_id = image_id + 1;
    end
    cd (folderlist(2).name);
end 