cd Dataset/
folderlist = dir();
folderlist = folderlist(3:end)
N = 60;
image_id = 1;
for i=1:size(folderlist):
    cd (folderlist(i).name);
    filelist = dir('*.jpg');
    for j=1:60%size(filelist)
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
%disp(size(features));
first = 1;
features = res;
temp_features = features(1:size(features,1), 1:64);
[idx, CC] = kmeans(temp_features,1);
database_centroid = CC;
A = features(1:size(features,1), 65);
temp_features = [temp_features, A];
res = temp_features(idx == 1,:);
graph = res;
layer_size = size(res,1);
%disp(layer_size);
count = 1;
last = (4*(4^2 - 1))/3 + 1;
i = 2;
while i<=last
    parent_index = int32((i-1)/4) + 1;
    if(parent_index == 1)
        start = 1;
    else
        start = layer_size(parent_index-1) + 1;
    end
    end_ = layer_size(parent_index);
    features = graph(start:end_, 1:65);
    A = features(1:size(features,1), 65);
    temp_features = features(1:size(features,1), 1:64);
    [idx, CC] = kmeans(temp_features,4);
    temp_features = [temp_features,A];
    for k=1:4
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
          if(first == 1)
              W = log10((N/count));
              first = 2;
          else
              W = [W, log10((N/count))];
          end
          i = i+1;
    end
end
    
%database image index
database = zeros(N,22);
first = 1;
for i=1:21
    if(first == 1)
        start = 1;
        first = 2;
    else
        start = layer_size(i-1) + 1;
    end
    last = layer_size(i);
    temp_features = graph(start:last, 65);
    for j=1:size(temp_features,1)
        database(temp_features(j),i) = database(temp_features(j),i)+ W(i);
        database(temp_features(j),22) = temp_features(j);
    end
end
%query vector
cd E:/Dataset/
folderlist = dir();
N = 12;
image_id = 1;
query_vector = zeros(N,21);
N = 60;
for i=3:3%size(folderlist)
    cd (folderlist(i).name);
    filelist = dir('*.jpg');
    for j=61:72%size(filelist)
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
            ind = [4*index-2, 4*index-1, 4*index, 4*index+1];
            level = 2;
            while (level<=3)
                D1  = sqrt(sum((features(k,1:64) - databse_centroid(4*index-2, 1:21)) .^ 2));
                D2  = sqrt(sum((features(k,1:64) - databse_centroid(4*index-1, 1:21)) .^ 2));
                D3  = sqrt(sum((features(k,1:64) - databse_centroid(4*index, 1:21)) .^ 2));
                D4  = sqrt(sum((features(k,1:64) - databse_centroid(4*index+1, 1:21)) .^ 2));
                
                [m,i_] = min(D1, D2, D3, D4);
                query_vector(image_id, ind(i_,1)) = query_vector(image_id, ind(i_,1)) + W(ind(i_));
                index = ind(i_);
                ind = [4*index-2, 4*index-1, 4*index, 4*index+1];
                level = level + 1;
            end
        end
        for l=1:N
            D  = sqrt(sum((query_vector(image_id,:) - databse_centroid(l, 1:21)) .^ 2));
        end
        B = sort(D);
        fprintf('image id:- %d; match id:- %d, %d, %d, %d\n',image_id,B(1,22),B(2,22),B(3,22),B(4,22));
        image_id = image_id + 1;
    end
end
