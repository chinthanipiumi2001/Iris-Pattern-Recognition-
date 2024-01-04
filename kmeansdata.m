% Generate sample data
rng default;  % For reproducibility
X = [randn(100,2)*0.75+ones(100,2); randn(100,2)*0.5-ones(100,2); randn(100,2)*0.5+2*ones(100,2)];

% Implement K-means clustering
K = 3; % Number of clusters
[idx,C] = kmeans(X, K);

% Visualize the clustering
figure;
gscatter(X(:,1), X(:,2), idx,'rgb','osd');
hold on;
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3);
legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW');
title 'K-means Clustering';

% Evaluate different number of clusters using the Elbow method
distortions = zeros(1, 10);
for k = 1:10
    [~,C] = kmeans(X, k);
    D = pdist2(X, C, 'squaredeuclidean');
    distortions(k) = sum(min(D,[],2));
end

rng default;  % For reproducibility
X = [randn(100,2)*0.75+ones(100,2); randn(100,2)*0.5-ones(100,2); randn(100,2)*0.5+2*ones(100,2)];

K_values = 3:5; % Different K values
silhouette_means = zeros(size(K_values));

% Loop through different K values
for i = 1:length(K_values)
    K = K_values(i);
    idx = kmeans(X, K);
    figure;
    silhouette(X,idx);
    title(sprintf('Silhouette for K = %d', K));

    % Compute and store the mean silhouette value
    silhouette_values = silhouette(X, idx);
    silhouette_means(i) = mean(silhouette_values);
end