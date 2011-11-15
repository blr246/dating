function [ weights,ideal,nonideal,trainingExamples,labels ] = getWeights(N,numNegs,numTrainingExamples)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% it's a naive way of getting the weights, can be improved.
neg = -1 + rand(numNegs,1);
neg = neg/(-1 * sum(neg));

pos = rand(N-numNegs,1);
pos = pos/sum(pos);

for i = 1:N

weights = vertcat(pos,neg);
weights = weights(randperm(numel(weights)));
end

ideal = double(weights > 0);
nonideal = double(weights < 0);

trainingExamples = rand(numTrainingExamples,N);
labels = trainingExamples * weights;

end
