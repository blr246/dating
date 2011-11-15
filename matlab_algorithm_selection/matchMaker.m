function [ Theta,candidate ] = matchMaker(trainingExamples,labels,lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(trainingExamples);
X = trainingExamples(:,1:n);

Y = labels;
regularizationMatrix = eye(size(X,2));
regularizationMatrix(1,1) = 0;
Theta = (X' * X + (lambda .* regularizationMatrix).^2) \ (X' * Y);
candidate = double(Theta > 0);
end


