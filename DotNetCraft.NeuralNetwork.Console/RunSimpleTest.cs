using System;
using System.Collections.Generic;
using System.Linq;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

namespace DotNetCraft.NeuralNetwork.Console
{
    public static class RunSimpleTest
    {
        private static void GetSets(int count, int featureSize, int labelSize, out NeuralNetworkDataSet trainSet, out NeuralNetworkDataSet dataSet)
        {
            var random = new Random();
            trainSet = new NeuralNetworkDataSet(8000, featureSize, labelSize);
            dataSet = new NeuralNetworkDataSet(2000, featureSize, labelSize);
            for (int i = 0; i < count; i++)
            {
                var features = new double[featureSize];
                for (int j = 0; j < featureSize; j++)
                {
                    features[j] = random.NextDouble();
                }

                var labels = new double[labelSize];
                int rndIndex = random.Next(0, labelSize);
                labels[rndIndex] = 1;

                var sum = labels.Sum();
                if (sum > 1)
                {
                    int k = 0;
                    k++;
                }

                if (i < 8000)
                    trainSet.AddSample(features, labels);
                else
                    dataSet.AddSample(features, labels);
            }
        }

        public static void Run()
        {
            int featureSize = 10;
            int labelSize = 5;

            var functions = new Dictionary<ActivationFunctionType, IActivationFunction>
            {
                {ActivationFunctionType.None, new NoneActivationFunction()},
                {ActivationFunctionType.Sigmoid, new SigmoidActivationFunction()},
                {ActivationFunctionType.Tanh, new TanhActivationFunction()},
            };
            var activationFunctionFactory = new ActivationFunctionFactory(functions);
            var weightBuilder = new SimpleWeightBuilder();

            var network = new NeuralNetwork(activationFunctionFactory, weightBuilder);
            network.AddLayer(featureSize, ActivationFunctionType.None);
            //network.AddLayer(2 * featureSize, ActivationFunctionType.Sigmoid);
            network.AddConvolutionalLayer(3, ActivationFunctionType.Sigmoid);
            network.AddLayer(4, ActivationFunctionType.Sigmoid);
            network.AddLayer(labelSize, ActivationFunctionType.Sigmoid);

            GetSets(10000, featureSize, labelSize, out var trainSet, out var dataSet);
            
            network.Train(trainSet, 1000, 100);

            var goodPrediction = 0;
            for (int i = 0; i < dataSet.SampleCount; i++)
            {
                var features = dataSet.GetFeatures(i);
                var labels = dataSet.GetLabels(i);

                var result = network.Calculate(features);

                var resIndex = result.MaxIndex();
                var labelIndex = labels.MaxIndex();

                if (resIndex == labelIndex)
                    goodPrediction++;
            }

            var prediction = 100 * (goodPrediction / (double)dataSet.SampleCount);
            System.Console.WriteLine($"Prediction: {prediction:F3}; Total: {dataSet.SampleCount}; Good: {goodPrediction}");
        }
    }
}
