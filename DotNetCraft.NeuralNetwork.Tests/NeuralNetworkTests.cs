using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DotNetCraft.NeuralNetwork.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        private void GetSets(int count, int featureSize, int labelSize, out NeuralNetworkDataSet trainSet, out NeuralNetworkDataSet dataSet)
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
                for (int j = 0; j < labelSize; j++)
                {
                    labels[j] = random.NextDouble();
                }

                if (i < 8000)
                    trainSet.AddSample(features, labels);
                else
                    dataSet.AddSample(features, labels);
            }
        }

        private void GetIntegerSets(int count, int featureSize, int labelSize, out NeuralNetworkDataSet trainSet, out NeuralNetworkDataSet dataSet)
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
                for (int j = 0; j < labelSize; j++)
                {
                    labels[j] = random.NextDouble();
                }

                //Normalize Features and labels;
                //features = features.NormalizeData(0, 1);
                //labels = labels.NormalizeData(0, 1);

                Assert.IsFalse(features.Any(double.IsNaN));
                Assert.IsFalse(labels.Any(double.IsNaN));

                if (i < 8000)
                    trainSet.AddSample(features, labels);
                else
                    dataSet.AddSample(features, labels);
            }
        }

        [TestMethod]
        public void NeuralNetworkFullTest()
        {
            int featureSize = 4;
            int labelSize = 1;

            var functions = new Dictionary<ActivationFunctionType, IActivationFunction>
            {
                {ActivationFunctionType.None, new NoneActivationFunction()},
                {ActivationFunctionType.Sigmoid, new SigmoidActivationFunction()},
                {ActivationFunctionType.Tanh, new TanhActivationFunction()},
            };
            var activationFunctionFactory = new ActivationFunctionFactory(functions);
            var weightBuilder = new SimpleWeightBuilder();

            var network = new NeuralNetwork(activationFunctionFactory, weightBuilder);
            network.AddLayer(featureSize, ActivationFunctionType.Tanh);
            network.AddLayer(2* featureSize, ActivationFunctionType.Sigmoid);
            network.AddLayer(2* featureSize, ActivationFunctionType.Sigmoid);
            network.AddLayer(labelSize, ActivationFunctionType.Tanh);

            //GetSets(10000, featureSize, labelSize, out var trainSet, out var dataSet);
            GetIntegerSets(10000, featureSize, labelSize, out var trainSet, out var dataSet);

            network.Train(trainSet, 1000, 100);

            var goodPrediction = 0;
            for (int i = 0; i < dataSet.SampleCount; i++)
            {
                var features = dataSet.GetFeatures(i);
                var labels = dataSet.GetLabels(i);

                var result = network.Calculate(features);

                var delta = Math.Abs(result[0] - labels[0]);
                if (delta < 0.0001)
                    goodPrediction++;
            }

            var prediction = 100 * (goodPrediction / (double)dataSet.SampleCount);
            Debug.WriteLine($"Prediction: {prediction:F3}; Total: {dataSet.SampleCount}; Good: {goodPrediction}");
        }
    }
}
