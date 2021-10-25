using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using CsvHelper.Configuration;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.Synapses;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;
using MLModel1_ConsoleApp1;

namespace DotNetCraft.NeuralNetwork.Console
{
    public static class ForexTest
    {
        private class Candle
        {
            public DateTime DATE { get; set; }
            public double OPEN { get; set; }
            public double HIGH { get; set; }
            public double LOW { get; set; }
            public double CLOSE { get; set; }
            public int TICKVOL { get; set; }
            public int VOL { get; set; }
            public int SPREAD { get; set; }
        }

        public static void Run()
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                Delimiter = "\t"
            };

            var data = new List<Candle>();
            using (var reader = new StreamReader(@"e:\_ForexExports\EURUSD_M1_201801020900_201912312059.csv"))
            {
                using (var csv = new CsvReader(reader, config))
                {
                    csv.Read();
                    csv.ReadHeader();
                    var features = new List<double>();
                    int index = 0;
                    while (csv.Read())
                    {
                        var record = csv.GetRecord<Candle>();
                        data.Add(record);
                    }
                }
            }

            int inputSize = 40;
            int predictionSize = 20;

            int featureSize = 4 * inputSize;
            int labelSize = 1;
            NeuralNetworkDataSet trainSet = new NeuralNetworkDataSet(1, featureSize, labelSize);
            NeuralNetworkDataSet dataSet = new NeuralNetworkDataSet(1, featureSize, labelSize);
            int sampleIndex = 0;
            var singleSample = new List<Candle>();
            foreach (var candle in data)
            {
                singleSample.Add(candle);
                if (singleSample.Count < inputSize + predictionSize)
                    continue;

                var items = singleSample.Take(inputSize).ToList();
                var sampleFeature = new double[featureSize];

                var item = items[0];
                var openZero = item.OPEN;
                var closeZero = item.CLOSE;
                var highZero = item.HIGH;
                var lowZero = item.LOW;

                var closePrice = sampleFeature[0];
                for (var index = 0; index < items.Count; index++)
                {
                    item = items[index];
                    sampleFeature[4 * index] = GetPoints(item.OPEN - openZero);
                    sampleFeature[4 * index + 1] = GetPoints(item.CLOSE - closeZero);
                    sampleFeature[4 * index + 2] = GetPoints(item.HIGH - highZero);
                    sampleFeature[4 * index + 3] = GetPoints(item.LOW - lowZero);

                    closePrice = item.CLOSE;
                }
                

                var predictionData = singleSample.Skip(inputSize).ToList();
                var maxPrice = predictionData.Max(x => x.HIGH);
                var minPrice = predictionData.Min(x => x.LOW);


                var maxPts = GetPoints(maxPrice);
                var closePts = GetPoints(closePrice);
                var delta = maxPts - closePts;

                var sampleLabel = new double[labelSize];

                if (delta >= 20)
                    sampleLabel[0] = 1;
                //if (delta <= -40)
                //    sampleLabel[0] = 1;
                //else if (delta <= -20)
                //    sampleLabel[1] = 1;
                //else if (delta <= 0)
                //    sampleLabel[2] = 1;
                //else if (delta <= 20)
                //    sampleLabel[3] = 1;
                //else
                //    sampleLabel[4] = 1;


                sampleFeature = sampleFeature.NormalizeData(0, 1);

                if (sampleIndex++ % 10 < 4)
                {
                    dataSet.AddSample(sampleFeature, sampleLabel);
                }
                else
                {
                    trainSet.AddSample(sampleFeature, sampleLabel);
                }
                

                singleSample.Clear();

                //if (trainSet.SampleCount > 1000)
                  //  break;
            }
            
            //trainSet.Normalize();
            //dataSet.Normalize();


            var functions = new Dictionary<ActivationFunctionType, IActivationFunction>
            {
                {ActivationFunctionType.None, new NoneActivationFunction()},
                {ActivationFunctionType.Sigmoid, new SigmoidActivationFunction()},
                {ActivationFunctionType.Tanh, new TanhActivationFunction()},
            };
            var activationFunctionFactory = new ActivationFunctionFactory(functions);
            var weightBuilder = new SimpleWeightBuilder();

            Synapse.Alpha = 0.1;
            Synapse.E = 0.4;

            var network = new NeuralNetwork(activationFunctionFactory, weightBuilder);
            network.AddLayer(featureSize, ActivationFunctionType.Sigmoid);
            //network.AddLayer(2 * featureSize, ActivationFunctionType.Sigmoid);
            //network.AddLayer(2 * featureSize, ActivationFunctionType.Sigmoid);
            //network.AddLayer(2 * featureSize, ActivationFunctionType.Sigmoid);

            int window = 12;
            network.AddConvolutionalLayer(window, ActivationFunctionType.Sigmoid);
            network.AddLayer(featureSize / window, ActivationFunctionType.Sigmoid);

            network.AddLayer(labelSize, ActivationFunctionType.Sigmoid);

            for (int megaIndex = 0; megaIndex < 100; megaIndex++)
            {
                network.Train(trainSet, 500, 100);

                var correctPrediction = 0;
                var goodPrediction = 0;
                var badPrediction = 0;
                for (int i = 0; i < dataSet.SampleCount; i++)
                {
                    var features = dataSet.GetFeatures(i);
                    var labels = dataSet.GetLabels(i);
                    var result = network.Calculate(features);

                    //var labelIndex = labels.MaxIndex();
                    //var resultIndex = result.MaxIndex();
                    //if (labelIndex == resultIndex)

                    if (labels[0] >= 0.9)
                    {
                        if (result[0] >= 0.4)
                            goodPrediction++;
                    }
                    else
                    {
                        if (result[0] >= 0.4)
                            badPrediction++;
                    }

                    if (Math.Abs(labels[0] - result[0]) <= 0.4)
                    {
                        correctPrediction++;
                        //if (goodPrediction % 5 == 0)
                        //{
                        //    System.Console.WriteLine(
                        //        $"Label: {labels[0]:F3}; Result: {result[0]:F3} => {labels[0] - result[0]:F3}");
                        //}
                    }
                }

                var prediction = 100 * (correctPrediction / (double) dataSet.SampleCount);
                System.Console.WriteLine(
                    $"Prediction: {prediction:F3}; Total: {dataSet.SampleCount}; Good: {correctPrediction}; Best: {goodPrediction}; Bad: {badPrediction}");
            }
        }

        private static int GetPoints(double price)
        {
            //0.7500123-> 123
            // 10000000  10000
            //1.20054 ->  054
            // 100000 -> 10000
            const int points = 100000;
            int x = (int)((price * points) % 1000);
            return x;
        }
    }
}
