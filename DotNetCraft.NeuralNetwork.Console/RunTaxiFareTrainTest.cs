using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
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
    public static class RunTaxiFareTrainTest
    {
        private class TaxiFareTrain
        {
            public string vendor_id { get; set; }
            public int rate_code { get; set; }
            public int passenger_count { get; set; }
            public double trip_time_in_secs { get; set; }
            public double trip_distance { get; set; }
            public string payment_type { get; set; }
            public double fare_amount { get; set; }
        }

        private static Dictionary<string, int> _strToInt = new Dictionary<string, int>();

        private static int ConvertToInt(string str)
        {
            if (_strToInt.TryGetValue(str, out var res))
                return res;

            var index = _strToInt.Count;
            _strToInt.Add(str, index);
            return index;
        }

        public static void Run()
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                Delimiter = ","
            };

            int featureSize = 6;
            int labelSize = 1;
            var samples = new List<MLModel1.ModelInput>();
            var sampleResults = new List<double>();
            NeuralNetworkDataSet trainSet = new NeuralNetworkDataSet(1, featureSize, labelSize);
            NeuralNetworkDataSet dataSet = new NeuralNetworkDataSet(1, featureSize, labelSize);
            using (var reader = new StreamReader(@"e:\Code\machinelearning-samples-main\samples\csharp\getting-started\Regression_TaxiFarePrediction\TaxiFarePrediction\Data\taxi-fare-train.csv"))
            {
                using (var csv = new CsvReader(reader, config))
                {
                    csv.Read();
                    csv.ReadHeader();
                    var features = new List<double>();
                    int index = 0;
                    while (csv.Read())
                    {
                        var record = csv.GetRecord<TaxiFareTrain>();

                        var feature = new double[featureSize];
                        feature[0] = ConvertToInt(record.vendor_id);
                        feature[1] = record.rate_code;
                        feature[2] = record.passenger_count;
                        feature[3] = record.trip_time_in_secs;
                        feature[4] = record.trip_distance;
                        feature[5] = ConvertToInt(record.payment_type);

                        var label = new double[labelSize];
                        label[0] = record.fare_amount;

                        if (index++ % 10 < 2)
                        {
                            dataSet.AddSample(feature, label);
                            samples.Add(new MLModel1.ModelInput
                            {
                                Vendor_id = record.vendor_id,
                                Rate_code = record.rate_code,
                                Passenger_count = record.passenger_count,
                                Trip_time_in_secs = (float)record.trip_time_in_secs,
                                Trip_distance = (float)record.trip_distance,
                                Payment_type = record.payment_type,
                            });
                            sampleResults.Add(record.fare_amount);
                        }
                        else
                        {
                            trainSet.AddSample(feature, label);
                        }
                    }
                }
            }

            trainSet.Normalize();
            dataSet.Normalize();

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
            network.AddLayer(4 * featureSize, ActivationFunctionType.Sigmoid);
            network.AddLayer(4 * featureSize, ActivationFunctionType.Sigmoid);
            
            //network.AddConvolutionalLayer(2, ActivationFunctionType.Sigmoid);
            //network.AddLayer(2*3, ActivationFunctionType.Sigmoid);

            network.AddLayer(labelSize, ActivationFunctionType.Sigmoid);

            network.Train(trainSet, 5, 1);

            var goodPrediction = 0;
            for (int i = 0; i < dataSet.SampleCount; i++)
            {
                var features = dataSet.GetFeatures(i);
                var labels = dataSet.GetLabels(i);
                var result = network.Calculate(features);
                if (Math.Abs(result[0] - labels[0]) <= 0.01)
                    goodPrediction++;
            }

            var prediction = 100 * (goodPrediction / (double)dataSet.SampleCount);
            System.Console.WriteLine($"Prediction: {prediction:F3}; Total: {dataSet.SampleCount}; Good: {goodPrediction}");
        }
    }
}
