using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace DotNetCraft.NeuralNetwork
{
    public class NeuralNetworkDataSet
    {
        private readonly int _featureSize;
        private readonly int _labelSize;
        public List<double> Features { get; set; }
        public List<double> Labels { get; set; }
        public int SampleCount { get; private set; }

        public NeuralNetworkDataSet(int sampleCount, int featureSize, int labelSize)
        {
            _featureSize = featureSize;
            _labelSize = labelSize;
            Features = new List<double>(sampleCount * featureSize);
            Labels = new List<double>(sampleCount * labelSize);
        }

        public void AddSample(double[] sampleFeatures, double[] sampleLabels)
        {
            Features.AddRange(sampleFeatures);
            Labels.AddRange(sampleLabels);
            SampleCount++;
        }

        public double[] GetFeatures(int sampleIndex)
        {
            var position = sampleIndex * _featureSize;
            var result = Features.GetRange(position, _featureSize).ToArray();
            return result;
        }

        public double[] GetLabels(int sampleIndex)
        {
            var position = sampleIndex * _labelSize;
            var result = Labels.GetRange(position, _labelSize).ToArray();
            return result;
        }

        public void Normalize()
        {
            //var minResult = new double[_featureSize];
            //var maxResult = new double[_featureSize];

            //for (int i = 0; i < _featureSize; i++)
            //{
            //    maxResult[i] = int.MinValue;
            //    minResult[i] = int.MaxValue;
            //}

            //for (int index = 0; index < Features.Count; index += Features.Count)
            //{
            //    for (int i = 0; i < _featureSize; i++)
            //    {
            //        var item = Features[index + i];
            //        if (maxResult[i] < item)
            //            maxResult[i] = item;
            //        if (minResult[i] > item)
            //            minResult[i] = item;
            //    }
            //}

            //for (int index = 0; index < Features.Count; index += Features.Count)
            //{
            //    for (int i = 0; i < _featureSize; i++)
            //    {
            //        var item = Features[index + i];
            //        double range = maxResult[i] - minResult[i];
            //        if (range == 0)
            //            item = 0;
            //        else
            //            item = (item - range) / range;
            //        Features[index+i] = item;
            //    }
            //}

            Features = Features.NormalizeData(0, 1).ToList();
            Labels = Labels.NormalizeData(0, 1).ToList();

            Debug.Assert(Features.Any(double.IsNaN) == false);
            Debug.Assert(Labels.Any(double.IsNaN) == false);
        }
    }
}
