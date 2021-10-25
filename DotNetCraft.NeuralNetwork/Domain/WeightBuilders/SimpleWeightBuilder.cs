using System;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.WeightBuilders
{
    public class SimpleWeightBuilder: IWeightBuilder
    {
        private static readonly Random _random = new Random(0);

        #region Implementation of IWeightBuilder

        public double BuildWeight()
        {
            return _random.NextDouble();
        }

        public double[] BuildWeights(int count)
        {
            var result = new double[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = _random.NextDouble();
            }

            return result;
        }

        #endregion
    }
}
