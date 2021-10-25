using System.Collections.Generic;
using System.Linq;

namespace DotNetCraft.NeuralNetwork
{
    public static class MyExtensions
    {
        public static double[] NormalizeData(this IEnumerable<double> data, int min, int max)
        {
            double dataMax = data.Max();
            double dataMin = data.Min();
            double range = dataMax - dataMin;

            return data
                .Select(d => (d - dataMin) / range)
                .Select(n => (1 - n) * min + n * max)
                .ToArray();
        }

        public static int MaxIndex(this double[] data)
        {
            var max = data[0];
            var maxIndex = 0;
            for (int i = 1; i < data.Length; i++)
            {
                if (max < data[i])
                {
                    max = data[i];
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }
    }
}
