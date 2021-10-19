using System;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.ActivationFunctions
{
    public class SigmoidActivationFunction: IActivationFunction
    {
        #region Implementation of IActivationFunction

        public double Calculate(double x)
        {
            var result = 1 + Math.Exp(-x);
            result = 1 / result;
            return result;
        }

        #endregion
    }
}
