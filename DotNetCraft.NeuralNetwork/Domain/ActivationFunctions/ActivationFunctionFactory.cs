using System;
using System.Collections.Generic;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.ActivationFunctions
{
    public class ActivationFunctionFactory: IActivationFunctionFactory
    {
        private readonly IDictionary<ActivationFunctionType, IActivationFunction> _activationFunctions;

        public ActivationFunctionFactory(IDictionary<ActivationFunctionType, IActivationFunction> activationFunctions)
        {
            if (activationFunctions == null) 
                throw new ArgumentNullException(nameof(activationFunctions));

            _activationFunctions = activationFunctions;
        }

        public IActivationFunction Build(ActivationFunctionType activationFunctionType)
        {
            return _activationFunctions[activationFunctionType];
        }
    }
}
