// Precompiled headers
#include "Pch.h"

#include "LogicGatesMatrices.h"
#include "Utility/Layer.h"

namespace LogicGatesMatrices
{
	namespace
	{
		// Training data format: [input1, input2, expected_output]
		const Matrix GTrainingDataOr
		({
			{ 0, 0, 0 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 1 },
		});

		const Matrix GTrainingDataAnd
		({
			{ 0, 0, 0 },
			{ 1, 0, 0 },
			{ 0, 1, 0 },
			{ 1, 1, 1 },
		});

		const Matrix GTrainingDataNand
		({
			{ 0, 0, 1 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 0 },
		});

		const Matrix GTrainingDataXor
		({
			{ 0, 0, 0 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 0 },
		});

		const Matrix& GTrainingData = GTrainingDataXor;
	}

	class XorModel
	{
	public:
		explicit XorModel(const bool randomise = true)
		{
			// XOR requires a hidden layer (not linearly separable)
			// Architecture: 2 inputs -> 2 hidden neurons -> 1 output
			HiddenLayers.emplace_back(2, 2, randomise);
			OutputLayer = Layer(HiddenLayers.back(), 1, randomise);
		}

		float ForwardToModel(const Matrix& inputs)
		{
			// Forward propagation through the network
			if (!HiddenLayers.empty())
			{
				HiddenLayers[0].Activate(inputs);

				for (size_t hiddenLayerIndex = 1; hiddenLayerIndex < HiddenLayers.size(); ++hiddenLayerIndex)
				{
					HiddenLayers[hiddenLayerIndex].Activate(HiddenLayers[hiddenLayerIndex - 1]);
				}

				OutputLayer.Activate(HiddenLayers.back());
			}

			return OutputLayer.GetActivations().At(0, 0);
		}

		void Print() const
		{
			printf("Hidden Layers:\n");
			for (size_t i = 0; i < HiddenLayers.size(); ++i)
			{
				printf("  Layer %llu:\n", i);
				HiddenLayers[i].PrintValues();
			}

			printf("Output Layer:\n");
			OutputLayer.PrintValues();
		}

		float CalculateCost() const
		{
			float result = 0.0f;
			const size_t sampleCount = GTrainingData.GetRowCount();

			for (size_t index = 0; index < sampleCount; ++index)
			{
				// Get the current training sample inputs
				const float x1 = GTrainingData.At(index, 0);
				const float x2 = GTrainingData.At(index, 1);
				const Matrix trainingSample({ { x1, x2 } });

				// Use the model to evaluate an output (need to cast away const for forward pass)
				const float modelPrediction = const_cast<XorModel*>(this)->ForwardToModel(trainingSample);

				// Get expected output
				const float expectedOutput = GTrainingData.At(index, 2);

				// Calculate prediction error
				const float predictionError = modelPrediction - expectedOutput;

				// Accumulate squared error
				result += predictionError * predictionError;
			}

			// Return Mean Squared Error
			result /= static_cast<float>(sampleCount);

			return result;
		}

		void Learn(const XorModel& gradients, const float learningRate = 1e-1f)
		{
			// Update hidden layer weights and biases
			for (size_t layerIndex = 0; layerIndex < HiddenLayers.size(); ++layerIndex)
			{
				Layer& currentLayer = HiddenLayers[layerIndex];
				const Layer& gradientLayer = gradients.HiddenLayers[layerIndex];

				// Update weights
				for (size_t rowIndex = 0; rowIndex < currentLayer.GetWeights().GetRowCount(); ++rowIndex)
				{
					for (size_t columnIndex = 0; columnIndex < currentLayer.GetWeights().GetColumnCount(); ++columnIndex)
					{
						float& currentWeight = currentLayer.GetWeights().At(rowIndex, columnIndex);
						const float gradient = gradientLayer.GetWeights().At(rowIndex, columnIndex);
						currentWeight -= learningRate * gradient;
					}
				}

				// Update biases
				for (size_t biasIndex = 0; biasIndex < currentLayer.GetBiases().GetColumnCount(); ++biasIndex)
				{
					float& currentBias = currentLayer.GetBiases().At(0, biasIndex);
					const float gradient = gradientLayer.GetBiases().At(0, biasIndex);
					currentBias -= learningRate * gradient;
				}
			}

			// Update output layer weights
			for (size_t rowIndex = 0; rowIndex < OutputLayer.GetWeights().GetRowCount(); ++rowIndex)
			{
				for (size_t columnIndex = 0; columnIndex < OutputLayer.GetWeights().GetColumnCount(); ++columnIndex)
				{
					float& currentWeight = OutputLayer.GetWeights().At(rowIndex, columnIndex);
					const float gradient = gradients.OutputLayer.GetWeights().At(rowIndex, columnIndex);
					currentWeight -= learningRate * gradient;
				}
			}

			// Update output layer biases
			for (size_t biasIndex = 0; biasIndex < OutputLayer.GetBiases().GetColumnCount(); ++biasIndex)
			{
				float& currentBias = OutputLayer.GetBiases().At(0, biasIndex);
				const float gradient = gradients.OutputLayer.GetBiases().At(0, biasIndex);
				currentBias -= learningRate * gradient;
			}
		}

		// Calculate gradients using finite differences (numerical approximation)
		void CalculateGradientsFiniteDiff(XorModel& gradients, const float epsilon = 1e-3f)
		{
			const float baseCost = CalculateCost();
			float savedValue;

			// Process hidden layers
			for (size_t layerIndex = 0; layerIndex < HiddenLayers.size(); ++layerIndex)
			{
				Layer& currentLayer = HiddenLayers[layerIndex];
				Layer& gradientLayer = gradients.HiddenLayers[layerIndex];

				// Calculate gradients for weights
				for (size_t rowIndex = 0; rowIndex < currentLayer.GetWeights().GetRowCount(); ++rowIndex)
				{
					for (size_t columnIndex = 0; columnIndex < currentLayer.GetWeights().GetColumnCount(); ++columnIndex)
					{
						float& currentWeight = currentLayer.GetWeights().At(rowIndex, columnIndex);
						savedValue = currentWeight;
						currentWeight += epsilon;
						gradientLayer.GetWeights().At(rowIndex, columnIndex) = (CalculateCost() - baseCost) / epsilon;
						currentWeight = savedValue;
					}
				}

				// Calculate gradients for biases
				for (size_t biasIndex = 0; biasIndex < currentLayer.GetBiases().GetColumnCount(); ++biasIndex)
				{
					float& currentBias = currentLayer.GetBiases().At(0, biasIndex);
					savedValue = currentBias;
					currentBias += epsilon;
					gradientLayer.GetBiases().At(0, biasIndex) = (CalculateCost() - baseCost) / epsilon;
					currentBias = savedValue;
				}
			}

			// Process output layer weights
			for (size_t rowIndex = 0; rowIndex < OutputLayer.GetWeights().GetRowCount(); ++rowIndex)
			{
				for (size_t columnIndex = 0; columnIndex < OutputLayer.GetWeights().GetColumnCount(); ++columnIndex)
				{
					float& currentWeight = OutputLayer.GetWeights().At(rowIndex, columnIndex);
					savedValue = currentWeight;
					currentWeight += epsilon;
					gradients.OutputLayer.GetWeights().At(rowIndex, columnIndex) = (CalculateCost() - baseCost) / epsilon;
					currentWeight = savedValue;
				}
			}

			// Process output layer biases
			for (size_t biasIndex = 0; biasIndex < OutputLayer.GetBiases().GetColumnCount(); ++biasIndex)
			{
				float& currentBias = OutputLayer.GetBiases().At(0, biasIndex);
				savedValue = currentBias;
				currentBias += epsilon;
				gradients.OutputLayer.GetBiases().At(0, biasIndex) = (CalculateCost() - baseCost) / epsilon;
				currentBias = savedValue;
			}
		}

	private:
		std::vector<Layer> HiddenLayers;
		Layer OutputLayer;
	};

	void Run()
	{
		// Create our neural network model
		XorModel model;

		printf("Starting Cost (MSE) = %f\n", model.CalculateCost());
		printf("------------------------------\n");

		// Training hyperparameters
		constexpr size_t maxIterations = 100000;
		constexpr float learningRate = 1.0f;
		constexpr float epsilon = 1e-3f;
		constexpr float convergenceThreshold = 1e-4f;

		// Training loop using finite difference gradient approximation
		for (size_t iteration = 0; iteration < maxIterations; ++iteration)
		{
			// Calculate gradients using finite differences
			XorModel gradients(false);
			model.CalculateGradientsFiniteDiff(gradients, epsilon);
			
			// Update model parameters
			model.Learn(gradients, learningRate);

			// Calculate the new cost after adjusting
			const float currentCost = model.CalculateCost();

			// Print progress (every 1000 iterations to reduce output, plus early iterations)
			if (iteration % 1000 == 0 || iteration < 10)
			{
				printf("Iteration %llu: Cost = %.6f\n", iteration + 1, currentCost);
			}

			// Check for convergence
			if (currentCost < convergenceThreshold)
			{
				printf("Converged at iteration %llu with cost %.6f\n", iteration + 1, currentCost);
				break;
			}
		}

		printf("------------------------------\n");
		printf("Final Cost (MSE) = %.6f\n", model.CalculateCost());
		printf("------------------------------\n");
		printf("Model Parameters:\n");
		model.Print();
		printf("------------------------------\n");

		printf("Results (XOR Truth Table):\n");
		printf("Input1 | Input2 | Predicted | Expected | Error\n");
		printf("-------|--------|-----------|----------|-------\n");

		for (size_t index = 0; index < GTrainingData.GetRowCount(); ++index)
		{
			const float x1 = GTrainingData.At(index, 0);
			const float x2 = GTrainingData.At(index, 1);
			const float expected = GTrainingData.At(index, 2);
			const float predicted = model.ForwardToModel(Matrix({ { x1, x2 } }));
			const float error = std::abs(predicted - expected);

			printf("  %.0f    |   %.0f    |   %.4f    |   %.0f      | %.4f\n", 
				x1, x2, predicted, expected, error);
		}
	}
}
