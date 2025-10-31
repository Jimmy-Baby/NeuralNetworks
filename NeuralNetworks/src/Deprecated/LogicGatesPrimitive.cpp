// Precompiled headers
#include "Pch.h"

#include "LogicGatesPrimitive.h"
#include "Utility/Math.h"

namespace LogicGatesPrimitive
{
	namespace
	{
		using TSample = float[3];

		// Training data format: [input1, input2, expected_output]
		constexpr TSample GTrainingDataOr[] =
		{
			{ 0, 0, 0 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 1 },
		};

		constexpr TSample GTrainingDataAnd[] =
		{
			{ 0, 0, 0 },
			{ 1, 0, 0 },
			{ 0, 1, 0 },
			{ 1, 1, 1 },
		};

		constexpr TSample GTrainingDataNand[] =
		{
			{ 0, 0, 1 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 0 },
		};

		constexpr TSample GTrainingDataXor[] =
		{
			{ 0, 0, 0 },
			{ 1, 0, 1 },
			{ 0, 1, 1 },
			{ 1, 1, 0 },
		};

		const TSample* GTrainingData = GTrainingDataXor;
		constexpr size_t TRAINING_DATA_SIZE = std::size(GTrainingDataXor);
	}

	class XorModel
	{
	public:
		XorModel()
			: OrWeight1(Math::RandomFloat()),
			  OrWeight2(Math::RandomFloat()),
			  OrBias(Math::RandomFloat()),
			  NandWeight1(Math::RandomFloat()),
			  NandWeight2(Math::RandomFloat()),
			  NandBias(Math::RandomFloat()),
			  AndWeight1(Math::RandomFloat()),
			  AndWeight2(Math::RandomFloat()),
			  AndBias(Math::RandomFloat())
		{
		}

		[[nodiscard]] float ForwardToModel(const float x1, const float x2) const
		{
			// Forward propagation through the XOR network
			const float orActivation = Math::Sigmoid(x1 * OrWeight1 + x2 * OrWeight2 + OrBias);
			const float nandActivation = Math::Sigmoid(x1 * NandWeight1 + x2 * NandWeight2 + NandBias);

			const float modelPrediction = Math::Sigmoid(orActivation * AndWeight1 + nandActivation * AndWeight2 + AndBias);
			
			return modelPrediction;
		}

		void PrintWeightsAndBiases() const
		{
			printf("Layer 1 (OR):   w1=%.4f, w2=%.4f, b=%.4f\n", OrWeight1, OrWeight2, OrBias);
			printf("Layer 1 (NAND): w1=%.4f, w2=%.4f, b=%.4f\n", NandWeight1, NandWeight2, NandBias);
			printf("Layer 2 (AND):  w1=%.4f, w2=%.4f, b=%.4f\n", AndWeight1, AndWeight2, AndBias);
		}

		[[nodiscard]] float CalculateCost() const
		{
			float result = 0.0f;

			for (size_t index = 0; index < TRAINING_DATA_SIZE; ++index)
			{
				// Get the current training sample
				const float x1 = GTrainingData[index][0];
				const float x2 = GTrainingData[index][1];
				const float expectedOutput = GTrainingData[index][2];

				// Forward propagation
				const float orActivation = Math::Sigmoid(x1 * OrWeight1 + x2 * OrWeight2 + OrBias);
				const float nandActivation = Math::Sigmoid(x1 * NandWeight1 + x2 * NandWeight2 + NandBias);
				const float modelPrediction = Math::Sigmoid(orActivation * AndWeight1 + nandActivation * AndWeight2 + AndBias);

				// Calculate prediction error
				const float predictionError = modelPrediction - expectedOutput;

				// Accumulate squared error
				result += predictionError * predictionError;
			}

			// Return Mean Squared Error
			result /= static_cast<float>(TRAINING_DATA_SIZE);

			return result;
		}

		// Calculate gradients using finite differences and update weights
		void TrainFiniteDiff(const float learningRate = 1e-1f, const float epsilon = 1e-3f)
		{
			// Calculate base cost
			const float baseCost = CalculateCost();
			float savedValue;

			// Calculate gradients for OR gate weights using finite differences
			savedValue = OrWeight1;
			OrWeight1 += epsilon;
			const float orGradientWeight1 = (CalculateCost() - baseCost) / epsilon;
			OrWeight1 = savedValue;

			savedValue = OrWeight2;
			OrWeight2 += epsilon;
			const float orGradientWeight2 = (CalculateCost() - baseCost) / epsilon;
			OrWeight2 = savedValue;

			savedValue = OrBias;
			OrBias += epsilon;
			const float orGradientBias = (CalculateCost() - baseCost) / epsilon;
			OrBias = savedValue;

			// Calculate gradients for NAND gate weights
			savedValue = NandWeight1;
			NandWeight1 += epsilon;
			const float nandGradientWeight1 = (CalculateCost() - baseCost) / epsilon;
			NandWeight1 = savedValue;

			savedValue = NandWeight2;
			NandWeight2 += epsilon;
			const float nandGradientWeight2 = (CalculateCost() - baseCost) / epsilon;
			NandWeight2 = savedValue;

			savedValue = NandBias;
			NandBias += epsilon;
			const float nandGradientBias = (CalculateCost() - baseCost) / epsilon;
			NandBias = savedValue;

			// Calculate gradients for AND gate weights
			savedValue = AndWeight1;
			AndWeight1 += epsilon;
			const float andGradientWeight1 = (CalculateCost() - baseCost) / epsilon;
			AndWeight1 = savedValue;

			savedValue = AndWeight2;
			AndWeight2 += epsilon;
			const float andGradientWeight2 = (CalculateCost() - baseCost) / epsilon;
			AndWeight2 = savedValue;

			savedValue = AndBias;
			AndBias += epsilon;
			const float andGradientBias = (CalculateCost() - baseCost) / epsilon;
			AndBias = savedValue;

			// Update weights and biases using gradient descent
			OrWeight1 -= learningRate * orGradientWeight1;
			OrWeight2 -= learningRate * orGradientWeight2;
			OrBias -= learningRate * orGradientBias;
			NandWeight1 -= learningRate * nandGradientWeight1;
			NandWeight2 -= learningRate * nandGradientWeight2;
			NandBias -= learningRate * nandGradientBias;
			AndWeight1 -= learningRate * andGradientWeight1;
			AndWeight2 -= learningRate * andGradientWeight2;
			AndBias -= learningRate * andGradientBias;
		}

	private:
		// Layer 1: OR gate parameters
		float OrWeight1;
		float OrWeight2;
		float OrBias;
		
		// Layer 1: NAND gate parameters
		float NandWeight1;
		float NandWeight2;
		float NandBias;
		
		// Layer 2: AND gate parameters
		float AndWeight1;
		float AndWeight2;
		float AndBias;
	};

	void Run()
	{
		XorModel model;

		printf("Initial Cost = %.6f\n", model.CalculateCost());
		printf("------------------------------\n");

		// Training hyperparameters
		constexpr size_t maxIterations = 100000;
		constexpr float learningRate = 1.0f;
		constexpr float epsilon = 1e-3f;
		constexpr float convergenceThreshold = 1e-4f;

		// Training loop using finite difference gradient approximation
		for (size_t iteration = 0; iteration < maxIterations; ++iteration)
		{
			// Train one step
			model.TrainFiniteDiff(learningRate, epsilon);

			// Calculate the cost after updating
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
		printf("Final Cost = %.6f\n", model.CalculateCost());
		printf("------------------------------\n");
		printf("Model Parameters:\n");
		model.PrintWeightsAndBiases();
		printf("------------------------------\n");

		printf("XOR Truth Table Results:\n");
		printf("Input1 | Input2 | Predicted | Expected | Error\n");
		printf("-------|--------|-----------|----------|-------\n");

		for (size_t index = 0; index < TRAINING_DATA_SIZE; ++index)
		{
			const float x1 = GTrainingData[index][0];
			const float x2 = GTrainingData[index][1];
			const float expected = GTrainingData[index][2];
			const float predicted = model.ForwardToModel(x1, x2);
			const float error = std::abs(predicted - expected);

			printf("  %.0f    |   %.0f    |   %.4f    |   %.0f      | %.4f\n", 
				x1, x2, predicted, expected, error);
		}
	}
}
