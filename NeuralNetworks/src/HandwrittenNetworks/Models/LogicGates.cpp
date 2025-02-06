#include "LogicGates.h"
#include "../CustomMath.h"

#include <cstdio>
#include <cstdlib>
#include <format>

namespace LogicGates
{
	using TSample = float[3];

	TSample g_TrainingDataOr[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 1 },
	};

	TSample g_TrainingDataAnd[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 1, 1, 1 },
	};

	TSample g_TrainingDataNand[] =
	{
		{ 0, 0, 1 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	};

	TSample g_TrainingDataXor[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	};

#define TRAINING_DATA g_TrainingDataXor

	TSample* g_TrainingData = TRAINING_DATA;
	constexpr size_t g_TrainingDataSize = std::size(TRAINING_DATA);

	float CalculateCost(const float Weight1, const float Weight2, const float Bias)
	{
		float Result = 0.0f;

		for (size_t Index = 0; Index < g_TrainingDataSize; ++Index)
		{
			// Get the current training sample
			const float TrainingSampleX = g_TrainingData[Index][0];
			const float TrainingSampleY = g_TrainingData[Index][1];

			// Use the model to evaluate an output
			const float ModelPrediction = Math::Sigmoid(TrainingSampleX * Weight1 + TrainingSampleY * Weight2 + Bias);

			// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
			const float PredictionError = ModelPrediction - g_TrainingData[Index][2];

			// Square the error to get a positive number, and add it to the result
			Result += PredictionError * PredictionError;
		}

		// Average out all the accumulated error
		Result /= g_TrainingDataSize;

		return Result;
	}

	// Expected formula:
	// y = x1 * 0.5 + 
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Set our starting weights
		float Weight1 = Math::RandomFloat();
		float Weight2 = Math::RandomFloat();
		float Bias = Math::RandomFloat();

		printf("Cost = %f, Weight1 = %f, Weight2 = %f, Bias = %f\n", CalculateCost(Weight1, Weight2, Bias), Weight1, Weight2, Bias);
		printf("------------------------------------------------\n");

#if 0
		// Finite Difference
		for (size_t Index = 0; Index < 1000; ++Index)
		{
			constexpr float RATE = 10;
			constexpr float EPSILON = 1e-3f;

			// Calculate cost
			const float Cost = CalculateCost(Weight1, Weight2, Bias);

			// Finite Difference -> f(a + h) - f(a)
			// Calculate The Wiggles(R)
			const float WiggleWeight1 = (CalculateCost(Weight1 + EPSILON, Weight2, Bias) - Cost) / EPSILON;
			const float WiggleWeight2 = (CalculateCost(Weight1, Weight2 + EPSILON, Bias) - Cost) / EPSILON;
			const float WiggleBias = (CalculateCost(Weight1, Weight2, Bias + EPSILON) - Cost) / EPSILON;

			// Adjust the weights and bias using The Wiggles(R)
			Weight1 -= RATE * WiggleWeight1;
			Weight2 -= RATE * WiggleWeight2;
			Bias -= RATE * WiggleBias;

			// Calculate the new cost after adjusting
			const float NewCost = CalculateCost(Weight1, Weight2, Bias);

			// Print results of this iteration
			printf("Cost = %f, Weight1 = %f, Weight2 = %f, Bias = %f\n", NewCost, Weight1, Weight2, Bias);

			if (std::format("{:.6f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}
#else
		// Gradient Descent
		for (size_t TrainIndex = 0; TrainIndex < 10000; ++TrainIndex)
		{
			constexpr float RATE = 1.0f;

			// Calculate cost
			const float Cost = CalculateCost(Weight1, Weight2, Bias);

			printf("Cost = %f, Weight1 = %f, Weight2 = %f, Bias = %f\n", Cost, Weight1, Weight2, Bias);

			// Gradient Descent
			// Calculate The Wiggles(R)
			float WiggleWeight1 = 0.0f, WiggleWeight2 = 0.0f, WiggleBias = 0.0f;

			for (size_t SampleIndex = 0; SampleIndex < g_TrainingDataSize; ++SampleIndex)
			{
				const float Xi = g_TrainingData[SampleIndex][0];
				const float Yi = g_TrainingData[SampleIndex][1];
				const float Zi = g_TrainingData[SampleIndex][2];

				const float Ai = Math::Sigmoid(Xi * Weight1 + Yi * Weight2 + Bias);
				const float ModelError = Ai - Zi;

				const float ScaledModelError = 2 * ModelError * Ai * (1 - Ai);

				WiggleWeight1 += ScaledModelError * Xi;
				WiggleWeight2 += ScaledModelError * Yi;
				WiggleBias += ScaledModelError;
			}

			WiggleWeight1 /= g_TrainingDataSize;
			WiggleWeight2 /= g_TrainingDataSize;
			WiggleBias /= g_TrainingDataSize;

			// Adjust the weights and bias using The Wiggles(R)
			Weight1 -= RATE * WiggleWeight1;
			Weight2 -= RATE * WiggleWeight2;
			Bias -= RATE * WiggleBias;

			// Calculate the new cost after adjusting
			const float NewCost = CalculateCost(Weight1, Weight2, Bias);

			// Print results of this iteration
			printf("Cost = %f, Weight1 = %f, Weight2 = %f, Bias = %f\n", NewCost, Weight1, Weight2, Bias);

			if (std::format("{:.6f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}
#endif

		printf("------------------------------\n");
		printf("Weight1 = %f, Weight2 = %f, Bias = %f\n", Weight1, Weight2, Bias);
		printf("------------------------------\n");
		printf("Results:\n");

		for (int Index = 0; Index < 4; ++Index)
		{
			const float X1 = g_TrainingData[Index][0];
			const float X2 = g_TrainingData[Index][1];
			const float W1 = Weight1;
			const float W2 = Weight2;
			const float B = Bias;

			const float Result = Math::Sigmoid(X1 * W1 + X2 * W2 + B);

			printf("{ %.0f | %.0f = %.2f }\n", X1, X2, Result);
		}
	}
}
